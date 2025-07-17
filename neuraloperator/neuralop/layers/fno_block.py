"""
FNO Block Implementation with Bregman Architecture Support

Based on NeuralOperator (https://github.com/neuraloperator/neuraloperator)
Original work Copyright (c) 2023 NeuralOperator developers

Modifications for Bregman architecture:
- Added architecture parameter with "bregman", "euclidean", and "standard" options
- Custom initialization schemes (init_std = 0 for Bregman networks)
- Modified forward passes for Bregman regularization

Modified by Abdel-Rahim Mezidi, 2024
Licensed under MIT License
"""

from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .mlp import MLP
from .normalization_layers import AdaIN
from .skip_connections import skip_connection
from .spectral_convolution import SpectralConv
from ..utils import validate_scaling_factor


Number = Union[int, float]


class FNOBlocks(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        n_layers=1,
        max_n_modes=None,
        fno_block_precision="full",
        use_mlp=False,
        mlp_dropout=0,
        mlp_expansion=0.5,
        non_linearity=F.gelu,
        non_linearity_inv=nn.Identity(),
        non_linearity_range= [-np.Inf, np.Inf],
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        use_fno_skip=True,
        architecture="standard",
        fno_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        SpectralConv=SpectralConv,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        fft_norm="forward",
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.n_dim = len(n_modes)

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.n_dim, n_layers)

        self.max_n_modes = max_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.architecture = architecture
        self.non_linearity = non_linearity
        self.non_linearity_inv = non_linearity_inv
        self.non_linearity_range = non_linearity_range
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.use_fno_skip = use_fno_skip
        self.mlp_skip = mlp_skip
        self.use_mlp = use_mlp
        self.mlp_expansion = mlp_expansion
        self.mlp_dropout = mlp_dropout
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.ada_in_features = ada_in_features
        
        if self.architecture in ["bregman"]:
            self.init_std = 0
        else:
            self.init_std = "auto"
        
        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            output_scaling_factor=output_scaling_factor,
            max_n_modes=max_n_modes,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
            init_std=self.init_std,
        )

        if use_fno_skip:   
            self.fno_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=fno_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            if self.architecture == 'bregman':
                for skip in self.fno_skips:
                    skip.weight.data.fill_(0)
        else:
            self.fno_skips = None

        if use_mlp:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * mlp_expansion),
                        dropout=mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.mlp_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=mlp_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.mlp = None

        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [
                    getattr(nn, f"InstanceNorm{self.n_dim}d")(
                        num_features=self.out_channels
                    )
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "batch_norm":
            self.norm = nn.ModuleList(
                [
                    getattr(nn, f"BatchNorm{self.n_dim}d")(
                        num_features=self.out_channels
                    )
                    for _ in range(n_layers * self.n_norms)
                ]
            )        
        # elif norm == 'layer_norm':
        #     self.norm = nn.ModuleList(
        #         [
        #             nn.LayerNorm(elementwise_affine=False)
        #             for _ in range(n_layers*self.n_norms)
        #         ]
        #     )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, layer_norm]"
            )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        if self.architecture == "bregman":
            x, x_skip = x
            x_skip = self.convs[index].transform(x_skip, output_shape=output_shape)
        elif self.architecture in ["euclidean", "residual_ffno"]:
            x_skip = self.convs[index].transform(x, output_shape=output_shape)
            
        if self.use_fno_skip:
            x_skip_fno = self.fno_skips[index](x)
            x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)
            
        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno
        if self.use_fno_skip:
            x = x + x_skip_fno
        if self.architecture in ["euclidean", "bregman"]:
            x = x + x_skip
            x_skip = x

        if (self.mlp is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            x = self.mlp[index](x) + x_skip_mlp

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)
                
        if self.architecture == "residual_ffno":
            x = x + x_skip

        if self.architecture == "bregman" and index < (
            self.n_layers - 1
        ):
            return (x, x_skip)
        else:
            return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.convs.n_modes = n_modes
        self._n_modes = n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)


class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)
