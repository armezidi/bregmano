import re
import warnings
from functools import partial
from math import prod
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

import wandb


# normalization, pointwise gaussian
class UnitGaussianNormalizer:
    def __init__(self, x, eps=0.00001, reduce_dim=[0], verbose=True):
        super().__init__()

        msg = ("neuralop.utils.UnitGaussianNormalizer has been deprecated. "
               "Please use the newer neuralop.datasets.UnitGaussianNormalizer instead.")
        warnings.warn(msg, DeprecationWarning)
        n_samples, *shape = x.shape
        self.sample_shape = shape
        self.verbose = verbose
        self.reduce_dim = reduce_dim

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, reduce_dim, keepdim=True).squeeze(0)
        self.std = torch.std(x, reduce_dim, keepdim=True).squeeze(0)
        self.eps = eps

        if verbose:
            print(
                f"UnitGaussianNormalizer init on {n_samples}, reducing over {reduce_dim}, samples of shape {shape}."
            )
            print(f"   Mean and std of shape {self.mean.shape}, eps={eps}")

    def encode(self, x):
        # x = x.view(-1, *self.sample_shape)
        x -= self.mean
        x /= self.std + self.eps
        # x = (x.view(-1, *self.sample_shape) - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps  # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:, sample_idx] + self.eps  # T*batch*n
                mean = self.mean[:, sample_idx]

        # x is in shape of batch*n or T*batch*n
        # x = (x.view(self.sample_shape) * std) + mean
        # x = x.view(-1, *self.sample_shape)
        x *= std
        x += mean

        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()
        return self

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()
        return self

    def to(self, device):
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def count_model_params(model):
    """Returns the total number of parameters of a PyTorch model
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    return sum(
        [p.numel() * 2 if p.is_complex() else p.numel() for p in model.parameters()]
    )

def count_tensor_params(tensor, dims=None):
    """Returns the number of parameters (elements) in a single tensor, optionally, along certain dimensions only

    Parameters
    ----------
    tensor : torch.tensor
    dims : int list or None, default is None
        if not None, the dimensions to consider when counting the number of parameters (elements)
    
    Notes
    -----
    One complex number is counted as two parameters (we count real and imaginary parts)'
    """
    if dims is None:
        dims = list(tensor.shape)
    else:
        dims = [tensor.shape[d] for d in dims]
    n_params = prod(dims)
    if tensor.is_complex():
        return 2*n_params
    return n_params


def wandb_login(api_key_file="../config/wandb_api_key.txt", key=None):
    if key is None:
        key = get_wandb_api_key(api_key_file)

    wandb.login(key=key)


def set_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        os.environ["WANDB_API_KEY"] = key.strip()


def get_wandb_api_key(api_key_file="../config/wandb_api_key.txt"):
    import os

    try:
        return os.environ["WANDB_API_KEY"]
    except KeyError:
        with open(api_key_file, "r") as f:
            key = f.read()
        return key.strip()


# Define the function to compute the spectrum
def spectrum_2d(signal, n_observations, normalize=True):
    """This function computes the spectrum of a 2D signal using the Fast Fourier Transform (FFT).

    Paramaters
    ----------
    signal : a tensor of shape (T * n_observations * n_observations)
        A 2D discretized signal represented as a 1D tensor with shape
        (T * n_observations * n_observations), where T is the number of time
        steps and n_observations is the spatial size of the signal.

        T can be any number of channels that we reshape into and
        n_observations * n_observations is the spatial resolution.
    n_observations: an integer
        Number of discretized points. Basically the resolution of the signal.

    Returns
    --------
    spectrum: a tensor
        A 1D tensor of shape (s,) representing the computed spectrum.
    """
    T = signal.shape[0]
    signal = signal.view(T, n_observations, n_observations)

    if normalize:
        signal = torch.fft.fft2(signal)
    else:
        signal = torch.fft.rfft2(
            signal, s=(n_observations, n_observations), normalized=False
        )

    # 2d wavenumbers following PyTorch fft convention
    k_max = n_observations // 2
    wavenumers = torch.cat(
        (
            torch.arange(start=0, end=k_max, step=1),
            torch.arange(start=-k_max, end=0, step=1),
        ),
        0,
    ).repeat(n_observations, 1)
    k_x = wavenumers.transpose(0, 1)
    k_y = wavenumers

    # Sum wavenumbers
    sum_k = torch.abs(k_x) + torch.abs(k_y)
    sum_k = sum_k

    # Remove symmetric components from wavenumbers
    index = -1.0 * torch.ones((n_observations, n_observations))
    k_max1 = k_max + 1
    index[0:k_max1, 0:k_max1] = sum_k[0:k_max1, 0:k_max1]

    spectrum = torch.zeros((T, n_observations))
    for j in range(1, n_observations + 1):
        ind = torch.where(index == j)
        spectrum[:, j - 1] = (signal[:, ind[0], ind[1]].sum(dim=1)).abs() ** 2

    spectrum = spectrum.mean(dim=0)
    return spectrum


Number = Union[float, int]


def validate_scaling_factor(
    scaling_factor: Union[None, Number, List[Number]],
    n_dim: int,
    n_layers: Optional[int] = None,
) -> Union[None, List[float], List[List[float]]]:
    """
    Parameters
    ----------
    scaling_factor : None OR float OR list[float]
    n_dim : int
    n_layers : int or None; defaults to None
        If None, return a single list (rather than a list of lists)
        with `factor` repeated `dim` times.
    """
    if scaling_factor is None:
        return None
    if isinstance(scaling_factor, (float, int)):
        if n_layers is None:
            return [float(scaling_factor)] * n_dim

        return [[float(scaling_factor)] * n_dim] * n_layers
    if (
        isinstance(scaling_factor, list)
        and len(scaling_factor) > 0
        and all([isinstance(s, (float, int)) for s in scaling_factor])
    ):
        return [[float(s)] * n_dim for s in scaling_factor]

    return None




"""
Definition of activation functions and their inverse (conjugate)

Enhanced activation function framework for Bregman Neural Operators

Based on NeuralOperator (https://github.com/neuraloperator/neuraloperator)
Original work Copyright (c) 2023 NeuralOperator developers

New additions:
- Comprehensive activation function library (relu, sigmoid, isru, tanh, etc.)
- Mathematical conjugate pairs for Bregman optimization
- Version-specific behavior for bregman/euclidean/standard architectures
- Range constraints with epsilon adjustments

Modified by Abdel-Rahim Mezidi, 2024
Licensed under MIT License
"""

eps = 1e-4


def isru(var_in):
    return torch.div(var_in, torch.sqrt(1 + var_in ** 2))


def isru_conjugate(var_in):
    return torch.div(var_in, torch.sqrt(1 - var_in ** 2))


def sigmoid_conjugate(var_in):
    return torch.log(torch.div(var_in, 1 + var_in))


def atanh(var_in):
    return torch.atanh(var_in)
    #return .5*(torch.log(1+var_in) - torch.log(1-var_in))
    #return .5 * (torch.log((1 + var_in) / (1 - var_in)))


def asinh(var_in):
    return torch.log(var_in + torch.sqrt(var_in ** 2 + 1))


def softplus_conjugate(var_in, beta=1, threshold=5):
    """
    Inverse of SoftPlus functional.
    - Option A: Standard implementation
    - Option B: Stable implementation inspired from the following TensorFlow code:
        [https://github.com/tensorflow/probability/blob/v0.15.0/tensorflow_probability/python/math/generic.py#L494-L545]
    """
    option = 'B'
    if option == 'A':
        return torch.log(torch.exp(beta * var_in) - 1) / beta
    else:
        # = (1 / beta) * torch.log(1 - torch.exp(-beta * var_in)) + var_in
        return torch.where(var_in * beta < np.exp(-threshold), torch.log(beta*var_in)/beta + var_in,
                           (1 / beta) * torch.log(1 - torch.exp(-beta * var_in)) + var_in)


def softplus(var_in, beta=1, threshold=10):
    """
    SoftPlus functional.
    - Option A: Standard implementation
    - Option B: Stable implementation without threshold from the following post
        [https://github.com/pytorch/pytorch/issues/31856]
    """
    option = 'B'
    if option == 'A':
        return torch.nn.functional.softplus(var_in, beta=beta, threshold=threshold)
    else:
        return - (1 / beta) * log_sigmoid(-beta * var_in)


def log_sigmoid(var_in):
    min_elem = torch.min(var_in, torch.zeros_like(var_in))
    z = torch.exp(min_elem) + torch.exp(min_elem - var_in)
    return min_elem - torch.log(z)


def asin(var_in, threshold=1e-5):
    return torch.where(var_in < -1+threshold, np.pi/2-np.sqrt(2)*torch.sqrt(var_in+1),
                       torch.where(var_in > 1-threshold, np.pi/2 + np.sqrt(2)*torch.sqrt(1-var_in),
                       torch.asin(var_in)))


def bent_identity(var_in, param=1.):
    return (var_in + torch.sqrt(var_in ** 2 + param)) / (2 * param)


def bent_identity_conjugate(var_in, param=1.):
    return param * var_in + 1 / var_in


def scaled_tanh(var_in,a=1.7159,b=0.6666):
    return a*torch.tanh(b*var_in)

def scaled_atanh(var_in,a=1.7159,b=0.6666):
    return torch.atanh(var_in/a)/b

class Zeros(nn.Module):
    def __init__(self):
        super(Zeros, self).__init__()

    @staticmethod
    def forward(var_input):
        return torch.zeros_like(var_input)


def get_non_linearity(activation_name, version='standard'):
    """ Get the couple (activation/offset) for Bregman, Euclidean and Standard neural networks """

    if activation_name == 'relu':
        activation = nn.ReLU()
        smooth_activation = bent_identity
        smooth_offset = bent_identity_conjugate
        v_range = [0, np.Inf]
    elif activation_name == 'sigmoid':
        activation = nn.Sigmoid()
        smooth_activation = nn.Sigmoid()
        smooth_offset = sigmoid_conjugate
        v_range = [0, 1]
    elif activation_name == 'isru':
        activation = isru
        smooth_activation = isru
        smooth_offset = isru_conjugate
        v_range = [-1, 1]
    elif activation_name == 'tanh':
        activation = torch.tanh
        smooth_activation = torch.tanh
        smooth_offset = torch.atanh
        v_range = [-1, 1]
    elif activation_name == 'scaled_tanh':
        a = 1.7159
        b = 0.6666
        activation = scaled_tanh
        smooth_activation = scaled_tanh
        smooth_offset = scaled_atanh
        v_range = [-a, a]
    elif activation_name == 'atan':
        activation = torch.atan
        smooth_activation = torch.atan
        smooth_offset = torch.tan
        v_range = [-np.pi / 2, np.pi / 2]
    elif activation_name == 'sin':
        activation = torch.sin
        smooth_activation = torch.sin
        smooth_offset = asin #torch.asin
        v_range = [-1, 1]
    elif activation_name == 'asinh':
        activation = asinh
        smooth_activation = asinh
        smooth_offset = torch.sinh
        v_range = [-np.Inf, np.Inf]
    elif 'softplus' in activation_name:
        beta = [float(s) for s in re.findall(r'[\d\.\d]+',activation_name)]
        if not beta:
            beta = 1000
        else:
            beta = beta[0]
        # activation = (lambda var: softplus(var, beta=beta))
        # smooth_activation = (lambda var: softplus(var, beta=beta))
        # smooth_offset = (lambda var: softplus_conjugate(var, beta=beta))
        activation = partial(softplus,beta=beta)
        smooth_activation = partial(softplus,beta=beta)
        smooth_offset = partial(softplus_conjugate,beta=beta)
        v_range = [0, np.Inf]
    else:
        activation = nn.functional.gelu
        smooth_activation = nn.functional.gelu
        smooth_offset = nn.Identity()
        v_range = [-np.Inf,np.Inf]
        print('default gelu regularization is used')

    if version == 'bregman':
        v_range = [v_range[0] + eps, v_range[1] - eps]
        return smooth_activation, smooth_offset, v_range
    elif version == 'euclidean':
        return activation, torch.nn.Identity(), [-np.Inf, np.Inf]
    elif version == 'standard':
        return activation, Zeros(), [-np.Inf, np.Inf]