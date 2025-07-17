#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This code belongs to the paper:
-- Tripura, T., & Chakraborty, S. (2022). Wavelet neural operator: a neural 
   operator for parametric partial differential equations. arXiv preprint arXiv:2205.02191.
   
This code is for 1-D wave advection equation (time-independent problem).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.wavelet_convolution import WaveConv1d,WaveConv2d
from .base_model import BaseModel
# from utils import *
from ..layers.wavelet_convolution import WaveConv1d, WaveConv2d

class WNO1d(BaseModel, name="WNO1d"):
    def __init__(
        self,
        width,
        level,
        layers,
        size,
        wavelet,
        in_channel,
        grid_range,
        padding=0,
        non_linearity=F.mish,
        architecture="standard",
    ):
        super(WNO1d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x) = g(K.v + W.v)(x).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 2-channel tensor, Initial condition and location (a(x), x)
              : shape: (batchsize * x=s * c=2)
        Output: Solution of a later timestep (u(x))
              : shape: (batchsize * x=s * c=1)
              
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : scalar, signal length
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: scalar (for 1D), right support of 1D domain
        padding   : scalar, size of zero padding
        """
        
        self.level = level
        self.width = width
        self.n_layers = layers
        self.size = round(size*(1+2*padding))
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        self.non_linearity = non_linearity
        self.architecture = architecture
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()

        # if padding is not None and (
        #     (isinstance(padding, list) and sum(padding) > 0)
        #     or (isinstance(padding, (float, int)) and padding > 0)
        # ):
        #     self.domain_padding = DomainPadding(
        #         domain_padding=padding,
        #         padding_mode="symmetric",
        #         output_scaling_factor=1,
        #     )
        # else:
        #     self.domain_padding = None



        # self.fc0 = nn.Linear(self.in_channel, self.width) # input channel is 2: (a(x), x)
        self.fc0 = nn.Conv1d(self.in_channel, 128, 1)
        self.fc1 = nn.Conv1d(128, self.width, 1)
        for i in range(self.n_layers):
            self.conv.append(
                WaveConv1d(
                    self.width,
                    self.width,
                    self.level,
                    self.size,
                    self.wavelet,
                    architecture=self.architecture,
                )
            )
            if self.architecture in ["euclidean", "bregman"]:
                w = nn.Conv1d(self.width, self.width, 1)
                w.weight.data.fill_(0)
                w.bias.data.fill_(0)
                self.w.append(w)
            else:
                self.w.append(nn.Conv1d(self.width, self.width, 1))
        self.fc2 = nn.Conv1d(self.width, 128, 1)
        self.fc3 = nn.Conv1d(128, 1, 1)

        # self.fc1 = nn.Linear(self.width, 128)
        # self.fc2 = nn.Linear(128, 1)

    def forward(self, x, **kwargs):
        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        nx = x.shape[-1]
        x = self.fc0(x)  # Shape: Batch * x * Channel
        x = self.fc1(x)  # Shape: Batch * x * Channel
        # x = x.permute(0, 2, 1)       # Shape: Batch * Channel * x
        if self.padding != 0:
            # print(x.shape)
            x = F.pad(x, (round(self.padding*nx), round(self.padding*nx)))
            # x =self.domain_padding.pad(x)
            # print(x.shape)
            # print(x)

        if self.architecture in ["euclidean", "bregman"]:
            # Add non linearity to be in the correct range
            x_skip = x
            
        for index, (convl, wl) in enumerate( zip(self.conv, self.w) ):
            x = convl(x) + wl(x) 
            
            if self.architecture in ["euclidean", "bregman"]:
                x += x_skip
                x_skip = x
            if index != self.n_layers - 1:  # Final layer has no activation
                x = self.non_linearity(x)  # Shape: Batch * Channel * x

        if self.padding != 0:
            # print(x.shape)
            x = x[..., round(self.padding*nx):-round(self.padding*nx)]
            # x =self.domain_padding.unpad(x)
            # print(x.shape)
            # print(x)

        # x = x.permute(0, 2, 1)       # Shape: Batch * x * Channel
        x = F.gelu(self.fc2(x))  # Shape: Batch * x * Channel
        x = self.fc3(x)  # Shape: Batch * x * Channel
        return x

    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, self.grid_range, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)


class WNO2d(BaseModel, name="WNO2d"):
    def __init__(
        self, width, level, layers, size, wavelet, in_channel, grid_range, padding=0, non_linearity=F.mish, architecture="standard"
    ):
        super(WNO2d, self).__init__()

        """
        The WNO network. It contains l-layers of the Wavelet integral layer.
        1. Lift the input using v(x) = self.fc0 .
        2. l-layers of the integral operators v(j+1)(x,y) = g(K.v + W.v)(x,y).
            --> W is defined by self.w; K is defined by self.conv.
        3. Project the output of last layer using self.fc1 and self.fc2.
        
        Input : 3-channel tensor, Initial input and location (a(x,y), x,y)
              : shape: (batchsize * x=width * x=height * c=3)
        Output: Solution of a later timestep (u(x,y))
              : shape: (batchsize * x=width * x=height * c=1)
        
        Input parameters:
        -----------------
        width : scalar, lifting dimension of input
        level : scalar, number of wavelet decomposition
        layers: scalar, number of wavelet kernel integral blocks
        size  : list with 2 elements (for 2D), image size
        wavelet: string, wavelet filter
        in_channel: scalar, channels in input including grid
        grid_range: list with 2 elements (for 2D), right supports of 2D domain
        padding   : scalar, size of zero padding
        """

        self.level = level
        self.width = width
        self.n_layers = layers
        self.size = size
        self.wavelet = wavelet
        self.in_channel = in_channel
        self.grid_range = grid_range 
        self.padding = padding
        self.non_linearity = non_linearity
        self.architecture = architecture
        
        self.conv = nn.ModuleList()
        self.w = nn.ModuleList()

        self.fc0 = nn.Conv2d(self.in_channel, 128, 1)
        self.fc1 = nn.Conv2d(128, self.width, 1)
        for i in range(self.n_layers):
            self.conv.append(
                WaveConv2d(
                    self.width,
                    self.width,
                    self.level,
                    self.size,
                    self.wavelet,
                    architecture=self.architecture,
                )
            )
            if self.architecture in ["euclidean", "bregman"]:
                w = nn.Conv2d(self.width, self.width, 1)
                w.weight.data.fill_(0)
                w.bias.data.fill_(0)
                self.w.append(w)
            else:
                self.w.append(nn.Conv2d(self.width, self.width, 1))
        self.fc2 = nn.Conv2d(self.width, 128, 1)
        self.fc3 = nn.Conv2d(128, 1, 1)

    def forward(self, x, **kwargs):

        x = self.fc0(x)  
        x = self.fc1(x)  
        if self.padding != 0:
            padding = int(self.padding//2)
            x = F.pad(x, [padding,padding,padding,padding])
            print(x.shape)
        for index, (convl, wl) in enumerate(zip(self.conv, self.w)):
            x = convl(x) + wl(x)
            if index != self.n_layers - 1:  # Final layer has no activation
                x = self.non_linearity(x) # Shape: Batch * Channel * x * y

        if self.padding != 0:
            print(x.shape)
            x = x[...,padding : -padding,padding : -padding]
            print(x.shape)


        x = F.gelu(self.fc2(x))  # Shape: Batch * x * Channel
        x = self.fc3(x)  # Shape: Batch * x * Channel
        return x
    
    def get_grid(self, shape, device):
        # The grid of the solution
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, self.grid_range[0], size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, self.grid_range[1], size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)
