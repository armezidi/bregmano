import os

import numpy as np
import scipy.io
import h5py
import torch
from einops import rearrange
from torch.utils.data import DataLoader, Dataset

from .base import Builder


class NS1tBuilder(Builder):
    name = 'ns_1t'

    def __init__(self, data_path: str, train_size: int, test_size: int,
                 ssr: int, Tinit: int, Tfinal: int, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.data_path = data_path
        self.Tinit = Tinit
        self.Tfinal = Tfinal
        self.train_size = train_size
        self.test_size = test_size
        with h5py.File(os.path.expandvars(self.data_path), 'r') as f:
            data = np.swapaxes(f['u'],0,-1).astype(np.float32)[:train_size+2*test_size,...,np.array([self.Tinit,self.Tfinal])]
            
        # data = scipy.io.loadmat(os.path.expandvars(data_path))[
        #     'u'].astype(np.float32)
        # For NavierStokes_V1e-5_N1200_T20.mat
        # data.shape == (1200, 64, 64, 20)

        data = torch.from_numpy(data)
        data = data[:, ::ssr, ::ssr]
        B, X, Y, T = data.shape
        print(data.shape)

        self.test_dataset = NavierStokesDataset(
            data[:test_size])
        print(len(self.test_dataset))
               
        self.val_dataset = NavierStokesDataset(
            data[test_size:2*test_size])
        print(len(self.val_dataset))
        
        self.train_dataset = NavierStokesTrainingDataset(
            data[2*test_size:2*test_size+train_size])
        print(len(self.train_dataset))
        
        # train_dataset.shape == [1000, 64, 64, 20]

    def train_dataloader(self) -> DataLoader:
        loader = DataLoader(self.train_dataset,
                            shuffle=True,
                            drop_last=False,
                            **self.kwargs)
        return loader

    def val_dataloader(self) -> DataLoader:
        loader = DataLoader(self.val_dataset,
                            shuffle=False,
                            drop_last=False,
                            **self.kwargs)
        return loader

    def test_dataloader(self) -> DataLoader:
        loader = DataLoader(self.test_dataset,
                            shuffle=False,
                            drop_last=False,
                            **self.kwargs)
        return loader

    def inference_data(self):
        with h5py.File(os.path.expandvars(self.data_path), 'r') as f:
            data = np.swapaxes(f['u'],0,-1).astype(np.float32)[:1,...,np.array([self.Tinit,self.Tfinal])]
          
        return {'data': data}


class NavierStokesTrainingDataset(Dataset):
    def __init__(self, data):
        # data.shape == [B, X, Y, T]
        x = data[..., 0:-1]
        y = data[..., 1:]

        # dx = data[..., 1:-1] - data[..., :-2]
        # dy = data[..., 2:] - data[..., 1:-1]

        x = rearrange(x, 'b m n t -> (b t) m n 1')
        y = rearrange(y, 'b m n t -> (b t) m n 1')

        # dx = rearrange(dx, 'b m n t -> (b t) m n 1')
        # dy = rearrange(dy, 'b m n t -> (b t) m n 1')

        self.x = x
        self.y = y
        # self.dx = dx
        # self.dy = dy

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {
            'x': self.x[idx],
            'y': self.y[idx],
            # 'dx': self.dx[idx],
            # 'dy': self.dy[idx],
        }


class NavierStokesDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.times = np.arange(0, 2, 1)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return {
            'data': self.data[idx],
            'times': self.times,
        }
