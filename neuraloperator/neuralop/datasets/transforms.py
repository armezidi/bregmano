from abc import abstractmethod
from typing import List

import torch
from torch.utils.data import Dataset
class Transform(torch.nn.Module):
    """
    Applies transforms or inverse transforms to 
    model inputs or outputs, respectively
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def transform(self):
        pass

    @abstractmethod
    def inverse_transform(self):
        pass

    @abstractmethod
    def cuda(self):
        pass

    @abstractmethod
    def cpu(self):
        pass

    @abstractmethod
    def to(self, device):
        pass
    
class Normalizer():
    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, data):
        return (data - self.mean)/(self.std + self.eps)

class Composite(Transform):
    def __init__(self, transforms: List[Transform]):
        """Composite transform composes a list of
        Transforms into one Transform object.

        Transformations are not assumed to be commutative

        Parameters
        ----------
        transforms : List[Transform]
            list of transforms to be applied to data
            in order
        """
        super.__init__()
        self.transforms = transforms
    
    def transform(self, data_dict):
        for tform in self.transforms:
            data_dict = tform.transform(self.data_dict)
        return data_dict
    
    def inverse_transform(self, data_dict):
        for tform in self.transforms[::-1]:
            data_dict = tform.transform(self.data_dict)
        return data_dict

    def to(self, device):
        # all Transforms are required to implement .to()
        self.transforms = [t.to(device) for t in self.transforms if hasattr(t, 'to')]
        return self

def regular_grid(spatial_dims, grid_boundaries=[[0, 1], [0, 1]]):
    """
    Appends grid positional encoding to an input tensor, concatenating as additional dimensions along the channels
    """
    height, width = spatial_dims

    xt = torch.linspace(grid_boundaries[0][0], grid_boundaries[0][1],
                        height + 1)[:-1]
    yt = torch.linspace(grid_boundaries[1][0], grid_boundaries[1][1],
                        width + 1)[:-1]

    grid_x, grid_y = torch.meshgrid(xt, yt, indexing='ij')

    grid_x = grid_x.repeat(1, 1)
    grid_y = grid_y.repeat(1, 1)

    return grid_x, grid_y


class PositionalEmbedding2D():
    """A simple positional embedding as a regular 2D grid
    """
    def __init__(self, grid_boundaries=[[0, 1], [0, 1]]):
        """PositionalEmbedding2D applies a simple positional 
        embedding as a regular 2D grid

        Parameters
        ----------
        grid_boundaries : list, optional
            coordinate boundaries of input grid, by default [[0, 1], [0, 1]]
        """
        self.grid_boundaries = grid_boundaries
        self._grid = None
        self._res = None

    def grid(self, spatial_dims, device, dtype):
        """grid generates 2D grid needed for pos encoding
        and caches the grid associated with MRU resolution

        Parameters
        ----------
        spatial_dims : torch.size
             sizes of spatial resolution
        device : literal 'cpu' or 'cuda:*'
            where to load data
        dtype : str
            dtype to encode data

        Returns
        -------
        torch.tensor
            output grids to concatenate 
        """
        # handle case of multiple train resolutions
        if self._grid is None or self._res != spatial_dims: 
            grid_x, grid_y = regular_grid(spatial_dims,
                                      grid_boundaries=self.grid_boundaries)
            grid_x = grid_x.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            grid_y = grid_y.to(device).to(dtype).unsqueeze(0).unsqueeze(0)
            self._grid = grid_x, grid_y
            self._res = spatial_dims

        return self._grid

    def __call__(self, data, batched=True):
        if not batched:
            if data.ndim == 3:
                data = data.unsqueeze(0)
        batch_size = data.shape[0]
        x, y = self.grid(data.shape[-2:], data.device, data.dtype)
        out =  torch.cat((data, x.expand(batch_size, -1, -1, -1),
                          y.expand(batch_size, -1, -1, -1)),
                         dim=1)
        # in the unbatched case, the dataloader will stack N 
        # examples with no batch dim to create one
        if not batched and batch_size == 1: 
            return out.squeeze(0)
        else:
            return out