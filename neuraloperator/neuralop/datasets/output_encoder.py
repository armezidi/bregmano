from ..utils import count_tensor_params
from .transforms import Transform
from abc import abstractmethod
from collections.abc import Iterable
import torch

class OutputEncoder(torch.nn.Module):
    """OutputEncoder: converts the output of a model
        into a form usable by some cost function.
    """
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
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


class UnitGaussianNormalizer(Transform):
    """
    UnitGaussianNormalizer normalizes data to be zero mean and unit std. 
    """
    def __init__(self, mean=None, std=None, eps=1e-7, dim=None):
        """
        mean : torch.tensor or None
            has to include batch-size as a dim of 1
            e.g. for tensors of shape ``(batch_size, channels, height, width)``,
            the mean over height and width should have shape ``(1, channels, 1, 1)``
        std : torch.tensor or None
        eps : float, default is 0
            for safe division by the std
        dim : int list, default is None
            if not None, dimensions of the data to reduce over to compute the mean and std.

            .. important:: 

                Has to include the batch-size (typically 0).
                For instance, to normalize data of shape ``(batch_size, channels, height, width)``
                along batch-size, height and width, pass ``dim=[0, 2, 3]``
        
        Notes
        -----
        The resulting mean will have the same size as the input MINUS the specified dims.
        If you do not specify any dims, the mean and std will both be scalars.

        Returns
        -------
        UnitGaussianNormalizer instance
        """
        super().__init__()

        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        self.eps = eps
        if mean is not None:
            self.ndim = mean.ndim
        if isinstance(dim, int):
            dim = [dim]
        self.dim = dim
        self.n_elements = 0
        # print(self.dim)
    def fit(self, data_batch):
        self.update_mean_std(data_batch)

    def partial_fit(self, data_batch, batch_size=1):
        if 0 in list(data_batch.shape):
            return
        count = 0
        n_samples = len(data_batch)
        while count < n_samples:
            samples = data_batch[count:count+batch_size]
            # print(samples.shape)
            # if batch_size == 1:
            #     samples = samples.unsqueeze(0)
            if self.n_elements:
                self.incremental_update_mean_std(samples)
            else:
                self.update_mean_std(samples)
            count += batch_size

    def update_mean_std(self, data_batch):
        self.ndim = data_batch.ndim  # Note this includes batch-size
        self.n_elements = count_tensor_params(data_batch, self.dim)
        self.mean = torch.mean(data_batch, dim=self.dim, keepdim=True)
        self.squared_mean = torch.mean(data_batch**2, dim=self.dim, keepdim=True)
        self.std = torch.sqrt(self.squared_mean - self.mean**2)
        # print(self.mean)
        # print(self.std)
        # print(self.n_elements)
        
    def incremental_update_mean_std(self, data_batch):
        n_elements = count_tensor_params(data_batch, self.dim)

        self.mean = (1.0/(self.n_elements + n_elements))*(
            self.n_elements*self.mean + torch.sum(data_batch, dim=self.dim, keepdim=True))
        self.squared_mean = (1.0/(self.n_elements + n_elements - 1))*(
            self.n_elements*self.squared_mean + torch.sum(data_batch**2, dim=self.dim, keepdim=True))
        self.n_elements += n_elements

        self.std = torch.sqrt(self.squared_mean - self.mean**2)

    def transform(self, x):
        return (x - self.mean)/(self.std + self.eps)
    
    def inverse_transform(self, x):
        return (x*(self.std + self.eps) + self.mean)
    
    def forward(self, x):
        return self.transform(x)
    
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
    
    @classmethod
    def from_dataset(cls, dataset, dim=None, keys=None):
        """Return a dictionary of normalizer instances, fitted on the given dataset
        
        Parameters
        ----------
        dataset : pytorch dataset
            each element must be a dict {key: sample}
            e.g. {'x': input_samples, 'y': target_labels}
        dim : int list, default is None
            * If None, reduce over all dims (scalar mean and std)
            * Otherwise, must include batch-dimensions and all over dims to reduce over
        keys : str list or None
            if not None, a normalizer is instanciated only for the given keys
        """
        for i, data_dict in enumerate(dataset):
            if not i:
                if not keys:
                    keys = data_dict.keys()
                instances = {key: cls(dim=dim) for key in keys}
            for key, sample in data_dict.items():
                instances[key].partial_fit(sample.unsqueeze(0))
        return instances
             
