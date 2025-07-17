from pathlib import Path
import torch

from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor
import numpy as np
import h5py

def load_NS_64(
    data_path,
    n_train,
    n_test,
    n_val,
    batch_size,
    test_batch_size,
    val_batch_size,
    test_resolution=64,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=False,
    encode_input=False,
    encode_output=False,
    encoding="channel-wise",
    channel_dim=1,
    non_linearity_range  = (-np.Inf,np.Inf),
    Tinit=0,
    Tfinal=30,
    use_all_T=False,
    keep_time_dim=False
):
    
    
    with h5py.File(data_path, 'r') as f:
        if keep_time_dim:
            data = np.swapaxes(f['u'],0,-1)
            np.random.shuffle(data)
            x_test = torch.Tensor(data[:n_test, :, :,Tinit:Tinit+1]).reshape(n_test,64,64,2).unsqueeze(channel_dim).type(torch.float32).clone()
            y_test = torch.Tensor(data[:n_test, :, :,Tfinal:Tfinal+1]).unsqueeze(channel_dim).clone()

            x_val = torch.Tensor(data[n_test:n_test+n_val, :, :,Tinit:Tinit+1]).reshape(n_val,64,64,2).unsqueeze(channel_dim).type(torch.float32).clone()
            y_val = torch.Tensor(data[n_test:n_test+n_val, :, :,Tfinal:Tfinal+1]).unsqueeze(channel_dim).clone()
            
            x_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train,:, :,Tinit:Tinit+1]).reshape(n_train,64,64,2).unsqueeze(channel_dim).type(torch.float32).clone()
            y_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train, :, :,Tfinal:Tfinal+1]).unsqueeze(channel_dim).clone()    
            
        else:
            data = np.moveaxis(f['u'][:,:,:,:],-1,0)
            np.random.shuffle(data)
            
            x_test = torch.Tensor(data[:n_test,Tinit, :, :]).unsqueeze(channel_dim).type(torch.float32).clone()
            y_test = torch.Tensor(data[:n_test,Tfinal, :, :]).unsqueeze(channel_dim).clone()

            x_val = torch.Tensor(data[n_test:n_test+n_val,Tinit, :, :]).unsqueeze(channel_dim).type(torch.float32).clone()
            y_val = torch.Tensor(data[n_test:n_test+n_val,Tfinal, :, :]).unsqueeze(channel_dim).clone()
            
            x_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train,Tinit,:, :]).unsqueeze(channel_dim).type(torch.float32).clone()
            y_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train,Tfinal, :, :]).unsqueeze(channel_dim).clone()    
        

    if encode_input:
        if encoding == "channel-wise":
            reduce_dims = list(range(x_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        input_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        input_encoder.fit(x_train)
        #x_train = input_encoder.transform(x_train)
        #x_test = input_encoder.transform(x_test.contiguous())
    else:
        input_encoder = None

    if encode_output:
        if encoding == "channel-wise":
            reduce_dims = list(range(y_train.ndim))
        elif encoding == "pixel-wise":
            reduce_dims = [0]

        output_encoder = UnitGaussianNormalizer(dim=reduce_dims)
        output_encoder.fit(y_train)
        #y_train = output_encoder.transform(y_train)
    else:
        output_encoder = None

    train_db = TensorDataset(
        x_train,
        y_train,
    )
    train_loader = torch.utils.data.DataLoader(
        train_db,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    test_db = TensorDataset(
        x_test,
        y_test,
    )
    test_loader = torch.utils.data.DataLoader(
        test_db,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    test_loader = {test_resolution: test_loader}
    
    val_db = TensorDataset(
        x_val,
        y_val,
    )
    val_loader = torch.utils.data.DataLoader(
        val_db,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    
    if positional_encoding:
        pos_encoding = PositionalEmbedding2D(grid_boundaries=grid_boundaries)
    else:
        pos_encoding = None
    data_processor = DefaultDataProcessor(
        in_normalizer=input_encoder,
        out_normalizer=output_encoder,
        positional_encoding=pos_encoding
    )
    return train_loader, test_loader,val_loader, data_processor
