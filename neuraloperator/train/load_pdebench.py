from pathlib import Path
import torch

from neuralop.datasets.output_encoder import UnitGaussianNormalizer
from neuralop.datasets.tensor_dataset import TensorDataset
from neuralop.datasets.transforms import PositionalEmbedding2D
from neuralop.datasets.data_transforms import DefaultDataProcessor
import numpy as np
import random
import h5py
from scipy.io import loadmat


def load_1d_nscomp(
    data_path,
    n_train,
    n_test,
    n_val,
    batch_size,
    test_batch_size,
    val_batch_size,
    test_resolution=64,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="pixel-wise",
    channel_dim=1,
    non_linearity_range  = (-np.Inf,np.Inf),
    Tinit=10,
    Tfinal=100,
    use_all_T=False,
    keep_time_dim=False
):
    
    
    with h5py.File(data_path, 'r') as f:
        if use_all_T:
            data =f['Vx'][:,Tinit:Tfinal+1,:]
        else:
            data =f['Vx'][:,np.array([Tinit,Tfinal]),:]
        # n_samples = data.shape[0]
        # indices = list(range(n_samples))
        # random.shuffle(indices)    
        np.random.shuffle(data)
        
        x_test_ = data[:n_test,0, :]
        y_test_ = data[:n_test,1, :]
        
        x_val_ = data[n_test:n_test+n_val,0, :]
        y_val_ = data[n_test:n_test+n_val,1, :]
        
        x_train_ = data[n_test+n_val:n_test+n_val+n_train,0, :]
        y_train_ = data[n_test+n_val:n_test+n_val+n_train,1, :]
        
        if use_all_T:
            for i in range(1,Tfinal-Tinit):
                x_test_ = np.concatenate((x_test_,data[:n_test,i, :]),axis=0)
                y_test_ = np.concatenate((y_test_,data[:n_test,i+1, :]),axis=0)

                x_val_ = np.concatenate((x_val_,data[n_test:n_test+n_val,i, :]),axis=0)
                y_val_ = np.concatenate((y_val_,data[n_test:n_test+n_val,i+1, :]),axis=0)
                
                x_train_ = np.concatenate((x_train_,data[n_test+n_val:n_test+n_val+n_train,i, :]),axis=0)
                y_train_ = np.concatenate((y_train_,data[n_test+n_val:n_test+n_val+n_train,i+1, :]),axis=0)
            
        x_test = torch.Tensor(x_test_).unsqueeze(channel_dim).type(torch.float32).clone()
        y_test = torch.Tensor(y_test_).unsqueeze(channel_dim).type(torch.float32).clone()
        
        x_val = torch.Tensor(x_val_).unsqueeze(channel_dim).type(torch.float32).clone()
        y_val = torch.Tensor(y_val_).unsqueeze(channel_dim).type(torch.float32).clone()
        
        x_train = torch.Tensor(x_train_).unsqueeze(channel_dim).type(torch.float32).clone()
        y_train = torch.Tensor(y_train_).unsqueeze(channel_dim).type(torch.float32).clone()
        

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

def load_1d_advection(
    data_path,
    n_train,
    n_test,
    n_val,
    batch_size,
    test_batch_size,
    val_batch_size,
    test_resolution=64,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="pixel-wise",
    channel_dim=1,
    non_linearity_range  = (-np.Inf,np.Inf),
    Tinit=10,
    Tfinal=200,
    use_all_T=False,
    keep_time_dim=False
):
    
    
    with h5py.File(data_path, 'r') as f:
        data =f['tensor'][:,np.array([Tinit,Tfinal]),:]
        # n_samples = data.shape[0]
        # indices = list(range(n_samples))
        # random.shuffle(indices)    
        np.random.shuffle(data)
        
        x_test = torch.Tensor(data[:n_test,0, :]).unsqueeze(channel_dim).type(torch.float32).clone()
        y_test = torch.Tensor(data[:n_test,-1, :]).unsqueeze(channel_dim).type(torch.float32).clone()

        x_val = torch.Tensor(data[n_test:n_test+n_val,0, :]).unsqueeze(channel_dim).type(torch.float32).clone()
        y_val = torch.Tensor(data[n_test:n_test+n_val,-1, :]).unsqueeze(channel_dim).type(torch.float32).clone()
        
        x_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train,0, :]).unsqueeze(channel_dim).type(torch.float32).clone()
        y_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train,-1, :]).unsqueeze(channel_dim).type(torch.float32).clone()


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

def load_1d_burgers(
    data_path,
    n_train,
    n_test,
    n_val,
    batch_size,
    test_batch_size,
    val_batch_size,
    test_resolution=64,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="pixel-wise",
    channel_dim=1,
    non_linearity_range  = (-np.Inf,np.Inf),
    Tinit=10,
    Tfinal=200,
    use_all_T=False,
    keep_time_dim=False
):
    
    
    with h5py.File(data_path, 'r') as f:
        
        if use_all_T:
            data =f['tensor'][:,np.arange(Tinit,Tfinal+1,25),:]
        else:
            data =f['tensor'][:,np.array([Tinit,Tfinal]),:]
        # n_samples = data.shape[0]
        # indices = list(range(n_samples))
        # random.shuffle(indices)    
        np.random.shuffle(data)
        
        
        x_test_ = data[:n_test,0, :]
        y_test_ = data[:n_test,1, :]
        
        x_val_ = data[n_test:n_test+n_val,0, :]
        y_val_ = data[n_test:n_test+n_val,1, :]
        
        x_train_ = data[n_test+n_val:n_test+n_val+n_train,0, :]
        y_train_ = data[n_test+n_val:n_test+n_val+n_train,1, :]
       
        if use_all_T: 
            for i in range(1,data.shape[1]-1):
                x_test_ = np.concatenate((x_test_,data[:n_test,i, :]),axis=0)
                y_test_ = np.concatenate((y_test_,data[:n_test,i+1, :]),axis=0)

                x_val_ = np.concatenate((x_val_,data[n_test:n_test+n_val,i, :]),axis=0)
                y_val_ = np.concatenate((y_val_,data[n_test:n_test+n_val,i+1, :]),axis=0)
                
                x_train_ = np.concatenate((x_train_,data[n_test+n_val:n_test+n_val+n_train,i, :]),axis=0)
                y_train_ = np.concatenate((y_train_,data[n_test+n_val:n_test+n_val+n_train,i+1, :]),axis=0)
                
                
        x_test = torch.Tensor(x_test_).unsqueeze(channel_dim).type(torch.float32).clone()
        y_test = torch.Tensor(y_test_).unsqueeze(channel_dim).type(torch.float32).clone()
        
        x_val = torch.Tensor(x_val_).unsqueeze(channel_dim).type(torch.float32).clone()
        y_val = torch.Tensor(y_val_).unsqueeze(channel_dim).type(torch.float32).clone()
        
        x_train = torch.Tensor(x_train_).unsqueeze(channel_dim).type(torch.float32).clone()
        y_train = torch.Tensor(y_train_).unsqueeze(channel_dim).type(torch.float32).clone()
        

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

def load_2d_darcy(
    data_path,
    n_train,
    n_test,
    n_val,
    batch_size,
    test_batch_size,
    val_batch_size,
    test_resolution=64,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="pixel-wise",
    channel_dim=1,
    non_linearity_range  = (-np.Inf,np.Inf),
    Tinit=10,
    Tfinal=200,
    use_all_T=False,
    keep_time_dim=False
):
    
    
    with h5py.File(data_path, 'r') as f:
        data = np.stack((f['nu'],f['tensor'][:,0,...]),-1)
        # n_samples = data.shape[0]
        # indices = list(range(n_samples))
        # random.shuffle(indices)    
        np.random.shuffle(data)
        
        x_test = torch.Tensor(data[:n_test,...,0]).unsqueeze(channel_dim).type(torch.float32).clone()
        y_test = torch.Tensor(data[:n_test,...,-1]).unsqueeze(channel_dim).type(torch.float32).clone()

        x_val = torch.Tensor(data[n_test:n_test+n_val,...,0]).unsqueeze(channel_dim).type(torch.float32).clone()
        y_val = torch.Tensor(data[n_test:n_test+n_val,...,-1]).unsqueeze(channel_dim).type(torch.float32).clone()
        
        x_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train,...,0]).unsqueeze(channel_dim).type(torch.float32).clone()
        y_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train,...,-1]).unsqueeze(channel_dim).type(torch.float32).clone()

        # x_test = torch.Tensor(f['nu'][:n_test,...]).unsqueeze(channel_dim).type(torch.float32).clone()
        # y_test = torch.Tensor(f['tensor'][:n_test,0,...]).unsqueeze(channel_dim).type(torch.float32).clone()

        # x_val = torch.Tensor(f['nu'][n_test:n_test+n_val,...]).unsqueeze(channel_dim).type(torch.float32).clone()
        # y_val = torch.Tensor(f['tensor'][n_test:n_test+n_val,0,...]).unsqueeze(channel_dim).type(torch.float32).clone()
        
        # x_train = torch.Tensor(f['nu'][n_test+n_val:n_test+n_val+n_train,...]).unsqueeze(channel_dim).type(torch.float32).clone()
        # y_train = torch.Tensor(f['tensor'][n_test+n_val:n_test+n_val+n_train,0,...]).unsqueeze(channel_dim).type(torch.float32).clone()


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

def load_1d_burgers_fno(
    data_path,
    n_train,
    n_test,
    n_val,
    batch_size,
    test_batch_size,
    val_batch_size,
    test_resolution=64,
    grid_boundaries=[[0, 1], [0, 1]],
    positional_encoding=True,
    encode_input=False,
    encode_output=True,
    encoding="pixel-wise",
    channel_dim=1,
    non_linearity_range  = (-np.Inf,np.Inf),
    Tinit=10,
    Tfinal=200,
    use_all_T=False,
    keep_time_dim=False
):
    

    f =  loadmat(data_path)
    data = np.stack((f['a'][:,::8],f['u'][:,::8]),-1)
    np.random.shuffle(data)
    # print(data.shape)
    x_test = torch.Tensor(data[:n_test,:, 0]).unsqueeze(channel_dim).type(torch.float32).clone()
    y_test = torch.Tensor(data[:n_test,:, 1]).unsqueeze(channel_dim).type(torch.float32).clone()
    
    x_val = torch.Tensor(data[n_test:n_test+n_val,:, 0]).unsqueeze(channel_dim).type(torch.float32).clone()
    y_val = torch.Tensor(data[n_test:n_test+n_val,:, 1]).unsqueeze(channel_dim).type(torch.float32).clone()
    
    x_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train,:, 0]).unsqueeze(channel_dim).type(torch.float32).clone()
    y_train = torch.Tensor(data[n_test+n_val:n_test+n_val+n_train,:, 1]).unsqueeze(channel_dim).type(torch.float32).clone()

    del data
    del f
    
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
