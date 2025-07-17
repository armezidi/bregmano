"""
Training a WNO on PDEBench
=============================

"""

# %%
#

from datetime import datetime
from torchviz import make_dot
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from load_pdebench import load_1d_nscomp, load_1d_advection, load_1d_burgers, load_2d_darcy
from load_NS_64_1e4 import load_NS_64
from neuralop import LpLoss, Trainer, FftLoss, BoundaryLoss, RmseLoss, NRmseLoss
from neuralop.models import FNO, WNO1d, WNO2d
from neuralop.training import BasicLoggerCallback, EarlyStoppingCallback
from neuralop.utils import count_model_params, get_non_linearity
import random

import wandb

seed = 44
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
# torch.use_deterministic_algorithms(True)
# rng = np.random.default_rng(seed=seed)
# %%
# Model and Trainer parameters

# 1D_NS_comp
# 1D_Advection_beta04
# 1D_Burgers_nu0001
# 2D_DarcyFlow_beta01
# NS_1e3
# NS_1e4

config = dict(
    dataset='2D_DarcyFlow_beta01',
    resolution=64,
    n_train=10,
    n_test=10,
    n_val=10 ,
    T_init=0,
    T_final=40,
    batch_size=16,
    n_epochs=2,
    n_layers=2,
    wavelet='db6',
    width=2,
    level=4,
    in_channel=1,
    grid_range=[1],
    padding=0.1,
    lifting_channels=128,
    projection_channels=128,
    non_linearity='softplus',
    architecture='standard',
    optimizer='Adam',
    lr=1e-3,
    weight_decay = 0,
    scheduler='StepLR',
    train_loss='L2',
    eval_losses=['L2'],
    device='cpu',
    log_interval=1,
    log_params=False,
    first_save=1,
    patience = 1,
    delta = 1e-3,
    domain_padding=None,
    domain_padding_mode='symmetric',
    seed=seed
)
config["T_max"] = int(config["n_epochs"]*1.1)

# %%
# Create the wandb logger

wandbargs = dict(project='FNO-NS-ECML',
                name=f'FNO_NS64__{datetime.now().strftime("%d-%m_%H%M")}',
                config=config,
                save_code=True,
                mode = 'disabled')
                
                
wandblogger = BasicLoggerCallback(wandb_kwargs=wandbargs,log_params=config['log_params'])

# %% Create the early stopping callback for the model 
name = f"{config['dataset']}/{config['architecture']}_{config['n_layers']}l_{'{:.0e}'.format(config['lr'])}_s{config['seed']}_{datetime.now().strftime('%d%m_%H%M%S')}"
wandb.run.name = name

model_path = Path(__file__).resolve().parent.joinpath(f"models/new/{wandbargs['name']}").as_posix()
earlystopper = EarlyStoppingCallback(patience=config["patience"],delta=config['delta'],metric='l2',
                                   save_dir=model_path,save_interval=config['log_interval'],
                                   first_save=config['first_save'],verbose=True)
# %% Create the checkpoint callback for the model

# model_path = Path(__file__).resolve().parent.joinpath("models/").as_posix()
# checkpoint = CheckpointCallback(model_path,'l2',save_interval=config['save_interval'],
#                                 first_save=config['first_save'])
# %%
# Loading the Navier-Stokes dataset in 64x64 resolution
if config['dataset'] == 'NS_1e3':
    data_path = Path(__file__).resolve().parent.parent.parent.joinpath("datasets/NS_FNO/ns_V1e-3_N5000_T50.mat").as_posix()
elif config['dataset'] == 'NS_1e4':
    data_path = Path(__file__).resolve().parent.parent.parent.joinpath("datasets/NS_FNO/ns_V1e-4_N10000_T30.mat").as_posix()
else:
    data_path = Path(__file__).resolve().parent.parent.parent.joinpath(f"datasets/PDEBench/{config['dataset']}.hdf5").as_posix()

if config["dataset"] == '1D_NS_comp':
    data_loader = load_1d_nscomp
elif config["dataset"] == '1D_Advection_beta04':
    data_loader = load_1d_advection
elif config["dataset"] == '1D_Burgers_nu0001':
    data_loader = load_1d_burgers
elif config["dataset"] == '2D_DarcyFlow_beta01':
    data_loader = load_2d_darcy
elif config['dataset'] == 'NS_1e3':
    data_loader = load_NS_64    
elif config['dataset'] == 'NS_1e4':
    data_loader = load_NS_64    
    
train_loader, val_loader, test_loader, data_processor = data_loader(data_path,
        n_train=config['n_train'], n_test=config['n_test'], n_val=config['n_val'], test_resolution=64,
        batch_size=config['batch_size'], test_batch_size=config['batch_size'], val_batch_size=config['batch_size'],
        encode_input=False, encode_output=False, positional_encoding=False, Tinit=config['T_init'], Tfinal=config['T_final'],
        channel_dim=1
        )
data_processor = data_processor.to(config["device"])

# %%
# get the non linearity and its inverse

if config['architecture'] == 'bregman':
    non_linearity, non_linearity_inv, non_linearity_range = get_non_linearity(config["non_linearity"],version=config["architecture"])
else:
    non_linearity, non_linearity_inv, non_linearity_range = torch.nn.functional.mish, torch.nn.Identity(), [-np.Inf,np.Inf]



# non_linearity, non_linearity_inv, non_linearity_range = torch.nn.functional.relu, torch.nn.Identity(), [-np.Inf,np.Inf]
# %%
# We create a FNO model

# model = FNO(in_channels=1,n_modes=config["n_modes"], hidden_channels=config["hidden_channels"],
#              lifting_channels=config['lifting_channels'], projection_channels=config['projection_channels'],
#              use_fno_skip=config["use_fno_skip"], factorization=config['factorization'], rank=config['rank'],
#              architecture=config["architecture"], non_linearity=non_linearity,
#              non_linearity_inv=non_linearity_inv, n_layers=config["n_layers"],
#              non_linearity_range=non_linearity_range,domain_padding=config["domain_padding"],
#              domain_padding_mode=config["domain_padding_mode"]
#             )
print(list(next(iter(train_loader))['x'].shape[2:]))
size = list(next(iter(train_loader))['x'].shape[2:])
if config['padding']!=0:
    padding= config['padding']*size[0]
    size = [int((s*(1+config['padding'])//2)*2) for s in size ]
print(size)
if  len(size) == 2:
    model = WNO2d(width=config['width'], level=config['level'], layers=config['n_layers'], size=size, wavelet=config['wavelet'],
                in_channel=config['in_channel'], grid_range=config['grid_range'], padding=padding)
elif len(size) == 1:
    model = WNO1d(width=config['width'], level=config['level'], layers=config['n_layers'], size=size[0], wavelet=config['wavelet'],
                in_channel=config['in_channel'], grid_range=config['grid_range'], padding=padding)

device = torch.device(config["device"])
model = model.to(device)

n_params = count_model_params(model)
config["n_params"] = n_params
print(f'\nOur model has {n_params} parameters.')
# %% Visualize the model graph

# x = next(iter(train_loader))['x'].to(device)
# y = model(x)
# make_dot(y.mean(),params=dict(model.named_parameters()),show_saved=True)
# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=config["lr"], 
                                weight_decay=config["weight_decay"])

# optimizer = torch.optim.SGD(model.parameters(),
#                                 lr=config["lr"])

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["T_max"])
scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1,total_iters=0)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.5)


# %%
# Creating the losses
l2loss = LpLoss(d=len(size), p=2)
fftloss = FftLoss(d=len(size), p=2)
boundaryloss = BoundaryLoss(d=len(size), p=2)
rmse = RmseLoss(d=len(size))
nrmse = NRmseLoss(d=len(size))
train_loss = l2loss
test_losses = {'l2': l2loss}
eval_losses={'l2': l2loss, 'fft': fftloss, 'boundary': boundaryloss, 'rmse': rmse, 'nrmse': nrmse}

                                        
# %% 
# Create the trainer

trainer = Trainer(model=model, n_epochs=config["n_epochs"],
                  device=device,
                  callbacks=[wandblogger,earlystopper],
                #   data_processor=data_processor,             
                  wandb_log=True,
                  verbose=True,
                  log_test_interval=config['log_interval'],
                  use_distributed=False)

# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader=train_loader,
              test_loaders=val_loader,
              optimizer=optimizer,
              scheduler=scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=test_losses)

# %% 
# Log the final test loss to wandb and finish the run

val_l2 = trainer.evaluate(eval_losses, val_loader[64])['_l2']
test_l2,fft_l2,boundary_l2,rmse_l2,nrmse_l2 = trainer.evaluate(eval_losses, test_loader).values()
wandb.log({"test_l2": test_l2, "test_fft": fft_l2, "test_boundary": boundary_l2, "test_rmse": rmse_l2, "test_nrmse": nrmse_l2})
wandb.finish()
print(f'Final validation L2 loss: {val_l2}')
print(f'Final test L2 loss: {test_l2}')
print(f'Final test FFT loss: {fft_l2}')
print(f'Final test Boundary loss: {boundary_l2}')
print(f'Final test RMSE loss: {rmse_l2}')
print(f'Final test NRMSE loss: {nrmse_l2}')

# %%


test_samples = test_loader.dataset
fig = plt.figure(figsize=(7, 7))
for i in range(3):
    data = test_samples[np.random.randint(0, len(test_samples))]
    data = data_processor.preprocess(data, batched=False)
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0))
    x = x.cpu()
    y = y.cpu()
    ax = fig.add_subplot(3, 3, i*3 + 1)
    ax.plot(x[0].cpu())
    if i == 0: 
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, i*3 + 2)
    ax.plot(y.squeeze())
    if i == 0: 
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, i*3 + 3)
    ax.plot(out.squeeze().detach().cpu().numpy())
    if i == 0: 
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
plt.show()
# %%
