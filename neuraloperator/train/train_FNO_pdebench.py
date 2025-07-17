"""
Training a FNO on Navier-Srokes
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
from neuralop import LpLoss, Trainer, FftLoss, BoundaryLoss, RmseLoss, NRmseLoss, MseLoss
from neuralop.models import FNO
from neuralop.training import BasicLoggerCallback, EarlyStoppingCallback
from neuralop.utils import count_model_params, get_non_linearity
import random

import wandb

seed = 43
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
    dataset='1D_Burgers_nu0001',
    resolution=1024,
    n_train=100,
    n_test=100,
    n_val=100,
    T_init=9,
    T_final=19,
    use_all_T=False,
    batch_size=32,
    n_epochs=100,
    n_modes=[16],
    n_layers=4,
    lifting_channels=128,
    hidden_channels=64,
    projection_channels=128,
    use_fno_skip=True,
    factorization=None,
    rank=1,
    non_linearity='softplus',
    architecture='standard',
    optimizer='Adam',
    lr=1e-3,
    weight_decay = 0,
    scheduler='CosineAnnealingLR',
    train_loss='L2',
    eval_losses=['L2'],
    device='cuda',
    log_interval=5,
    log_params=False,
    first_save=100,
    patience = 30,
    delta = 1e-3,
    domain_padding=None,
    domain_padding_mode='one-sided',
    seed=seed,
    norm = None
)
config["T_max"] = int(config["n_epochs"]*1.1)

torch.autograd.set_detect_anomaly(True)
# %%
# Create the wandb logger

wandbargs = dict(project='hybrid',
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
if config["dataset"] == 'NS_1e3':
    data_path = Path(__file__).resolve().parent.parent.parent.joinpath("datasets/NS_FNO/ns_V1e-3_N5000_T50.mat").as_posix()
elif config["dataset"] == 'NS_1e4':
    data_path = Path(__file__).resolve().parent.parent.parent.joinpath("datasets/NS_FNO/ns_V1e-4_N10000_T30.mat").as_posix()
else:   
    data_path = Path(__file__).resolve().parent.parent.parent.joinpath(f"datasets/PDEBench/{wandb.config.dataset}.hdf5").as_posix()

if config["dataset"] == '1D_NS_comp':
    data_loader = load_1d_nscomp
elif config["dataset"] == '1D_Advection_beta04':
    data_loader = load_1d_advection
elif config["dataset"] == '1D_Burgers_nu0001':
    data_loader = load_1d_burgers
elif config["dataset"] == '2D_DarcyFlow_beta01':
    data_loader = load_2d_darcy
elif config["dataset"] == 'NS_1e3':
    data_loader = load_NS_64    
elif config["dataset"] == 'NS_1e4':
    data_loader = load_NS_64 
    
train_loader, val_loader, test_loader, data_processor = data_loader(data_path,
        n_train=config['n_train'], n_test=config['n_test'], n_val=config['n_val'], test_resolution=64,
        batch_size=config['batch_size'], test_batch_size=config['batch_size'], val_batch_size=config['batch_size'],
        encode_input=True, encode_output=True, positional_encoding=False, Tinit=config['T_init'], Tfinal=config['T_final'],
        use_all_T=config['use_all_T']
        )
data_processor = data_processor.to(config["device"])

print("dataset shape: ", next(iter(train_loader))['x'].shape)
# %%
# get the non linearity and its inverse

if config['architecture'] == 'bregman':
    non_linearity, non_linearity_inv, non_linearity_range = get_non_linearity(config["non_linearity"],version=config["architecture"])
else:
    non_linearity, non_linearity_inv, non_linearity_range = torch.nn.functional.relu, torch.nn.Identity(), [-np.Inf,np.Inf]



# non_linearity, non_linearity_inv, non_linearity_range = torch.nn.functional.relu, torch.nn.Identity(), [-np.Inf,np.Inf]
# %%
# We create a FNO model

# torch.autograd.set_detect_anomaly(True, check_nan=True)

model = FNO(in_channels=1,n_modes=config["n_modes"], hidden_channels=config["hidden_channels"],
             lifting_channels=config['lifting_channels'], projection_channels=config['projection_channels'],
             use_fno_skip=config["use_fno_skip"], factorization=config['factorization'], rank=config['rank'],
             architecture=config["architecture"], non_linearity=non_linearity,
             non_linearity_inv=non_linearity_inv, n_layers=config["n_layers"],
             non_linearity_range=non_linearity_range,domain_padding=config["domain_padding"],
             domain_padding_mode=config["domain_padding_mode"],norm=config["norm"]
            )

device = torch.device(config["device"])
model = model.to(device)

n_params = count_model_params(model)
config["n_params"] = n_params
print(f'\nOur model has {n_params} parameters.')
print(model)
# %% Visualize the model graph

x = next(iter(train_loader))['x'].to(device)
y = model(x)
make_dot(y.mean(),params=dict(model.named_parameters()),show_saved=True).render("FNO", format="pdf")


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
l2loss = LpLoss(d=len(config['n_modes']), p=2)
# l2loss = MseLoss(d=len(config['n_modes']))
fftloss = FftLoss(d=len(config['n_modes']), p=2)
boundaryloss = BoundaryLoss(d=len(config['n_modes']), p=2)
rmse = RmseLoss(d=len(config['n_modes']))
nrmse = NRmseLoss(d=len(config['n_modes']))
train_loss = l2loss
test_losses = {'l2': l2loss}
eval_losses={'l2': l2loss, 'fft': fftloss, 'boundary': boundaryloss, 'rmse': rmse, 'nrmse': nrmse}

                                        
# %% 
# Create the trainer

trainer = Trainer(model=model, n_epochs=config["n_epochs"],
                  device=device,
                  callbacks=[wandblogger, earlystopper],
                # callbacks=[wandblogger],
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
# for name, param in model.named_parameters():
#     print(name, param.grad.norm().item())

# %% 
torch.autograd.set_detect_anomaly(False, check_nan=True)


from graphviz import Digraph
import torch
from torch.autograd import Variable, Function

def iter_graph(root, callback):
    queue = [root]
    seen = set()
    while queue:
        fn = queue.pop()
        if fn in seen:
            continue
        seen.add(fn)
        for next_fn, _ in fn.next_functions:
            if next_fn is not None:
                queue.append(next_fn)
        callback(fn)

def register_hooks(var):
    fn_dict = {}
    def hook_cb(fn):
        def register_grad(grad_input, grad_output):
            fn_dict[fn] = grad_input
        fn.register_hook(register_grad)
    iter_graph(var.grad_fn, hook_cb)

    def is_bad_grad(grad_output):
        if grad_output is None:
            print('None')
            return False
        return grad_output.isnan().any() or (grad_output.abs() >= 1e5).any()

    def make_dot():
        node_attr = dict(style='filled',
                        shape='box',
                        align='left',
                        fontsize='12',
                        ranksep='0.1',
                        height='0.2')
        dot = Digraph(node_attr=node_attr, graph_attr=dict(size="12,12"))

        def size_to_str(size):
            return '('+(', ').join(map(str, size))+')'

        def build_graph(fn):
            if hasattr(fn, 'variable'):  # if GradAccumulator
                u = fn.variable
                node_name = 'Variable\n ' + size_to_str(u.size())
                dot.node(str(id(u)), node_name, fillcolor='lightblue')
            else:
                assert fn in fn_dict, fn
                fillcolor = 'white'
                if any(is_bad_grad(gi) for gi in fn_dict[fn]):
                    fillcolor = 'red'
                dot.node(str(id(fn)), str(type(fn).__name__), fillcolor=fillcolor)
            for next_fn, _ in fn.next_functions:
                if next_fn is not None:
                    next_id = id(getattr(next_fn, 'variable', next_fn))
                    dot.edge(str(next_id), str(id(fn)))
        iter_graph(var.grad_fn, build_graph)

        return dot

    return make_dot
sample = next(iter(train_loader))
x = sample['x'].to(device)
z = model(x)
z = train_loss(z, sample['y'].to(device))
get_dot = register_hooks(z)
z.backward()
dot = get_dot()
#dot.save('tmp.dot') # to get .dot
#dot.render('tmp') # to get SVG
dot # in Jupyter, you can just render the variable

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


# test_samples = test_loader.dataset
# fig = plt.figure(figsize=(7, 7))
# for i in range(3):
#     data = test_samples[i]
#     data = data_processor.preprocess(data, batched=False)
#     # Input x
#     x = data['x']
#     # Ground-truth
#     y = data['y']
#     # Model prediction
#     out = model(x.unsqueeze(0))
#     x = x.cpu()
#     y = y.cpu()
#     ax = fig.add_subplot(3, 3, i*3 + 1)
#     ax.plot(x[0].cpu())
#     if i == 0: 
#         ax.set_title('Input x')
#     plt.xticks([], [])
#     plt.yticks([], [])

#     ax = fig.add_subplot(3, 3, i*3 + 2)
#     ax.plot(y.squeeze())
#     if i == 0: 
#         ax.set_title('Ground-truth y')
#     plt.xticks([], [])
#     plt.yticks([], [])

#     ax = fig.add_subplot(3, 3, i*3 + 3)
#     ax.plot(out.squeeze().detach().cpu().numpy())
#     if i == 0: 
#         ax.set_title('Model prediction')
#     plt.xticks([], [])
#     plt.yticks([], [])

# fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
# plt.tight_layout()
# fig.show()
# plt.show()
# %%
