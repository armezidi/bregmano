# %%
# 
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
import random
from load_pdebench import load_1d_nscomp, load_1d_advection, load_1d_burgers, load_2d_darcy, load_1d_burgers_fno
from load_NS_64_1e4 import load_NS_64
from neuralop import LpLoss, Trainer, FftLoss, BoundaryLoss, RmseLoss, NRmseLoss
from neuralop.models import FNO, WNO1d, WNO2d
from neuralop.training import BasicLoggerCallback, EarlyStoppingCallback
from neuralop.utils import get_non_linearity

import wandb
# %%
# wandb sweep parameters

# 1D_NS_comp
# 1D_Advection_beta04
# 1D_Burgers_nu0001
# 2D_DarcyFlow_beta01
# NS_1e3
# NS_1e4

sweep_config = {
    'name': f'NS_1e3: archi_layers_seed_lr',
    'method': 'grid',
    'parameters': {
    'dataset' : {'values':['NS_1e3']},
    'resolution' : {'values':[64]},
    'n_train' : {'values':[1000]},
    'n_test' : {'values':[1000]},
    'n_val' : {'values':[1000]},
    'Tinit' : {'values':[10]},
    'Tfinal' : {'values':[49]},
    'batch_size' : {'values':[16]},
    'n_epochs' : {'values':[2000]},
    'T_max': {'values': [int(1.1*2001)]},
    'wavelet' : {'values':['db4']},
    'width' : {'values':[32]},
    'level' : {'values':[4]},
    'in_channel' : {'values':[1]},
    'grid_range' : {'values':[[1]]},
    'padding' : {'values':[0]},
    'n_layers' : {'values':[4,8,16]},
    'non_linearity' : {'values':['softplus']},
    'architecture' : {'values':['bregman','standard']},
    'optimizer' : {'values':['Adam']},
    'lr' : {'values':list(np.logspace(-5,-3.5,4))},
    'weight_decay' : {'values':[0]},
    'scheduler' : {'values':['ConstantLR']},
    'train_loss' : {'values':['l2']},
    'eval_losses' : {'values':[['l2']]},
    'device' : {'values':['cuda']},
    'log_interval':{'values':[5]},
    'log_params':{'values':[False]},
    'first_save':{'values':[5]},
    'patience':{'values':[20]},
    'delta':{'values':[1e-3]},
    'seed':{'values':[42,43,44,45]},
    }
}



sweep_id = wandb.sweep(sweep=sweep_config, project="Bregman-WNO")
# %%
# Create the main loop for the sweep

def main():
    wandbargs = dict(save_code=True)               
    wandblogger = BasicLoggerCallback(wandbargs, log_params=False)
    name = f"{wandb.config.architecture}_{wandb.config.n_layers}l_lr{'{:.0e}'.format(wandb.config.lr)}_s{wandb.config.seed}_{datetime.now().strftime('%d%m_%H%M%S')}"
    wandb.run.name = name
    
    model_path = Path(__file__).resolve().parent.joinpath(f"models/WNO/{wandb.config.dataset}/{name}").as_posix()
    print(f"Model path: {model_path}")
    earlystopper = EarlyStoppingCallback(patience=wandb.config.patience, delta=wandb.config.delta, metric='l2',
                                         save_dir=model_path, save_interval=wandb.config.log_interval,
                                         first_save=wandb.config.first_save, verbose=True) 

    torch.manual_seed(wandb.config.seed)
    np.random.seed(wandb.config.seed)
    random.seed(wandb.config.seed)
    # torch.use_deterministic_algorithms(True)
    if wandb.config.dataset == 'NS_1e3':
        data_path = Path(__file__).resolve().parent.parent.parent.joinpath("datasets/NS_FNO/ns_V1e-3_N5000_T50.mat").as_posix()
    elif wandb.config.dataset == 'NS_1e4':
        data_path = Path(__file__).resolve().parent.parent.parent.joinpath("datasets/NS_FNO/ns_V1e-4_N10000_T30.mat").as_posix()
    else:   
        data_path = Path(__file__).resolve().parent.parent.parent.joinpath(f"datasets/PDEBench/{wandb.config.dataset}.hdf5").as_posix()
    
    if wandb.config.dataset == '1D_NS_comp':
        data_loader = load_1d_nscomp
    elif wandb.config.dataset == '1D_Advection_beta04':
        data_loader = load_1d_advection
    elif wandb.config.dataset == '1D_Burgers_nu0001':
        data_loader = load_1d_burgers
    elif wandb.config.dataset == '2D_DarcyFlow_beta01':
        data_loader = load_2d_darcy
    elif wandb.config.dataset == '1D_Burgers_nu01':
        data_loader = load_1d_burgers_fno
    elif wandb.config.dataset == 'NS_1e3':
        data_loader = load_NS_64    
    elif wandb.config.dataset == 'NS_1e4':
        data_loader = load_NS_64
        
    
    train_loader, val_loader, test_loader, data_processor = data_loader(data_path, test_resolution=64,
            n_train=wandb.config.n_train, n_test=wandb.config.n_test, n_val=wandb.config.n_val,
            batch_size=wandb.config.batch_size, test_batch_size=wandb.config.batch_size, val_batch_size=wandb.config.batch_size,
            encode_input=False, encode_output=False, positional_encoding=False, Tinit=wandb.config.Tinit, Tfinal=wandb.config.Tfinal,
            channel_dim=1
            )
    data_processor = data_processor.to(wandb.config.device)
    
    if wandb.config.architecture == 'bregman':
        non_linearity, _, _ = get_non_linearity(wandb.config.non_linearity, version=wandb.config.architecture)
    else:
        non_linearity, _, _ = torch.nn.functional.mish, torch.nn.Identity(), [-np.Inf,np.Inf]


    size = list(next(iter(train_loader))['x'].shape[2:])
    if wandb.config.padding!=0:
        padding= wandb.config.padding*size[0]
        size = [int((s*(1+wandb.config.padding)//2)*2) for s in size ]

    if len(size) == 1:
        model = WNO1d(in_channel=wandb.config.in_channel,
                        grid_range=wandb.config.grid_range, 
                        width=wandb.config.width, 
                        level=wandb.config.level, 
                        wavelet=wandb.config.wavelet, 
                        padding=padding, 
                        layers=wandb.config.n_layers, 
                        non_linearity=non_linearity,
                        architecture=wandb.config.architecture,
                        size=size[-1],
                        )
    else:
        model = WNO2d(in_channel=wandb.config.in_channel,
                        grid_range=wandb.config.grid_range, 
                        width=wandb.config.width, 
                        level=wandb.config.level, 
                        layers=wandb.config.n_layers, 
                        size=size, wavelet=wandb.config.wavelet,
                        padding=padding, 
                        non_linearity=non_linearity, 
                        architecture=wandb.config.architecture)
    
    device = torch.device(wandb.config.device)
    model = model.to(device)

    if wandb.config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=wandb.config.lr, 
                                     weight_decay=wandb.config.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=wandb.config.lr)
    
    if wandb.config.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=wandb.config.T_max)
    elif wandb.config.scheduler == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.5)
    else:   
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1,total_iters=0)

    l2loss = LpLoss(d=len(size),p=2)
    fftloss= FftLoss(d=len(size),p=2)
    bdloss = BoundaryLoss(d=len(size),p=2)
    rmse = RmseLoss()
    nrmse = NRmseLoss()
    train_loss = l2loss
    eval_losses={'l2': l2loss}
    final_losses = {'l2': l2loss, 'fft': fftloss, "bd": bdloss, "rmse": rmse, "nrmse": nrmse}


    trainer = Trainer(model=model, n_epochs=wandb.config.n_epochs,
                    device=device,
                    callbacks=[wandblogger,earlystopper],
                    data_processor=data_processor,           
                    wandb_log=True,
                    verbose=True,
                    log_test_interval=wandb.config.log_interval,
                    use_distributed=False)


    trainer.train(train_loader=train_loader,
                test_loaders=val_loader,
                optimizer=optimizer,
                scheduler=scheduler, 
                regularizer=False, 
                training_loss=train_loss,
                eval_losses=eval_losses)
    
    val_l2, val_fft, val_bd, val_rmse, val_nrsme  = trainer.evaluate(final_losses, val_loader[64]).values()
    test_l2, test_fft, test_bd, test_rmse, test_nrsme  = trainer.evaluate(final_losses, test_loader).values()

    wandb.log({'val_l2': val_l2, 'val_fftlow': val_fft[0], 'val_fftmid': val_fft[1], 'val_ffthigh': val_fft[2], 'val_bd': val_bd, 'val_rmse': val_rmse, 'val_nrsme': val_nrsme,
                'test_l2': test_l2, 'test_fftlow': test_fft[0], 'test_fftmid': test_fft[1], 'test_ffthigh': test_fft[2], 'test_bd': test_bd, 'test_rmse': test_rmse, 'test_nrsme': test_nrsme})
    wandb.finish()
    print(f'Final validation L2 loss: {val_l2}'
        f'Final test L2 loss: {test_l2}')
    del model, optimizer, scheduler, trainer
    
    
wandb.agent(sweep_id, main)
