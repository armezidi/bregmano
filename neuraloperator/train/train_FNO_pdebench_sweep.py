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
from neuralop.models import FNO
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
    'name': f'NS_1e4 relu',
    'method': 'grid',
    'parameters': {
    'dataset' : {'values':['NS_1e4']},
    'resolution' : {'values':[64]},
    'n_train' : {'values':[1000]},
    'n_test' : {'values':[1000]},
    'n_val' : {'values':[1000]},
    'Tinit' : {'values':[9]},
    'Tfinal' : {'values':[19]},
    'use_all_T':{'values':[False]},
    'batch_size' : {'values':[32]},
    'n_epochs' : {'values':[2001]},
    'T_max': {'values': [int(1.1*2001)]},
    'n_modes' : {'values':[[12,12]]},
    'n_layers' : {'values':[4,8,16]},
    'lifting_channels' : {'values':[128]},
    'hidden_channels' : {'values':[32]},
    'projection_channels' : {'values':[128]},
    'use_fno_skip' : {'values':[True]},
    'factorization' : {'values':[None]},
    'rank' : {'values':[1]},
    'non_linearity' : {'values':['gelu']},
    'architecture' : {'values':["standard","bregman"]}, # Choose between standard, bregman, or residual_ffno
    'optimizer' : {'values':['Adam']},
    'lr' : {'values':list(np.logspace(-4,-2.5,4))},
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
    'domain_padding':{'values':[None]},
    'domain_padding_mode':{'values':["symmetric"]},
    'encode_input':{'values':[False]},
    'encode_output':{'values':[False]},
    }
}



sweep_id = wandb.sweep(sweep=sweep_config, project="Bregman-FNO")
# %%
# Create the main loop for the sweep

def main():
    wandbargs = dict(save_code=True)               
    wandblogger = BasicLoggerCallback(wandbargs, log_params=False)
    name = f"{wandb.config.architecture}_{wandb.config.n_layers}l_b{wandb.config.batch_size}_lr{'{:.0e}'.format(wandb.config.lr)}_skip{str(wandb.config.use_fno_skip)}_s{wandb.config.seed}_{wandb.config.domain_padding}_{wandb.config.domain_padding_mode}_{datetime.now().strftime('%d%m_%H%M%S')}"
    wandb.run.name = name
    
    model_path = Path(__file__).resolve().parent.joinpath(f"models/{wandb.config.dataset}/{name}").as_posix()
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
            encode_input=wandb.config.encode_input, encode_output=wandb.config.encode_output, positional_encoding=False, 
            Tinit=wandb.config.Tinit, Tfinal=wandb.config.Tfinal, use_all_T=wandb.config.use_all_T
            )
    data_processor = data_processor.to(wandb.config.device)
    
    if wandb.config.architecture == 'bregman':
        non_linearity, non_linearity_inv, non_linearity_range = get_non_linearity(wandb.config.non_linearity,
                                                                                  version=wandb.config.architecture)
    else:
        # Default to relu for standard architecture, as per original FNO paper
        non_linearity, non_linearity_inv, non_linearity_range = torch.nn.functional.relu, torch.nn.Identity(), [-np.Inf,np.Inf]
        
    
    model = FNO(in_channels=1,n_modes=wandb.config.n_modes, hidden_channels=wandb.config.hidden_channels,
                lifting_channels=wandb.config.lifting_channels, projection_channels=wandb.config.projection_channels,
                use_fno_skip=wandb.config.use_fno_skip, factorization=wandb.config.factorization, rank=wandb.config.rank,
                architecture=wandb.config.architecture, non_linearity=non_linearity,
                non_linearity_inv=non_linearity_inv, n_layers=wandb.config.n_layers,
                non_linearity_range=non_linearity_range, domain_padding=wandb.config.domain_padding,
                domain_padding_mode=wandb.config.domain_padding_mode
                )

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

    l2loss = LpLoss(d=len(wandb.config.n_modes), p=2)
    fftloss= FftLoss(d=len(wandb.config.n_modes),p=2)
    bdloss = BoundaryLoss(d=len(wandb.config.n_modes),p=2)
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

wandb.agent(sweep_id, main)