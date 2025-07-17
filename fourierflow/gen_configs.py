# %%
import yaml
import os
import numpy as np

base_config = {
    'wandb': {
        'project': 'FFNOr',
        'tags': ['ffno_32w_2layerlifting'],
        'notes': '',
        'log_model': True,
    },
    'builder': {
        '_target_': 'fourierflow.builders.NS1tBuilder',
        'data_path': '${oc.env:DATA_ROOT}/NS_FNO/ns_V1e-3_N5000_T50.mat',
        # 'data_path': '${oc.env:DATA_ROOT}/NS_FNO/ns_V1e-4_N10000_T30.mat',
        # '_target_': 'fourierflow.builders.Darcy1tBuilder',
        # 'data_path': '${oc.env:DATA_ROOT}/PDEBench/2D_DarcyFlow_beta01.hdf5',
        'train_size': 1000,
        'test_size': 1000,
        'ssr': 1,
        'batch_size': 16,
        'Tinit': 10,
        'Tfinal': 45,
        'num_workers': 0,
        'pin_memory': True
    },
    'routine': {
        '_target_': 'fourierflow.routines.Grid2D1tExperiment',
        'conv': {
            '_target_': 'fourierflow.modules.FNOFactorized2DBlock',
            'modes': 12,
            'width': 32,
            'n_layers': 4,
            'input_dim': 1,
            'share_weight': True,
            'factor': 4,
            'ff_weight_norm': True,
            'gain': 0.1,
            'dropout': 0.0,
            'in_dropout': 0.0,
            'two_layers_lifting': True,
        },
        'n_steps': 1,
        'use_position': False,
        'should_normalize': False,
        'max_accumulations': 1000,
        'noise_std': 0.0,
        'optimizer': {
            '_target_': 'functools.partial',
            '_args_': ['${get_method: torch.optim.Adam}'],
            'lr': 0.0001,
            'weight_decay': 0.0
        },
        'scheduler': {
            'scheduler': {
                '_target_': 'functools.partial',
                '_args_': ['${get_method: torch.optim.lr_scheduler.ConstantLR}'],
                'optimizer': 'optimizer',
                'factor': 1,
                'total_iters': 0
            },
            'name': 'learning_rate'
        }
    },
    'trainer': {
        'accelerator': 'gpu',
        'devices': 1,
        'precision': 32,
        'max_epochs': 2000,
        'log_every_n_steps': 320,
        'track_grad_norm': -1,
        'fast_dev_run': False,
        'limit_train_batches': 1.0,
        'check_val_every_n_epoch': 5
    },
    'callbacks': [
        {
            '_target_': 'fourierflow.callbacks.CustomModelCheckpoint',
            'filename': '{epoch}-{step}-{valid_loss:.5f}',
            'save_top_k': 1,
            'save_last': False,
            'monitor': 'valid_loss',
            'mode': 'min',
            'every_n_train_steps': None,
            'every_n_epochs': 5,
            'verbose': True

        },
        {
            '_target_': 'pytorch_lightning.callbacks.LearningRateMonitor',
            'logging_interval': 'step'
        },
        {
            '_target_': 'pytorch_lightning.callbacks.ModelSummary',
            'max_depth': 4
        },
        {
            '_target_': 'pytorch_lightning.callbacks.EarlyStopping',
            'monitor': 'valid_loss',
            'patience': 50,
            'min_delta': 1e-3,
            'mode': 'min',
            'verbose': True
        }
    ]
}


learning_rates = list(np.logspace(-4,-2.5,4))
n_layers_values = [4,8,16]
seeds = [42,43,44,45]

# use_position =  [False,True]
# should_normalize = [False, True]
# two_layers_lifting = [False, True]

group_name = 'ffno_12m_32w_2lift'
# base_dir = f'experiments/torus_li/{group_name}/{seeds[0]}'
base_dir = f'experiments/NS_1e3/{group_name}/'

# remove all directories and sub files (Windows)
for root, dirs, files in os.walk(base_dir):
    for dir in dirs:
        exp_dir = os.path.join(base_dir, dir)
        print(f'Removing {dir}')
        for root, dirs, files in os.walk(exp_dir):
            for file in files:
                os.remove(os.path.join(exp_dir, file))
        os.rmdir(exp_dir)
        
        
for lr in learning_rates:
    for n_layers in n_layers_values:
        for seed in seeds:
            config = base_config.copy()         
            config['wandb']['tags'] = ['ffno_12m_32w_2lift','pat50','ns1e3']
            
            
            config['routine']['conv']['n_layers'] = n_layers
            config['routine']['optimizer']['lr'] = float(lr)
            config['seed'] = seed
            config['wandb']['group'] = f'{group_name}/{n_layers}/{lr:.0e}/{seed}'
            dir_name = f"{n_layers}layers_lr{'{:.0e}'.format(lr)}_seed{seed}"
            config_dir = os.path.join(base_dir, dir_name)
            os.makedirs(config_dir, exist_ok=True)
            config_filename = os.path.join(config_dir, 'config.yaml')
            with open(config_filename, 'w') as file:
                yaml.dump(config, file)
            print(f'Created {config_filename}')
            
            
# for lr in learning_rates:
#     for n_layers in n_layers_values:
#         for seed in seeds:
#             for lift in two_layers_lifting:
#                 config = base_config.copy()
#                 config['routine']['conv']['n_layers'] = n_layers
#                 config['routine']['optimizer']['lr'] = float(lr)
#                 config['seed'] = seed
#                 config['wandb']['group'] = f'ffno_basic/{n_layers}/{lr:.0e}/{seed}'
#                 config['routine']['conv']['two_layers_lifting'] = lift
#                 dir_name = f"{n_layers}layers_lr{'{:.0e}'.format(lr)}_seed{seed}_lift{lift}"
#                 config_dir = os.path.join(base_dir, dir_name)
#                 os.makedirs(config_dir, exist_ok=True)
#                 config_filename = os.path.join(config_dir, 'config.yaml')
#                 with open(config_filename, 'w') as file:
#                     yaml.dump(config, file)
#                 print(f'Created {config_filename}')
