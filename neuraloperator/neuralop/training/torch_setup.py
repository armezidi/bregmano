import torch

def setup(config):
    """A convenience function to intialize the device, setup torch settings and
    check multi-grid and other values. It sets up distributed communitation, if used.
    
    Parameters
    ----------
    config : dict 
        this function checks:
        * config.data (n_train, batch_size, test_batch_sizes, n_tests, test_resolutions)
    
    Returns
    -------
    device, is_logger
        device : torch.device
        is_logger : bool
    """

    is_logger = True
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    if 'seed' in config.distributed:
        seed = config.distributed.seed

    #Set device, random seed and optimization
    if torch.cuda.is_available():

        torch.cuda.set_device(device.index)

        if 'seed' in config.distributed:
            torch.cuda.manual_seed(seed)
        increase_l2_fetch_granularity()
        try:
            torch.set_float32_matmul_precision('high')
        except AttributeError:
            pass
        
        torch.backends.cudnn.benchmark = True

    if 'seed' in config.distributed:
        torch.manual_seed(seed)

    return device, is_logger


def increase_l2_fetch_granularity():
    try:
        import ctypes

        _libcudart = ctypes.CDLL('libcudart.so')
        # Set device limit on the current device
        # cudaLimitMaxL2FetchGranularity = 0x05
        pValue = ctypes.cast((ctypes.c_int*1)(), ctypes.POINTER(ctypes.c_int))
        _libcudart.cudaDeviceSetLimit(ctypes.c_int(0x05), ctypes.c_int(128))
        _libcudart.cudaDeviceGetLimit(pValue, ctypes.c_int(0x05))
        assert pValue.contents.value == 128
    except:
        return
