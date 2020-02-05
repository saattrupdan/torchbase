import torch
from pathlib import Path
from .typing import *

def save(wrapper: Wrapper, postfix: str = '') -> Wrapper:
    ''' Save a dictionary with the history and weights. '''

    params = {
        'model_state_dict': wrapper.model.state_dict(),
        'optimiser_state_dict': wrapper.optimiser.state_dict(),
        'history': wrapper.history
    }

    if wrapper.scheduler is not None:
        params['scheduler_state_dict'] = wrapper.scheduler.state_dict()

    torch.save(params, wrapper.data_dir / f'{wrapper.model_name}_{postfix}.pt')
    return wrapper

def save_model(wrapper: Wrapper, postfix: str = '') -> Wrapper:
    ''' Save model for inference. '''
    path = wrapper.data_dir / f'{wrapper.model_name}_{postfix}.pkl'
    torch.save(wrapper.model, path)
    return wrapper

def load(cls: Wrapper, model_name: str, data_dir: Pathlike = '.') -> Wrapper:

    # Fetch the full model path
    path = next(Path(str(data_dir)).glob(f'{model_name}*.pt'))

    # Load the checkpoint
    checkpoint = torch.load(path)

    params = {**checkpoint['params'], **{
        'optimiser': None,
        'scheduler': None,
    }}

    wrapper = cls(params)
    wrapper.load_state_dict(checkpoint['model_state_dict'])

    wrapper.history = checkpoint['history']
    wrapper.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    if wrapper.scheduler is not None:
        scheduler_state_dict = checkpoint['scheduler_state_dict']
        wrapper.scheduler.load_state_dict(scheduler_state_dict)

    return wrapper
