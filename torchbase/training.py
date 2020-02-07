import torch
from tqdm.auto import tqdm
from itertools import count
from functools import cmp_to_key
from .utils import str2sched
from .typing import *

def train_model(wrapper: Wrapper,
    train_loader: DataLoader, 
    val_loader: nDataLoader = None,
    epochs: nNumeric = None,
    monitor: nStr = None,
    minimise_monitor: nBool = None,
    target_value: nFloat = None,
    patience: nNumeric = 10,
    smoothing: float = 0.99, 
    pred_threshold: float = 0.5, 
    save_model: bool = True,
    overwrite: bool = True) -> Wrapper:

    # Alias the loss function
    crit = wrapper.criterion

    # Set epochs and patience to infinity if they are None
    if epochs is None: epochs = float('inf')
    if patience is None: patience = float('inf')
    if monitor is None:
        monitor = 'loss' if val_loader is None else 'val_loss'

    # Initialise scheduler
    if isinstance(wrapper.scheduler, str):
        wrapper.scheduler = str2sched(wrapper.scheduler, 
            optimiser = wrapper.optimiser, 
            dataloader = train_loader,
            epochs = epochs,
            patience = patience
        )

    # Bundle together the metrics and their names and values
    metric_data = {
        metric.name: {
            'metric': metric, 
            'value': 0.,
            'train': metric.name[:4] != 'val_'
        }
        for metric in wrapper.metrics
    }

    # Initialise minimise_monitor
    if minimise_monitor is None:
        if monitor == 'loss' or monitor == 'val_loss':
            minimise_monitor = True
        else:
            minimise_monitor = False

    # Initialise score_cmp and best_score
    if minimise_monitor:
        score_cmp = lambda x, y: x < y
        best_score = float('inf')
    else:
        score_cmp = lambda x, y: x > y
        best_score = 0.

    # Initialise history
    for name in metric_data.keys():
        if wrapper.history.get(name) is None:
            wrapper.history[name] = []
        elif wrapper.history[name]:
            metric_data[name]['value'] = wrapper.history[name][-1]

    # Initialise scores.
    # We're working with all scores rather than the current score since
    # lists are mutable and floats are not, allowing us to update
    # the score on the fly
    scores = wrapper.history[monitor]
    best_score = max([best_score] + scores, key = cmp_to_key(score_cmp))

    # Initialise starting epoch
    start_epoch = len(wrapper.history['loss'])

    # Initialise the number of bad epochs; i.e. epochs with no improvement
    bad_epochs = 0

    # Training announcement
    ntrain = len(train_loader) * train_loader.batch_size
    if val_loader is None: 
        wrapper.logger.info(f'Training on {ntrain:,d} samples')
    else: 
        nval = len(val_loader) * val_loader.batch_size
        wrapper.logger.info(f'Training on {ntrain:,d} samples and '\
                         f'validating on {nval:,d} samples')

    wrapper.logger.info(f'Number of trainable parameters: '\
                     f'{wrapper.trainable_params():,d}')

    # Training loop
    for epoch in count(start = start_epoch):

        # Epoch announcement
        wrapper.logger.debug(f'Commencing epoch {epoch}')

        # Enable training mode
        wrapper.train()

        # Stop if we have reached the total number of epochs
        if epoch >= start_epoch + epochs:
            wrapper.logger.info(f'Reached {epochs} epochs, stopping training')
            break

        # Epoch loop
        nsamples = len(train_loader) * train_loader.batch_size
        with tqdm(total = nsamples) as epoch_pbar:
            epoch_pbar.set_description(f'Epoch {epoch:2d}')
            for idx, (xtrain, ytrain) in enumerate(train_loader):

                # Reset the gradients
                wrapper.optimiser.zero_grad()

                # Do a forward pass, calculate loss and backpropagate
                yhat = wrapper.forward(xtrain)
                try:
                    loss = crit(yhat, ytrain)
                except ValueError:
                    raise ValueError(f'The shape of the prediction, '\
                        f'{yhat.shape}, is different to the shape of '\
                        f'the target value, {ytrain.shape}, which is not '\
                        f'allowed when using the loss function {type(crit)}. '\
                        f'Maybe you need to change it to a categorical '\
                        f'variant?')
                loss.backward()
                wrapper.optimiser.step()

                # Exponentially moving average of training metrics
                # Note: The float() is there to copy the loss by value
                #       and not by reference, to allow it to be garbage
                #       collected and avoid an excessive memory leak
                for name, data in metric_data.items():
                    if data['train']:

                        # Get metric value
                        if name == 'loss':
                            value = float(loss)
                        else:
                            value = data['metric'](yhat.detach(), ytrain)

                        # Moving average
                        value = smoothing * data['value'] + \
                                (1 - smoothing) * value

                        # Bias correction
                        exponent: float = epoch * nsamples 
                        exponent += (idx + 1) * train_loader.batch_size
                        exponent /= 1 - smoothing
                        value /= 1 - smoothing ** exponent

                        # Update the metric value
                        metric_data[name]['value'] = value

                # Update progress bar description
                desc = f'Epoch {epoch:2d}'
                for name, data in metric_data.items():
                    if data['train']:
                        desc += f' - {name} {data["value"]:.4f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.update(train_loader.batch_size)

            # Add training scores to history
            for name, data in metric_data.items():
                if data['train']:
                    wrapper.history[name].append(data['value'])

            # Compute validation metrics
            if val_loader is not None:
                with torch.no_grad():

                    # Enable validation mode
                    wrapper.eval()

                    for xval, yval in val_loader:
                        yhat = wrapper.forward(xval)
                        for name, data in metric_data.items():
                            if not data['train']:
                                if name == 'val_loss':
                                    value = float(crit(yhat, yval))
                                else:
                                    value = data['metric'](yhat.detach(), yval)
                                metric_data[name]['value'] += value

                    # Calculate average values of validation metrics
                    for name, data in metric_data.items():
                        if not data['train']:
                            metric_data[name]['value'] /= len(val_loader)

                    # Add training scores to history
                    for name, data in metric_data.items():
                        if not data['train']:
                            wrapper.history[name].append(data['value'])

                    # Update progress bar description
                    desc = f'Epoch {epoch:2d}'
                    for name, data in metric_data.items():
                        desc += f' - {name} {data["value"]:.4f}'
                    epoch_pbar.set_description(desc)

        # Add score to learning scheduler
        if wrapper.scheduler is not None: wrapper.scheduler.step(scores[-1])

        # Save model if score is best so far
        if score_cmp(scores[-1], best_score):  
            best_score = scores[-1]

        # Delete older models and save the current one
        if save_model:
            if overwrite:
                for f in wrapper.data_dir.glob(f'{wrapper.model_name}*.pt'): 
                    f.unlink()
            wrapper.save(f'{scores[-1]:.4f}_{monitor}')

        # Stop if score has not improved for <patience> many epochs
        if score_cmp(best_score, scores[-1]):
            bad_epochs += 1
            if bad_epochs > patience:
                wrapper.logger.info('Model is not improving, stopping '\
                                 'training.')

                # Load the model with the best score
                glob = list(wrapper.data_dir.glob(f'{wrapper.model_name}*.pt'))
                if save_model and glob != []:
                    checkpoint = torch.load(glob[0])
                    wrapper.history = checkpoint['history']
                    wrapper.load_state_dict(checkpoint['model_state_dict'])

                break

        # If score *has* improved then reset <bad_epochs>
        else: bad_epochs = 0

        # Stop when we perfom better than <target_value>
        if target_value is not None:
            if score_cmp(scores[-1], target_value):
                wrapper.logger.info('Reached target performance, stopping '\
                                 'training.')
                break

    return wrapper
