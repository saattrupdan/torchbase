import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from itertools import count
from functools import cmp_to_key
from .utils import getname
from .typing import *

def train_model(wrapper: Wrapper,
    train_loader: DataLoader, 
    val_loader: nDataLoader = None,
    epochs: nNumeric = None,
    monitor: nStr = None,
    minimise_monitor: nBool = None,
    target_value: nFloat = None,
    patience: nNumeric = 10,
    smoothing: float = 0.999, 
    pred_threshold: float = 0.5, 
    save_model: bool = True,
    overwrite: bool = True,
    #train_metrics: Floats = [],
    metrics: Metriclikes = []) -> Wrapper:

    # Set epochs and patience to infinity if they are None
    if epochs is None: epochs = float('inf')
    if patience is None: patience = float('inf')
    if monitor is None:
        monitor = 'loss' if val_loader is None else 'val_loss'

    ## Construct list of training metric names
    #train_metric_names = []
    #train_metrics = ['loss'] + train_metrics
    #for train_metric in train_metrics:
    #    if isinstance(train_metric, str): 
    #        train_metric_names.append(train_metric)
    #    elif isinstance(train_metric, Metric):
    #        train_metric_names.append(train_metric.__name__)
    #    else:
    #        train_metric_names.append(type(train_metric).__name__)

    ## Construct list of validation metric names
    #val_metric_names = []
    #val_metrics = ['loss'] + val_metrics
    #for val_metric in val_metrics:
    #    if isinstance(val_metric, str): 
    #        val_metric_names.append('val_' + val_metric)
    #    elif isinstance(val_metric, Metric):
    #        val_metric_names.append('val_' + val_metric.__name__)
    #    else:
    #        val_metric_names.append('val_' + type(val_metric).__name__)

    ## Initialise training metrics
    #for idx, metric in enumerate(train_metrics):
    #    if metric == 'loss':
    #        train_metrics[idx] = wrapper.criterion

    ## Initialise validation metrics
    #for idx, metric in enumerate(val_metrics):
    #    if metric == 'loss':
    #        val_metrics[idx] = wrapper.criterion

    metric_data = [(getname(metric), str2metric(metric, wrapper), 0.) 
                   for metric in metrics]

    # Initialise history
    if wrapper.history == {}:
        for name, _, _ in metric_data: 
            wrapper.history[name] = []

    # Initialise the number of bad epochs; i.e. epochs with no improvement
    bad_epochs = 0

    # Initialise starting epoch
    if wrapper.history != {}:
        start_epoch = len(wrapper.history[list(wrapper.history.keys())[0]])
    else:
        start_epoch = 0

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
        best_score = 0

    # Initialise list of scores.
    # We're working with all scores rather than the current score since
    # lists are mutable and floats are not, allowing us to update
    # the score on the fly
    scores = wrapper.history[monitor]

    # If we're resuming training then fetch the best score
    if start_epoch > 0:
        for idx, name in enumerate(train_metric_names):
            train_metric_vals[idx] = wrapper.history[name][-1]
        for idx, name in enumerate(val_metric_names):
            val_metric_vals[idx] = wrapper.history[name][-1]
        best_score = max(scores, key = cmp_to_key(score_cmp))

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

                # Batch announcement
                wrapper.logger.debug(f'Commencing batch {idx}')

                # Reset the gradients
                wrapper.optimiser.zero_grad()

                # Do a forward pass, calculate loss and backpropagate
                yhat = wrapper.forward(xtrain)
                loss = wrapper.criterion(yhat, ytrain)
                loss.backward()
                wrapper.optimiser.step()

                # Exponentially moving average of training metrics
                # Note: The float() is there to copy the loss by value
                #       and not by reference, to allow it to be garbage
                #       collected and avoid an excessive memory leak
                yhat = yhat.detach()
                for idx, metric in enumerate(train_metrics):
                    if not isinstance(metric, str):
                        if idx == 0:
                            metric_val = float(loss)
                        else:
                            yhat = yhat > 0.5
                            metric_val = float(metric(yhat, ytrain))
                        train_metric_vals[idx] = \
                            smoothing * train_metric_vals[idx] + \
                            (1 - smoothing) * metric_val

                # Bias correction
                exponent = epoch * nsamples 
                exponent += (idx + 1) * train_loader.batch_size
                exponent /= 1 - smoothing
                for idx in range(len(train_metric_vals)):
                    train_metric_vals[idx] /= 1 - smoothing ** exponent

                # Update progress bar description
                desc = f'Epoch {epoch:2d}'
                for idx, name in enumerate(train_metric_names):
                    desc += f' - {name} {train_metric_vals[idx]:.4f}'
                epoch_pbar.set_description(desc)
                epoch_pbar.update(train_loader.batch_size)

            # Add training scores to history
            for n, v in zip(train_metric_names, train_metric_vals):
                wrapper.history[n].append(v)

            # Compute validation metrics
            if val_loader is not None:
                with torch.no_grad():

                    # Enable validation mode
                    wrapper.eval()

                    # Initialise validation metrics
                    for idx in range(len(val_metric_vals)): 
                        val_metric_vals[idx] = 0.

                    for xval, yval in val_loader:
                        yhat = wrapper.forward(xval)
                        for idx, metric in enumerate(val_metrics):
                            if not isinstance(metric, str):
                                val_metric_vals[idx] += metric(yhat, yval)

                    # Calculate average values of validation metrics
                    for idx in range(len(val_metric_vals)):
                        val_metric_vals[idx] /= len(val_loader)

                    # Add validation scores to history
                    for n, v in zip(val_metric_names, val_metric_vals):
                        wrapper.history[n].append(v)

                    # Update progress bar description
                    desc = f'Epoch {epoch:2d}'
                    for n, v in zip(train_metric_names, train_metric_vals):
                        desc += f' - {n} {v:.4f}'
                    for n, v in zip(val_metric_names, val_metric_vals):
                        desc += f' - {n} {v:.4f}'
                    epoch_pbar.set_description(desc)

        # Add score to learning scheduler
        if wrapper.scheduler is not None: wrapper.scheduler.step(scores[-1])

        # Save model if score is best so far
        if score_cmp(scores[-1], best_score):  best_score = scores[-1]

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
                glob = wrapper.data_dir.glob(f'{wrapper.model_name}*.pt')
                if save_model and list(glob):
                    checkpoint = torch.load(path)
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
