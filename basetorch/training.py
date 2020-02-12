import torch
from tqdm.auto import tqdm
from itertools import count
from functools import cmp_to_key
from .utils import str2sched
from .typing import *

def train_model(wrapper: Wrapper, train_loader: DataLoader, 
    val_loader: nDataLoader = None, epochs: nNumeric = None) -> Wrapper:

    # Initialise TensorBoard and add model graph
    if wrapper.tensorboard:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        writer.add_graph(wrapper.model, next(iter(train_loader))[0])
        wrapper.logger.info('Tensorboard active! Run "tensorboard '\
            '--logdir=runs" in a new terminal window to open it.')

    # Alias the loss function
    crit = wrapper.criterion

    # Set epochs and patience to infinity if they are None
    epochs = float('inf') if epochs is None else epochs
    patience = float('inf') if wrapper.patience is None else wrapper.patience
    if wrapper.monitor is None:
        monitor = 'loss' if val_loader is None else 'val_loss'
    else:
        monitor = wrapper.monitor
    wrapper.logger.debug(f'Monitoring {monitor}')

    # Initialise scheduler
    if isinstance(wrapper.scheduler, str):
        wrapper.scheduler = str2sched(wrapper.scheduler, 
            optimiser = wrapper.optimiser, 
            dataloader = train_loader,
            epochs = epochs,
            patience = patience
        )
        wrapper.logger.debug('Initialised '\
            f'{type(wrapper.scheduler).__name__} scheduler')

    # Bundle together the metrics and their names and values
    metric_data = {
        metric.name: {
            'metric': metric, 
            'value': 0.,
            'train': metric.name[:4] != 'val_'
        }
        for metric in wrapper.metrics
    }
    wrapper.logger.debug('Initialised metrics '\
        f'{[m.name for m in wrapper.metrics]}')

    # Initialise minimise_monitor
    if wrapper.minimise_monitor is None:
        if monitor == 'loss' or monitor == 'val_loss':
            minimise_monitor = True
        else:
            minimise_monitor = False
    else:
        minimise_monitor = wrapper.minimise_monitor

    # Initialise score_cmp and best_score
    if minimise_monitor:
        score_cmp = lambda x, y: x < y
        best_score = float('inf')
        wrapper.logger.debug(f'Minimising {monitor}')
    else:
        score_cmp = lambda x, y: x > y
        best_score = 0.
        wrapper.logger.debug(f'Maximising {monitor}')

    # Initialise history
    for name in metric_data.keys():
        if wrapper.history.get(name) is None:
            wrapper.history[name] = []
        elif wrapper.history[name]:
            metric_data[name]['value'] = wrapper.history[name][-1]
    wrapper.logger.debug('History initialised')

    # Initialise scores.
    # We're working with all scores rather than the current score since
    # lists are mutable and floats are not, allowing us to update
    # the score on the fly
    scores = wrapper.history[monitor]
    best_score = max([best_score] + scores, key = cmp_to_key(score_cmp))
    wrapper.logger.debug(f'Best score initialised to {best_score}')

    # Initialise starting epoch
    start_epoch = len(wrapper.history['loss'])
    wrapper.logger.debug(f'Starting epoch initialised to {start_epoch}')

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
    niter: int = 0
    for epoch in count(start = start_epoch):

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

                # Update the number of iterations
                niter += train_loader.batch_size

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
                        value = wrapper.smoothing * data['value'] + \
                                (1 - wrapper.smoothing) * value

                        # Bias correction
                        value /= 1 - wrapper.smoothing ** \
                                 (niter / (1 - wrapper.smoothing))

                        # Update the metric value
                        metric_data[name]['value'] = value

                        # Update Tensorboard
                        if wrapper.tensorboard:
                            wrapper.logger.debug(f'Adding {name} to '\
                                                 f'Tensorboard')
                            writer.add_scalar(name + '/train', value, niter)

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
                    wrapper.logger.debug(f'Adding {name} to history')
                    wrapper.history[name].append(data['value'])

            # Compute validation metrics
            if val_loader is not None:
                wrapper.logger.debug('Computing validation metrics')
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
                            wrapper.logger.debug(f'Adding {name} to history')
                            wrapper.history[name].append(data['value'])

                            # Update Tensorboard
                            if wrapper.tensorboard:
                                wrapper.logger.debug(f'Adding {name} to '\
                                                     f'Tensorboard')
                                name = name[4:] + '/val'
                                writer.add_scalar(name, data['value'], niter)

                    # Update progress bar description
                    desc = f'Epoch {epoch:2d}'
                    for name, data in metric_data.items():
                        desc += f' - {name} {data["value"]:.4f}'
                    epoch_pbar.set_description(desc)

        # Add score to learning scheduler
        if wrapper.scheduler is not None: wrapper.scheduler.step(scores[-1])

        # Update Tensorboard
        if wrapper.tensorboard:
            wrapper.logger.debug('Adding learning rate to Tensorboard')
            lr: float = wrapper.optimiser.param_groups[0]['lr']
            writer.add_scalar('learning_rate', lr, niter)

        # Save model if score is best so far
        if score_cmp(round(scores[-1], 4), round(best_score, 4)):
            best_score = scores[-1]
            wrapper.logger.debug(f'Best score is now {best_score:.4f}')

            # Delete older models and save the current one
            if wrapper.save_model:
                wrapper.logger.debug('Saving model')
                if wrapper.overwrite:
                    files = wrapper.data_dir.glob(f'{wrapper.model_name}*.pt')
                    for f in files: f.unlink()
                wrapper.save(f'{scores[-1]:.4f}_{monitor}')

            # Reset number of bad epochs
            if bad_epochs:
                wrapper.logger.debug('Bad epochs reset')
            bad_epochs = 0

        # Stop if score has not improved for <patience> many epochs
        else:
            bad_epochs += 1
            if bad_epochs == 1:
                wrapper.logger.debug(f'There is now {bad_epochs} bad epoch')
            else:
                wrapper.logger.debug(f'There are now {bad_epochs} bad epochs')

            if bad_epochs > patience:
                wrapper.logger.info('Model is not improving, stopping '\
                                    'training.')

                # Load the model with the best score
                glob = list(wrapper.data_dir.glob(f'{wrapper.model_name}*.pt'))
                if wrapper.save_model and glob != []:
                    checkpoint = torch.load(glob[0])
                    wrapper.history = checkpoint['history']
                    wrapper.load_state_dict(checkpoint['model_state_dict'])

                break

        # Stop when we perfom better than target_value
        if wrapper.target_value is not None:
            if score_cmp(scores[-1], wrapper.target_value):
                wrapper.logger.info('Reached target performance, stopping '\
                                    'training.')
                break

    return wrapper
