import torch
import logging
import matplotlib.pyplot as plt
from typing import Union, Dict, Callable, Sequence, Iterable
from tqdm.auto import tqdm
from itertools import count
from pathlib import Path
from functools import cmp_to_key, partial

# Type aliases
Metric = Callable[[torch.Tensor, torch.Tensor], float]

def parametrised(decorator):
    ''' A meta-decorator that enables parameters in inner decorator. '''
    def parametrised_decorator(*args, **kwargs):
        def repl(cls):
            return decorator(cls, *args, **kwargs)
        return repl
    return parametrised_decorator

@parametrised
def magic(cls,
    optimiser: Union[torch.optim.Optimizer, str, None] = 'adamw',
    scheduler: Union[torch.optim.lr_scheduler._LRScheduler, str, None]\
        = 'reduce_on_plateau',
    criterion: Union[Metric, str] = 'binary_cross_entropy',
    data_dir: Union[Path, str] = '.',
    verbose: int = 1):
    ''' Adds more functionality to a PyTorch Module. '''

    class MagicModule(Module):
        _model_class = cls
        _optimiser = optimiser
        _scheduler = scheduler
        _criterion = criterion
        _data_dir = data_dir
        _verbose = verbose

    return MagicModule

class Module:
    ''' A PyTorch module with logging and training built-in.

    INPUT
        model_class: class
            A subclass of torch.nn.Module
        criterion: Metric or str = 'binary_cross_entropy'
            The loss function used for training. The following string
            values are permitted: 
                'mean_absolute_error'
                'mean_squared_error'
                'binary_cross_entropy'
                'categorical_cross_entropy'
                'neg_log_likelihood'
                'binary_cross_entropy_with_logits'
                'ctc'
        optimiser: Optimiser, str or None
            The optimiser used for training. The following string values
            are permitted: 
                'adam'
                'adadelta'
                'adagrad'
                'adamw'
                'sparse_adam'
                'adamax'
                'rmsprop'
                'sgd'
        scheduler: Scheduler, str or None
            The optimiser used for training. The following string values
            are permitted: 
                'reduce_on_plateau'
        data_dir: Path or str = '.'
            The data directory
        verbose: int = 1
            Verbosity level, can be 0, 1 or 2
    '''

    _model_class = None
    _optimiser = None
    _scheduler = None
    _criterion = None
    _data_dir = None
    _verbose = None

    def __init__(self, *args, **kwargs):
        self.model = type(self)._model_class(*args, **kwargs)
        self.model_name = type(self)._model_class.__name__
        self.optimiser = type(self)._optimiser
        self.scheduler = type(self)._scheduler
        self.criterion = type(self)._criterion
        self.data_dir = type(self)._data_dir
        self.verbose = type(self)._verbose
        self.history = {}
        self.compile()

    def compile(self):
        ''' Compile the criterion, optimiser and scheduler. '''

        # Initialise loss function
        if self.criterion == 'mean_absolute_error':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion == 'mean_squared_error':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion == 'binary_cross_entropy':
            self.criterion = torch.nn.BCELoss()
        elif self.criterion == 'categorical_cross_entropy':
            self.criterion = torch.nn.NLLLoss()
        elif self.criterion == 'neg_log_likelihood':
            self.criterion = torch.nn.NLLLoss()
        elif self.criterion == 'binary_cross_entropy_with_logits':
            self.criterion = torch.nn.BCEWithLogitsLoss()
        elif self.criterion == 'ctc':
            self.criterion = torch.nn.CTCLoss()
        else:
            raise RuntimeError(f'Criterion {self.criterion} not found.')

        # Initialise self.optimiser
        params = self.model.parameters()
        if self.optimiser == 'adam':
            self.optimiser = torch.optim.Adam(params)
        elif self.optimiser == 'adadelta':
            self.optimiser = torch.optim.AdaDelta(params)
        elif self.optimiser == 'adagrad':
            self.optimiser = torch.optim.AdaGrad(params)
        elif self.optimiser == 'adamw':
            self.optimiser = torch.optim.AdamW(params)
        elif self.optimiser == 'sparse_adam':
            self.optimiser = torch.optim.SparseAdam(params)
        elif self.optimiser == 'adamax':
            self.optimiser = torch.optim.Adamax(params)
        elif self.optimiser == 'rmsprop':
            self.optimiser = torch.optim.RMSProp(params)
        elif self.optimiser == 'sgd':
            self.optimiser = torch.optim.SGD(params)
        else:
            raise RuntimeError(f'Optimiser {self.optimiser} not found.')

        # Initialise self.scheduler
        if isinstance(self.scheduler, str):
            if self.scheduler == 'reduce_on_plateau':
                self.scheduler = \
                    torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer = self.optimiser 
                    )
            else:
                raise RuntimeError(f'Scheduler {self.scheduler} not found.')
        else:
            self.scheduler = self.scheduler(optimizer = self.optimiser)

        # Set up logging
        logging.basicConfig()
        logging.root.setLevel(logging.NOTSET)
        self.logger = logging.getLogger()

        # Set logging level
        if self.verbose == 0:
            self.logger.setLevel(logging.WARNING)
        elif self.verbose == 1:
            self.logger.setLevel(logging.INFO)
        elif self.verbose == 2:
            self.logger.setLevel(logging.DEBUG)

        # Initialise data_dir
        self.data_dir = Path(str(self.data_dir))

        return self

    def trainable_params(self) -> int:
        ''' Returns the number of trainable parameters in the model. '''
        return sum(param.numel() for param in self.model.parameters() 
                if param.requires_grad)

    def is_cuda(self) -> bool:
        return next(self.model.parameters()).is_cuda

    def plot(self, 
        metrics: Union[list, str], 
        save_to: Union[str, None] = None, 
        title: Union[str, None] = 'Model performance by epoch',
        xlabel: Union[str, None] = 'epochs', 
        ylabel: Union[str, None] = 'score',
        show_legend: bool = True,
        show_plot: bool = True):
        ''' Plot the training history. '''

        if self.history == {}:
            raise RuntimeError('No training data found.')
        else:
            plt.style.use('ggplot')
            fig, ax = plt.subplots()

            if isinstance(metrics, str): 
                metrics = [metrics]

            for metric in metrics: 
                ax.plot(self.history[metric], label = metric)

            if show_legend: ax.legend(loc = 'best')
            if title is not None: ax.set_title(title)
            if xlabel is not None: ax.set_xlabel(xlabel)
            if ylabel is not None: ax.set_ylabel(ylabel)
            if save_to is not None: plt.savefig(save_to)
            if show_plot: plt.show()
            return self

    def save(self, postfix: str = ''):
        ''' Save a dictionary with the history and weights. '''

        params = {
            'model_state_dict': self.model.state_dict(),
            'optimiser_state_dict': self.optimiser.state_dict(),
            'history': self.history
        }

        if self.scheduler is not None:
            params['scheduler_state_dict'] = self.scheduler.state_dict()

        torch.save(params, self.data_dir / f'{self.model_name}_{postfix}.pt')
        return self

    def save_model(self, postfix: str = ''):
        ''' Save model for inference. '''
        path = self.data_dir / f'{self.model_name}_{postfix}.pkl'
        torch.save(self.model, path)
        return self

    @classmethod
    def load(cls, model_name: str, data_dir: Union[Path, str] = '.'):

        # Fetch the full model path
        path = next(Path(str(data_dir)).glob(f'{model_name}*.pt'))

        # Load the checkpoint
        checkpoint = torch.load(path)

        params = {**checkpoint['params'], **{
            'optimiser': None,
            'scheduler': None,
        }}

        model = cls(params)
        model.load_state_dict(checkpoint['model_state_dict'])

        model.history = checkpoint['history']
        model.optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
        if model.scheduler is not None:
            scheduler_state_dict = checkpoint['scheduler_state_dict']
            model.scheduler.load_state_dict(scheduler_state_dict)

        return model

    def fit(self, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: Union[torch.utils.data.DataLoader, None] = None,
        epochs: Union[int, float, None] = None,
        monitor: Union[str, None] = None,
        minimise_monitor: Union[bool, None] = None,
        target_value: Union[float, None] = None,
        patience: Union[int, float, None] = 10,
        smoothing: float = 0.999, 
        pred_threshold: float = 0.5, 
        save_model: bool = True,
        overwrite: bool = True,
        train_metrics: list = [],
        val_metrics: list = []):

        # Set epochs and patience to infinity if they are None
        if epochs is None: epochs = float('inf')
        if patience is None: patience = float('inf')
        if monitor is None:
            monitor = 'loss' if val_loader is None else 'val_loss'

        # Construct list of training metric names
        train_metric_names = []
        train_metrics = ['loss'] + train_metrics
        for train_metric in train_metrics:
            if isinstance(train_metric, str): 
                train_metric_names.append(train_metric)
            elif isinstance(train_metric, Metric):
                train_metric_names.append(train_metric.__name__)
            else:
                train_metric_names.append(type(train_metric).__name__)

        # Construct list of validation metric names
        val_metric_names = []
        val_metrics = ['loss'] + val_metrics
        for val_metric in val_metrics:
            if isinstance(val_metric, str): 
                val_metric_names.append('val_' + val_metric)
            elif isinstance(val_metric, Metric):
                val_metric_names.append('val_' + val_metric.__name__)
            else:
                val_metric_names.append('val_' + type(val_metric).__name__)

        # Initialise training metrics
        for idx, metric in enumerate(train_metrics):
            if metric == 'loss':
                train_metrics[idx] = self.criterion

        # Initialise validation metrics
        for idx, metric in enumerate(val_metrics):
            if metric == 'loss':
                val_metrics[idx] = self.criterion

        # Initialise metric values
        train_metric_vals = [0. for _ in train_metrics]
        val_metric_vals = [0. for _ in val_metrics]

        # Initialise history
        if self.history == {}:
            for name in set(train_metric_names).union(val_metric_names): 
                self.history[name] = []

        # Initialise the number of bad epochs; i.e. epochs with no improvement
        bad_epochs = 0

        # Initialise starting epoch
        if self.history != {}:
            start_epoch = len(self.history[list(self.history.keys())[0]])
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
        scores = self.history[monitor]

        # If we're resuming training then fetch the best score
        if start_epoch > 0:
            for idx, name in enumerate(train_metric_names):
                train_metric_vals[idx] = self.history[name][-1]
            for idx, name in enumerate(val_metric_names):
                val_metric_vals[idx] = self.history[name][-1]
            best_score = max(scores, key = cmp_to_key(score_cmp))

        # Training announcement
        ntrain = len(train_loader) * train_loader.batch_size
        if val_loader is None: 
            self.logger.info(f'Training on {ntrain:,d} samples')
        else: 
            nval = len(val_loader) * val_loader.batch_size
            self.logger.info(f'Training on {ntrain:,d} samples and '\
                             f'validating on {nval:,d} samples')

        self.logger.info(f'Number of trainable parameters: '\
                         f'{self.trainable_params():,d}')

        # Training loop
        for epoch in count(start = start_epoch):

            # Epoch announcement
            self.logger.debug(f'Commencing epoch {epoch}')

            # Enable training mode
            self.train()

            # Stop if we have reached the total number of epochs
            if epoch >= start_epoch + epochs:
                self.logger.info(f'Reached {epochs} epochs, stopping training')
                break

            # Epoch loop
            nsamples = len(train_loader) * train_loader.batch_size
            with tqdm(total = nsamples) as epoch_pbar:
                epoch_pbar.set_description(f'Epoch {epoch:2d}')
                for idx, (xtrain, ytrain) in enumerate(train_loader):

                    # Batch announcement
                    self.logger.debug(f'Commencing batch {idx}')

                    # Reset the gradients
                    self.optimiser.zero_grad()

                    # Do a forward pass, calculate loss and backpropagate
                    yhat = self.forward(xtrain)
                    loss = self.criterion(yhat, ytrain)
                    loss.backward()
                    self.optimiser.step()

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
                    self.history[n].append(v)

                # Compute validation metrics
                if val_loader is not None:
                    with torch.no_grad():

                        # Enable validation mode
                        self.eval()

                        # Initialise validation metrics
                        for idx in range(len(val_metric_vals)): 
                            val_metric_vals[idx] = 0.

                        for xval, yval in val_loader:
                            yhat = self.forward(xval)
                            for idx, metric in enumerate(val_metrics):
                                if not isinstance(metric, str):
                                    val_metric_vals[idx] += metric(yhat, yval)

                        # Calculate average values of validation metrics
                        for idx in range(len(val_metric_vals)):
                            val_metric_vals[idx] /= len(val_loader)

                        # Add validation scores to history
                        for n, v in zip(val_metric_names, val_metric_vals):
                            self.history[n].append(v)

                        # Update progress bar description
                        desc = f'Epoch {epoch:2d}'
                        for n, v in zip(train_metric_names, train_metric_vals):
                            desc += f' - {n} {v:.4f}'
                        for n, v in zip(val_metric_names, val_metric_vals):
                            desc += f' - {n} {v:.4f}'
                        epoch_pbar.set_description(desc)

            # Add score to learning scheduler
            if self.scheduler is not None: self.scheduler.step(scores[-1])

            # Save model if score is best so far
            if score_cmp(scores[-1], best_score):  best_score = scores[-1]

            # Delete older models and save the current one
            if save_model:
                if overwrite:
                    for f in self.data_dir.glob(f'{self.model_name}*.pt'): 
                        f.unlink()
                self.save(f'{scores[-1]:.4f}_{monitor}')

            # Stop if score has not improved for <patience> many epochs
            if score_cmp(best_score, scores[-1]):
                bad_epochs += 1
                if bad_epochs > patience:
                    self.logger.info('Model is not improving, stopping '\
                                     'training.')

                    # Load the model with the best score
                    glob = self.data_dir.glob(f'{self.model_name}*.pt')
                    if save_model and list(glob):
                        checkpoint = torch.load(path)
                        self.history = checkpoint['history']
                        self.load_state_dict(checkpoint['model_state_dict'])

                    break

            # If score *has* improved then reset <bad_epochs>
            else: bad_epochs = 0

            # Stop when we perfom better than <target_value>
            if target_value is not None:
                if score_cmp(scores[-1], target_value):
                    self.logger.info('Reached target performance, stopping '\
                                     'training.')
                    break

        return self

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __repr__(self, *args, **kwargs):
        return self.model.__repr__(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.model.train(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.model.eval(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self.model.to(*args, **kwargs)

    def cuda(self, *args, **kwargs):
        return self.model.cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        return self.model.cpu(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.model.load_state_dict(*args, **kwargs)
