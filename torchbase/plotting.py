import matplotlib.pyplot as plt
from .typing import *

def plot(wrapper: Wrapper,
    metrics: Metriclikes, 
    save_to: nStr = None, 
    title: nStr = 'Model performance by epoch',
    xlabel: nStr = 'epochs', 
    ylabel: nStr = 'score',
    show_legend: bool = True,
    show_plot: bool = True) -> Wrapper:
    ''' Plot the training history. '''

    if wrapper.history == {}:
        raise RuntimeError('No training data found.')
    else:
        plt.style.use('ggplot')
        fig, ax = plt.subplots()

        if isinstance(metrics, str): 
            metrics = [metrics]

        for metric in metrics: 
            ax.plot(wrapper.history[metric], label = metric)

        if show_legend: ax.legend(loc = 'best')
        if title is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if save_to is not None: plt.savefig(save_to)
        if show_plot: plt.show()

        return wrapper
