import matplotlib.pyplot as plt
from .typing import *

def plot(wrapper: Wrapper,
    metrics: nMetriclikesOrString = None,
    save_to: nStr = None, 
    title: nStr = 'Model performance by epoch',
    xlabel: nStr = 'epochs', 
    ylabel: nStr = 'score',
    show_legend: bool = True,
    show_plot: bool = True,
    plot_style: str = 'ggplot') -> Wrapper:
    ''' Plot the training history. '''

    if wrapper.history == {}:
        raise RuntimeError('No training data found.')
    else:
        plt.style.use(plot_style)
        fig, ax = plt.subplots()

        if metrics is None:
            metrics = wrapper.metrics
        elif isinstance(metrics, str): 
            metrics = [metric for metric in wrapper.metrics 
                       if metric.name == metrics]
        elif isinstance(metrics, list): 
            metrics = [metric for metric in wrapper.metrics 
                       if metric.name in metrics]

        for metric in metrics: 
            ax.plot(wrapper.history[metric.name], label = metric.name)

        if show_legend: ax.legend(loc = 'best')
        if title is not None: ax.set_title(title)
        if xlabel is not None: ax.set_xlabel(xlabel)
        if ylabel is not None: ax.set_ylabel(ylabel)
        if save_to is not None: plt.savefig(save_to)
        if show_plot: plt.show()

        return wrapper
