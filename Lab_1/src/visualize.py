import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# def visualize_histogram(data: np.ndarray, n_bins: int, range: tuple[int], title: str, ylim: float=None, density: bool=False):
#     plt.hist(data, n_bins, range=range, density=density)
#     plt.xlabel("Value")
#     plt.ylabel("Frequency")
#     if ylim:
#         plt.ylim(ylim)
#     plt.title(title)

def visualize_histogram_subp(data: np.ndarray, n_bins: int, range: tuple[int], ax: plt.Axes, title: str, color: str=None, ylim: float=None, density: bool=False):
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.hist(data, n_bins, range=range, density=density, color=color)
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    if ylim:
        ax.set_ybound(0, ylim)
    ax.set_title(title)

def visualize_kde_subp(data: np.ndarray, ax: plt.Axes, title: str, bandwidth=0.5, color: str=None, fill: bool=True):
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.kdeplot(data, ax=ax, bw_adjust=bandwidth, fill=fill, color=color)
    ax.set_xlabel("Value")
    ax.set_title(title)