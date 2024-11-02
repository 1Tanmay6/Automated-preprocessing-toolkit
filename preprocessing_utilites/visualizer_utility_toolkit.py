import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union, Optional
import numpy as np
import os


class VisualizerUtilityToolkit:
    def __init__(self, save_dir: str = 'plots', display: bool = False):
        """
        Initializes the VisualizerUtilityToolkit with optional parameters for plot style, save directory, and display settings.

        :param save_dir: Directory where plots will be saved. Default is 'plots'.
        :param display: Boolean indicating whether to display the plots. Default is False.
        """
        self.style = 'darkgrid'
        self.palette = 'deep'
        self.figure_size = (10, 6)
        self.save_dir = save_dir
        self.plot_count = 0
        self.display = display
        sns.set_style(self.style)
        sns.set_palette(self.palette)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def __call__(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray]) -> plt.Any:
        self.change_style('whitegrid')
        self.change_palette('Set2')
        self.set_figure_size(12, 8)

        # Generate all plots with new style
        print("Generating all plots with new style...")
        self.generate_all_plots(
            x, y, x_label='X values', y_label='Y values')

    def _create_figure(self, figsize: Optional[tuple] = None) -> tuple:
        """
        Creates a figure with a specified size.

        :param figsize: Tuple specifying the width and height of the figure. Defaults to self.figure_size.
        :return: A tuple containing the figure and axis objects.
        """
        return plt.subplots(figsize=figsize or self.figure_size)

    def _save_plot(self, plot_type: str, display: bool = False):
        """
        Saves the plot to the specified directory and optionally displays it.

        :param plot_type: Type of the plot to be saved (e.g., 'scatter', 'line').
        :param display: Boolean indicating whether to display the plot after saving.
        """
        self.plot_count += 1
        filename = f"{self.plot_count:03d}_{plot_type}.png"
        plt.savefig(os.path.join(self.save_dir, filename))
        if display:
            plt.show()
        else:
            plt.close()

    def scatter_plot(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                     x_label: str = 'X', y_label: str = 'Y', title: str = 'Scatter Plot'):
        """
        Generates and saves a scatter plot.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param x_label: Label for the x-axis. Default is 'X'.
        :param y_label: Label for the y-axis. Default is 'Y'.
        :param title: Title of the plot. Default is 'Scatter Plot'.
        """
        fig, ax = self._create_figure()
        sns.scatterplot(x=x, y=y, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self._save_plot('scatter', self.display)
