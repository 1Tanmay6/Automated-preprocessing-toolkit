import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Union, Optional
import numpy as np
import os
from scipy import stats


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

    def line_plot(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                  x_label: str = 'X', y_label: str = 'Y', title: str = 'Line Plot'):
        """
        Generates and saves a line plot.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param x_label: Label for the x-axis. Default is 'X'.
        :param y_label: Label for the y-axis. Default is 'Y'.
        :param title: Title of the plot. Default is 'Line Plot'.
        """
        fig, ax = self._create_figure()
        sns.lineplot(x=x, y=y, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self._save_plot('line', self.display)

    def regression_plot(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                        x_label: str = 'X', y_label: str = 'Y', title: str = 'Regression Plot'):
        """
        Generates and saves a regression plot.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param x_label: Label for the x-axis. Default is 'X'.
        :param y_label: Label for the y-axis. Default is 'Y'.
        :param title: Title of the plot. Default is 'Regression Plot'.
        """
        fig, ax = self._create_figure()
        sns.regplot(x=x, y=y, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self._save_plot('regression', self.display)

    def histogram(self, data: Union[List, np.ndarray], label: str = 'Data',
                  title: str = 'Histogram', bins: int = 'auto', kde: bool = True):
        """
        Generates and saves a histogram.

        :param data: Data for the histogram.
        :param label: Label for the x-axis. Default is 'Data'.
        :param title: Title of the plot. Default is 'Histogram'.
        :param bins: Number of bins for the histogram. Default is 'auto'.
        :param kde: Boolean indicating whether to include a KDE plot. Default is True.
        """
        fig, ax = self._create_figure()
        sns.histplot(data=data, bins=bins, kde=kde, ax=ax)
        ax.set_xlabel(label)
        ax.set_title(title)
        self._save_plot('histogram', self.display)

    def box_plot(self, data: Union[List, np.ndarray], label: str = 'Data',
                 title: str = 'Box Plot'):
        """
        Generates and saves a box plot.

        :param data: Data for the box plot.
        :param label: Label for the y-axis. Default is 'Data'.
        :param title: Title of the plot. Default is 'Box Plot'.
        """
        fig, ax = self._create_figure()
        sns.boxplot(y=data, ax=ax)
        ax.set_ylabel(label)
        ax.set_title(title)
        self._save_plot('box', self.display)

    def violin_plot(self, data: Union[List, np.ndarray], label: str = 'Data',
                    title: str = 'Violin Plot'):
        """
        Generates and saves a violin plot.

        :param data: Data for the violin plot.
        :param label: Label for the y-axis. Default is 'Data'.
        :param title: Title of the plot. Default is 'Violin Plot'.
        """
        fig, ax = self._create_figure()
        sns.violinplot(y=data, ax=ax)
        ax.set_ylabel(label)
        ax.set_title(title)
        self._save_plot('violin', self.display)

    def joint_plot(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                   x_label: str = 'X', y_label: str = 'Y', title: str = 'Joint Plot',
                   kind: str = 'scatter'):
        """
        Generates and saves a joint plot.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param x_label: Label for the x-axis. Default is 'X'.
        :param y_label: Label for the y-axis. Default is 'Y'.
        :param title: Title of the plot. Default is 'Joint Plot'.
        :param kind: Type of joint plot. Default is 'scatter'.
        """
        joint_plot = sns.jointplot(x=x, y=y, kind=kind)
        joint_plot.fig.suptitle(title, y=1.02)
        joint_plot.set_axis_labels(x_label, y_label)
        self._save_plot('joint', self.display)

    def residual_plot(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                      x_label: str = 'X', y_label: str = 'Y', title: str = 'Residual Plot'):
        """
        Generates and saves a residual plot.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param x_label: Label for the x-axis. Default is 'X'.
        :param y_label: Label for the y-axis. Default is 'Y'.
        :param title: Title of the plot. Default is 'Residual Plot'.
        """
        fig, ax = self._create_figure()
        sns.residplot(x=x, y=y, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self._save_plot('residual', self.display)

    def kde_plot(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                 x_label: str = 'X', y_label: str = 'Y', title: str = 'KDE Plot'):
        """
        Generates and saves a KDE plot.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param x_label: Label for the x-axis. Default is 'X'.
        :param y_label: Label for the y-axis. Default is 'Y'.
        :param title: Title of the plot. Default is 'KDE Plot'.
        """
        fig, ax = self._create_figure()
        sns.kdeplot(x=x, y=y, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self._save_plot('kde', self.display)

    def correlation_heatmap(self, data: pd.DataFrame, title: str = 'Correlation Heatmap'):
        """
        Generates and saves a correlation heatmap.

        :param data: DataFrame containing the data for the heatmap.
        :param title: Title of the plot. Default is 'Correlation Heatmap'.
        """
        fig, ax = self._create_figure()
        sns.heatmap(data.corr(), annot=True,
                    cmap='coolwarm', linewidths=0.5, ax=ax)
        ax.set_title(title)
        self._save_plot('heatmap', self.display)

    def pairplot(self, data: pd.DataFrame, title: str = 'Pairplot'):
        """
        Generates and saves a pairplot.

        :param data: DataFrame containing the data for the pairplot.
        :param title: Title of the plot. Default is 'Pairplot'.
        """
        pairplot = sns.pairplot(data, diag_kind='kde')
        pairplot.fig.suptitle(title, y=1.02)
        self._save_plot('pairplot', self.display)

    def bar_plot(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                 x_label: str = 'X', y_label: str = 'Y', title: str = 'Bar Plot'):
        """
        Generates and saves a bar plot.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param x_label: Label for the x-axis. Default is 'X'.
        :param y_label: Label for the y-axis. Default is 'Y'.
        :param title: Title of the plot. Default is 'Bar Plot'.
        """
        fig, ax = self._create_figure()
        sns.barplot(x=x, y=y, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self._save_plot('bar', self.display)

    def count_plot(self, x: Union[List, np.ndarray], label: str = 'Categories',
                   title: str = 'Count Plot'):
        """
        Generates and saves a count plot.

        :param x: Data for the x-axis.
        :param label: Label for the x-axis. Default is 'Categories'.
        :param title: Title of the plot. Default is 'Count Plot'.
        """
        fig, ax = self._create_figure()
        sns.countplot(x=x, ax=ax)
        ax.set_xlabel(label)
        ax.set_title(title)
        self._save_plot('count', self.display)

    def distribution_plot(self, data: Union[List, np.ndarray], label: str = 'Data',
                          title: str = 'Distribution Plot'):
        """
        Generates and saves a distribution plot.

        :param data: Data for the distribution plot.
        :param label: Label for the x-axis. Default is 'Data'.
        :param title: Title of the plot. Default is 'Distribution Plot'.
        """
        fig, ax = self._create_figure()
        sns.histplot(x=data, ax=ax)
        ax.set_xlabel(label)
        ax.set_title(title)
        self._save_plot('distribution', self.display)

    def swarm_plot(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                   x_label: str = 'X', y_label: str = 'Y', title: str = 'Swarm Plot'):
        """
        Generates and saves a swarm plot.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param x_label: Label for the x-axis. Default is 'X'.
        :param y_label: Label for the y-axis. Default is 'Y'.
        :param title: Title of the plot. Default is 'Swarm Plot'.
        """
        fig, ax = self._create_figure()
        sns.swarmplot(x=x, y=y, ax=ax)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        self._save_plot('swarm', self.display)

    def qq_plot(self, data: Union[List, np.ndarray], label: str = 'Data',
                title: str = 'Q-Q Plot'):
        """
        Generates and saves a Q-Q plot.

        :param data: Data for the Q-Q plot.
        :param label: Label for the x-axis. Default is 'Data'.
        :param title: Title of the plot. Default is 'Q-Q Plot'.
        """
        fig, ax = self._create_figure()
        stats.probplot(data, dist="norm", plot=ax)
        ax.set_title(title)
        self._save_plot('qq', self.display)

    def change_style(self, style: str):
        """
        Changes the plot style.

        :param style: Style name (e.g., 'darkgrid', 'whitegrid').
        """
        self.style = style
        sns.set_style(self.style)

    def change_palette(self, palette: str):
        """
        Changes the plot color palette.

        :param palette: Palette name (e.g., 'deep', 'pastel').
        """
        self.palette = palette
        sns.set_palette(self.palette)

    def set_figure_size(self, width: int, height: int):
        """
        Sets the size of the figures.

        :param width: Width of the figure.
        :param height: Height of the figure.
        """
        self.figure_size = (width, height)

    def set_save_directory(self, save_dir: str):
        """
        Sets the directory where plots will be saved.

        :param save_dir: Directory path.
        """
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

    def generate_all_plots(self, x: Union[List, np.ndarray], y: Union[List, np.ndarray],
                           x_label: str = 'X', y_label: str = 'Y'):
        """
        Generates and saves all types of plots for given x and y data.

        :param x: Data for the x-axis.
        :param y: Data for the y-axis.
        :param x_label: Label for the x-axis. Default is 'X'.
        :param y_label: Label for the y-axis. Default is 'Y'.
        """
        self.scatter_plot(x, y, x_label, y_label)
        self.line_plot(x, y, x_label, y_label)
        self.regression_plot(x, y, x_label, y_label)
        self.histogram(y, y_label)
        self.box_plot(y, y_label)
        self.violin_plot(y, y_label)
        self.joint_plot(x, y, x_label, y_label)
        self.residual_plot(x, y, x_label, y_label)
        self.kde_plot(x, y, x_label, y_label)
        self.bar_plot(x, y, x_label, y_label)
        self.count_plot(x, x_label)
        self.distribution_plot(y, y_label)
        self.swarm_plot(x, y, x_label, y_label)
        self.qq_plot(y, y_label)

        data = pd.DataFrame({x_label: x, y_label: y})
        self.correlation_heatmap(data)
        self.pairplot(data)


def main():
    # Create sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 100)
    y = 2 * x + 1 + np.random.normal(0, 1, 100)

    # Create a more complex dataset for correlation heatmap and pairplot
    data = pd.DataFrame({
        'X': x,
        'Y': y,
        'Z': np.sin(x) + np.random.normal(0, 0.5, 100),
        'W': np.exp(x/10) + np.random.normal(0, 1, 100),
        'Category': np.random.choice(['A', 'B', 'C'], 100)
    })

    # Initialize the VisualizerToolkit
    viz_toolkit = VisualizerUtilityToolkit(display=False)

    # Generate individual plots
    print("Generating individual plots...")
    viz_toolkit.scatter_plot(x, y, x_label='X values', y_label='Y values')
    viz_toolkit.line_plot(x, y, x_label='X values', y_label='Y values')
    viz_toolkit.regression_plot(x, y, x_label='X values', y_label='Y values')
    viz_toolkit.histogram(y, label='Y values')
    viz_toolkit.box_plot(y, label='Y values')
    viz_toolkit.violin_plot(y, label='Y values')
    viz_toolkit.joint_plot(x, y, x_label='X values', y_label='Y values')
    viz_toolkit.residual_plot(x, y, x_label='X values', y_label='Y values')
    viz_toolkit.kde_plot(x, y, x_label='X values', y_label='Y values')
    viz_toolkit.bar_plot(data['Category'], data['Y'],
                         x_label='Category', y_label='Y values')
    viz_toolkit.count_plot(data['Category'], label='Categories')
    viz_toolkit.distribution_plot(y, label='Y values')
    viz_toolkit.swarm_plot(data['Category'], data['Y'],
                           x_label='Category', y_label='Y values')
    viz_toolkit.qq_plot(y, label='Y values')

    # Generate correlation heatmap and pairplot
    print("Generating correlation heatmap and pairplot...")
    viz_toolkit.correlation_heatmap(data[['X', 'Y', 'Z', 'W']])
    viz_toolkit.pairplot(data[['X', 'Y', 'Z', 'W']])

    # Change style and palette
    print("Changing style and palette...")
    viz_toolkit.change_style('whitegrid')
    viz_toolkit.change_palette('Set2')
    viz_toolkit.set_figure_size(12, 8)

    # Generate all plots with new style
    print("Generating all plots with new style...")
    viz_toolkit.generate_all_plots(
        x, y, x_label='X values', y_label='Y values')


if __name__ == "__main__":
    main()
