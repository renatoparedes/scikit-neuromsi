#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2025, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""Utilities for plotting NDResultCollection data.

NDResultCollectionPlotter provides methods for plotting NDResultCollection
data, including line plots for reports and biases.

"""

# =============================================================================
# IMPORTS
# =============================================================================


import seaborn as sns

from ..utils import AccessorABC

# =============================================================================
# PLOTTER
# =============================================================================


class NDResultCollectionPlotter(AccessorABC):
    """NDResultCollection plotting utilities.

    Parameters
    ----------
    ndcollection : NDResultCollection
        The NDResultCollection object to be plotted.



    """

    _default_kind = "unity_report"

    def __init__(self, ndcollection):
        self._nd_collection = ndcollection

    def _line_report(self, report, ax, kws):
        """Generate a line plot for a given report.

        Parameters
        ----------
        report : pandas.DataFrame
            The report data to be plotted.
        ax : matplotlib.axes.Axes
            The axes object to plot on.
        kws : dict
            Additional keyword arguments for the plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The plotted axes object.

        """
        parameter = report.columns[0]
        x = report.index
        y = report[parameter]

        kws.setdefault("label", parameter)
        ax = sns.lineplot(x=x, y=y, ax=ax, **kws)

        return ax

    def n_report(self, n, *, parameter=None, ax=None, **kws):
        """Line plot of the N-report for a given number of causes.

        Parameters
        ----------
        n : int
            The number of causes for the N-report.
        parameter : str, optional
            The parameter to plot, by default None.
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on, by default None.
        **kws
            Additional keyword arguments for the plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The plotted axes object.

        """
        causes_acc = self._nd_collection.causes
        the_report = causes_acc.n_report(n, parameter=parameter)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel(f"Proportion of {n} causes")
        return ax

    def unity_report(self, *, parameter=None, ax=None, **kws):
        """Line plot of the unity report.

        Parameters
        ----------
        parameter : str, optional
            The parameter to plot, by default None.
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on, by default None.
        **kws
            Additional keyword arguments for the plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The plotted axes object.
        """
        causes_acc = self._nd_collection.causes
        the_report = causes_acc.unity_report(parameter=parameter)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel("Proportion of unique causes")
        return ax

    def mean_report(self, *, parameter=None, ax=None, **kws):
        """Line plot of the mean report.

        Parameters
        ----------
        parameter : str, optional
            The parameter to plot, by default None.
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on, by default None.
        **kws
            Additional keyword arguments for the plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The plotted axes object.
        """
        causes_acc = self._nd_collection.causes
        the_report = causes_acc.mean_report(parameter=parameter)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel("Mean of causes")
        return ax

    def bias(
        self,
        influence_parameter,
        *,
        changing_parameter=None,
        dim=None,
        mode=None,
        quiet=False,
        show_iterations=True,
        show_mean=True,
        ax=None,
        legend=True,
        it_linestyle="--",
        it_alpha=0.25,
        mean_color="black",
        **kws,
    ):
        """Line plot of bias.

        Parameters
        ----------
        influence_parameter : object
            The parameter that influences the bias calculation.
        changing_parameter : object, optional
            The parameter that changes across the bias calculation.
        dim : object, optional
            The dimension to use for the bias calculation.
        mode : object, optional
            The mode to use for the bias calculation.
        quiet : bool, default False
            If True, suppress output messages.
        show_iterations : bool, default True
            If True, plot individual iterations of bias data.
        show_mean : bool, default True
            If True, plot the mean bias data.
        ax : matplotlib.axes.Axes, optional
            The matplotlib axes to plot on. If None, a new figure and axes
            will be created.
        legend : bool, default True
            If True, add a legend to the plot.
        it_linestyle : str, default '--'
            The line style to use for iteration plots.
        it_alpha : float, default 0.25
            The alpha (transparency) value for iteration plots.
        mean_color : str, default 'black'
            The color to use for the mean bias plot.
        **kws : dict
            Additional keyword arguments to pass to seaborn's lineplot.

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot.

        Raises
        ------
        ValueError
            If both show_iterations and show_mean are False.

        """
        if not (show_iterations or show_mean):
            raise ValueError(
                "If 'show_iterations' and 'show_mean' are False, "
                "yout plot are empty"
            )

        coll = self._nd_collection
        changing_parameter = coll.coerce_parameter(changing_parameter)
        dim = coll.coerce_dimension(dim)
        mode = coll.coerce_mode(mode)

        bias_acc = coll.bias

        kws.setdefault("estimator", "mean")

        if show_iterations:
            the_bias = bias_acc.bias(
                influence_parameter,
                changing_parameter=changing_parameter,
                dim=dim,
                mode=mode,
                quiet=quiet,
            )

            x = the_bias.index
            for column in the_bias.columns:
                y = the_bias[column]
                label = (
                    "Iterations" if (column == the_bias.columns[0]) else None
                )
                ax = sns.lineplot(
                    x=x,
                    y=y,
                    ax=ax,
                    linestyle=it_linestyle,
                    alpha=it_alpha,
                    legend=False,
                    label=label,
                    **kws,
                )

        if show_mean:
            the_bias_mean = bias_acc.bias_mean(
                influence_parameter,
                changing_parameter=changing_parameter,
                dim=dim,
                mode=mode,
                quiet=quiet,
            )
            x_mean = the_bias_mean.index
            y_mean = the_bias_mean[the_bias_mean.columns[0]]
            ax = sns.lineplot(
                x=x_mean,
                y=y_mean,
                ax=ax,
                legend=False,
                label="Bias Mean",
                color=mean_color,
                **kws,
            )

        ax.set_ylabel("Bias")
        ax.set_title(
            f"Fixed={influence_parameter}, Target={changing_parameter}\n"
            f"Mode={mode}, Dimension={dim}"
        )

        if legend:
            ax.legend()
            leg = ax.get_legend()
            leg.legend_handles[0].set_color(mean_color)

        return ax
