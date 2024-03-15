#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of the
#   Scikit-NeuroMSI Project (https://github.com/renatoparedes/scikit-neuromsi).
# Copyright (c) 2021-2022, Renato Paredes; Cabral, Juan
# License: BSD 3-Clause
# Full Text:
#     https://github.com/renatoparedes/scikit-neuromsi/blob/main/LICENSE.txt

# =============================================================================
# DOCS
# =============================================================================

"""
This module contains classes for plotting NDResultCollection data.

NDResultCollectionPlotter provides methods for plotting NDResultCollection
data, including line plots for reports and biases.

"""

# =============================================================================
# IMPORTS
# =============================================================================


import seaborn as sns

from ...utils import AccessorABC

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
        ax.set_ylabel("Proportion of unit causes")
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
        ax=None,
        **kws,
    ):
        """Line plot of the bias for a given influence parameter.

        Parameters
        ----------
        influence_parameter : str
            The influence parameter for the bias plot.
        changing_parameter : str, optional
            The changing parameter, by default None.
        dim : str, optional
            The dimension to plot, by default None.
        mode : str, optional
            The mode of the plot, by default None.
        quiet : bool, optional
            Whether to suppress output, by default False.
        ax : matplotlib.axes.Axes, optional
            The axes object to plot on, by default None.
        **kws
            Additional keyword arguments for the plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The plotted axes object.
        """
        bias_acc = self._nd_collection.bias
        the_bias = bias_acc.bias(
            influence_parameter,
            changing_parameter=changing_parameter,
            dim=dim,
            mode=mode,
            quiet=quiet,
        )

        legend = kws.pop("legend", True)

        kws.setdefault("estimator", "mean")

        changing_parameter, influence_parameter, _ = the_bias.columns[0]
        estimator = kws["estimator"].title()

        kws.setdefault(
            "label", f"{estimator} {changing_parameter}({influence_parameter})"
        )

        ax = sns.lineplot(data=the_bias, ax=ax, legend=False, **kws)

        if legend:
            ax.legend()

        ax.set_ylabel("Bias")

        return ax

    def bias_mean(
        self,
        influence_parameter,
        *,
        changing_parameter=None,
        dim=None,
        mode=None,
        quiet=False,
        ax=None,
        **kws,
    ):
        bias_acc = self._nd_collection.bias
        the_bias = bias_acc.bias_mean(
            influence_parameter,
            changing_parameter=changing_parameter,
            dim=dim,
            mode=mode,
            quiet=quiet,
        )

        legend = kws.pop("legend", True)

        changing_parameter, influence_parameter, _ = the_bias.columns[0]

        kws.setdefault(
            "label", f"Mean {changing_parameter}({influence_parameter})"
        )

        ax = sns.lineplot(data=the_bias, ax=ax, legend=False, **kws)

        if legend:
            ax.legend()

        ax.set_ylabel("Bias")

        return ax
