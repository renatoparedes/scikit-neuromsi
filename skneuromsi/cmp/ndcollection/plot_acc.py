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

""""""

# =============================================================================
# IMPORTS
# =============================================================================


import seaborn as sns


from ...utils import AccessorABC

# =============================================================================
# PLOTTER
# =============================================================================


class NDResultCollectionPlotter(AccessorABC):
    _default_kind = "unity_report"

    def __init__(self, ndcollection):
        self._nd_collection = ndcollection

    def _line_report(self, report, ax, kws):
        parameter = report.columns[0]
        x = report.index
        y = report[parameter]

        kws.setdefault("label", parameter)
        ax = sns.lineplot(x=x, y=y, ax=ax, **kws)

        return ax

    def n_report(self, n, *, parameter=None, ax=None, **kws):
        causes_acc = self._nd_collection.causes
        the_report = causes_acc.n_report(n, parameter=parameter)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel(f"Proportion of {n} causes")
        return ax

    def unity_report(self, *, parameter=None, ax=None, **kws):
        causes_acc = self._nd_collection.causes
        the_report = causes_acc.unity_report(parameter=parameter)
        ax = self._line_report(the_report, ax, kws)
        ax.set_ylabel("Proportion of unit causes")
        return ax

    def mean_report(self, *, parameter=None, ax=None, **kws):
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
        show_progress=True,
        ax=None,
        **kws,
    ):
        bias_acc = self._nd_collection.bias
        the_bias = bias_acc.bias(
            influence_parameter,
            changing_parameter=changing_parameter,
            dim=dim,
            mode=mode,
            show_progress=show_progress,
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
        show_progress=True,
        ax=None,
        **kws,
    ):
        bias_acc = self._nd_collection.bias
        the_bias = bias_acc.bias_mean(
            influence_parameter,
            changing_parameter=changing_parameter,
            dim=dim,
            mode=mode,
            show_progress=show_progress,
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
