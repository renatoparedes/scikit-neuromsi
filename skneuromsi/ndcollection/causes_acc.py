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

"""Implementation of causes analysis in multisensory integration.

The NDResultCausesAcc class provides an accessor, NDResultCausesAcc, for
analyzing causes in the context of multisensory integration. It offers methods
to calculate causes and check the uniqueness of causes.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from collections import defaultdict

import numpy as np

import pandas as pd

from ..utils import AccessorABC

# =============================================================================
# NDResultCauses ACC
# =============================================================================


class NDResultCollectionCausesAcc(AccessorABC):
    """Accessor for calculating causes in an NDResultCollection.

    Causal analysis in the context of multisensory integration refers to the
    study of factors and conditions that contribute to how the brain combines
    and processes information from different senses, such as vision, hearing,
    touch, and others. It involves identifying these factors, studying their
    interactions, evaluating the consistency of sensory information, and
    developing models to explain how the brain integrates sensory inputs
    effectively.


    Parameters
    ----------
    ndcollection : NDResultCollection
        The NDResultCollection object to be accessed and analyzed.

    """

    _default_kind = "unity_report"

    def __init__(self, ndcollection):
        self._nd_collection = ndcollection

    def causes_by_parameter(self, *, parameter=None):
        """Get causes by a specific parameter.

        Parameters
        ----------
        parameter : str, optional
            The parameter to group by, by default None.
            If None then the parameter with more than one value is selected.
            Check the documentation of the NDResultCollection.coerce_parameter.

        Returns
        -------
        cdf : pandas.DataFrame
            The DataFrame containing causes grouped by the specified parameter.

        """
        nd_collection = self._nd_collection

        parameter = nd_collection.coerce_parameter(parameter)

        run_parameters_values = nd_collection.run_parameters_values
        causes = nd_collection.causes_

        columns = defaultdict(list)
        for rp_value, causes in zip(run_parameters_values, causes):
            columns[("Parameters", parameter)].append(rp_value[parameter])
            columns[("", "Causes")].append(causes)

        cdf = pd.DataFrame.from_dict(columns)

        cdf.index.name = "Iteration"

        # put al the parameters together
        cdf = cdf[np.sort(cdf.columns)[::-1]]

        return cdf

    def unique_causes(self, *, parameter=None):
        """Get unique values of number of causes based on a specific parameter.

        Parameters
        ----------
        parameter : str, optional
            The parameter to group by, by default None.
            If None then the parameter with more than one value is selected.
            Check the documentation of the NDResultCollection.coerce_parameter.

        Returns
        -------
        unique_causes : numpy.ndarray
            Array containing unique causes based on the specified parameter.

        """
        cba = self.causes_by_parameter(parameter=parameter)
        return cba[("", "Causes")].unique()

    def n_report(self, n, *, parameter=None):
        """Generate an N-report for a given number of causes.

        Analogous to Unity's "n_report," this analysis function assesses how
        information from different sensory modalities converges or combines in
        the brain to create a arbitrary number of  perceptual experiences.

        For instance, if we consider three modalities - vision, hearing, and
        touch - and the brain identifies two experiences, two of these
        modalities are perceived as a single experience, while one remains
        separate.

        The primary aim of this analysis is to determine how different
        modalities are integrated to form a coherent perceptual experience.

        Parameters
        ----------
        n : int
            The number of causes for the N-report.
        parameter : str, optional
            The parameter to group by, by default None.
            If None then the parameter with more than one value is selected.
            Check the documentation of the NDResultCollection.coerce_parameter.

        Returns
        -------
        the_report : pandas.DataFrame
            The generated N-report DataFrame.

        """
        nd_collection = self._nd_collection

        parameter = nd_collection.coerce_parameter(parameter)
        cdf = self.causes_by_parameter(parameter=parameter)

        values = cdf[("Parameters", parameter)]
        crosstab = pd.crosstab(values, cdf["", "Causes"])
        n_ity = crosstab.get(n, 0) / crosstab.sum(axis="columns")

        the_report = pd.DataFrame(n_ity, columns=[parameter])
        the_report.index.name = parameter
        the_report.columns = pd.Index(["Causes"], name="")

        return the_report

    def unity_report(self, *, parameter=None):
        """Generate a unity report.

        In the context of multisensory integration, the unity report typically
        refers to an analysis or visualization that assesses the extent to
        which information from different sensory modalities converges or
        combines in the brain to create a unified perceptual experience. This
        type of report aims to quantify the degree of integration or coherence
        between sensory inputs, such as visual, auditory, tactile, and other
        sensory stimuli.

        For instance, researchers may use a unity report to analyze how well
        sensory information from vision and touch aligns when perceiving a
        textured object. They might measure the degree of synchronization or
        correlation between neural responses in visual and somatosensory brain
        regions to determine the level of multisensory integration.

        The unity report in multisensory integration studies helps researchers
        understand the mechanisms underlying the brain's ability to merge
        information from different senses, leading to a cohesive and unified
        perception of the environment. It provides quantitative insights into
        the integration process and contributes to our understanding of how the
        brain creates a seamless perceptual representation from diverse sensory
        inputs.

        Parameters
        ----------
        parameter : str, optional
            The parameter to group by, by default None.
            If None then the parameter with more than one value is selected.
            Check the documentation of the NDResultCollection.coerce_parameter.

        Returns
        -------
        unity_report : pandas.DataFrame
            The generated unity report DataFrame.

        """
        return self.n_report(1, parameter=parameter)

    def mean_report(self, *, parameter=None):
        """Generate a mean report.

        Similar to Unity's "mean_report," this analysis function evaluates how
        information from different sensory modalities converges or combines in
        the brain to create an average number of perceptual experiences.

        For example, if we consider three modalities - vision, hearing, and
        touch - and the brain identifies 1.5 experiences, the model generally
        struggles to combine stimuli into a single sensory experience in most
        cases.

        The main goal of this analysis is to determine how, in most instances,
        different modalities integrate to create a unified perceptual
        experience.

        Parameters
        ----------
        parameter : str, optional
            The parameter to group by, by default None.
            If None then the parameter with more than one value is selected.
            Check the documentation of the NDResultCollection.coerce_parameter.

        Returns
        -------
        mean_report : pandas.DataFrame
            The generated mean report DataFrame.

        """
        nd_collection = self._nd_collection
        parameter = nd_collection.coerce_parameter(parameter)
        cdf = self.causes_by_parameter(parameter=parameter)

        groups = cdf.groupby(("Parameters", parameter))
        report = groups.mean()

        report.index.name = parameter
        report.columns = pd.Index(["Causes"], name="")

        return report

    def describe_causes(self, *, parameter=None):
        """Generate descriptive statistics for causes.

        This method generates descriptive statistics for all convergence values
        of the modalities concerning the given parameter.

        These values aim to provide a general idea of how the integration model
        combined stimuli from different modalities.

        Parameters
        ----------
        parameter : str, optional
            The parameter to group by, by default None.
            If None then the parameter with more than one value is selected.
            Check the documentation of the NDResultCollection.coerce_parameter.

        Returns
        -------
        describe_report : pandas.DataFrame
            The generated descriptive statistics report DataFrame.

        """
        nd_collection = self._nd_collection
        parameter = nd_collection.coerce_parameter(parameter)
        cdf = self.causes_by_parameter(parameter=parameter)

        groups = cdf.groupby(("Parameters", parameter))
        report = groups.describe()

        report.index.name = parameter

        columns = report.columns
        report.columns = pd.MultiIndex.from_product(
            [[""], columns.levels[-1]], names=["Causes", None]
        )
        report.columns.name = "Causes"

        return report
