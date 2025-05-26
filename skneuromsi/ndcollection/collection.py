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

"""Implementation of NDCollection.

The ndcollection module implements the NDCollection class, which is a
collection of NDResult objects.

"""

# =============================================================================
# IMPORTS
# =============================================================================

from collections.abc import Sequence

import methodtools

import numpy as np

import pandas as pd

import tqdm

from . import bias_acc, causes_acc, cplot_acc
from .. import core
from ..utils import Bunch, dict_cmp

# =============================================================================
# RESULT COLLECTION
# =============================================================================


def _modes_describe(ndres):
    """Describe the modes of an NDResult.

    This function calculates and returns a dictionary containing information
    about the modes of an NDResult.

    Parameters
    ----------
    ndres : NDResult
        The NDResult object for which to describe the modes.

    Returns
    -------
    dict
        A dictionary with the following key:

        - 'var' : float
            Variance of the modes.

    """
    modes = ndres.get_modes()
    return {"var": modes.var()}


def _common_metadata_cache(ndres):
    """Get the metadata cache from an NDResult object.

    This function returns the metadata cache from an NDResult object.

    Parameters
    ----------
    ndres : NDResult
        The NDResult object from which to get the metadata cache.

    Returns
    -------
    Bunch
        The metadata cache of the NDResult object.

    """
    return {
        "modes": ndres.modes_,
        "output_mode": ndres.output_mode,
        "run_parameters": tuple(ndres.run_parameters.to_dict()),
        "dims": ndres.dims,
    }


def _make_metadata_cache(ndresults):
    """Create a metadata cache from a collection of NDResult objects.

    This function iterates over a collection of NDResult objects and extracts
    various metadata information to create a metadata cache.

    Parameters
    ----------
    ndresults : iterable
        Iterable containing NDResult objects.

    Returns
    -------
    cache : Bunch
        A dict like object containing the metadata cache with the following
        attributes:

        - modes : array-like
            Modes associated with the NDResult collection.
        - run_parameters : tuple
            Tuple representing the run parameters of the NDResult collection.
        - mnames : ndarray
            Names of the NDResults.
        - mtypes : ndarray
            Types of the NDResults.
        - output_mode : str
            Output mode of the NDResult collection.
        - nmaps : ndarray
            Number of maps associated with each NDResult.
        - time_ranges : ndarray
            Time ranges associated with each NDResult.
        - position_ranges : ndarray
            Position ranges associated with each NDResult.
        - time_resolutions : ndarray
            Time resolutions associated with each NDResult.
        - position_resolutions : ndarray
            Position resolutions associated with each NDResult.
        - run_parameters_values : ndarray
            Run parameter values for each NDResult.
        - causes : ndarray
            Causes information for each NDResult.
        - modes_variances_sum : pandas.Series
            Sum of variances for modes.
        - dims : array-like
            Dimensions associated with the NDResult collection.

    """
    mnames = []
    mtypes = []
    nmaps = []
    time_ranges = []
    position_ranges = []
    time_resolutions = []
    position_resolutions = []
    run_parameters_values = []
    causes = []
    modes_variances = []

    common_cache = None

    for ndres in ndresults:
        mnames.append(ndres.mname)
        mtypes.append(ndres.mtype)
        nmaps.append(ndres.nmap_)
        time_ranges.append(ndres.time_range)
        position_ranges.append(ndres.position_range)
        time_resolutions.append(ndres.time_res)
        position_resolutions.append(ndres.position_res)
        run_parameters_values.append(ndres.run_parameters.to_dict())
        causes.append(ndres.causes_)
        modes_describe_dict = _modes_describe(ndres)
        modes_variances.append(modes_describe_dict["var"])

        # all the run_parameters/modes are the same, so lets take the first one
        if common_cache is None:
            common_cache = _common_metadata_cache(ndres)
        else:
            other_ndres = _common_metadata_cache(ndres)
            if not dict_cmp.dict_allclose(common_cache, other_ndres):
                same_keys = sorted(common_cache.keys())
                raise ValueError(
                    "All NDResults must have "
                    f"the same metadata in {same_keys}."
                )

    # Resume the series collection into a single one we use sum instead of
    # numpy sum, because we want a pandas.Series and not a numpy array.
    # Also we assign the name to the Series.
    modes_variances_sum = sum(modes_variances)
    modes_variances_sum.name = "VarSum"

    cache = {
        "mnames": np.asarray(mnames),
        "mtypes": np.asarray(mtypes),
        "nmaps": np.asarray(nmaps),
        "time_ranges": np.asarray(time_ranges),
        "position_ranges": np.asarray(position_ranges),
        "time_resolutions": np.asarray(time_resolutions),
        "position_resolutions": np.asarray(position_resolutions),
        "run_parameters_values": np.asarray(run_parameters_values),
        "causes": np.asarray(causes),
        "modes_variances_sum": modes_variances_sum,
    }
    cache.update(common_cache)

    return cache


class NDResultCollection(Sequence):
    """Collection of NDResult objects.

    Note that NDResult objects can be very memory hungry, so the
    NDResultCollection is designed to store the data in a compressed format.
    This is why the compressed_results parameter is an iterable of
    CompressedNDResult objects.

    Check ``NDResultCollection.from_ndresults`` if you want to create an
    instance from a list of uncompressed NDResult objects.

    Parameters
    ----------
    name : str
        Name of the NDResultCollection.
    compressed_results : iterable
        Iterable containing CompressedNDResult objects.

    """

    def __init__(self, name, compressed_results, *, tqdm_cls=None):
        self._name = str(name)
        self._cndresults = np.asarray(compressed_results)
        self._tqdm_cls = tqdm_cls

        # this is where we cache all the cpu intensive stuff
        self._cache = None

        if not len(self._cndresults):
            cls_name = type(self).__name__
            raise ValueError(f"Empty {cls_name} not allowed")
        if not (
            self._tqdm_cls is None or issubclass(self._tqdm_cls, tqdm.tqdm)
        ):
            raise TypeError(
                "'tqdm_cls' must be an instance of tqdm.tqdm or None"
            )
        if not all(
            isinstance(ndr, core.CompressedNDResult)
            for ndr in self._cndresults
        ):
            raise ValueError("Not all results are CompressedNDResult objects")

        # populate the cache
        self._populate_cache()

    @classmethod
    def from_ndresults(
        cls,
        name,
        results,
        *,
        compression_params=core.DEFAULT_COMPRESSION_PARAMS,
        tqdm_cls=None,
    ):
        """Create an instance of NDResultCollection from a list of \
        NDResult objects.

        Parameters
        ----------
        name : str
            The name of the NDResultCollection.
        results : List[NDResult]
            The list of NDResult objects to be compressed and stored in the
            collection.
        compression_params : Tuple[str, int], optional
            The compression parameters for the NDResult objects. Defaults to
             core.DEFAULT_COMPRESSION_PARAMS.
        tqdm_cls : tqdm.tqdm, optional
            The tqdm class to use. Defaults to None.

        Returns
        -------
        NDResultCollection
            An instance of NDResultCollection containing the compressed
            NDResult objects.
        """
        generator = (
            core.compress_ndresult(r, compression_params=compression_params)
            for r in results
        )
        compressed_results = np.fromiter(generator, dtype=object)
        return cls(name, compressed_results, tqdm_cls=tqdm_cls)

    # Because is a Sequence ==================================================

    def __len__(self):
        """Return the number of NDResult objects in the collection."""
        return len(self._cndresults)

    def __getitem__(self, slicer):
        """Return the NDResult object at the given index."""
        cndresults = self._cndresults.__getitem__(slicer)
        if isinstance(cndresults, core.CompressedNDResult):
            return core.decompress_ndresult(cndresults)

        generator = (core.decompress_ndresult(cr) for cr in cndresults)
        ndresults = np.fromiter(generator, dtype=object)
        return ndresults

    # PROPERTIES =============================================================

    def _populate_cache(self):
        """Populate the cache of the NDResultCollection."""
        # if the cache is None, we need to collect metadata
        if self._cache is None:
            ndresults = iter(self)

            # if tqdm_cls is not None, we need to show a progress bar
            if self._tqdm_cls:
                ndresults = self._tqdm_cls(
                    iterable=ndresults,
                    total=len(self),
                    desc="Collecting metadata",
                )

            # collect metadata
            cache = _make_metadata_cache(ndresults)
            self._cache = Bunch("_cache", cache)

    @property
    def name(self):
        """Name of the NDResultCollection."""
        return self._name

    @property
    def tqdm_cls(self):
        """The tqdm class to use."""
        return self._tqdm_cls

    @property
    def modes_(self):
        """Modes of all the results in the NDResultCollection."""
        return self._cache["modes"]

    @property
    def dims_(self):
        """Dimensions of all the results in the \
        NDResultCollection."""
        return self._cache["dims"]

    @property
    def output_mode_(self):
        """Output mode of all the results in the \
        NDResultCollection."""
        return self._cache["output_mode"]

    @property
    def run_parameters_(self):
        """Run parameters of all the results in the \
        NDResultCollection."""
        return self._cache["run_parameters"]

    @property
    def causes_(self):
        """Causes of all the results in the \
        NDResultCollection."""
        return self._cache["causes"]

    @property
    def run_parameters_values(self):
        """Run parameters values of all the results in the \
        NDResultCollection."""
        return self._cache["run_parameters_values"]

    @property
    def input_modes_(self):
        """Input modes of all the results in the NDResultCollection.

        Returns all modes that are not the output mode.

        """
        candidates = self.modes_
        return candidates[~(candidates == self.output_mode_)]

    def __repr__(self):
        """x.__repr__() <==> repr(x)."""
        cls_name = type(self).__name__
        name = self.name
        length = len(self)
        return f"<{cls_name} {name!r} len={length}>"

    # PARAMETERS ANALYSIS =====================================================

    def disparity_matrix(self):
        """Generate a disparity matrix from run parameters values.

        The resulting DataFrame has iterations as rows and parameters as
        columns.

        Returns
        -------
        pandas.DataFrame
            A DataFrame representing the disparity matrix.

        """
        run_parameters_values = self._cache["run_parameters_values"]
        df = pd.DataFrame(list(run_parameters_values))
        df.columns.name = "Parameters"
        return df

    def changing_parameters(self):
        """Determine run parameters wich has multiple values.

        This method calculates parameters that exhibit changing values
        across the disparity matrix. It identifies parameters for which
        the unique values are not consistent across all data points.

        Returns
        -------
        pandas.Series
            A series indicating whether each parameter has changing values
            across the disparity matrix.

        """
        dm = self.disparity_matrix()
        changes = dm.nunique() > 1
        changes.name = "Changes"
        return changes

    def coerce_parameter(self, prefer=None):
        """Coerce the provided run parameter or select a preferred one.

        If 'prefer' is None, this method selects a run parameter based on which
        parameter is called with more than one value. If multiple changing
        parameters are available, The method fails.

        If 'prefer' is provided, it is validated against the available run
        parameters.

        Parameters
        ----------
        prefer : str or None, optional
            The run parameter to be coerced or selected.

        Returns
        -------
        str
            The coerced or preferred run parameter.

        Raises
        ------
        ValueError
            If the value of 'parameter' is ambiguous due to multiple candidates
            or if the provided run parameter is not in the available run
            parameters.

        """
        if prefer is None:
            wpc = self.changing_parameters()
            candidates = wpc[wpc].index.to_numpy()
            candidates_len = len(candidates)
            if candidates_len != 1:
                candidates_str = (
                    f"Candidates: {candidates}" if candidates_len > 0 else ""
                )
                raise ValueError(
                    "The value of 'run_parameter' is ambiguous since it has "
                    f"{candidates_len} candidates. {candidates_str}".strip()
                )
            prefer = candidates[0]
        elif prefer not in self.run_parameters_:
            raise ValueError(f"Unknown run_parameter {prefer!r}")
        return prefer

    def modes_variance_sum(self):
        """Get the sum of variances for modes.

        This method returns the sum of variances associated with modes.

        Returns
        -------
        pandas.Series
            The sum of variances for modes.

        """
        modes_variances_sum = self._cache["modes_variances_sum"]
        varsum = modes_variances_sum.copy()
        return varsum

    def coerce_mode(self, prefer=None):
        """Coerces the provided preferred mode or selects a mode with maximum \
        variance.

        If 'prefer' is None, this method selects the mode with the maximum
        variance from available modes. If multiple modes have the same
        variance, the method fails.

        If 'prefer' is provided, it is validated against the available modes.

        Parameters
        ----------
        prefer : str or None, optional
            The mode to be validated or selected.

        Returns
        -------
        str
            The validate or selected mode.

        Raises
        ------
        ValueError
            If the value of 'prefer' is ambiguous due to multiple candidates
            with the same variance or if the provided preferred mode is not in
            the available modes.

        """
        if prefer is None:
            # maybe two modes have exactly the same variance_sum
            # for this reason we dont use argmax that only return the first max
            modes_varsum = self.modes_variance_sum()
            maxvalue = modes_varsum.max()

            candidates = modes_varsum.index[modes_varsum == maxvalue]
            candidates_len = len(candidates)

            if candidates_len != 1:
                raise ValueError(
                    "The value of 'mode' is ambiguous since it has "
                    f"{candidates_len} candidates. "
                    f"Candidates: {candidates.to_numpy()}"
                )

            prefer = candidates[0]

        elif prefer not in self.modes_:
            raise ValueError(f"Unknown mode {prefer!r}")

        return prefer

    def coerce_dimension(self, prefer=None):
        """Coerce and validate the provided preferred dimension or select the \
        default dimension if None.

        If no dimension is provided, the method prefers to use the 'time'
        dimension; otherwise, it check if the provided dimension exists
        criteria.

        Parameters
        ----------
        prefer : str or None, optional
            The dimension to be coerced, or the default dimension if None is
            provided.

        Returns
        -------
        str
            The coerced or selected dimension.

        Raises
        ------
        ValueError
            If the provided preferred dimension is not in the available
            dimensions.

        """
        if prefer is None:
            prefer = core.constants.D_TIMES
        elif prefer not in self.dims_:
            raise ValueError(f"Unknown dimension {prefer!r}")
        return prefer

    # ACCESORS ================================================================

    @methodtools.lru_cache(maxsize=None)
    @property
    def causes(self):
        """Accessor for NDResultCausesAcc providing access to causes \
        analysis."""
        return causes_acc.NDResultCollectionCausesAcc(self)

    @methodtools.lru_cache(maxsize=None)
    @property
    def bias(self):
        """Accessor for NDResultBiasAcc providing access to bias analysis."""
        return bias_acc.NDResultCollectionBiasAcc(self, self._tqdm_cls)

    @methodtools.lru_cache(maxsize=None)
    @property
    def plot(self):
        """Accessor for NDResultCollectionPlotter providing access to \
        plotting utilities."""
        return cplot_acc.NDResultCollectionPlotter(self)

    # IO ======================================================================

    def to_ndc(self, path_or_stream, metadata=None, quiet=False, **kwargs):
        """Store the NDResultCollection in a NMSI Collection (NDC) format.

        Parameters
        ----------
        path_or_stream : str or file-like
            File path or file-like object to save the NDC file.
        metadata : dict, optional
            Additional metadata to include in the NDC file.
        quiet : bool, optional
            If True, suppress tqdm progress bar. Defaults to False.
        **kwargs
            Additional keyword arguments passed to store_ndrcollection
            function.

        """
        from ..io import store_ndresults_collection  # noqa

        tqdm_cls = None if quiet else self._tqdm_cls
        store_ndresults_collection(
            path_or_stream,
            ndrcollection=self,
            tqdm_cls=tqdm_cls,
            metadata=metadata,
            **kwargs,
        )
