"""Research-only zero-lag correction; never imported by the clinical worker.

The direct correction is REJECTED: the existing score aliases grid periods.
This module exposes the experiment for regression audits only.
"""
from types import MethodType


def zero_lag_grid_search(finder, autocorrelation, aspect):
    # np.correlate(..., 'full') has zero lag at its middle element.
    # The vendor's reversed prefix starts at lag one instead.
    correlation = autocorrelation[autocorrelation.shape[0] // 2:].clone()
    correlation -= correlation.mean()
    minimum = correlation.shape[-1] / (finder.max_number_of_grid_lines / aspect)
    maximum = correlation.shape[-1] / (finder.min_number_of_grid_lines / aspect)
    period = finder._grid_search_min_distance(correlation, finder.samples, minimum, maximum)
    for _ in range(finder.max_zoom):
        width = (maximum - minimum) / finder.zoom_factor
        minimum = max(period - width, minimum)
        maximum = min(period + width, maximum)
        period = finder._grid_search_min_distance(correlation, finder.samples, minimum, maximum)
    return period


def install_research_correction(finder):
    finder._zoom_grid_search_min_distance = MethodType(zero_lag_grid_search, finder)
    return finder

