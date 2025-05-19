"""Lightweight helper utilities for testing."""

import math


def modify_dsbuffer(dsbuffer, granularity):
    """Limit dataset buffer values to the given granularity."""
    result = {}
    for key, value in dsbuffer.items():
        if value is not None:
            result[key] = min(granularity, value)
        else:
            result[key] = None
    return result


def weighting_function(x, x0, k):
    """Simple exponential decay after an offset."""
    if x <= x0:
        return 1
    return math.exp(-math.log(2) * (x - x0) / k)
