from scipy.stats import iqr as quartile_range
from scipy.fftpack import fft
import numpy as np
import math
import scipy

# Agg functions

# For slope of time series
def linear_fit(array_like):
    # Linear features
    if (len(array_like) == 0):
        return [0, 0]
    p = np.polyfit(np.arange(len(array_like)), array_like, 1)
    return [p[0], p[1]]


def poly_fit(array_like):
    # Poly features
    if (len(array_like) == 0):
        return [0, 0, 0]
    p = np.polyfit(np.arange(len(array_like)), array_like, 2)
    return [p[0], p[1], p[2]]


def iqr(array_like):
    # inter quartile range.
    result = quartile_range(array_like)
    return result if not math.isnan(result) else 0


def kurt(array_like):
    result = scipy.stats.kurtosis(array_like)
    return result if not math.isnan(result) else 0


def mcr(array_like):
    # returns how many times the mean has been crossed.
    mean = np.mean(array_like)
    array_like = array_like - mean
    return np.sum(np.diff(np.sign(array_like)).astype(bool))


def fourier_transform(array_like):
    # Return Fast Fourier transfor of array.
    result = fft(array_like)
    return 0


def adjust_stress_values(stress_level):
    mapping = {
        1: 2,
        2: 3,
        3: 4,
        4: 1,
        5: 0
    }

    try:
        return mapping[stress_level]
    except KeyError:
        return None