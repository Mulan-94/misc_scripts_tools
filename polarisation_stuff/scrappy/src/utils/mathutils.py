import numpy as np

def rms(data):
    """Root mean square"""
    return np.sqrt(np.square(data).mean())


def ssq(data):
    """Sum of squares"""
    return np.nansum(np.square(data))


def sqrt_ssq(*args):
    """Square root of sum of squares"""
    squares = [np.square(x) for x in args]
    squares = np.sqrt(np.sum(squares, axis=0))
    return squares


def are_all_nan(data):
    return np.all(np.isnan(data))


def are_all_zeroes(data):
    return np.all(data==0)


def are_all_masked(data):
    size = data.compressed().size
    return True if size==0 else False


def is_infinite(data):
    return np.isinf(data)


def signal_to_noise(signal, rms_noise):
    return np.abs(signal/rms_noise)


def snr_is_above_threshold(snr, thresh):
    """
    Return True if the snr is above the threshold
    similar to return true if:
        signal > threshold * noise
    """
    return True if snr>=thresh else False


def nancount(data):
    """Count the number of nans in the array"""
    return np.isnan(data).sum()