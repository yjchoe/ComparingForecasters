"""
Mercer kernel functions.
"""

import numpy as np


def get_kernel(kernel_name, *args):
    """Get a kernel function given its arguments."""
    assert kernel_name in kernels, \
        f"invalid kernel type {kernel_name}. use one of: {list(kernels.keys())}"

    def _kernel_fn(x, y):
        return kernels[kernel_name](x, y, *args)

    return _kernel_fn


def linear_kernel(x, y, *args):
    # NK29 version: K(p, p') = (1-2p) * (1-2p')
    # return np.dot(1 - 2 * x, 1 - 2 * y)
    return np.dot(x, y)


def polynomial_kernel(x, y, d=2, *args):
    # NK29 version: K(p, p') = (1-2p) * (1-2p')
    # return np.dot(1 - 2 * x, 1 - 2 * y) ** d
    return (1. + np.dot(x, y)) ** d


def gaussian_kernel(x, y, sigma=1., *args):
    return np.exp(-(x - y) ** 2 / (2 * sigma ** 2 + 1e-9))


def epanechnikov_kernel(x, y, normalizer=2., *args):
    return 0.75 * (1 - ((x - y) / normalizer) ** 2)


# a str-to-function mapping for imports
kernels = {
    "linear": linear_kernel,
    "polynomial": polynomial_kernel,
    "poly": polynomial_kernel,
    "gaussian": gaussian_kernel,
    "rbf": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
    "epa": epanechnikov_kernel,
}
