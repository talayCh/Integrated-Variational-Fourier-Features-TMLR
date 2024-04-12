"""Convenience functions, mainly for abstracting torch's linear algebra with terser names and better batching.
"""

import numpy as np
import lab
import lab.tensorflow
from typing import Callable
import spherical_harmonics.tensorflow
from scipy import integrate
from scipy.special import gamma


def sinc(x, delta=1e-12):
    return lab.sin(x+delta)/(x+delta)

def _funk_hecke(shape_function: Callable[[float], float], n: int, dim: int) -> float:
    r"""
    Implements Funk-Hecke [see 1] where we integrate over the sphere of dim-1
    living in \Re^dim.
    Using these coefficients we can approximate shape function s combined with
    the CollapsedEigenfunctions C as follows:
    For x, x' in \Re^dim
    s(x^T x') = \sum_n a_n C_n(x^T x')
    for which, when we uncollapse C_n, we get
    s(x^T x') = \sum_n \sum_k^N(dim, n) a_n \phi_{n, k}(x) \phi_{n, k}(x')
    [1] Variational Inducing Spherical Harmonics (appendix)
    :param shape_function: [-1, 1] -> \Re
    :param n: degree (level)
    :param dim: x, x' in \Re^dim
    """
    assert dim >= 3, "Sphere needs to be at least S^2."
    omega_d = surface_area_sphere(dim - 1) / surface_area_sphere(dim)
    alpha = (dim - 2.0) / 2.0
    C = spherical_harmonics.gegenbauer_polynomial.Gegenbauer(n, alpha)
    C_1 = C(1.0)

    def integrand(t: float) -> float:
        return shape_function(t) * C(t) * (1.0 - t ** 2) ** (alpha - 0.5)

    v = integrate.quad(integrand, -1.0, 1.0)[0]
    return v * omega_d / C_1

def surface_area_sphere(d: int) -> float:
    return 2 * (np.pi ** (d / 2)) / gamma(d / 2)