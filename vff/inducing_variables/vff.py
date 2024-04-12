"""Variational Fourier Features. In one dimension:
    <φ_m, f> = <e^(i2π m/(b-a)), f>_H
    where H is reproducing kernel Hilbert space corresponding to the kernel of f and the inner product is taken over 
    the interval [a, b]. The range of m is assumed to be symmetric about 0 and include 0; for higher dimensions, we can 
    use the product of these (that is, a regular rectangular grid)
The elements of K_uu are RKHS norms of the form <φ_m, φ_m>_h (the off-diagonals are zero). For lower order Matérn 
kernels in one dimension, these norms are known and given in the reference. For higher dimensions, we only have simple 
sum-over-dimensions or product-over-dimensions compositions of these features.

The key reference is 'Variational Fourier Features for Gaussian Processes', James Hensman, Nicolas Durrande and Arno 
Solin. https://jmlr.org/papers/v18/16-579.html.

@article{JMLR:v18:16-579,
  author  = {James Hensman and Nicolas Durrande and Arno Solin},
  title   = {Variational Fourier Features for Gaussian Processes},
  journal = {Journal of Machine Learning Research},
  year    = {2018},
  volume  = {18},
  number  = {151},
  pages   = {1--52},
  url     = {http://jmlr.org/papers/v18/16-579.html}
}

TODO: The paper has methods to compute Kuf outside of [a, b], but these terms are hyperparameter dependent 
(so they're incompatible with precomputing). We need a way to use a separate code path at test time... one 
possibility is a custom posterior method which implements the corrections.
"""

import gpflow as gp
import tensorflow as tf
import math
import lab
import lab.tensorflow
import functools
import numpy as np
from gpflow.mean_functions import MeanFunction
from gpflow.likelihoods import Gaussian
from gpflow.kernels import Kernel
from gpflow.posteriors import PrecomputedValue
from check_shapes import Shape
from typing import Optional, Union
from ..objectives import make_krp
from ..dispatch import sgpr_elbo, sgpr_precompute, posterior_precompute, Kuu, Kuf, spectral_density
from ..posteriors import PrecomputedOperator
from .composition import MaskedProductKronecker, AdditiveInducingVariable

MASKS = {
    "none" : lambda freqs, Md, eps : freqs < math.inf,
    "spherical" : lambda freqs, Md, eps : lab.sum((freqs/(Md*eps/2))**2, axis=-1) <= 1.,
    "symplectic" : lambda freqs, Md, eps : lab.sum(lab.abs(freqs/(Md*eps/2)), axis=-1) <= 1.,
}

class VariationalFourierFeature1D(gp.inducing_variables.InducingVariables):
    """Variational Fourier features in one dimension.
    """

    def __init__(self, a : float, b : float, M : int, num_outputs : int=1):
        self.Z = lab.expand_dims(lab.to_numpy(lab.range(M) / (b-a)), axis=-1)
        self.Z = lab.concat(self.Z, self.Z[..., 1:, :], axis=-2)
        self.a = a
        self.b = b
        self.M = 2*M - 1
        self.D = 1
        self.P = num_outputs

    @property
    def num_inducing(self) -> tf.Tensor:
        return self.M
    
    @property
    def shape(self) -> Shape:
        return (self.num_inducing, self.D, self.P)
    
class VariationalFourierFeatureProduct(MaskedProductKronecker):

    def __init__(self, a : lab.Numeric, b : lab.Numeric, Md : lab.Numeric, 
                 mask_type: str="none", D : Optional[int] = None):
        """Provide either the inducing frequencies z or the maximum number in each dimension and mask type"""
        if isinstance(Md, int):
            if D is None:
                D = 1
            Md = np.array([Md for _ in range(D)], dtype=np.int64)
        # now Md is iterable -- just confirm it's the right length
        if not len(Md) == D:
            raise ValueError("Md should be a single value or length D but got length {} and D={}".format(
                len(Md), D
            ))
        if isinstance(a, float) or lab.is_scalar(a) or len(a) == 1:
            a = lab.uprank(lab.squeeze(np.array([a for _ in range(D)])), 1)
        if isinstance(b, float) or lab.is_scalar(b) or len(b) == 1:
            b = lab.uprank(lab.squeeze(np.array([b for _ in range(D)])), 1)

        super().__init__([VariationalFourierFeature1D(a[d], b[d], Md[d]) for d in range(D)], mask_type=mask_type)
        self.a = a
        self.b = b

class VariationalFourierFeatureAdditive(AdditiveInducingVariable):
    def __init__(self, a : lab.Numeric, b : lab.Numeric, Md : lab.Numeric, D : Optional[int] = None):
        """Provide either the inducing frequencies z or the maximum number in each dimension and mask type"""
        if isinstance(Md, int):
            if D is None:
                D = 1
            Md = np.array([Md for _ in range(D)], dtype=np.int64)
        # now Md is iterable -- just confirm it's the right length
        if not len(Md) == D:
            raise ValueError("Md should be a single value or length D but got length {} and D={}".format(
                len(Md), D
            ))
        if isinstance(a, float) or lab.is_scalar(a) or len(a) == 1:
            a = lab.uprank(lab.squeeze(np.array([a for _ in range(D)])), 1)
        if isinstance(b, float) or lab.is_scalar(b) or len(b) == 1:
            b = lab.uprank(lab.squeeze(np.array([b for _ in range(D)])), 1)

        super().__init__([VariationalFourierFeature1D(a[d], b[d], Md[d]) for d in range(D)])
        self.a = a
        self.b = b

@Kuf.register(VariationalFourierFeature1D, Kernel, object)
def kuf_vff_1d(inducing_variable: VariationalFourierFeature1D, 
               kernel: Kernel,
               x):
    if lab.any(x > inducing_variable.b) or lab.any(x < inducing_variable.a):
        raise ValueError("VFF is implemented only within the boundary [a, b] = [{}, {}]".format(inducing_variable.a, 
                    inducing_variable.b) + " but some inputs were found outside the boundary.")
    prod = 2*math.pi*lab.squeeze(x, axis=-1) * inducing_variable.Z[..., :int((inducing_variable.num_inducing+1)//2), :]
    out = lab.concat(lab.cos(prod), lab.sin(prod[..., 1:, :]), axis=-2)
    return tf.linalg.LinearOperatorFullMatrix(out)

@Kuu.register(VariationalFourierFeature1D, gp.kernels.Matern12)
def kuu_vff_matern12(inducing_variable: VariationalFourierFeature1D, kernel: gp.kernels.Matern12):
    omegas = 2.*np.pi*lab.squeeze(inducing_variable.Z[..., :int((inducing_variable.num_inducing+1)//2), :], axis=-1)
    two_or_four = gp.utilities.to_default_float(tf.where(omegas==0, 2., 4.))
    diag = ((inducing_variable.b - inducing_variable.a) 
            * (1/kernel.lengthscales**2 + omegas**2)
            * kernel.lengthscales / kernel.variance / two_or_four
    )
    vec = tf.ones_like(diag) / lab.sqrt(1.*kernel.variance)
    cos_term = tf.linalg.LinearOperatorLowRankUpdate(
        tf.linalg.LinearOperatorDiag(diag, is_positive_definite=True, is_self_adjoint=True), 
        lab.expand_dims(vec, axis=-1) 
    )

    omegas = omegas[omegas!=0]
    diag = ((inducing_variable.b - inducing_variable.a) 
            * (1/kernel.lengthscales**2 + omegas**2)
            * kernel.lengthscales / kernel.variance / 4.
    )
    sin_term = tf.linalg.LinearOperatorDiag(diag, is_positive_definite=True, is_self_adjoint=True)

    return tf.linalg.LinearOperatorBlockDiag([cos_term, sin_term],
                                             is_non_singular=True,
                                             is_positive_definite=True,
                                             is_self_adjoint=True,
                                             is_square=True)

@Kuu.register(VariationalFourierFeature1D, gp.kernels.Matern32)
def kuu_vff_matern32(inducing_variable: VariationalFourierFeature1D, kernel: gp.kernels.Matern32):
    omegas = 2.*np.pi*lab.squeeze(inducing_variable.Z[..., :int((inducing_variable.num_inducing+1)//2), :], axis=-1)
    four_or_eight = gp.utilities.to_default_float(tf.where(omegas==0, 4., 8.))
    diag = ((inducing_variable.b - inducing_variable.a) 
            * (3/kernel.lengthscales**2 + omegas**2)**2
            * kernel.lengthscales**3 / lab.sqrt(3)**3 / kernel.variance / four_or_eight
    )
    vec = tf.ones_like(diag) / lab.sqrt(1.*kernel.variance)
    cos_term = tf.linalg.LinearOperatorLowRankUpdate(
        tf.linalg.LinearOperatorDiag(diag, is_positive_definite=True, is_self_adjoint=True), 
        lab.expand_dims(vec, axis=-1) 
    )

    omegas = omegas[omegas!=0]
    diag = ((inducing_variable.b - inducing_variable.a) 
            * (3/kernel.lengthscales**2 + omegas**2)**2
            * kernel.lengthscales**3 / lab.sqrt(3)**3 / kernel.variance / 8.
    )
    vec = omegas * kernel.lengthscales / lab.sqrt(5 * kernel.variance)
    sin_term = tf.linalg.LinearOperatorLowRankUpdate(
        tf.linalg.LinearOperatorDiag(diag, is_positive_definite=True, is_self_adjoint=True), 
        lab.expand_dims(vec, axis=-1) 
    )

    return tf.linalg.LinearOperatorBlockDiag([cos_term, sin_term],
                                             is_non_singular=True,
                                             is_positive_definite=True,
                                             is_self_adjoint=True,
                                             is_square=True)

@Kuu.register(VariationalFourierFeature1D, gp.kernels.Matern52)
def kuu_vff_matern52(inducing_variable: VariationalFourierFeature1D, kernel: gp.kernels.Matern52):
    omegas = 2.*np.pi*lab.squeeze(inducing_variable.Z[..., :int((inducing_variable.num_inducing+1)//2), :], axis=-1)
    sixteen_or_thirtytwo = gp.utilities.to_default_float(tf.where(omegas==0, 16., 32.))
    v1 = (3/5*(omegas * kernel.lengthscales)**2 - 1) / lab.sqrt(8*kernel.variance)
    v2 = tf.ones_like(v1) / lab.sqrt(1.*kernel.variance)
    vec = lab.concat(lab.expand_dims(v1, axis=-1), lab.expand_dims(v2, axis=-1), axis=-1)
    diag = (3 * (inducing_variable.b - inducing_variable.a) 
            * (5/kernel.lengthscales**2 + omegas**2)**3
            * kernel.lengthscales**5 / lab.sqrt(5)**5 / kernel.variance / sixteen_or_thirtytwo
    )
    cos_term = tf.linalg.LinearOperatorLowRankUpdate(
        tf.linalg.LinearOperatorDiag(diag, is_positive_definite=True, is_self_adjoint=True), 
        vec 
    )

    omegas = omegas[1:, ...]
    diag = (3*(inducing_variable.b - inducing_variable.a) 
            * (5/kernel.lengthscales**2 + omegas**2)**3
            * kernel.lengthscales**5 / lab.sqrt(5)**5 / kernel.variance / 32.
    )
    vec = lab.sqrt(3/5/kernel.variance) * omegas * kernel.lengthscales
    sin_term = tf.linalg.LinearOperatorLowRankUpdate(
        tf.linalg.LinearOperatorDiag(diag, is_positive_definite=True, is_self_adjoint=True), 
        lab.expand_dims(vec, axis=-1) 
    )
    return tf.linalg.LinearOperatorBlockDiag([cos_term, sin_term],
                                             is_non_singular=True,
                                             is_positive_definite=True,
                                             is_self_adjoint=True,
                                             is_square=True)