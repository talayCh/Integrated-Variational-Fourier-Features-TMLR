"""
Variational inducing spherical harmonics:
    <φ_m, f> = <φ_m, f>_H
    where H is the reproducing kernel Hilbert space corresponding to a stationary kernel on the sphere, and f is either 
    directly modelled on a sphere, or the kernel on the original domain includes an affine mapping onto the sphere,
    and φ_m are the spherical harmonics, chosen in increasing order.

The values for K_uu are the spherical harmonic expansion coefficients of the kernel. These are calculated using the
Funk-Hecke theorem.

The key reference is 'Sparse Gaussian Processes with Spherical Harmonic Features', Vincent Dutordoir, Nicolas Durrande 
and James Hensman, ICML 2020. http://proceedings.mlr.press/v119/dutordoir20a

    
@InProceedings{pmlr-v119-dutordoir20a,
    title = 	 {Sparse {G}aussian Processes with Spherical Harmonic Features},
    author =       {Dutordoir, Vincent and Durrande, Nicolas and Hensman, James},
    booktitle = 	 {Proceedings of the 37th International Conference on Machine Learning},
    pages = 	 {2793--2802},
    year = 	 {2020},
    editor = 	 {III, Hal Daumé and Singh, Aarti},
    volume = 	 {119},
    series = 	 {Proceedings of Machine Learning Research},
    month = 	 {13--18 Jul},
    publisher =    {PMLR},
    pdf = 	 {http://proceedings.mlr.press/v119/dutordoir20a/dutordoir20a.pdf},
    url = 	 {https://proceedings.mlr.press/v119/dutordoir20a.html},
}

"""

import gpflow as gp
import tensorflow as tf
import math
import numpy as np
import lab
import lab.tensorflow
import spherical_harmonics.tensorflow
from gpflow.inducing_variables import InducingVariables
from gpflow.base import MeanAndVariance, TensorType
from gpflow.mean_functions import MeanFunction
from gpflow.likelihoods import Gaussian
from gpflow.posteriors import PrecomputedValue, PrecomputeCacheType
from check_shapes import Shape
from typing import cast, Tuple
from ..functional.functional import _funk_hecke
from ..kernels import IsotropicSpherical, Chordal
from ..spectral_densities import spectral_density
from ..dispatch import sgpr_elbo, sgpr_precompute, posterior_precompute, conditional_posterior_with_precompute, Kuu, Kuf
from ..posteriors import PrecomputedOperator

class SphericalHarmonicFeature(InducingVariables):
    """Inducing Spherical Harmonic class. This is intended for modelling on R^D by mapping it onto S^D, and using 
    covariance functions on R^{D+1} restricted to S^D. 
    """

    def __init__(self, dimension: int, max_degree: int):
        self.dimension = dimension
        self.max_degree = max_degree
        self.basis_functions = spherical_harmonics.SphericalHarmonics(dimension+1, max_degree)
        self._eigenvalues = {}

    @property
    def num_inducing(self) -> tf.Tensor:
        return len(self.basis_functions)
    
    @property
    def shape(self) -> Shape:
        """Only single output"""
        return (len(self.basis_functions), self.dimension, 1)
    
    def eigenvalues(self, max_degree: int, shape_function) -> tf.Tensor:
        if max_degree not in self._eigenvalues:
            values = []
            for n in range(max_degree):
                v = _funk_hecke(shape_function, n, self.dimension+1)
                values.append(lab.expand_dims(lab.to_numpy(v), axis=-1))
            self._eigenvalues[max_degree] = lab.concat(*values)
        return self._eigenvalues[max_degree]


@sgpr_precompute.register(SphericalHarmonicFeature, Chordal, object)
def sgpr_precompute_vish(inducing_variable : InducingVariables, kernel : Chordal, model : object):
    """Chordal spherical harmonics precompute. The only meaningful computation is the signal energy
        ν^2 = ∑_n y_n^2
    and the eigenvalues of the covariance function, which are computed using the Funk-Hecke method.
    """
    model.signal_energy = tf.squeeze(tf.reduce_sum(model.data[1]**2, axis=-2))
    _ = inducing_variable.eigenvalues(inducing_variable.max_degree, kernel.shape_function)

@sgpr_elbo.register(SphericalHarmonicFeature, Chordal, object)
def sgpr_elbo_vish(inducing_variable: SphericalHarmonicFeature, kernel: Chordal, model: object):
    x = tf.concat([ (model.data[0]/kernel.lengthscales), kernel.bias*tf.ones_like(model.data[0][..., :1])], axis=1)
    r = tf.linalg.norm(x, axis=-1, keepdims=True)
    x = x/r
    # then treat as standard SGPR
    sigma = tf.sqrt(model.likelihood.variance)
    kuf = Kuf(inducing_variable, kernel, x)
    kuu = Kuu(inducing_variable, kernel)
    # Linear operators appear to mess with the optimiser, so try customising the calculations
    kuu_diag_inv = kuu._operator.diag

    kff_diag = kernel(model.data[0], full_cov=False)
    M, N = kuf.shape[-2:]
    #kuf_rotated = kuu.cholesky().solve(kuf.to_dense()/sigma)
    kuf_rotated = lab.expand_dims(lab.sqrt(kuu_diag_inv), axis=-1) * kuf.to_dense()/sigma
    B = tf.linalg.matmul(kuf_rotated, kuf_rotated, transpose_b=True)
    A = tf.linalg.eye(M, dtype=gp.config.default_float()) + B
    rtA = tf.linalg.cholesky(A)
    ybar = tf.linalg.matmul(kuf_rotated, (model.data[1] - model.mean_function(model.data[0]))/sigma)

    trace_term = tf.reduce_sum(kff_diag) / model.likelihood.variance - tf.reduce_sum(tf.linalg.diag_part(B))
    vec = tf.linalg.triangular_solve(rtA, ybar, lower=True)
    quadratic_term = model.signal_energy / model.likelihood.variance - tf.reduce_sum(vec**2)
    logdet_term = (2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(rtA))) 
                   + N * tf.math.log(2*math.pi*model.likelihood.variance)
    )

    return - 0.5 * (trace_term + logdet_term + quadratic_term)

@conditional_posterior_with_precompute.register(SphericalHarmonicFeature, Chordal, tuple, (tf.Tensor, np.ndarray),
                                                bool, bool, int)
def conditional_posterior_with_precompute_vish_chordal(inducing_variable : SphericalHarmonicFeature, kernel : Chordal, 
                                          cache: Tuple[tf.Tensor, ...], Xnew: TensorType,
                                          full_cov: bool, full_output_cov: bool,
                                          num_latent_gps: int) -> MeanAndVariance:
    """VISH chordal posterior"""
    #L, rtA, c = cache
    Linv, rtA, c = cache


    Xnew = cast(tf.Tensor, Xnew)
    x = tf.concat([ (Xnew/kernel.lengthscales), kernel.bias*tf.ones_like(Xnew[..., :1])], axis=1)
    r = tf.linalg.norm(x, axis=-1, keepdims=True)
    x = x/r

    Kus = Kuf(inducing_variable, kernel, x)
    #tmp1 = L.solve(Kus).to_dense()
    tmp1 = lab.expand_dims(Linv, axis=-1) * Kus.to_dense()
    tmp2 = tf.linalg.triangular_solve(rtA, tmp1, lower=True)
    mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
    if full_cov:
        var = (kernel(Xnew) + tf.linalg.matmul(tmp2, tmp2, transpose_a=True) 
               - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
        )
        var = tf.tile(var[None, ...], [num_latent_gps, 1, 1]) # [P, N, N]
    else:
        var = kernel(Xnew, full_cov=False) + tf.reduce_sum(tf.square(tmp2), 0) - tf.reduce_sum(tf.square(tmp1), 0)
        var = tf.tile(var[:, None], [1, num_latent_gps]) # [N, P]
    
    return mean, var

@posterior_precompute.register(SphericalHarmonicFeature, Chordal, MeanFunction, Gaussian, (tf.Tensor, np.ndarray),
                               (tf.Tensor, np.ndarray))
def posterior_precompute_vish_chordal(inducing_variable: SphericalHarmonicFeature,
                                      kernel: Chordal,
                                      mean_function: MeanFunction,
                                      likelihood: Gaussian,
                                      X_data: tf.Tensor,
                                      Y_data: tf.Tensor):
    X_data = cast(tf.Tensor, X_data)
    x = tf.concat([ (X_data/kernel.lengthscales), kernel.bias*tf.ones_like(X_data[..., :1])], axis=1)
    r = tf.linalg.norm(x, axis=-1, keepdims=True)
    x = x/r

    M = inducing_variable.num_inducing
    err = Y_data - mean_function(x)

    kuf = Kuf(inducing_variable, kernel, x)
    kuu = Kuu(inducing_variable, kernel, jitter=gp.config.default_jitter())

    sigma_sq = tf.squeeze(likelihood.variance_at(x), axis=-1)
    sigma = tf.sqrt(sigma_sq)

    #L = kuu.cholesky()  
    Linv = lab.sqrt(kuu._operator.diag)
    #kuf_rotated = L.solve(kuf.to_dense()/sigma)
    kuf_rotated = lab.expand_dims(Linv, axis=-1) * kuf.to_dense()/sigma
    A = tf.linalg.matmul(kuf_rotated, kuf_rotated, transpose_b=True) + tf.eye(
        M, dtype=gp.config.default_float()
    )  
    rtA = tf.linalg.cholesky(A)  
    ybar = tf.linalg.matmul(kuf_rotated, err / sigma[..., None])
    c = tf.linalg.triangular_solve(rtA, ybar, lower=True)

    D = err.shape[1]
    M = X_data.shape[0]
    D_dynamic = D is None
    M_dynamic = M is None

    #TODO: for now, this has been modified ad hoc to get linear operator L to work. Fine for fused predict, but not 
    # suitable more generally.
    return (
        PrecomputedOperator(Linv), 
        PrecomputedValue(rtA, (M_dynamic, D_dynamic)), 
        PrecomputedValue(c, (M_dynamic, D_dynamic))
    )

@Kuu.register(SphericalHarmonicFeature, Chordal)
def Kuu_vish_chordal(inducing_variable: SphericalHarmonicFeature, kernel: Chordal, jitter=None):
    vals = kernel.variance * inducing_variable.eigenvalues(inducing_variable.max_degree, kernel.shape_function)
    num_per_level = [spherical_harmonics.fundamental_set.num_harmonics(inducing_variable.dimension+1, n) 
                     for n in range(inducing_variable.max_degree)]
    kuu_inv_diag = tf.concat(values=[tf.repeat(vals[i], r) for i, r in enumerate(num_per_level)], axis=0)
    return tf.linalg.LinearOperatorInversion(tf.linalg.LinearOperatorDiag(kuu_inv_diag,
                is_non_singular=True,
                is_self_adjoint=True,
                is_positive_definite=True,
                is_square=True
        ))

@Kuf.register(SphericalHarmonicFeature, Chordal, object)
def Kuf_vish_chordal(inducing_variable: SphericalHarmonicFeature, kernel: Chordal, x):
    return tf.linalg.LinearOperatorFullMatrix(tf.transpose(inducing_variable.basis_functions(x)))