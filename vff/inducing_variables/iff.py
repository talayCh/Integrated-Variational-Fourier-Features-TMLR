"""
Integrated Fourier Features:
Conceptually, the features are:
    <φ_m, f> = \int_D \int f(x) e^(-i2π ξ^T x) dx / s(ξ) dξ
    D = [z_md - ε_1/2, z_md + ε_1/2] x ... x [z_md - ε_d/2, z_m + ε_d/2]
But for each z_md, it is assumed that -z_md is also present, and we change co-ordinates make the features real valued.
Also, we approximate the integral over ξ using a midpoint approximation for the purposes of calculating K_uf and K_uu.

(This approximation is mathematically equivalent to normalising by √s(ξ), which is more convenient for theory. For 
implementation, it suits us to use the definition above since then K_uf is not hyperparameter dependent, without any 
fiddly renormalisation in the lower bound calculation.)
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
from .composition import MaskedProductIFF, AdditiveInducingVariableIFF
from .vff import VariationalFourierFeature1D, VariationalFourierFeatureAdditive, VariationalFourierFeatureProduct
#from .wavelet import InducingWaveletFeature1D

MASKS = {
    "none" : lambda freqs, Md, eps : freqs < math.inf,
    "spherical" : lambda freqs, Md, eps : lab.sum((freqs/(Md*eps/2))**2, axis=-1) <= 1.,
    "symplectic" : lambda freqs, Md, eps : lab.sum(lab.abs(freqs/(Md*eps/2)), axis=-1) <= 1.,
}

class IntegratedFourierFeature1D(gp.inducing_variables.InducingVariables):

    def __init__(self, eps : float, M : int, num_outputs : int=1):
        self.Z = lab.expand_dims(lab.to_numpy(lab.range(M) * eps + eps/2), axis=-1)
        self.Z = lab.concat(self.Z, self.Z, axis=-2)
        self.epsilon = eps
        self.M = 2*M
        self.D = 1
        self.P = num_outputs

    @property
    def num_inducing(self) -> tf.Tensor:
        return self.M
    
    @property
    def shape(self) -> Shape:
        return (self.num_inducing, self.D, self.P)

class IntegratedFourierFeature(gp.inducing_variables.InducingVariables):
    """Integrated Fourier Feature class
    """

    def __init__(self, epsilon : lab.Numeric, Md : lab.Numeric,
                 mask_type : Optional[Union[str, int]] = "spherical", D : Optional[int] = None):
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
        if isinstance(epsilon, float) or lab.is_scalar(epsilon) or len(epsilon) == 1:
            epsilon = lab.uprank(lab.squeeze(np.array([epsilon for _ in range(D)])), 1)
        self.freqs1d = [lab.to_numpy(lab.range(int(Md[d]//2)) * epsilon[d] + epsilon[d]/2) for d in range(D)]
        self.Md = Md
        self.mask_type = mask_type
        self.epsilon = epsilon

        # duplicate to account for cos and sin components
        freqs = [lab.concat(f, f, axis=-1) for f in self.freqs1d]

        # if the frequency grid is small or the there's no mask, then just do this directly
        if mask_type == "none" or lab.prod(Md) < 100:
            freqs = [*np.meshgrid(*freqs)]
            freqs = [lab.expand_dims(freqs[d], axis=-1) for d in range(D-1, -1, -1)]
            freqs = lab.reshape(lab.concat(*freqs, axis=-1), lab.prod(Md), D)
            self._mask = MASKS[mask_type](freqs, Md, epsilon)
            self.z = freqs[self._mask, :]

        else:
            # build up the frequencies carefully, saving memory
            def add_dim(f, f_new):
                """Given d dimensional frequencies f obeying the mask constraint and 1-dimensional frequencies f_new,
                construct d+1 dimensional frequencies and apply the mask constraint"""
                f_new = lab.to_numpy(lab.uprank(f_new, 2))
                num_freqs = f.shape[-2]
                d = f.shape[-1]
                f_out = lab.concat(*[lab.concat(*[lab.concat(
                                    lab.expand_dims(f[j, :], axis=-2), lab.uprank(eff, 2), axis=-1) 
                                    for j in range(num_freqs)]) for eff in f_new]
                                    )
                mask = MASKS[mask_type](f_out, Md[:d+1], epsilon[:d+1])
                return f_out[mask, :]
            self.z = functools.reduce(add_dim, 
                                    [lab.expand_dims(lab.to_numpy(freqs[d]), axis=-1) for d in range(D)])
         
        # temp
        self.Z = self.z
    @property
    def num_inducing(self) -> tf.Tensor:
        return self.z.shape[-2]
    
    @property
    def shape(self) -> Shape:
        """Only single output for now"""
        return (*self.z.shape, 1)
    
@sgpr_elbo.register((IntegratedFourierFeature, MaskedProductIFF, AdditiveInducingVariableIFF,
                     VariationalFourierFeature1D, VariationalFourierFeatureProduct, VariationalFourierFeatureAdditive,
                     #InducingWaveletFeature1D
                     ), 
                     Kernel, object)
def sgpr_elbo_iff(inducing_variable : IntegratedFourierFeature, kernel : Kernel, model : object):
    noise_var = model.likelihood.variance
    kuu = Kuu(inducing_variable, kernel)
    rt_kuu = kuu.cholesky()
    N = model.data[0].shape[-2]

    B = rt_kuu.solve(lab.transpose(rt_kuu.solve(model.B))/noise_var)
    rtA = lab.chol(lab.eye(model.B.dtype, model.B.shape[-1]) + B)
    ybar = rt_kuu.solve(model.ybar/noise_var)

    logdet_term = 2 * lab.sum(lab.log(lab.diag_extract(rtA)))  + N * tf.math.log(2*math.pi*model.likelihood.variance)
    
    trace_term = tf.reduce_sum(kernel(model.data[0], full_cov=False))/noise_var - tf.reduce_sum(tf.linalg.diag_part(B))
    quadratic_term = (model.signal_energy/noise_var 
                      - tf.reduce_sum(tf.linalg.triangular_solve(rtA, ybar, lower=True)**2)
    )
    
    objective = - 0.5 * (trace_term + logdet_term + quadratic_term)

    return lab.squeeze(objective)

@sgpr_precompute.register((IntegratedFourierFeature, MaskedProductIFF, AdditiveInducingVariableIFF, IntegratedFourierFeature1D,
                    VariationalFourierFeature1D, VariationalFourierFeatureProduct, VariationalFourierFeatureAdditive,
                    #InducingWaveletFeature1D
                    ), 
                    Kernel, object)
def sgpr_precompute_iff(inducing_variable : IntegratedFourierFeature, kernel : Kernel, model : object):
    model.signal_energy = lab.sum(model.data[1]**2)
    M = inducing_variable.num_inducing
    out_dim = model.data[1].shape[-1]
    model.B = lab.to_numpy(lab.zeros(gp.config.default_float(), M, M))
    model.ybar = lab.to_numpy(lab.zeros(gp.config.default_float(), M, out_dim))

    
    # chunk 10^4 at a time
    N = model.data[0].shape[-2]
    for i in range(0, N, 10000):
        x_chunk = model.data[0][..., i:(i+10000), :]
        y_chunk = model.data[1][..., i:(i+10000), :]
        
        kuf_chunk = Kuf(inducing_variable, kernel, lab.to_numpy(x_chunk))
        
        model.B = model.B + (kuf_chunk @ kuf_chunk.adjoint()).to_dense() # M x M
        model.ybar = model.ybar + kuf_chunk @ y_chunk # M x P
    model.B = (model.B + lab.transpose(model.B))/2

@posterior_precompute.register((IntegratedFourierFeature, MaskedProductIFF, AdditiveInducingVariableIFF, IntegratedFourierFeature1D,
                    VariationalFourierFeature1D, VariationalFourierFeatureProduct, VariationalFourierFeatureAdditive), 
                    Kernel, MeanFunction, Gaussian, (tf.Tensor, np.ndarray),
                               (tf.Tensor, np.ndarray))
def posterior_precompute_iff(inducing_variable: IntegratedFourierFeature,
                                      kernel: Kernel,
                                      mean_function: MeanFunction,
                                      likelihood: Gaussian,
                                      X_data: tf.Tensor,
                                      Y_data: tf.Tensor):
    """Precompute training data related terms. The only difference from the fallback is we push the Kuf computation 
    to numpy and cpu."""

    M = inducing_variable.num_inducing
    err = Y_data - mean_function(X_data)

    kuf = Kuf(inducing_variable, kernel, lab.to_numpy(X_data))
    kuu = Kuu(inducing_variable, kernel)

    sigma_sq = tf.squeeze(likelihood.variance_at(X_data), axis=-1)
    sigma = tf.sqrt(sigma_sq)

    L = kuu.cholesky()  
    kuf_rotated = L.solve(kuf.to_dense()/sigma)
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
        PrecomputedOperator(L), 
        PrecomputedValue(rtA, (M_dynamic, D_dynamic)), 
        PrecomputedValue(c, (M_dynamic, D_dynamic))
    )

@Kuf.register(IntegratedFourierFeature, Kernel, object)
def Kuf_iff(inducing_variable: IntegratedFourierFeature, kernel: Kernel, x):
    """When rescaled to be independent of the hyperparameters, Kuf in 1D has the cosine and sine elements

        c(z_m, x_n)_cos = √2 cos (2π z_m x_n)
        c(z_m, x_n)_sin = √2 sin (2π z_m x_n)

    and in higher dimension, Kuf is the Khatri-Rao product of Kuf in each dimension, assuming the inducing frequencies 
    are gridded. We assume the frequencies are indeed on a subset of grid points, specified by the mask.
    """
    D = inducing_variable.z.shape[-1]
    if lab.prod(inducing_variable.Md) < 100:
        # small number of features, don't worry too much about memory
        corr_matrix = functools.reduce(make_krp, [make_single_corr_matrix(x[..., d], 
                                                inducing_variable.freqs1d[d]) for d in range(D)]
        )[inducing_variable._mask, :]
    else:
        _, corr_matrix = functools.reduce(lambda x, y: add_corr_dim(x, y, inducing_variable), [
            (lab.expand_dims(lab.concat(inducing_variable.freqs1d[d], inducing_variable.freqs1d[d], axis=-1), 
                             axis=-1),
                make_single_corr_matrix(x[..., d], inducing_variable.freqs1d[d]), 
            ) 
                for d in range(D)]
        )
    return tf.linalg.LinearOperatorFullMatrix(corr_matrix)
        
@Kuu.register(IntegratedFourierFeature, Kernel)
def Kuu_iff(inducing_variable: IntegratedFourierFeature, kernel: Kernel):
    """When rescaled as Kuf, so that Kuf is independent of the hyperparameters, Kuu is just a diagonal matrix of 
    spectral density evaluations scaled by 1/(ε_1...ε_D)"""
    vol = lab.prod(inducing_variable.epsilon)
    return tf.linalg.LinearOperatorInversion(tf.linalg.LinearOperatorDiag(
                (spectral_density(kernel, inducing_variable.z)*vol),
                is_non_singular=True,
                is_self_adjoint=True,
                is_positive_definite=True,
                is_square=True
            ))

@Kuf.register(IntegratedFourierFeature1D, Kernel, object)
def Kuf_iff_1D(inducing_variable: IntegratedFourierFeature1D, kernel: Kernel, x):
    prod = 2*math.pi*lab.squeeze(x, axis=-1) * inducing_variable.Z[..., :int(inducing_variable.num_inducing//2), :]
    out = lab.concat(lab.sqrt(2) * lab.cos(prod), lab.sqrt(2) * lab.sin(prod), axis=-2)
    return tf.linalg.LinearOperatorFullMatrix(out)

def make_single_corr_matrix(x, f):
    prod = 2*math.pi*lab.expand_dims(x, axis=-1) * lab.expand_dims(f, axis=-2)
    out = lab.transpose(
        lab.concat(lab.sqrt(2) * lab.cos(prod), lab.sqrt(2) * lab.sin(prod), axis=-1)
    )
    return out

def add_corr_dim(f_and_mat, f_and_mat_new, inducing_variable):
    f, a = f_and_mat
    f_new, b = f_and_mat_new
    num_freqs = f.shape[-2]
    f_out = lab.concat(*[lab.concat(*[lab.concat(
        lab.expand_dims(f[j, :], axis=-2), lab.uprank(eff, 2), axis=-1)
        for j in range(num_freqs)]) for eff in f_new]
    )
    D_loc = f_out.shape[-1]
    mask = MASKS[inducing_variable.mask_type](f_out, inducing_variable.Md[:D_loc], inducing_variable.epsilon[:D_loc])

    out = make_krp(a, b)
    return (f_out[mask, :], out[mask, :])