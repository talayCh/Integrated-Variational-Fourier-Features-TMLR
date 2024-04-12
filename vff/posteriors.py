"""SGPR posterior objects with multiple dispatch across inducing variables etc"""

import gpflow as gp
import tensorflow as tf
import numpy as np
from gpflow.posteriors import PrecomputedValue, PrecomputeCacheType
from gpflow.utilities import assert_params_false
from gpflow.base import RegressionData,  MeanAndVariance, TensorType
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.likelihoods import Gaussian
from check_shapes import check_shapes, inherit_check_shapes
from typing import Optional, Tuple, cast
from dataclasses import dataclass
from .dispatch import posterior_precompute, conditional_posterior_with_precompute, Kuu, Kuf

class SGPRPosterior(gp.posteriors.AbstractPosterior):
    """
    SGPR posteriors with multiple dispatch for the precompute. 
    
    For new inducing variables or kernels where additional efficiencies can be gained, implement posterior_precompute 
    and conditional_posterior_with_precompute.
    """
    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, Q]",
        "inducing_variable: [M, D, 1]"
    )
    def __init__(
        self,
        kernel : Kernel,
        data : RegressionData,
        inducing_variable : InducingVariables,
        likelihood : Gaussian,
        num_latent_gps : int,
        mean_function: MeanFunction,
        *args,
        precompute_cache: Optional[PrecomputeCacheType]
    ) -> None:
        x, y  = data
        super().__init__(kernel, x, mean_function=mean_function)
        self.Y_data = y
        self.likelihood = likelihood
        self.inducing_variable = inducing_variable
        self.num_latent_gps = num_latent_gps

        if precompute_cache is not None:
            self.update_cache(precompute_cache)
    
    @inherit_check_shapes
    def _conditional_with_precompute(self, 
                                     cache: Tuple[tf.Tensor, ...], 
                                     Xnew: TensorType, 
                                     full_cov: bool = False, 
                                     full_output_cov: bool = False
        ) -> MeanAndVariance:
        assert_params_false(self._conditional_with_precompute, full_output_cov=full_output_cov)
        return conditional_posterior_with_precompute(self.inducing_variable, self.kernel, cache, Xnew, 
                                                     full_cov, full_output_cov, self.num_latent_gps)
    
    def _precompute(self) -> Tuple[PrecomputedValue, ...]:
        assert self.mean_function is not None
        return posterior_precompute(self.inducing_variable, self.kernel, self.mean_function, 
                                    self.likelihood, self.X_data, self.Y_data)
    
    @inherit_check_shapes
    def _conditional_fused(
        self, Xnew: TensorType, full_cov: bool = False, full_output_cov: bool=False,
    ) -> MeanAndVariance:
        """Compute the mean and variance of the function evaluations at new points Xnew, without making use of caching.
        """
        temp_cache = tuple(c.value for c in self._precompute())
        return self._conditional_with_precompute(temp_cache, Xnew, full_cov, full_output_cov)
    
# multiple dispatch for the SGPR posteriors
@conditional_posterior_with_precompute.register(InducingVariables, Kernel, tuple, (tf.Tensor, np.ndarray),
                                                bool, bool, int)
def conditional_posterior_with_precompute_fallback(inducing_variable : InducingVariables, kernel : Kernel, 
                                          cache: Tuple[tf.Tensor, ...], Xnew: TensorType,
                                          full_cov: bool, full_output_cov: bool,
                                          num_latent_gps: int) -> MeanAndVariance:
    """Fallback SGPR posterior."""
    L, rtA, c = cache

    Kus = Kuf(inducing_variable, kernel, Xnew)
    tmp1 = L.solve(Kus).to_dense()
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

@posterior_precompute.register(InducingVariables, Kernel, MeanFunction, Gaussian, (tf.Tensor, np.ndarray), 
                               (tf.Tensor, np.ndarray))
def posterior_precompute_fallback(inducing_variable : InducingVariables, kernel : Kernel, mean_function : MeanFunction,
                         likelihood : Gaussian,
                         X_data : tf.Tensor, Y_data : tf.Tensor) -> Tuple[PrecomputedValue]:
    X_data = cast(tf.Tensor, X_data)
    M = inducing_variable.num_inducing
    err = Y_data - mean_function(X_data)

    kuf = Kuf(inducing_variable, kernel, X_data)
    kuu = Kuu(inducing_variable, kernel, jitter=gp.config.default_jitter())

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

@dataclass
class PrecomputedOperator:
    value : tf.linalg.LinearOperator