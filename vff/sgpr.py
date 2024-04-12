"""Collapsed variational sparse Gaussian process regression (SGPR) classes for use with gpflow. 

These classes allow using different inducing features, and should eventually be able to be used for multioutput 
regression.

TODO: (non-urgent) check that differing M and D across dimensions work

TODO: IFF posterior 
TODO: VISH posterior
TODO: (ideally) put VFF in this framework too
TODO: make sure this can work well for additive covariance functions also
TODO: faster evaluation for IFF on gridded data
TODO: Tackle the multi-output case
"""

import gpflow as gp
import tensorflow as tf
import math
import lab
import lab.tensorflow
from gpflow.posteriors import PrecomputeCacheType
from gpflow.base import RegressionData, TensorData
from gpflow.models.util import InducingVariablesLike, inducingpoint_wrapper
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel
from gpflow.mean_functions import MeanFunction
from gpflow.likelihoods import Gaussian
from check_shapes import check_shapes, inherit_check_shapes
from typing import Optional
from .posteriors import SGPRPosterior
from .dispatch import sgpr_elbo, sgpr_precompute, Kuu, Kuf


class SGPR(gp.models.GPModel, gp.models.InternalDataTrainingLossMixin):
    """Base class for SGPR with flexibility to work with different inducing features
    """

    def __init__(
            self,
            data: RegressionData,
            kernel: Kernel,
            inducing_variable: InducingVariablesLike,
            *args,
            mean_function: Optional[MeanFunction] = None,
            num_latent_gps: Optional[int] = None,
            noise_variance: Optional[TensorData] = None,
            likelihood: Optional[Gaussian] = None,
    ):
        assert (noise_variance is None) or (likelihood is None), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = Gaussian(noise_variance)
        x_data, y_data = gp.models.util.data_input_to_tensor(data)
        num_latent_gps = y_data.shape[-1] if num_latent_gps is None else num_latent_gps
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = x_data, y_data
        self.num_data = x_data.shape[0]
        self.inducing_variable: InducingVariables = inducingpoint_wrapper(inducing_variable)

        sgpr_precompute(self.inducing_variable, self.kernel, self)

    #@inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()
    
    #@check_shapes(
    #    "return: []"
    #)
    @tf.function
    def elbo(self) -> tf.Tensor:
        return sgpr_elbo(self.inducing_variable, self.kernel, self)
    
    def posterior(self, precompute_cache: PrecomputeCacheType = PrecomputeCacheType.TENSOR,
                  ) -> SGPRPosterior:
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """

        return SGPRPosterior(
            kernel=self.kernel,
            data=self.data,
            inducing_variable=self.inducing_variable,
            likelihood=self.likelihood,
            num_latent_gps=self.num_latent_gps,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache
        )
    
    @inherit_check_shapes
    def predict_f(
        self, Xnew: gp.base.InputData, full_cov: bool = False, full_output_cov: bool = False,
    ) -> gp.base.MeanAndVariance:
        """Basic cache-free predict_f for training. Otherwise use the posterior's predict function.
        """
        return self.posterior(PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )

@sgpr_elbo.register(InducingVariables, Kernel, object)
def sgpr_elbo_fallback(inducing_variable : InducingVariables, kernel : Kernel, model : object, jitter : float=None):
    """Fallback lower bound computation. The equation is

       - 0.5 [tr(K_ff)/σ^2 - tr(B)/σ^2 + N log2πσ^2 + log|A| + ν^2/σ^2 - ybar^T A^{-1} ybar / σ^4]
    where
       A = I + B/σ^2
       B = K_uu^{-1/2} K_uf K_fu K_uu^{1/2}
       ybar = K_{uu}^{-1/2} K_{uf} y
       ν^2 = ∑_n y_n^2

    and it's convenient here to absorb the scaling by the noise into the definitions of various of these.

    We arrange the computation this way because it leads to focusing the O(M^3) computational cost into just on Cholesky 
    decomposition (of A) which has been chosen to be fairly numerically stable. The dominant cost in practice is the 
    O(NM^2) associated with multiplications involving K_uf (to rotate by K_uf and to calculate B).
    """
    if jitter is None:
        jitter = gp.config.default_jitter()

    sigma = tf.sqrt(model.likelihood.variance)
    kuf = Kuf(inducing_variable, kernel, model.data[0])
    kuu = Kuu(inducing_variable, kernel, jitter=jitter)

    kff_diag = kernel(model.data[0], full_cov=False)
    M, N = kuf.shape[-2:]
    #kuf_rotated = tf.linalg.triangular_solve(tf.linalg.cholesky(kuu), kuf/sigma, lower=True)
    kuf_rotated = kuu.cholesky().solve(kuf.to_dense()/sigma)
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

@sgpr_precompute.register(InducingVariables, Kernel, object)
def sgpr_precompute_fallback(inducing_variable : InducingVariables, kernel : Kernel, model : object):
    """Fallback precompute. The only meaningful computation is the signal energy
        ν^2 = ∑_n y_n^2
    """
    model.signal_energy = tf.squeeze(tf.reduce_sum(model.data[1]**2, axis=-2))

@Kuu.register(gp.inducing_variables.InducingPoints, Kernel)
def Kuu_inducing_points(inducing_variable: gp.inducing_variables.InducingPoints, 
                        kernel: Kernel, *args, jitter: float=0.0):
    """Standard inducing point covariance matrix but returning a linear operator."""
    kuu = kernel(inducing_variable.Z)
    kuu += jitter * tf.eye(inducing_variable.num_inducing, dtype=kuu.dtype)
    return tf.linalg.LinearOperatorFullMatrix(
        kuu,
        is_non_singular=True,
        is_self_adjoint=True,
        is_positive_definite=True,
        is_square=True
    )

@Kuf.register(gp.inducing_variables.InducingPoints, Kernel, object)
def Kuf_inducing_points(inducing_variable: gp.inducing_variables.InducingPoints, kernel: Kernel, Xnew):
    return tf.linalg.LinearOperatorFullMatrix(kernel(inducing_variable.Z, Xnew))