from cmath import sin
import math
import lab
import gc
import numpy as np
import tensorflow as tf
import scipy
import gpflow
import functools

def log_marginal_likelihood(y, x, k, lik):
    """"Convenience function to directly evaluate the log marginal likelihood for exact
    GP regression with covariance function k and Gaussian likelihood lik."""

    N = y.shape[-1]
    rtA = lab.chol(lab.eye(y.dtype, N) * lik.variance + k.K(x, x))
    logdet = 2*lab.sum(lab.log(lab.diag_extract(rtA)))
    quadform = lab.sum(y * lab.squeeze(lab.cholesky_solve(rtA, lab.expand_dims(y, axis=-1)), axis=-1))
    lml = - 0.5 * (N*math.log(2 * math.pi) + logdet + quadform)
    return lml