"""
Additional kernels
"""

import gpflow as gp
import tensorflow as tf
import math
import lab
import lab.tensorflow
import numpy as np
from typing import Union

class Chordal(gp.kernels.Kernel):
    """Kernel on R^D defined by mapping the inputs onto S^D and applying a kernel for R^{D+1} restricted to S^D.
    This wrapper is designed for base kernels which are stationary, and have a variance and lengthscale.
    The plan for such kernels is that they: 
     - have parameterised lengthscales and bias for mapping onto the unit hypersphere. The base kernel shouldn't have 
     lengthscale parameters.
     - the base kernel has an associated shape function s(x^\top x) = k(x, x') for which we can use multiple dispatch
     - the spherical harmonic coefficients can be calculated by computing the Funk-Hecke integral of the shape function
     (scaled up by the kernel variance) OR by evaluating the spectral density at a distance \sqrt{l(l+d-1)} from the 
     origin.
     - We might need to rescale the function by the norm of the transformation after
    """
    def __init__(self, base_kernel: gp.kernels.IsotropicStationary, dimension: int, variance: float = 1., 
                 lengthscales: Union[float, np.ndarray] = 1., bias: float=1., **kwargs):
        super().__init__(**kwargs)
        self.lengthscales = gp.base.Parameter(lengthscales, transform=gp.utilities.positive())
        self.variance = gp.base.Parameter(variance, transform=gp.utilities.positive())
        self.bias = gp.base.Parameter(bias, transform=gp.utilities.positive())
        self.D = dimension # the proper dimension, i.e. the dimension of the sphere, and the original domain

        self.base_kernel = base_kernel()
        gp.utilities.set_trainable(self.base_kernel.variance, False)
        gp.utilities.set_trainable(self.base_kernel.lengthscales, False)

    def shape_function(self, t):
        return lab.to_numpy(self.base_kernel.K_r2(tf.convert_to_tensor([2. * (1. - t)], 
                                                                       dtype=gp.config.default_float()))
        )
            
    def K(self, X, X2=None) -> tf.Tensor:
        X = tf.concat([ (X/self.lengthscales), self.bias*tf.ones_like(X[..., :1])], axis=1)
        r1 = tf.linalg.norm(X, axis=-1, keepdims=True)
        X = X/r1
        if X2 is not None:
            X2 = tf.concat([ (X2/self.lengthscales), self.bias*tf.ones_like(X2[..., :1])], axis=1)
            r2 = tf.linalg.norm(X2, axis=-1, keepdims=True)
            X2 = X2/r2
        else:
            r2 = r1
        return self.variance * self.base_kernel.K(X, X2)
        
    def K_diag(self, X) -> tf.Tensor:
        X = tf.concat([ (X/self.lengthscales), self.bias*tf.ones_like(X[..., :1])], axis=1)
        r1 = tf.linalg.norm(X, axis=-1, keepdims=True)
        X = X/r1
        return self.variance * self.base_kernel.K_diag(X)

        

class SpectralMixtureOneComponent(gp.kernels.Kernel):
    def __init__(self, num_dims=1., active_dims=None):
        super().__init__(active_dims)
        self.variance = gp.Parameter(1.0, transform=gp.utilities.positive())
        self.freq = gp.Parameter(lab.squeeze(lab.randn(num_dims)))
        self.lengthscales = gp.Parameter(lab.squeeze(1. + 0.1*lab.randn(num_dims)),
                                             transform = gp.utilities.positive())
        self.num_dims = num_dims

    def scale1(self, X):
        X_scaled = X / self.lengthscales if X is not None else X
        return X_scaled
    
    def scale2(self, X):
        X_scaled = self.freq * X if X is not None else X
        return X_scaled
    
    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        return (self.variance 
                * lab.exp(-0.5*gp.utilities.ops.square_distance(self.scale1(X), self.scale1(X2)))
                * lab.prod(lab.cos(2*math.pi*gp.utilities.ops.difference_matrix(self.scale2(X), self.scale2(X2))), 
                           axis=-1)
                )
    
    def K_diag(self, X):
        return self.variance * lab.ones(*X.shape[:-1])

class SpectralMixture(gp.kernels.Sum):
    def __init__(self, num_mixtures, num_dims):
        components = []
        for _ in range(num_mixtures):
            components = components + [SpectralMixtureOneComponent(num_dims)]
        super().__init__(components)
        self.num_mixtures = num_mixtures
        self.num_dims = num_dims

    @property
    def variance(self):
        return lab.squeeze(self.K(lab.zeros(1,self.num_dims), lab.zeros(1, self.num_dims)))