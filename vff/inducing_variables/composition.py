"""Compositions of multiple inducing variables.

THe plan is to have a convenient formalism for composition over dimensions.

MaskedProductInducingVariable:
    Inducing variables for multiple dimensions, where the input parameters are structure such that, with 
    compatible kernels, Kuf is Khatri-Rao structured, and Kuu is Kronecker structured, when there is no 
    mask. The mask filters out unneeded, extreme, values of z. The corresponding rows of Kuf, and rows and 
    columns of Kuu are deleted. For this to be memory efficient, we need an additional method for building 
    up Kuf incrementally (that is, adding and masking one dimension at a time).

AdditiveInducingVariable:
    Inducing variables for an additive (across dimensions) covariance function, where each kernel gets its 
    own set of inducing values. Then Kuf is formed by stacking these, and Kuu is formed by block diagonal 
    composition.
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
from typing import Optional, Union, List
from ..objectives import make_krp
from ..dispatch import sgpr_elbo, sgpr_precompute, posterior_precompute, Kuu, Kuf, spectral_density
from ..posteriors import PrecomputedOperator


def spherical_iff(freqs, u):
    eps = np.array([lab.squeeze(uu.epsilon) for uu in u])
    Md = np.array([lab.squeeze(uu.M) for uu in u])
    return lab.sum((freqs/(Md*eps/2))**2, axis=-1) <= 1.

def spherical_vff(freqs, u):
    eps = np.array([1/lab.squeeze(uu.b - uu.a) for uu in u])
    Md = np.array([lab.squeeze((uu.M+1)//2) for uu in u])
    return lab.sum((freqs/(Md*eps))**2, axis=-1) <= 1.

MASKS = {
    "none" : lambda freqs, u : freqs < math.inf,
    "spherical" : spherical_iff,
    "spherical_vff" : spherical_vff,
}

class MaskedProductInducingVariable(gp.inducing_variables.InducingVariables):

    def __init__(self, components : List[gp.inducing_variables.InducingVariables], mask_type: str="none",
                 cache_z = False):
        
        D = len(components)
        P = components[0].shape[-1]
        for u in components:
            if not u.shape[-2] == 1:
                raise ValueError("For product-structure inducing variables, each component must be for 1D inputs."
                                 + " Received {}".format(u.shape[-2]))
            if not u.shape[-1] == P:
                raise ValueError("For product-structured inducing variables, number of outputs must be consistent."
                                 + " Received {} and {} (and possible others)".format(P, u.shape[-1]))
        

        self.components = components
        self.Md = np.array([int(u.num_inducing) for u in self.components])
        self.D = D
        self.P = P
        self.mask_type = mask_type

        if mask_type == "none" and cache_z is False:
            self.M = lab.prod(self.Md)
        else:
            z = get_z(self)
            if cache_z:
                self.Z = z
            self.M = z.shape[-2]
        
    def reinit(self, kernel):
        """Simple temporary reinitialiser. Keep total M the same but redistribute"""
        pass

    @property
    def num_inducing(self) -> tf.Tensor:
        return self.M
    
    @property
    def shape(self) -> Shape:
        return (self.M, self.D, self.P)
    
class MaskedProductIFF(MaskedProductInducingVariable):
    def __init__(self, components, mask_type: str="spherical"):
        super().__init__(components, mask_type, cache_z=True)

class MaskedProductKronecker(MaskedProductInducingVariable):
    """Special case where Kuu is a kronecker product. """
    
@Kuf.register(MaskedProductInducingVariable, Kernel, object)
def Kuf_masked_product(inducing_variable: MaskedProductInducingVariable, kernel: Kernel, x):
    """Kuf as a Khatri-Rao product over the input dimensions, for inducing variable and compatible kernels set up for 
    this case. This is the fallback method, which is suitable for example where the kernel doesn't appear in Kuf 
    calculations."""
    if inducing_variable.mask_type=="none":
        corr_matrix = functools.reduce(make_krp, [Kuf(inducing_variable.components[d], kernel, 
                                                      lab.expand_dims(x[..., d], axis=-1)).to_dense() 
                                                  for d in range(inducing_variable.D)]
        )
    elif lab.prod(inducing_variable.Md) < 100000:
        corr_matrix = functools.reduce(make_krp, 
                                       [lab.to_numpy(
            Kuf(inducing_variable.components[d], kernel, lab.expand_dims(x[..., d], axis=-1)).to_dense())
                                                  for d in range(inducing_variable.D)]
        )[inducing_variable._mask, :]
    else:
        _, corr_matrix = functools.reduce(lambda x, y: add_kuf_dim(x, y, inducing_variable), 
                                          [(lab.to_numpy(inducing_variable.components[d].Z),
                lab.to_numpy(
            Kuf(inducing_variable.components[d], kernel, lab.expand_dims(x[..., d], axis=-1)).to_dense())
                                          )
                for d in range(inducing_variable.D)]
        )
    return tf.linalg.LinearOperatorFullMatrix(corr_matrix)

@Kuu.register(MaskedProductIFF, Kernel)
def Kuu_iff(inducing_variable: MaskedProductIFF, kernel: Kernel):
    """When rescaled as Kuf, so that Kuf is independent of the hyperparameters, Kuu is just a diagonal matrix of 
    spectral density evaluations scaled by 1/(ε_1...ε_D)"""
    vol = lab.prod(np.array([u.epsilon for u in inducing_variable.components]))
    return tf.linalg.LinearOperatorInversion(tf.linalg.LinearOperatorDiag(
                (spectral_density(kernel, inducing_variable.Z)*vol),
                is_non_singular=True,
                is_self_adjoint=True,
                is_positive_definite=True,
                is_square=True
            ))

@Kuu.register(MaskedProductKronecker, gp.kernels.Product)
def Kuu_kron(inducing_variable: MaskedProductKronecker, kernel: gp.kernels.Product):
    """TODO: we should treat this in the same way as kuf. If M is sufficiently small, build it directly and use the 
    cached mask. If M is large, build it bit by bit, culling the extras along the way."""
    if not len(kernel.kernels) == inducing_variable.D:
        raise ValueError("Inconsistent num dims in masked Kronecker product: "
                         + "kernel has {} but inducing variable {}".format(len(kernel.kernels, inducing_variable.D)))
    if not kernel.on_separate_dimensions:
        raise ValueError("Kronecker product inducing variables must have kronecker structure covariance "
                         + "(found kernel.on_separate_dimensions==False)")
    kuu = tf.linalg.LinearOperatorKronecker([Kuu(inducing_variable.components[d], kernel.kernels[d]) 
                                             for d in range(inducing_variable.D)],
                                             is_non_singular=True,
                                             is_self_adjoint=True,
                                             is_positive_definite=True,
                                             is_square=True)
    if inducing_variable.mask_type == "none":
        return tf.linalg.LinearOperatorKronecker([Kuu(inducing_variable.components[d], kernel.kernels[d]) 
                                             for d in range(inducing_variable.D)],
                                             is_non_singular=True,
                                             is_self_adjoint=True,
                                             is_positive_definite=True,
                                             is_square=True)
    elif inducing_variable.num_inducing < 100000:
        kuu = tf.linalg.LinearOperatorKronecker([Kuu(inducing_variable.components[d], kernel.kernels[d]) 
                                             for d in range(inducing_variable.D)]).to_dense()
        if lab.rank(kuu) == 2:
            kuu = lab.transpose(tf.ragged.boolean_mask(
                lab.transpose(tf.ragged.boolean_mask(kuu, inducing_variable._mask)), 
                inducing_variable._mask))
        else:
            mask = lab.tile(lab.expand_dims(inducing_variable._mask), axis=0, 
                            times=lab.rank(kuu) - lab.rank(inducing_variable._mask ),
                                *kuu.shape[:-lab.rank(inducing_variable._mask)], 
                                *((1,) * lab.rank(inducing_variable._mask)))
            kuu = lab.transpose(tf.ragged.boolean_mask(
                lab.transpose(tf.ragged.boolean_mask(kuu, mask).to_tensor), mask))
        return tf.linalg.LinearOperatorFullMatrix(kuu, is_non_singular=True, is_self_adjoint=True, 
                                                  is_positive_definite=True, is_square=True)
    else:
        raise NotImplementedError

def add_kuf_dim(z_and_kuf, z_and_kuf_new, inducing_variable):

    z, kuf = z_and_kuf
    z_new, kuf_new = z_and_kuf_new
    num_inputs = z.shape[-2]
    z_out = lab.concat(*[lab.concat(*[lab.concat(
                        lab.expand_dims(z[j, :], axis=-2), lab.uprank(zed, 2), axis=-1) 
                        for j in range(num_inputs)]) for zed in z_new]
                        )
    d = z_out.shape[-1]
    mask = MASKS[inducing_variable.mask_type](z_out, inducing_variable.components[:d])
    kuf_out = make_krp(kuf, kuf_new)
    return (z_out[..., mask, :], kuf_out[..., mask, :])
    
def add_dim(z, z_new, inducing_variable):
    """Given d dimensional inputs z obeying the mask constraint and 1-dimensional inputs z_new,
    construct d+1 dimensional inputs and apply the mask constraint"""
    z_new = lab.to_numpy(lab.uprank(z_new, 2))
    num_inputs = z.shape[-2]
    d = z.shape[-1]
    z_out = lab.concat(*[lab.concat(*[lab.concat(
                        lab.expand_dims(z[j, :], axis=-2), lab.uprank(zed, 2), axis=-1) 
                        for j in range(num_inputs)]) for zed in z_new]
                        )
    mask = MASKS[inducing_variable.mask_type](z_out, inducing_variable.components[:d+1])
    return z_out[mask, :]

def get_z(inducing_variable: MaskedProductInducingVariable):
    z1d = [lab.to_numpy(u.Z) for u in inducing_variable.components]
    if inducing_variable.mask_type == "none" or lab.prod(inducing_variable.Md) < 100000:
        z = [*np.meshgrid(*z1d)]
        z = [lab.expand_dims(z[d], axis=-1) for d in range(inducing_variable.D-1, -1, -1)]
        z = lab.reshape(lab.concat(*z, axis=-1), lab.prod(inducing_variable.Md), inducing_variable.D)
        inducing_variable._mask = MASKS[inducing_variable.mask_type](z, inducing_variable.components)
        if not inducing_variable.mask_type == "none":
            z = z[..., inducing_variable._mask, :]
    else:
        z = functools.reduce(lambda x, y: add_dim(x, y, inducing_variable), [z1d[d]
                                       for d in range(inducing_variable.D)])

    return z

class AdditiveInducingVariable(gp.inducing_variables.InducingVariables):

    def __init__(self, components : List[gp.inducing_variables.InducingVariables], mask_type: str="none",
                 cache_z = False):
        
        D = len(components)
        P = components[0].shape[-1]
        for u in components:
            if not u.shape[-2] == 1:
                raise ValueError("For additive-structure inducing variables, each component must be for 1D inputs."
                                 + " Received {}".format(u.shape[-2]))
            if not u.shape[-1] == P:
                raise ValueError("For additive-structured inducing variables, number of outputs must be consistent."
                                 + " Received {} and {} (and possible others)".format(P, u.shape[-1]))
        

        self.components = components
        self.Md = np.array([int(u.num_inducing) for u in self.components])
        self.D = D
        self.P = P
        
        self.M = lab.sum(self.Md)

    @property
    def num_inducing(self) -> tf.Tensor:
        return self.M
    
    @property
    def shape(self) -> Shape:
        return (self.M, self.D, self.P)
    
class AdditiveInducingVariableIFF(AdditiveInducingVariable):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
@Kuf.register(AdditiveInducingVariable, gp.kernels.Sum, object)
def kuf_additive(inducing_variable: AdditiveInducingVariable, kernel: gp.kernels.Sum, x):
    if not len(kernel.kernels) == inducing_variable.D:
        raise ValueError("Inconsistent num dims in additive model: "
                         + "kernel has {} but inducing variable {}".format(len(kernel.kernels, inducing_variable.D)))
    if not kernel.on_separate_dimensions:
        raise ValueError("Additive inducing variables must have additive covariance "
                         + "(found kernel.on_separate_dimensions==False)")
    kufs = [Kuf(inducing_variable.components[d], kernel.kernels[d], 
                lab.expand_dims(x[..., d], axis=-1)).to_dense() for d in range(inducing_variable.D)]
    return tf.linalg.LinearOperatorFullMatrix(lab.concat(*kufs, axis=-2))

@Kuu.register(AdditiveInducingVariable, gp.kernels.Sum)
def kuu_additive(inducing_variable: AdditiveInducingVariable, kernel: gp.kernels.Sum):
    if not len(kernel.kernels) == inducing_variable.D:
        raise ValueError("Inconsistent num dims in additive model: "
                         + "kernel has {} but inducing variable {}".format(len(kernel.kernels, inducing_variable.D)))
    if not kernel.on_separate_dimensions:
        raise ValueError("Additive inducing variables must have additive covariance "
                         + "(found kernel.on_separate_dimensions==False)")

    kuus = [Kuu(inducing_variable.components[d], kernel.kernels[d]) for d in range(inducing_variable.D)]
    return tf.linalg.LinearOperatorBlockDiag(kuus,
                                             is_non_singular=True,
                                             is_self_adjoint=True,
                                             is_positive_definite=True,
                                             is_square=True)