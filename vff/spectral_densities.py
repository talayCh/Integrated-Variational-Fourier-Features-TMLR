"""
Spectral densities corresponding to stationary kernels

TODO: implement a fallback for stationary covariance functions using the (?) fft
TODO: implement a fallback for sums, taking into account active_dims, and for products across dimensions
"""

import gpflow as gp
import math
import lab
import lab.tensorflow
import scipy
from .kernels import SpectralMixture
from .dispatch import spectral_density

@spectral_density.register(gp.kernels.SquaredExponential, object)
def spectral_density_gaussian(kernel: gp.kernels.SquaredExponential, freq: lab.Numeric):
    D = freq.shape[-1]
    return (math.sqrt(2*math.pi)**D * kernel.variance
                * lab.exp(-( lab.sum((2*math.pi*freq * kernel.lengthscales)**2, axis=-1) / 2))
                * lab.prod(kernel.lengthscales*1.)
                )

def matern_spectral_density(k, freq, nu):
    """Matérn-ν spectral density function (because the gpflow implementation of the Matérn covariance functions does 
    not contain the order except implicitly in the class name).

    The spectral density of a Matérn covariance function of order ν with lengthscale matrix Λ is a multivariate 
    centred Student t distribution with scale matrix Λ^{-2} and order 2ν. That is

    s(ξ) = (2π)^D ( Γ(ν+D/2) / (Γ(ν)(2π)^(D/2) ν^(D/2)) ) |Λ| [1 + ξ^⊤ Λ^2 ξ / (2ν)]^{-(ν+D/2)}

    where the leading (2π)^D factor is so that it is a proper density.
    """
    D = freq.shape[-1]
    a = 1 + lab.sum(( 2*math.pi * freq * k.lengthscales)**2, axis=-1)/(2*nu)
    b = scipy.special.gamma(nu+D/2)/scipy.special.gamma(nu) / math.sqrt(
        (2*nu*math.pi)**D) * lab.prod(k.lengthscales*1)
    return (2*math.pi)**D * k.variance * b / (lab.sqrt(a)**(2*nu + D))

@spectral_density.register(gp.kernels.Matern12, object)
def spectral_density_matern12(kernel: gp.kernels.Matern12, freq: lab.Numeric):
    return matern_spectral_density(kernel, freq, 0.5)

@spectral_density.register(gp.kernels.Matern32, object)
def spectral_density_matern32(kernel: gp.kernels.Matern32, freq: lab.Numeric):
    return matern_spectral_density(kernel, freq, 1.5)

@spectral_density.register(gp.kernels.Matern52, object)
def spectral_density_matern52(kernel: gp.kernels.Matern52, freq: lab.Numeric):
    return matern_spectral_density(kernel, freq, 2.5)

@spectral_density.register(SpectralMixture, object)    
def spectral_mixture_spectral_density(k, freqs):
    # first, gather the parameters
    for q in range(k.num_mixtures):
        component = k.kernels[q]
        if q == 0:
            weight = lab.uprank(component.variance*1, rank=1)
            invscale = lab.expand_dims(lab.uprank(component.lengthscales*1, rank=1), axis=-2)
            loc = lab.expand_dims(lab.uprank(component.freq*1, rank=1), axis=-2)
        else:
            weight = lab.concat(weight, lab.uprank(component.variance*1, rank=1),
                               axis=-1)
            invscale = lab.concat(invscale, 
                               lab.expand_dims(lab.uprank(component.lengthscales*1, rank=1), axis=-2),
                               axis=-2
                              )
            loc = lab.concat(loc, lab.expand_dims(lab.uprank(component.freq*1, rank=1), axis=-2), axis=-2)
    # then do stuff
    freqs = lab.expand_dims(freqs, axis=-2)

    spd = lab.sum(
                weight * (lab.exp(-0.5 * lab.sum((2*math.pi*(freqs - loc) * invscale)**2, axis=-1))
                         + lab.exp(-0.5 * lab.sum((2*math.pi*(freqs + loc) * invscale)**2, axis=-1))
                         ) * lab.sqrt(2*math.pi)**k.num_dims * lab.prod(invscale, axis=-1),# / lab.sqrt((2*math.pi)),
                axis=-1)
    
    return spd * 0.5