from gpflow.utilities import Dispatcher

# create abstract functions for multiple dispatch
sgpr_elbo = Dispatcher("sgpr_elbo")
sgpr_precompute = Dispatcher("sgpr_precompute")
Kuu = Dispatcher("Kuu")
Kuf = Dispatcher("Kuf")
conditional_posterior_with_precompute = Dispatcher("conditional_posterior_with_precompute")
posterior_precompute = Dispatcher("posterior_precompute")
spectral_density = Dispatcher("spectral_density")
