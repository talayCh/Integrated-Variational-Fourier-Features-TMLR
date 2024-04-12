import getopt
import lab
import math
import os
import numpy as np
import json
import time
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU') # should stop tf eating all the GPU memory
import gpflow
import torch
import gpytorch
from .sgpr import SGPR
from . import initialise
from . import kernels
from .inducing_variables import IntegratedFourierFeature1D
from .inducing_variables import SphericalHarmonicFeature, MaskedProductIFF, VariationalFourierFeatureProduct, VariationalFourierFeature1D
from .inducing_variables import VariationalFourierFeatureAdditive, AdditiveInducingVariableIFF
from .objectives import log_marginal_likelihood
from scipy.cluster.vq import kmeans
from gpflow.utilities import Dispatcher
import pandas as pd
from netCDF4 import Dataset as NetCDFFile
import pyproj
import tqdm
import itertools
import sys
import lab.tensorflow
import lab.torch

METHODS = {
    "SGPR-points-km",
    "SGPR-points-cv",
    "SGPR-points-gd",
    "SVGP",
    "VFF",
    "IFF-inf",
    "IFF",
    "VISH-chordal",
    "ASVGP",
}

DATASETS = {
    "precipitation",
    "temperature",
    "houseprice",
    "power",
    "airline",
}

printable_kernel_params = Dispatcher("printable_kernel_params")

def se_(pred_mean, y):
    """Unnormalised squared predictive error from normalised data and predictions, and normalising scale for 
    y.
    
    RMSE = \sqrt((y-μ)^T (y-υ)/N)

    where y is length N, and μ is the predictive mean at the same points.
    """
    return lab.sum((pred_mean - y)**2)

def nlpd_(pred_mean, pred_var, y, stdy):
    """Unnormalised marginal negative log predictive probability density from normalised data and predictions, and 
    normalising scale for y.
    
    NLPD(m) = 0.5 N log2πs^2 + 0.5 * ∑_n (log σ_n^2 + (y_n-μ_n)^2 / σ_n^2 )

    where (μ_n, σ_n) are the marginal predictive mean and standard deviation at the same point as y_n, and s is the 
    normalising scale for y (stdy).
    """
    nt = y.shape[-1]
    op = math.log(2*math.pi) * nt/2 + math.log(stdy)*nt
    op = op + lab.sum(lab.log(pred_var))/2
    op = op + lab.sum((y-pred_mean)**2 / pred_var)/2
    return op

class SKI(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, grid_size, num_dims, base_kernel, base_kernel_args):
        super().__init__(train_x, train_y, likelihood)
        
        #grid_size = gpytorch.utils.grid.choose_grid_size(train_x)
        grid_size = grid_size
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.GridInterpolationKernel(
                base_kernel(**base_kernel_args, ard_num_dims=num_dims), 
                grid_size=grid_size, 
                num_dims=num_dims,
            )
        )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def setup_logging(dataset_name, args):
    done = False
    suffix = 0
    while not done:
        try:
            logdirname = args['root'] + args['logdir'] + dataset_name + '_' + str(suffix)
            os.mkdir(logdirname)
            done = True
        except FileExistsError:
            suffix += 1

    return logdirname

def run_experiment(seed, dataset_name, method_name, args, logdirname, suffix, **params):
    print(method_name, seed)
    lab.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    N, D, x_train, y_train, x_test, y_test, meanx, stdx, meany, stdy = load_training_data(
        dataset_name,
        args['root']+args['data'], 
        rng
    )

    wc = time.perf_counter()
    model, minimise = initialise_model[method_name](seed, rng, N, D, x_train, y_train, x_test, y_test, **params)
    minimise(model)
    wc = time.perf_counter() - wc 
    nvfe = train_evaluation[method_name](N, x_train, y_train, model)
    rmse, nlpd = test_evaluation[method_name](x_test, y_test, stdy, model)
    
    loggable_params = printable_log_params[method_name](model)

    results_dict = {
        "NVFE" : lab.to_numpy(nvfe).item(),
        "RMSE" : lab.to_numpy(rmse).item(),
        "NLPD" : lab.to_numpy(nlpd).item(),
        "time" : wc,
        **loggable_params
    }

    fpath = logdirname + '/' + method_name + '_{}_'.format(suffix) + str(seed)

    fname = fpath + ".json"
    with open(fname, "w") as fp:
        json.dump(results_dict, fp, indent=4)

def initialise_sgpr_km(seed, rng, N, D, x_train, y_train, x_test, y_test,
                  M = None,
                  likelihood_variance = 1.,
                  kernel = gpflow.kernels.SquaredExponential,
                  initial_kernel_hypers = {},
                  maxiter=1000,
                  max_data=20000,
                  ):
    if M is None:
        raise ValueError("For SGPR/SVGP, must specify M (number of inducing points)")
    if N > max_data:
        x_train_ = rng.permutation(x_train, axis=-2)[..., :max_data, :]
    else:
        x_train_ = x_train

    k = kernel(**initial_kernel_hypers)
    z, _ = kmeans(lab.to_numpy(x_train_), M)
    model = SGPR(
        data = (x_train, lab.expand_dims(y_train, axis=-1)),
        kernel = k,
        noise_variance = likelihood_variance,
        inducing_variable = z,
    )
    optim = gpflow.optimizers.Scipy()
    def minimise(m):
        optim.minimize(m.training_loss,
                        (*m.kernel.trainable_variables, *m.likelihood.trainable_variables), 
                        options={"maxiter":maxiter}
                        )
    return model, minimise

def initialise_sgpr_cv(seed, rng, N, D, x_train, y_train, x_test, y_test,
                  M = None,
                  likelihood_variance = 1.,
                  kernel = gpflow.kernels.SquaredExponential,
                  initial_kernel_hypers = {},
                  ):
    if M is None:
        raise ValueError("For SGPR/SVGP, must specify M (number of inducing points)")

    k = kernel(**initial_kernel_hypers)
    initialiser = initialise.ConditionalVariance(seed=seed)
    z, _ = initialiser(lab.to_numpy(x_train), M, k)
    model = SGPR(
        data = (x_train, lab.expand_dims(y_train, axis=-1)),
        kernel = k,
        noise_variance = likelihood_variance,
        inducing_variable = z,
    )
    optim = gpflow.optimizers.Scipy()
    def minimise(m):
        initialise.minimise_reinit(optim, m, initialiser)
    return model, minimise

def initialise_svgp(seed, rng, N, D, x_train, y_train, x_test, y_test,
                  M = None,
                  likelihood_variance = 1.,
                  kernel = gpflow.kernels.SquaredExponential,
                  initial_kernel_hypers = {},
                  maxiter=10000,
                  max_data=20000,
                  minibatch_size=100,
                  lr=1e-2,
                  ):
    if M is None:
        raise ValueError("For SGPR/SVGP, must specify M (number of inducing points)")
    if N > max_data:
        x_train_ = rng.permutation(x_train, axis=-2)[..., :max_data, :]
    else:
        x_train_ = x_train
    k = kernel(**initial_kernel_hypers)
    z, _ = kmeans(lab.to_numpy(x_train_), M)
    model = gpflow.models.SVGP(
        kernel = k,
        likelihood = gpflow.likelihoods.Gaussian(),
        inducing_variable = z,
        num_data = N
    )
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, lab.expand_dims(y_train, axis=-1))
                                        ).repeat().shuffle(N)
    train_iter = iter(train_dataset.batch(minibatch_size))
    training_loss = model.training_loss_closure(train_iter, compile=True)
    optim = tf.optimizers.Adam(learning_rate=lr)
    def minimise(m):
        for _ in tqdm.tqdm(range(maxiter)):
            optim.minimize(training_loss,
                        m.trainable_variables,
                        )
    return model, minimise

def initialise_iff(seed, rng, N, D, x_train, y_train, x_test, y_test,
                  M = None,
                  likelihood_variance = 1.,
                  kernel = gpflow.kernels.SquaredExponential,
                  initial_kernel_hypers = {},
                  maxiter=1000
                  ):
    if M is None:
        raise ValueError("For SGPR/SVGP, must specify M (number of inducing points)")

    k = kernel(**initial_kernel_hypers)
    width = [lab.max(x_train[..., d]) - lab.min(x_train[..., d]) for d in range(D)]
    eps = [0.95/w for w in width]
    u = MaskedProductIFF(
            [IntegratedFourierFeature1D(eps[d], M) for d in range(D)]
             )
    model = SGPR(
        data = (x_train, lab.expand_dims(y_train, axis=-1)),
        kernel = k,
        noise_variance = likelihood_variance,
        inducing_variable = u,
    )
    optim = gpflow.optimizers.Scipy()
    def minimise(m):
        optim.minimize(m.training_loss,
                        (*m.kernel.trainable_variables, *m.likelihood.trainable_variables), 
                        options={"maxiter":maxiter}
                        )
    return model, minimise

def initialise_vff(seed, rng, N, D, x_train, y_train, x_test, y_test,
                  M = None,
                  likelihood_variance = 1.,
                  kernel = gpflow.kernels.SquaredExponential,
                  initial_kernel_hypers = {},
                  maxiter=1000,
                  cheat_ab=False
                  ):
    if M is None:
        raise ValueError("For VFF, must specify M (number of inducing points)")

    k = kernel(**initial_kernel_hypers)
    if cheat_ab:
        a = [min(lab.min(x_train[..., d]), lab.min(x_test[..., d]))-0.1 for d in range(D)]
        b = [max(lab.max(x_train[..., d]), lab.max(x_test[..., d]))+0.1 for d in range(D)]
    else:
        a = [lab.min(x_train[..., d])-0.1 for d in range(D)]
        b = [lab.max(x_train[..., d])-0.1 for d in range(D)]
    if D == 1:
        u = VariationalFourierFeature1D(a=a[0], b=b[0], M=M)
    else:
        u = VariationalFourierFeatureProduct(a=np.array(a), b=np.array(b), Md=M, D=D, mask_type="none")
    model = SGPR(
        data = (x_train, lab.expand_dims(y_train, axis=-1)),
        kernel = k,
        noise_variance = likelihood_variance,
        inducing_variable = u,
    )
    optim = gpflow.optimizers.Scipy()
    def minimise(m):
        optim.minimize(m.training_loss,
                        (*m.kernel.trainable_variables, *m.likelihood.trainable_variables), 
                        options={"maxiter":maxiter}
                        )
    return model, minimise

def initialise_asvgp(seed, rng, N, D, x_train, y_train, x_test, y_test,
                  M = None,
                  kernel = gpflow.kernels.SquaredExponential,
                  initial_kernel_hypers = {},
                  maxiter=1000,
                  root_path=None,
                  rel_path=None,
                  ):
    if M is None:
        raise ValueError("For SGPR/SVGP, must specify M (number of inducing points)")
    os.chdir(root_path)
    os.chdir(rel_path)
    sys.path.append(os.getcwd())
    import asvgp.basis as basis
    from asvgp.gpr import GPR_kron
    os.chdir(root_path)
    sys.path.append(os.getcwd())

    k = [kernel(**initial_kernel_hypers) for d in range(D)]
    a = [lab.min(x_train[..., d])-0.1 for d in range(D)]
    b = [lab.max(x_train[..., d])+0.1 for d in range(D)]
    bases = [basis.B4Spline(a[d], b[d], M) for d in range(D)]

    model = GPR_kron((x_train, y_train), 
                               k,
                               bases)
    optim = gpflow.optimizers.Scipy()
    def minimise(m):
        optim.minimize(m.training_loss,
                        (*m.kernel.trainable_variables, *m.likelihood.trainable_variables), 
                        options={"maxiter":maxiter}
                        )
    return model, minimise

def initialise_vish_chordal(seed, rng, N, D, x_train, y_train, x_test, y_test,
                  ell = None,
                  likelihood_variance = 1.,
                  base_kernel = gpflow.kernels.SquaredExponential,
                  initial_kernel_hypers = {},
                  maxiter=1000,
                  ):
    if ell is None:
        raise ValueError("For VISH, must specify the max degree ell (we use levels up to ell-1 inclusive)")  
    k = kernels.Chordal(base_kernel=base_kernel, dimension=D, **initial_kernel_hypers)
    u = SphericalHarmonicFeature(dimension=D, max_degree=ell)
    model = SGPR(
        data = (x_train, lab.expand_dims(y_train, axis=-1)),
        kernel = k,
        noise_variance = likelihood_variance,
        inducing_variable = u,
    )
    optim = gpflow.optimizers.Scipy()
    def minimise(m):
        optim.minimize(m.training_loss, m.trainable_variables, options={"maxiter":maxiter})
    return model, minimise

def initialise_ski(seed, rng, N, D, x_train, y_train, x_test, y_test,
                   M = None,
                   base_kernel = gpytorch.kernels.RBFKernel,
                   base_kernel_args = {},
                   maxiter = 1000
                   ):
    if M is None:
        raise ValueError("For SKI, must specify the grid size M")
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    x_train_ = torch.tensor(x_train).double().detach().cuda()
    y_train_ = torch.tensor(y_train).double().detach().cuda()
    model = SKI(x_train_,
                y_train_,
                likelihood,
                grid_size=M,
                num_dims=x_train.shape[-1],
                base_kernel=base_kernel,
                base_kernel_args=base_kernel_args).double().cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.1)
    lml = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    def minimise(ml):
        m, lml = ml
        prev = -math.inf
        for _ in tqdm.tqdm(range(maxiter)):
            optim.zero_grad()
            output = m(x_train_)
            loss = -lml(output, y_train_)
            loss.backward()
            optim.step()
            change = torch.abs(loss - prev)
            prev = loss
            if change < 1e-6:
                break
    return (model, lml), minimise

initialise_model = {
    "SGPR-points-km" : initialise_sgpr_km,
    "SGPR-points-cv" : initialise_sgpr_cv,
    "SVGP" : initialise_svgp,
    "VISH-chordal" : initialise_vish_chordal,
    "VFF" : initialise_vff,
    "IFF" : initialise_iff,
    "SKI" : initialise_ski,
}

def nvfe_sgpr(N, x_train, y_train, model):
    return model.training_loss()/N

def nvfe_svgp(N, x_train, y_train, model):
    nvfe = 0
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, lab.expand_dims(y_train, axis=-1))
                                        ).repeat().shuffle(N)
    minibatch_size=100
    train_iter = iter(train_dataset.batch(minibatch_size))
    for i, batch in enumerate(itertools.islice(train_iter, 1000)):
        nvfe += model.elbo(batch)
    nvfe = nvfe/(i+1)/N
    return nvfe

def nvfe_torch(N, x_train, y_train, model):
    m, lml = model
    nvfe = -lml(m(torch.tensor(x_train).double().detach().cuda()), 
                torch.tensor(y_train).double().detach().cuda())
    return nvfe

train_evaluation = {
    "SGPR-points-km" : nvfe_sgpr,
    "SGPR-points-cv" : nvfe_sgpr,
    "SVGP" : nvfe_svgp,
    "VISH-chordal" : nvfe_sgpr,
    "VFF" : nvfe_sgpr,
    "IFF" : nvfe_sgpr,
    "ASVGP" : nvfe_sgpr,
    "SKI" : nvfe_torch,
}

def metrics_gpflow(x_test, y_test, stdy, model):
    mse = 0
    nlpd = 0
    for i in range(0, x_test.shape[-2], 10000):
        xc = x_test[..., i : (i + 10000), :]
        yc = y_test[..., i : (i + 10000)]
        mean, sigma = model.predict_f(xc)
        mse += se_(lab.squeeze(1.*mean), yc)
        nlpd += nlpd_(lab.squeeze(1.*mean), lab.squeeze(sigma + model.likelihood.variance),
                        yc, stdy)
    rmse = lab.sqrt(mse/y_test.shape[-1])*stdy
    nlpd = nlpd/y_test.shape[-1]
    return rmse, nlpd

def metrics_torch(x_test, y_test, stdy, model):
    m = model[0].cpu()
    m.eval()
    x_test_ = torch.tensor(x_test).double().cpu()
    y_test_ = torch.tensor(y_test).double().cpu()
    mse = 0
    nlpd = 0
    with torch.no_grad():
        nt = 0 # number of valid values for nlpd
        for i in range(0, x_test.shape[-2], 10000):

            xc = x_test_[..., i : (i + 10000), :]
            yc = y_test_[..., i : (i + 10000)]
            with gpytorch.settings.fast_pred_var():
                pred = m.likelihood(m(xc))
                mean = pred.mean 
                sigma = torch.diagonal(pred.covariance_matrix)
                
            mse += se_(lab.squeeze(mean), yc)
            nlpd += nlpd_(lab.squeeze(mean)[sigma > 0], lab.squeeze(sigma)[sigma > 0],
                        yc[sigma > 0], stdy)
            nt += yc[sigma > 0].shape[-1]
        rmse = lab.sqrt(mse/y_test.shape[-1])*stdy
        nlpd = nlpd/nt
    return rmse, nlpd

test_evaluation = {
    "SGPR-points-km" : metrics_gpflow,
    "SGPR-points-cv" : metrics_gpflow,
    "SVGP" : metrics_gpflow,
    "VISH-chordal" : metrics_gpflow,
    "VFF" : metrics_gpflow,
    "IFF" : metrics_gpflow,
    "ASVGP" : metrics_gpflow,
    "SKI" : metrics_torch,
}

def printable_params_sgpr(model):
    return {
        "M" : int(model.inducing_variable.num_inducing),
        "noise variance" : lab.to_numpy(1*model.likelihood.variance).item(),
        **printable_kernel_params(model.kernel)
    }

def printable_params_asvgp(model):
    return {
        "M" : int(model.inducing_variable.num_inducing),
        "noise variance" : lab.to_numpy(1*model.likelihood.variance).item(),
        **printable_kernel_params(model.kernels)
    }

def printable_params_ski(model):
    return {
        "Grid size" : lab.to_numpy(model[0].covar_module.base_kernel.grid_sizes),
        "noise variance" : lab.to_numpy(model[0].likelihood.noise).item(),
        **printable_kernel_params(model[0])
    }

printable_log_params = {
    "SGPR-points-km" : printable_params_sgpr,
    "SGPR-points-cv" : printable_params_sgpr,
    "SVGP" : printable_params_sgpr,
    "VISH-chordal" : printable_params_sgpr,
    "VFF" : printable_params_sgpr,
    "IFF" : printable_params_sgpr,
    "ASVGP" : printable_params_asvgp,
    "SKI" : printable_params_ski,
}

@printable_kernel_params.register((gpflow.kernels.IsotropicStationary))
def printable_radial_kernel_params(kernel : gpflow.kernels.IsotropicStationary):
    return {
        "kernel variance" : lab.to_numpy(1*kernel.variance).item(),
        "kernel lengthscales": [*lab.to_numpy(1*kernel.lengthscales)]
    }

@printable_kernel_params.register((gpflow.kernels.Product, gpflow.kernels.Sum))
def printable_composition_kernel_params(kernel):
    k_vars = []
    k_ells = []
    for d in range(len(kernel.kernels)):
        k_vars = k_vars + [lab.to_numpy(1*kernel.kernels[d].variance).item()]
        k_ells = k_ells + [lab.to_numpy(1*kernel.kernels[d].lengthscales).item()]
    return {
            "kernel variance" : k_vars,
            "kernel lengthscales": k_ells
    }

@printable_kernel_params.register(list)
def prinatable_listed_kernel_params(kernels):
    k_vars = []
    k_ells = []
    for d in range(len(kernels)):
        k_vars = k_vars + [lab.to_numpy(1*kernels[d].variance).item()]
        k_ells = k_ells + [lab.to_numpy(1*kernels[d].lengthscales).item()]
    return {
            "kernel variance" : k_vars,
            "kernel lengthscales": k_ells
    }

@printable_kernel_params.register(kernels.Chordal)
def printable_chordal_kernel_params(kernel : kernels.Chordal):
    return {
        "kernel variance" : lab.to_numpy(1*kernel.variance).item(),
        "kernel lengthscales": [*lab.to_numpy(1*kernel.lengthscales)],
        "kernel bias": lab.to_numpy(1*kernel.bias).item()
    }

@printable_kernel_params.register(SKI)
def printable_ski_kernel_params(model : SKI):
    return {
        "kernel variance" : lab.to_numpy(model.covar_module.outputscale).item(),
        "kernel lengthscales": [*lab.squeeze(
                        lab.to_numpy(model.covar_module.base_kernel.base_kernel.lengthscale))]
}

### data loading
def load_training_data(dataset_name, datadir, rng):
    x, y = load_data[dataset_name](datadir)

    stdx = lab.std(x, axis=-2)
    meanx = lab.mean(x, axis=-2)
    x = (x -  meanx) / stdx
    stdy = lab.std(y, axis=-1)
    meany = lab.mean(y, axis=-1)
    y = (y - meany) / stdy

    num_train = math.ceil(80/100 * y.shape[0])
    idx = np.arange(0, y.shape[0], 1)
    rng.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    x_train = x[..., train_idx, :]
    y_train = y[..., train_idx]
    x_test = x[..., test_idx, :]
    y_test = y[..., test_idx]

    N = y_train.shape[0]
    D = x.shape[-1]
    return N, D, x_train, y_train, x_test, y_test, meanx, stdx, meany, stdy

def load_temperature_data(datadir):
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    fname = datadir + 'trends_feb_2020_2021.nc'
    nc = NetCDFFile(fname)
    data = nc['TEMPTREND'][:, :]
    lons = nc['lon'][:]
    lats = nc['lat'][:]

    lons, lats = np.meshgrid(lons, lats)

    lons_real = lons[~data.mask]
    lats_real = lats[~data.mask]
    data_real = data[~data.mask]

    x = np.array(np.vstack((lons_real.flatten(), lats_real.flatten())).T).astype(np.float64)
    y = np.array(data_real.flatten()).astype(np.float64)
    np.seterr(all='raise')

    return x, y

def load_precipitation_data(datadir):
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    fname = datadir + 'nws_precip_mtd_20210101_conus.nc'
    nc = NetCDFFile(fname)
    data = 25.4*nc.variables['normal'][:][::4, ::4]
    x_n, y_n = nc.variables['x'][:][::4], nc.variables['y'][::-1][::4]
    x_grid, y_grid= np.meshgrid(x_n,y_n)
    p = pyproj.Proj(projparams='+proj=stere +lat_0=90 +lat_ts=60 +lon_0=-105 +x_0=0 +y_0=0 +a=6371200 +b=6371200 +units=m +no_defs')
    lons, lats = p(x_grid, y_grid, inverse=True)

    lons_real = lons[~data.mask]
    lats_real = lats[~data.mask]
    data_real = data[~data.mask]

    x = np.array(np.vstack((lons_real.flatten(), lats_real.flatten())).T).astype(np.float64)
    y = np.array(data_real.flatten()).astype(np.float64)
    np.seterr(all='raise')

    return x, y

def load_houseprice_data(datadir, frac=0.2):
    fname = datadir + 'england_wales_house_prices.csv'
    df = pd.read_csv(fname)
    df = df.sample(frac=frac, random_state=1)

    x = np.array(np.vstack((df['longitude'], df['latitude'])).T).astype(np.float64)
    y = np.array(np.log(df['price']).to_numpy()).astype(np.float64)

    return x, y

def load_power_data(datadir):
    fname = datadir + 'uci_power.csv'
    df = pd.read_csv(fname)

    y = np.array(df['PE'])
    x = np.array(df.iloc[:, :-1])
    return x, y

def load_airline_data(datadir, frac=1.0):
    fname = datadir + 'airline.pickle'
    df = pd.read_pickle(fname)
    df = df.sample(frac=frac, random_state=1)

    # convert to minutes after midnight
    df.ArrTime = 60 * np.floor(df.ArrTime / 100) + np.mod(df.ArrTime, 100)
    df.DepTime = 60 * np.floor(df.DepTime / 100) + np.mod(df.DepTime, 100)

    names = [
        "Month",
        "DayofMonth",
        "DayOfWeek",
        "plane_age",
        "AirTime",
        "Distance",
        "ArrTime",
        "DepTime",
    ]
    x = df[names].values
    y = df["ArrDelay"].values

    return x, y

load_data = {
    "temperature" : load_temperature_data,
    "precipitation" : load_precipitation_data,
    "power" : load_power_data,
    "houseprice" : load_houseprice_data,
    "airline" : load_airline_data,
}

