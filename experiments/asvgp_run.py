
import os
import sys
import getopt
import tqdm
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import gpflow
import getopt
import lab
import math
import numpy as np
import json
import time
import tensorflow as tf
import gpflow
from scipy.cluster.vq import kmeans
import pandas as pd
import tqdm
import itertools
import lab.tensorflow

SEEDS = [12, 17, 44, 92, 101]
CONFIG = { # M^(1/D) 
    "ASVGP" : {
        "temperature" : [12, 24, 36, 48, 60, 72, 84, 96],
        "precipitation" : [12, 24, 36, 48, 60, 72, 84, 96],
        "houseprice" : [12, 24, 36, 48, 60, 72, 84, 96],
    },
}

DATASETS = { # dimensions
    "temperature" : 2,
    "precipitation" : 2,
    "houseprice" : 2,
}

def parse_args(argv):
    """Get command line arguments, set defaults and return a dict of settings."""

    args_dict = {
        ## File paths and logging
            'data'          : '/data/',             # relative path to data dir from root
            'root'          : '.',                 # path to root directory from which paths are relative
            'logdir'        : '/experiments/logs/', # relative path to logs from root
            'asvgp_dir'       : '../asvgp/'         # relative path to the directory with the ASVGP code

    }

    try:
        opts, _ = getopt.getopt(argv, '', [name + '=' for name in args_dict.keys()])
    except getopt.GetoptError:
        helpstr = 'Check options. Permitted long options are '
        print(helpstr, args_dict.keys())

    for opt, arg in opts:
        opt_name = opt[2:] # remove initial --
        args_dict[opt_name] = arg
    
    return args_dict



METHODS = {
    "ASVGP",
}

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

    k = [kernel(**initial_kernel_hypers) for d in range(D)]
    a = [lab.min(x_train[..., d])-0.1 for d in range(D)]
    b = [lab.max(x_train[..., d])+0.1 for d in range(D)]
    bases = [basis.B4Spline(a[d], b[d], M) for d in range(D)]

    model = GPR_kron((x_train, lab.expand_dims(y_train, axis=-1)), 
                               k,
                               bases)
    optim = gpflow.optimizers.Scipy()
    def minimise(m):
        optim.minimize(m.training_loss,
                        m.trainable_variables, 
                        options={"maxiter":maxiter}
                        )
    return model, minimise

initialise_model = {
    "ASVGP" : initialise_asvgp,
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

train_evaluation = {
    "ASVGP" : nvfe_sgpr,
}

def metrics_gpflow(x_test, y_test, stdy, model):
    mse = 0
    nlpd = 0
    for i in range(0, x_test.shape[-2], 10000):
        xc = x_test[..., i : (i + 10000), :]
        yc = y_test[..., i : (i + 10000)]
        mean, sigma = model.predict_f(xc)
        mean = lab.to_numpy(mean)
        mse += se_(lab.squeeze(mean), yc)
        nlpd += nlpd_(lab.squeeze(mean), lab.squeeze(lab.to_numpy(sigma + model.likelihood.variance)),
                        yc, stdy)
    rmse = lab.sqrt(mse/y_test.shape[-1])*stdy
    nlpd = nlpd/y_test.shape[-1]
    return rmse, nlpd

test_evaluation = {
    "ASVGP" : metrics_gpflow,
}

def printable_params_asvgp(model):
    return {
        "M" : int(model.m**model.d),
        "noise variance" : lab.to_numpy(1*model.likelihood.variance).item(),
        **printable_kernel_params(model.kernels)
    }

printable_log_params = {
    "ASVGP" : printable_params_asvgp,
}


def printable_kernel_params(kernels):
    k_vars = []
    k_ells = []
    for d in range(len(kernels)):
        k_vars = k_vars + [lab.to_numpy(1*kernels[d].variance).item()]
        k_ells = k_ells + [lab.to_numpy(1*kernels[d].lengthscales).item()]
    return {
            "kernel variance" : k_vars,
            "kernel lengthscales": k_ells
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




if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    os.chdir(args['root'])
    os.chdir(args["asvgp_dir"])
    sys.path.append(os.getcwd())
    print(sys.path)
    import asvgp.basis as basis
    from asvgp.gpr import GPR_kron
    os.chdir(args["root"])
    sys.path.append(os.getcwd())

    for dataset in DATASETS.keys():

        logdirname = setup_logging(dataset, args)


        ################################################################################################################
        ############## ASVGP (B splines) #################
        ################################################################################################################
        if dataset in CONFIG["ASVGP"].keys():
            for M in tqdm.tqdm(CONFIG["ASVGP"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "kernel" : gpflow.kernels.Matern52,
                        "initial_kernel_hypers" : {"lengthscales" : 0.2},
                    }
                    run_experiment(seed, dataset, "ASVGP", args, logdirname, 
                                                   suffix="M52_{}".format(M),root_path=args["root"],
                                                   rel_path=args["asvgp_dir"],
                                                **params)
