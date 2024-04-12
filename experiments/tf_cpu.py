
import os
import sys
import getopt
import tqdm
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')
import gpflow

SEEDS = [12, 17, 44, 92, 101]
CONFIG = { 
    "SGPR-points-cv-SE" : {
        "temperature" : [100, 200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3600], 
        "precipitation" : [100, 200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3600],
        "houseprice" : [100, 200, 400, 600, 800, 1000, 1200, 1500, 1750, 2000, 2250, 2500, 2750, 3000],
        },
    "SGPR-points-cv-M52" : {
        "temperature" : [100, 200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3600], 
        "precipitation" : [100, 200, 400, 600, 800, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3600],
        "houseprice" : [100, 200, 400, 600, 800, 1000, 1200, 1500, 1750, 2000, 2250, 2500, 2750, 3000],
        },
    "SVGP-SE" : {
        #"houseprice" : [100, 200, 400, 600, 800, 1200],
    },
    "SVGP-M52" : {
        #"houseprice" : [100, 200, 400, 600, 800, 1200],
    },
    "IFF-SE" : {
        "temperature" : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], 
        "precipitation" : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "houseprice" : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    },
    "IFF-M52" : {
        "temperature" : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60], 
        "precipitation" : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
        "houseprice" : [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60],
    },
    "VFF" : {
        "temperature" : [6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61], 
        "precipitation" : [6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61],
        "houseprice" : [6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61],
    },
    "VISH-chordal-SE" : { 
       # "temperature" : [9, 18, 27, 36, 45],
       # "precipitation" : [9, 18, 27, 36],
       # "houseprice" : [9, 18, 27, 36, 45],
    },
    "VISH-chordal-M52" : { 
        #"temperature" : [9, 18, 27, 36, 45],
        #"precipitation" : [9, 18, 27, 36],
        #"houseprice" : [9, 18, 27, 36, 45],
        #"airline" : [2, 3, 4, 5, 6],
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

if __name__ == '__main__':

    args = parse_args(sys.argv[1:])
    os.chdir(args['root'])
    sys.path.append(os.getcwd())
    import vff

    for dataset in DATASETS.keys():

        logdirname = vff.expmt_utils.setup_logging(dataset, args)

        ################################################################################################################
        ############## SGPR with inducing points (re)initialised using the conditional variance method #################
        ################################################################################################################
        if dataset in CONFIG["SGPR-points-cv-SE"].keys():
            for M in tqdm.tqdm(CONFIG["SGPR-points-cv-SE"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "likelihood_variance" : 1.,
                        "kernel" : gpflow.kernels.SquaredExponential,
                        "initial_kernel_hypers" : {"lengthscales" : [0.2]*DATASETS[dataset]},
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "SGPR-points-cv", args, logdirname, 
                                                   suffix="SE_{}".format(M),
                                                **params)

        if dataset in CONFIG["SGPR-points-cv-M52"].keys():
            for M in tqdm.tqdm(CONFIG["SGPR-points-cv-M52"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "likelihood_variance" : 1.,
                        "kernel" : gpflow.kernels.Matern52,
                        "initial_kernel_hypers" : {"lengthscales" : [0.2]*DATASETS[dataset]},
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "SGPR-points-cv", args, logdirname,
                                                   suffix="M52_{}".format(M),
                                                **params)

        ################################################################################################################
        ############## SVGP with inducing points optimised #################
        ################################################################################################################
        if dataset in CONFIG["SVGP-SE"].keys():
            for M in tqdm.tqdm(CONFIG["SVGP-SE"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "likelihood_variance" : 1.,
                        "kernel" : gpflow.kernels.SquaredExponential,
                        "initial_kernel_hypers" : {"lengthscales" : [0.2]*DATASETS[dataset]},
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "SVGP", args, logdirname, 
                                                   suffix="SE_{}".format(M),
                                                **params)

        if dataset in CONFIG["SVGP-M52"].keys():
            for M in tqdm.tqdm(CONFIG["SVGP-M52"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "likelihood_variance" : 1.,
                        "kernel" : gpflow.kernels.Matern52,
                        "initial_kernel_hypers" : {"lengthscales" : [0.2]*DATASETS[dataset]},
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "SVGP", args, logdirname,
                                                   suffix="M52_{}".format(M),
                                                **params)
        ################################################################################################################
        ############## IFF #################
        ################################################################################################################
        if dataset in CONFIG["IFF-SE"].keys():
            for M in tqdm.tqdm(CONFIG["IFF-SE"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "likelihood_variance" : 1.,
                        "kernel" : gpflow.kernels.SquaredExponential,
                        "initial_kernel_hypers" : {"lengthscales" : [0.2]*DATASETS[dataset]},
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "IFF", args, logdirname, 
                                                   suffix="SE_{}".format(M),
                                                **params)

        if dataset in CONFIG["IFF-M52"].keys():
            for M in tqdm.tqdm(CONFIG["IFF-M52"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "likelihood_variance" : 1.,
                        "kernel" : gpflow.kernels.Matern52,
                        "initial_kernel_hypers" : {"lengthscales" : [0.2]*DATASETS[dataset]},
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "IFF", args, logdirname, 
                                                   suffix="M52_{}".format(M),
                                                **params)
                    
        ################################################################################################################
        ############## VFF #################
        ################################################################################################################
        if dataset in CONFIG["VFF"].keys():
            for M in tqdm.tqdm(CONFIG["VFF"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "likelihood_variance" : 1.,
                        "kernel" : gpflow.kernels.Product,
                        "initial_kernel_hypers" : {"kernels" : 
                                                   [gpflow.kernels.Matern52(lengthscales=0.2, active_dims=[d]) for d in range(DATASETS[dataset])]
                            },
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "VFF", args, logdirname, 
                                                   suffix="M52_{}".format(M),
                                                **params)

        ################################################################################################################
        ############## VISH using a chordal kernel #################
        ################################################################################################################
        if dataset in CONFIG["VISH-chordal-SE"].keys():
            for ell in tqdm.tqdm(CONFIG["VISH-chordal-SE"][dataset]):
                for seed in SEEDS:
                    params = {
                        "ell" : ell,
                        "likelihood_variance" : 1.,
                        "base_kernel" : gpflow.kernels.SquaredExponential,
                        "initial_kernel_hypers" : {"lengthscales" : [0.2]*DATASETS[dataset]},
                    }

                    vff.expmt_utils.run_experiment(seed, dataset, "VISH-chordal", args, logdirname, 
                                                   suffix="SE_{}".format(ell),
                                                **params)

        if dataset in CONFIG["VISH-chordal-M52"].keys():
            for ell in tqdm.tqdm(CONFIG["VISH-chordal-M52"][dataset]):
                for seed in SEEDS:
                    params = {
                        "ell" : ell,
                        "likelihood_variance" : 1.,
                        "base_kernel" : gpflow.kernels.Matern52,
                        "initial_kernel_hypers" : {"lengthscales" : [0.2]*DATASETS[dataset]},
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "VISH-chordal", args, logdirname, 
                                                   suffix="M52_{}".format(ell),
                                                **params)
