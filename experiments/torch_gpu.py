
import os
import sys
import getopt
import tqdm
import torch
import gpytorch

SEEDS = [12, 17, 44, 92, 101]

CONFIG = {
    "SKI-SE" : {
        "temperature" : [28, 56, 113, 169],
        "precipitation" : [38, 76, 152, 227],
        "houseprice" : [81, 163, 326, 499],
    },
    "SKI-M52" : {
        "temperature" : [28, 56, 113, 169],
        "precipitation" : [38, 76, 152, 227],
        "houseprice" : [81, 163, 326, 499],
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

        if dataset in CONFIG["SKI-SE"].keys():
            for M in tqdm.tqdm(CONFIG["SKI-SE"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "base_kernel" : gpytorch.kernels.RBFKernel,
                        "base_kernel_args" : {},
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "SKI", args, logdirname, suffix="SE_{}".format(M),
                                                   **params)
        if dataset in CONFIG["SKI-M52"].keys():
            for M in tqdm.tqdm(CONFIG["SKI-SE"][dataset]):
                for seed in SEEDS:
                    params = {
                        "M" : M,
                        "base_kernel" : gpytorch.kernels.MaternKernel,
                        "base_kernel_args" : {"nu" : 2.5}
                    }
                    vff.expmt_utils.run_experiment(seed, dataset, "SKI", args, logdirname, suffix="M52_{}".format(M),
                                                   **params)