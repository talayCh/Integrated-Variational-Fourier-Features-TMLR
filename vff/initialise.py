"""Conditional variance initialiser from Burt et al 2019/2020 
(https://github.com/markvdw/RobustGP/tree/master/robustgp/init_methods)
"""

import numpy as np
import scipy
import warnings
from typing import Callable, Optional

import gpflow as gp
import tensorflow as tf
import lab.tensorflow

class InducingPointInitializer:
    def __init__(self, seed: Optional[int] = 0, randomized: Optional[bool] = True, **kwargs):
        self._randomized = randomized
        self.seed = seed if self.randomized else None

    def __call__(self, training_inputs: np.ndarray, M: int,
                 kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        if self.seed is not None:
            restore_random_state = np.random.get_state()
            np.random.seed(self.seed)
        else:
            restore_random_state = None

        Z = self.compute_initialisation(training_inputs, M, kernel)

        if self.seed is not None:
            np.random.set_state(restore_random_state)

        return Z

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        raise NotImplementedError

    @property
    def randomized(self):
        return self._randomized

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def __repr__(self):
        params = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if k not in ['_randomized']])
        return f"{type(self).__name__}({params})"

class ConditionalVariance(InducingPointInitializer):
    def __init__(self, sample: Optional[bool] = False, threshold: Optional[int] = 0.0, seed: Optional[int] = 0,
                 **kwargs):
        """
        :param sample: bool, if True, sample points into subset to use with weights based on variance, if False choose
        point with highest variance at each iteration
        :param threshold: float or None, if not None, if tr(Kff-Qff)<threshold, stop choosing inducing points as the approx.
        has converged.
        """
        super().__init__(seed=seed, randomized=True, **kwargs)
        self.sample = sample
        self.threshold = threshold

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        """
        The version of this code without sampling follows the Greedy approximation to MAP for DPPs in
        @incollection{NIPS2018_7805,
                title = {Fast Greedy MAP Inference for Determinantal Point Process to Improve Recommendation Diversity},
                author = {Chen, Laming and Zhang, Guoxin and Zhou, Eric},
                booktitle = {Advances in Neural Information Processing Systems 31},
                year = {2018},
            }
        and the initial code is based on the implementation of this algorithm (https://github.com/laming-chen/fast-map-dpp)
        It is equivalent to running a partial pivoted Cholesky decomposition on Kff (see Figure 2 in the below ref.),
        @article{fine2001efficient,
                title={Efficient SVM training using low-rank kernel representations},
                author={Fine, Shai and Scheinberg, Katya},
                journal={Journal of Machine Learning Research},
                year={2001}
            }
        TODO: IF M ==1 this throws errors, currently throws an assertion error, but should fix
        Initializes based on variance of noiseless GP fit on inducing points currently in active set
        Complexity: O(NM) memory, O(NM^2) time
        :param training_inputs: [N,D] numpy array,
        :param M: int, number of points desired. If threshold is None actual number returned may be less than M
        :param kernel: kernelwrapper object
        :return: inducing inputs, indices,
        [M,D] np.array to use as inducing inputs,  [M], np.array of ints indices of these inputs in training data array
        """
        N = training_inputs.shape[0]
        perm = np.random.permutation(N)  # permute entries so tiebreaking is random
        training_inputs = training_inputs[perm]
        # note this will throw an out of bounds exception if we do not update each entry
        indices = np.zeros(M, dtype=int) + N
        di = kernel(training_inputs, None, full_cov=False) + 1e-12  # jitter
        if self.sample:
            indices[0] = sample_discrete(di)
        else:
            indices[0] = np.argmax(di)  # select first point, add to index 0
        if M == 1:
            indices = indices.astype(int)
            Z = training_inputs[indices]
            indices = perm[indices]
            return Z, indices
        ci = np.zeros((M - 1, N))  # [M,N]
        for m in range(M - 1):
            j = int(indices[m])  # int
            new_Z = training_inputs[j:j + 1]  # [1,D]
            dj = np.sqrt(di[j])  # float
            cj = ci[:m, j]  # [m, 1]
            Lraw = np.array(kernel(training_inputs, new_Z, full_cov=True))
            L = np.round(np.squeeze(Lraw), 20)  # [N]
            L[j] += 1e-12  # jitter
            ei = (L - np.dot(cj, ci[:m])) / dj
            ci[m, :] = ei
            try:
                di -= ei ** 2
            except FloatingPointError:
                pass
            di = np.clip(di, 0, None)
            if self.sample:
                indices[m + 1] = sample_discrete(di)
            else:
                indices[m + 1] = np.argmax(di)  # select first point, add to index 0
            # sum of di is tr(Kff-Qff), if this is small things are ok
            if np.sum(np.clip(di, 0, None)) < self.threshold:
                indices = indices[:m]
                warnings.warn("ConditionalVariance: Terminating selection of inducing points early.")
                break
        indices = indices.astype(int)
        Z = training_inputs[indices]
        indices = perm[indices]
        return Z, indices

    def __repr__(self):
        params = ', '.join([f'{k}={v}' for k, v in self.__dict__.items()
                            if
                            k not in ['_randomized'] and
                            not (k == "threshold" and self.threshold == 0.0)])
        return f"{type(self).__name__}({params})"



class KdppMCMC(ConditionalVariance):

    def __init__(self, num_steps: Optional[int] = 10000, seed: Optional[int] = 0, **kwargs):
        """
        Implements the MCMC approximation to sampling from a k-DPP developed in
        @inproceedings{anari2016monte,
                       title={Monte Carlo Markov chain algorithms for sampling strongly Rayleigh distributions and determinantal point processes},
                       author={Anari, Nima and Gharan, Shayan Oveis and Rezaei, Alireza},
                       booktitle={Conference on Learning Theory},
                       pages={103--115},
                       year={2016}
                    }
        and used for initializing inducing point in
        @inproceedings{burt2019rates,
                       title={Rates of Convergence for Sparse Variational Gaussian Process Regression},
                       author={Burt, David and Rasmussen, Carl Edward and Van Der Wilk, Mark},
                       booktitle={International Conference on Machine Learning},
                       pages={862--871},
                      year={2019}
            }
        More information on determinantal point processes and related algorithms can be found at:
        https://github.com/guilgautier/DPPy
        :param sample: int, number of steps of MCMC to run
        :param threshold: float or None, if not None, if tr(Kff-Qff)<threshold, stop choosing inducing points as the approx.
        has converged.
        """
        super().__init__(seed=seed, **kwargs)
        self.num_steps = num_steps

    def compute_initialisation(self, training_inputs: np.ndarray, M: int,
                               kernel: Callable[[np.ndarray, Optional[np.ndarray], Optional[bool]], np.ndarray]):
        """
        :param training_inputs: training_inputs: [N,D] numpy array
        :param M: int, number of inducing inputs to return
        :param kernel: kernelwrapper object
        :param num_steps: number of swap steps to perform.
        :param init_indices: array of M indices or None, set used to initialize mcmc alg. if None, we use the greedy MAP
        init. (variance, with sample=False)
        :return: inducing inputs, indices, [M], np.array of ints indices of these inputs in training data array
        """
        N = training_inputs.shape[0]
        _, indices = super().compute_initialisation(training_inputs, M, kernel)
        kzz = kernel(training_inputs[indices], None, full_cov=True)
        Q, R = scipy.linalg.qr(kzz, overwrite_a=True)
        if np.min(np.abs(np.diag(R))) == 0:
            warnings.warn("Determinant At initialization is numerically 0, MCMC was not run")
            return training_inputs[indices], indices
        for _ in range(self.num_steps):
            if np.random.rand() < .5:  # lazy MCMC, half the time, no swap is performed
                continue
            indices_complement = np.delete(np.arange(N), indices)
            s = np.random.randint(M)
            t = np.random.choice(indices_complement)
            swap, Q, R = accept_or_reject(training_inputs, kernel, indices, s, t, Q, R)
            if swap:
                indices = np.delete(indices, s, axis=0)
                indices = np.append(indices, [t], axis=0)
        return training_inputs[indices], indices


def _delete_qr_square(Q, R, s):
    """
    Given a QR decomposition of a square matrix K, remove row and column s. Note we do not overwrite the original
    matrices
    :param Q: orthogonal matrix from QR decomposition of K
    :param R: upper triangular matrix. K = QR
    :param s: index of row/column to remove
    :return: QR decomposition of matrix formed by deleting row/column s from K
    """
    # remove the corresponding row and column from QR-decomposition
    Qtemp, Rtemp = scipy.linalg.qr_delete(Q, R, s, which='row')
    Qtemp, Rtemp = scipy.linalg.qr_delete(Qtemp, Rtemp, s, which='col', overwrite_qr=True)
    return Qtemp, Rtemp


def _build_new_row_column(X, kernel, indices, s, t):
    ZminusS = np.delete(X[indices], s, axis=0)
    xt = X[t:t + 1]
    ktt = kernel(xt, None, full_cov=True)[0]
    row_to_add = kernel(ZminusS, xt)[:, 0]
    col_to_add = np.concatenate([row_to_add, ktt], axis=0)
    return row_to_add, col_to_add


def _add_qr_square(Q, R, row, col):
    # get the shape, Note that QR is square. We insert the column and row in the last position
    M = Q.shape[0]
    # Add the row and column to the matrix
    Qtemp, Rtemp = scipy.linalg.qr_insert(Q, R, row, M, which='row', overwrite_qru=True)
    Qtemp, Rtemp = scipy.linalg.qr_insert(Qtemp, Rtemp, col, M, which='col', overwrite_qru=True)
    return Qtemp, Rtemp


def _get_log_det_ratio(X, kernel, indices, s, t, Q, R):
    """
    Returns the log determinant ratio, as well as the updates to Q and R if the point is swapped
    :param X: training inputs
    :param kernel: kernelwrapper
    :param indices: X[indices]=Z
    :param s: current point we might (X[indices])[s] is the current point we might swap out
    :param t: X[t] is the point we might add
    :param Q: orthogonal matrix
    :param R: upper triangular matrix, QR = Kuu (for Z=X[indices])
    :return: log determinant ratio , Qnew, Rnew. QR decomposition of Z-{s}+{t}.
    """
    log_denominator = np.sum(np.log(np.abs(np.diag(R))))  # current value of det(Kuu)
    # remove s from the QR decomposition
    Qtemp, Rtemp = _delete_qr_square(Q, R, s)
    # build new row and column to add to QR decomposition
    row_to_add, col_to_add = _build_new_row_column(X, kernel, indices, s, t)
    # add the corresponding row and column to QR-decomposition
    Qnew, Rnew = _add_qr_square(Qtemp, Rtemp, row_to_add, col_to_add)
    # sometimes Rnew will have zero entries along the diagonal for numerical reasons, in these case, we should always
    # reject the swap as one Kuu should not be able to have (near) zero determinant, we force numpy to raise an error
    # and catch it
    try:
        log_numerator = np.sum(np.log(np.abs(np.diag(Rnew))))  # det(Kuu) if we perform swap
    except FloatingPointError:
        return - np.inf, Qnew, Rnew
    log_det_ratio = log_numerator - log_denominator  # ratio of determinants
    return log_det_ratio, Qnew, Rnew


def accept_or_reject(X, kernel, indices, s, t, Q, R):
    """
    Decides whether or not to swap points. Updates QR accordingly. Seems reasonably stable. Error of QR will get big if
    10k or more iterations are run (e.g. increases by about a factor of 10 over 10k
    iterations). Consider recomputing occasionally.
    :param X: candidate points
    :param k: kernel
    :param indices: Current active set (Z)
    :param s: index of point that could be removed in Z (Note this point is (X[indices])[s], not X[s]!)
    :param t: index of point that could be added in X
    :return: swap, Q, R: bool, [M,M], [M,M]. If swap, we removed s and added t. Return updated Q and R accodingly.
    """
    log_det_ratio, Qnew, Rnew = _get_log_det_ratio(X, kernel, indices, s, t, Q, R)
    acceptance_prob = np.exp(log_det_ratio)  # P(new)/P(old), probability of swap
    if np.random.rand() < acceptance_prob:
        return True, Qnew, Rnew  # swapped
    return False, Q, R  # stayed in same state


np.seterr(all='raise')

def sample_discrete(unnormalized_probs):
    unnormalized_probs = np.clip(unnormalized_probs, 0, None)
    N = unnormalized_probs.shape[0]
    normalization = np.sum(unnormalized_probs)
    if normalization == 0:  # if all of the probabilities are numerically 0, sample uniformly
        warnings.warn("Trying to sample discrete distribution with all 0 weights")
        return np.random.choice(a=N, size=1)[0]
    probs = unnormalized_probs / normalization
    return np.random.choice(a=N, size=1, p=probs)[0]


def minimise_reinit(optim, model, initialiser):
    """Helper function to minimise with inducing point initialisation"""
    gp.utilities.set_trainable(model.inducing_variable, False)
    
    def init_z(model, initialiser):
        z, _ = initialiser(lab.to_numpy(model.data[0]), model.inducing_variable.num_inducing, 
                           model.kernel)

        model.inducing_variable.Z.assign(z)
    
    def local_run(optim, model):
        opt_logs = optim.minimize(model.training_loss, model.trainable_variables,
                                  options = dict(maxiter=1000)
                                 )
    for i in range(20):
        reinit = True
        try:
            local_run(optim, model)
        except tf.errors.InvalidArgumentError as e:
            if e.message[1:9] != "Cholesky":
                raise e
            init_z(model, initialiser)
            print(model.elbo().numpy())  # Check whether Cholesky fails
            reinit = False

        if reinit:
            old_z = model.inducing_variable.Z.numpy().copy()
            old_elbo = model.elbo()
            init_z(model, initialiser)
            if model.elbo() <= old_elbo:
                # Restore old Z, and finish optimisation
                model.inducing_variable.Z.assign(old_z)
                print("{} Stopped reinit_Z procedure because new ELBO was smaller than old ELBO.".format(i))
                break