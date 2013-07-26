import numpy as np
from scipy.optimize import minimize_scalar

from ..externals.six.moves import xrange
from .. import cluster
from .gmm import GMM
from .dpgmm import log_normalize, digamma
from .dpmfmm import log_c3k, sample_sphere_3d

def coth(x):
    return np.cosh(x) / np.sinh(x)

def mean_rel_abs_diff(a, b):
    return (abs(a - b) / abs(a)).mean()

class VBMFMM(GMM):
    def __init__(self, n_components=10,
            alpha=1.0, k_0=0.1, mu_0=None, a=5, b=4.7,
            n_iter=100, thresh=1e-6, random_state=None, verbose=False,
            min_covar=None, params='wmc', init_params='wmc'):

        self.alpha = float(alpha) / n_components
        self.k_0 = k_0
        self.mu_0 = mu_0
        self.a = a
        self.b = b

        self.verbose = verbose

        super(VBMFMM, self).__init__(
            n_components, 'diag', random_state=random_state,
            thresh=thresh, min_covar=min_covar,
            n_iter=n_iter, params=params, init_params=init_params)

    def _get_covars(self):
        return 1. / self._get_precisions()

    def _get_precisions(self):
        return self.precs_

    def _set_covars(self, covars):
        raise NotImplementedError('''The variational algorithm does
        not support setting the covariance parameters.''')

    def normalize_means_(self):
        self.means_ /= np.sqrt((self.means_**2.).sum(axis=1))[:,np.newaxis]

    def eval(self, X):
        dg = (digamma(self.alpha + self.N_bar_) -
              digamma(1 + self.N_bar_.sum()))[np.newaxis]
        log_lik = (log_c3k(self.precs_)[np.newaxis] +
                   np.dot(X, (self.precs_[:,np.newaxis] * self.means_).T))

        gamma = (dg + log_lik)
        r = log_normalize(gamma, axis=-1)
        logprob = np.sum(r * log_lik, axis=-1)
        return logprob, r

    def do_e_step_(self, X, r):
        N_bar = r.sum(axis=0)
        dg = (digamma(self.alpha + N_bar) -
              digamma(1 + N_bar.sum()))[np.newaxis]
        log_lik = (log_c3k(self.precs_)[np.newaxis] +
                   np.dot(X, (self.precs_[:,np.newaxis] * self.means_).T))

        gamma = (dg + log_lik)
        r = log_normalize(gamma, axis=-1)
        logprob = np.sum(r * log_lik, axis=-1)
        return logprob, r

    def do_m_step_(self, X, r, k_0_mu_0, updates):
        if 'm' in updates:
            N_bar = r.sum(axis=0)
            x_nn = (X[:,np.newaxis] * r[:,:,np.newaxis]).sum(axis=0)
            x_bar = x_nn / N_bar[:,np.newaxis]

            self.means_ = (N_bar * self.precs_)[:,np.newaxis] * x_bar + k_0_mu_0
            self.normalize_means_()

        if 'p' in updates:
            c = -(self.b + (self.means_ * x_nn).sum(axis=1)) / (self.a + N_bar)

            for k in xrange(self.n_components):
                self.precs_[k] = minimize_scalar(
                        lambda x : 0.5*( (1./x) - coth(x) - c[k] )**2.,
                        method='bounded', bounds=(1e-6, 200.),
                        options={'maxiter':20}).x

    def fit(self, X, updates='mp'):
        if X.ndim == 1:
            raise ValueError('Need more than 1 data point to fit.')

        if X.shape[1] != 3:
            raise ValueError('Can only fit 3-dimensional data.')

        N = X.shape[0]

        if self.mu_0 is None:
            mu_0 = X.mean(axis=0)
        else:
            mu_0 = self.mu_0
        mu_0 /= np.sqrt((mu_0**2.).sum())

        k_0_mu_0 = self.k_0 * mu_0

        # Initialize parameters and state.
        self.means_ = sample_sphere_3d(self.n_components)
        self.precs_ = 1e-2*np.ones((self.n_components,))
        r = np.ones((N, self.n_components)) / float(self.n_components)

        self.converged_ = False
        for it in xrange(self.n_iter):
            prev_means = self.means_.copy()
            prev_precs = self.precs_.copy()

            logprob, r = self.do_e_step_(X, r)
            self.do_m_step_(X, r, k_0_mu_0, updates)

            mrad_mu = mean_rel_abs_diff(self.means_, prev_means)
            mrad_k = mean_rel_abs_diff(self.precs_, prev_precs)
            if self.verbose:
                print("Logprob at iteration %d: %1.5e" % (it, logprob.mean()))

            if mrad_mu < self.thresh and mrad_k < self.thresh:
                if self.verbose:
                    print('Converged at iteration', it)
                self.converged_ = True
                break

        self.N_bar_ = r.sum(axis=0)
        self.weights_ = ((self.alpha + self.N_bar_) /
                         (self.n_components * self.alpha + N))
        self.weights_ /= self.weights_.sum()

        return self

    def fit_predict(self, X, updates='mp'):
        return self.fit(X, updates).predict()
