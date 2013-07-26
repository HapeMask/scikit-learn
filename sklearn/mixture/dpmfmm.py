"""Dirichlet Process von-Mises Fisher Mixture Model"""
#
# Author:
#         Gabriel Schwartz <gbs25@drexel.edu>
#

import numpy as np

from ..externals.six.moves import xrange
from .dpgmm import log_normalize

def axisangle2rotmat(axis, angle=None):
    if angle is None:
        angle = np.sqrt((axis**2.).sum())
        axis /= angle

    skew = np.array([
        [0., -axis[2], axis[1]],
        [axis[2], 0., -axis[0]],
        [-axis[1], axis[0], 0.]])
    skew2 = np.outer(axis, axis) - np.eye(3)

    return np.eye(3) + np.sin(angle)*skew + (1.-np.cos(angle))*skew2

def sample_sphere_3d(N=1):
    x = np.random.normal(size=(N,3))
    return np.squeeze(x / np.sqrt((x**2.).sum(axis=1))[:,np.newaxis])

def sample_vmf_3d(mu, kappa, size=1):
    if kappa < 1e-8:
        return sample_sphere_3d(size)

    mu = np.asarray(mu, np.float64)
    mu /= np.sqrt((mu**2.).sum())

    if np.allclose(mu, [0.,0.,1.]):
        R = np.eye(3)
    else:
        angle = np.arccos(np.dot(mu, [0.,0.,1.]))
        if abs(angle - np.pi) < 1e-8:
            R = np.eye(3)
            R[2,2] = -1.
        else:
            axis = np.cross([0.,0.,1.], mu)
            axis /= np.sqrt((axis**2.).sum())
            R = axisangle2rotmat(axis, angle)

    xi = np.squeeze(np.random.rand(size))[...,np.newaxis]
    phi = np.squeeze(2*np.pi*np.random.rand(size))[...,np.newaxis]

    w = (1. + (1. / kappa) * (np.log(xi) +
        np.log(1. - ((xi - 1.) / xi) * np.exp(-2.*kappa))))

    n = np.sqrt(1. - w**2.)
    pts = np.hstack([n*np.cos(phi), n*np.sin(phi), w])
    return np.dot(pts, R.T)

def c3k(k):
    return k / (4.*np.pi*np.sinh(k))

def log_c3k(k):
    return np.log(k) - np.log(2.*np.pi) - np.log(np.exp(k) - np.exp(-k))

def pdf_vmf_3d(X, mu, kappa):
    if kappa < 1e-8:
        return np.ones((X.shape[0],)) / (4.*np.pi)

    if X.ndim == 1:
        X = X[np.newaxis]

    normalization = c3k(kappa)
    exp = np.exp(kappa * np.dot(X, mu))

    if normalization < 1e-32:
        return np.zeros((X.shape[0],))
    else:
        exp[exp < 1e-16] = 0
        return normalization * exp

def log_pdf_vmf_3d(X, mu, kappa):
    if kappa < 1e-8:
        return np.ones((X.shape[0],)) / (4.*np.pi)

    if X.ndim == 1:
        X = X[np.newaxis]

    return log_c3k(kappa) + kappa * np.dot(X, mu)

def fit_vmf_3d(X):
    r = X.sum(axis=0)
    r_mag = np.sqrt((r**2.).sum())

    mu = r / r_mag

    r_bar = r_mag / X.shape[0]
    if abs(r_bar - 1.) < 1e-8:
        kappa = 1.
    else:
        kappa = (r_bar * (3. - r_bar**2.)) / (1. - r_bar**2.)
        # For stability. Kappa greater than ~200 causes underflow later.
        kappa = min(kappa, 200.)
    return mu, kappa

class vMFComponent:
    def __init__(self, mu, kappa, X=None):
        self.mu = np.asarray(mu)
        self.kappa = kappa
        if X is None:
            self.X = np.array((0, mu.shape[0]))
        else:
            self.X = np.asarray(X)

    def sample(self, size=1):
        return sample_vmf_3d(self.mu, self.kappa, size)

    def pdf(self, X):
        return pdf_vmf_3d(X, self.mu, self.kappa)

    def log_pdf(self, X):
        return log_pdf_vmf_3d(X, self.mu, self.kappa)

    def fit(self, X):
        self.mu, self.kappa = fit_vmf_3d(X)

    def remove_point(self, x):
        assert(self.size > 0)
        ps = self.X.shape[0]

        ind = abs(self.X - x).argmin(axis=0)[0]
        self.X = np.delete(self.X, ind, axis=0)

        if self.size > 0:
            self.fit(self.X)

    def add_point(self, x):
        self.X = np.vstack([self.X, [x]])
        self.fit(self.X)

    @property
    def size(self):
        return self.X.shape[0]

class DPMFMM:
    def __init__(self, n_components=10, alpha=1.0, n_iter = 10):
        self.initial_components = n_components
        self.alpha = alpha
        self.n_iter = n_iter
        self.components = {}

    @property
    def covars_(self):
        return np.array([c.kappa for c in self.components.values()])

    @property
    def means_(self):
        return np.array([c.mu for c in self.components.values()])

    @property
    def n_components(self):
        return len(self.components)

    def _remap_z(self):
        # Re-assign indicator variables to span [0, K-1] where K is the
        # number of occupied clusters, sorted by size.
        sorted_ids = list(sorted(self.components.keys(),
            key = lambda i : self.components[i].size)[::-1])
        sorted_comps = [self.components[i] for i in sorted_ids]

        self.components = dict(zip(xrange(len(sorted_comps)), sorted_comps))
        old_z = self.z.copy()
        for i in xrange(self.n_components):
            self.z[old_z == sorted_ids[i]] = i

    def fit(self, X, eps = 1e-8):
        if X.ndim == 1:
            raise ValueError("Need more than 1 data point to fit.")

        assert(X.shape[1] == 3)

        N = X.shape[0]
        log_alpha_plus_n_minus_1 = np.log(self.alpha + N - 1.)
        self.z = np.random.randint(0, self.initial_components, size=N)
        x_mean = X.mean(axis=0)

        for zi in np.unique(self.z):
            X_zi = X[self.z == zi]
            self.components[zi] = vMFComponent(x_mean, 1., X_zi)
        self._remap_z()

        self.converged_ = False
        for it in xrange(self.n_iter):
            prev_means = self.means_.copy()
            for i in np.random.permutation(N):
                self.components[self.z[i]].remove_point(X[i])
                # If the cluster is empty, remove it.
                if self.components[self.z[i]].size == 0:
                    del self.components[self.z[i]]
                    self._remap_z()

                p_zi = np.zeros((self.n_components+1,))
                # Fill p_zi with probabilities of joining an existing cluster.
                for k, comp in self.components.items():
                    p_zi[k] = np.log(comp.X.shape[0]) - log_alpha_plus_n_minus_1 + comp.log_pdf(X[i])

                # Compute probability of joining a new cluster. This is
                # actually approximating an integral over the cluster
                # parameters, but it is doing so via MC integration with one
                # step.
                # The prior on cluster centers is non-informative (uniform on
                # sphere, 1/4pi), and the prior on k is a Gamma distribution.
                p_new = np.log(self.alpha) - log_alpha_plus_n_minus_1
                p_zi[-1] = p_new + log_pdf_vmf_3d(X[i], sample_vmf_3d(x_mean, 1), np.random.gamma(2,30))

                # Normalize p_zi.
                p_zi = log_normalize(p_zi)

                # Sample from p_zi to get the new cluster for this point.
                r = np.random.rand()
                total = p_zi[0]
                k_new = 0

                while r > total and k_new < self.n_components:
                    k_new += 1
                    total += p_zi[k_new]

                self.z[i] = k_new
                if k_new in self.components:
                    self.components[k_new].add_point(X[i])
                else:
                    new_kappa = np.random.gamma(2,10)
                    #new_mean = sample_vmf_3d(X[i], new_kappa)
                    new_mean = X[i]
                    self.components[k_new] = vMFComponent(new_mean, new_kappa, [X[i]])

            print("NC:", self.n_components)
        self.X_ = X

    def predict(self, X):
        if self.n_components < 1:
            raise RuntimeError("predict() called before fit().")

        """
        p = np.hstack([comp.pdf(X)[:,np.newaxis] for comp in self.components.values()])
        return np.argmax(p, axis = 1)
        """

        if np.any(self.X_ != X):
            self.fit(X)

        return self.z
