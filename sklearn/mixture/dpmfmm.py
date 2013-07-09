"""Dirichlet Process von-Mises Fisher Mixture Model"""
#
# Author:
#         Gabriel Schwartz <gbs25@drexel.edu>
#

import numpy as np
from scipy.special import iv

from ..externals.six.moves import xrange

def axisangle2rotmat(axis, angle=None):
    if angle is None:
        angle = np.sqrt((axis**2).sum())
        axis /= angle

    skew = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]])
    skew2 = np.outer(axis, axis) - np.eye(3)

    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*skew2

def sample_sphere_3d(N=1):
    x = np.random.normal(size=(N,3))
    return np.squeeze(x / np.sqrt((x**2).sum(axis=1))[:,np.newaxis])

def sample_vmf_3d(mu, kappa, size=1):
    if kappa < 1e-8:
        return sample_sphere_3d(size)

    mu = np.asarray(mu, np.float64)
    mu /= np.sqrt((mu**2).sum())

    if np.allclose(mu, [0,0,1]):
        R = np.eye(3)
    else:
        angle = np.arccos(np.dot(mu, [0,0,1]))
        if abs(angle - np.pi) < 1e-8:
            R = np.eye(3)
            R[2,2] = -1
        else:
            axis = np.cross([0,0,1], mu)
            axis /= np.sqrt((axis**2).sum())
            R = axisangle2rotmat(axis, angle)

    xi = np.squeeze(np.random.rand(size))[...,np.newaxis]
    phi = np.squeeze(2*np.pi*np.random.rand(size))[...,np.newaxis]

    w = (1. + (1. / kappa) * (np.log(xi) +
        np.log(1 - ((xi - 1) / xi) * np.exp(-2*kappa))))

    n = np.sqrt(1 - w**2)
    pts = np.hstack([n*np.cos(phi), n*np.sin(phi), w])
    return np.dot(pts, R.T)

def c3k(k):
    return k / (2*np.pi*(np.exp(k) - np.exp(-k)))

def pdf_vmf_3d(X, mu, kappa):
    if kappa < 1e-8:
        return 1/(4*np.pi)

    try:
        normalization = c3k(kappa)
        exp = np.exp(kappa * np.dot(X, mu))
        if normalization < 1e-32 or exp < 1e-32:
            return 0
        else:
            return normalization * exp
    except:
        print("Bad pdf:", X.__repr__(), mu.__repr__(), kappa)
        print("c3k:", c3k(kappa))
        raise

def Ap(kappa):
    try:
        return iv(3./2., kappa) / iv(3./2.-1, kappa)
    except:
        print("bad kappa:", kappa)
        raise

def fit_vmf_3d(X):
    r = X.sum(axis=0)
    r_mag = np.sqrt((r**2).sum())

    mu = r / r_mag

    r_bar = r_mag / X.shape[0]
    try:
        if abs(r_bar - 1) < 1e-8:
            kappa = 1.
        else:
            kappa = (r_bar * (3 - r_bar**2)) / (1 - r_bar**2)
            # For stability. Kappa greater than ~200 causes underflow in c3k later.
            kappa = min(kappa, 200)
            kappa -= (Ap(kappa) - r_bar) / (1 - Ap(kappa)**2 - (2./kappa) * Ap(kappa))
            kappa = min(kappa, 200)
    except:
        print("Fit failed with X:\n", X.__repr__())
        exit()
    return mu, kappa

class vMFComponent:
    def __init__(self, mu, kappa, X=None):
        self.mu = np.asarray(mu, dtype=np.float64)
        self.kappa = kappa
        self.X = np.asarray(X, dtype=np.float64)

    def sample(self, size=1):
        return sample_vmf_3d(self.mu, self.kappa, size)

    def pdf(self, X):
        return pdf_vmf_3d(X, self.mu, self.kappa)

    def fit(self, X):
        self.mu, self.kappa = fit_vmf_3d(X)

    def remove_point(self, x):
        assert(self.X is not None)
        ps = self.X.shape[0]

        ind = abs(self.X - x).argmin(axis=0)[0]
        self.X = np.delete(self.X, ind, axis=0)

        assert(self.X.shape[0] == ps-1)
        if self.X.shape[0] == 0:
            del self.X
            self.X = None
        else:
            self.fit(self.X)

    def add_point(self, x):
        if self.X is None:
            self.X = np.array([x])
        else:
            self.X = np.vstack([self.X, [x]])

        self.fit(self.X)

class DPMFMM:
    def __init__(self, n_components=10, alpha=1.0, n_iter = 10):
        self.initial_components = n_components
        self.alpha = alpha
        self.n_iter = n_iter
        self.components = {}

    @property
    def means_(self):
        return np.array([c.mu for c in self.components.values()])

    @property
    def n_components(self):
        return len(self.components)

    def fit(self, X, eps = 1e-8):
        assert(X.shape[1] == 3)

        N = X.shape[0]
        self.z = np.random.randint(0, self.initial_components, size=N)
        x_mean = X.mean(axis=0)

        for zi in np.unique(self.z):
            X_zi = X[self.z == zi]
            self.components[zi] = vMFComponent(x_mean, 1., X_zi)

        for it in xrange(self.n_iter):
            prev_means = self.means_.copy()
            for i in np.random.permutation(N):
                self.components[self.z[i]].remove_point(X[i])
                # If the cluster is empty, remove it.
                if self.components[self.z[i]].X is None:
                    del self.components[self.z[i]]

                    # Re-assign indicator variables to span [0, K-1] where K is the
                    # number of occupied clusters.
                    old_ids = self.components.keys(); new_ids = xrange(self.n_components)
                    old_new = dict(zip(old_ids, new_ids))
                    self.components = dict(zip(new_ids, [self.components[k] for k in old_ids]))
                    old_new[self.z[i]] = -1
                    self.z = np.array([old_new[z] for z in self.z])

                p_zi = np.zeros((self.n_components+1,), np.float64)
                # Fill p_zi with probabilities of joining an existing cluster.
                for k, comp in self.components.items():
                    p_zi[k] = (comp.X.shape[0] / (self.alpha + N - 1.)) * comp.pdf(X[i])

                # Compute probability of joining a new cluster. This is
                # actually approximating an integral over the cluster
                # parameters, but it is doing so via MC integration with one
                # step.
                # The prior on cluster centers is non-informative (uniform on
                # sphere, 1/4pi), and the prior on k is a Gamma distribution.
                p_new = self.alpha / (self.alpha + N - 1)
                p_zi[-1] = p_new * pdf_vmf_3d(X[i], sample_sphere_3d(), np.random.gamma(2,5))

                # For stability issues (so numpy doesn't raise underflowo errors).
                p_zi[p_zi < 1e-8] = 0

                # Normalize p_zi.
                try:
                    p_zi /= p_zi.sum()
                except:
                    print("Normalize failed.")
                    print(p_zi)
                    print(p_zi.sum())
                    exit()

                # Sample from p_zi to get the new cluster for this point.
                r = np.random.rand()
                total = p_zi[0]
                k_new = 0
                try:
                    assert(list(sorted(self.components.keys())) == list(xrange(self.n_components)))
                except:
                    print("Bad ordering of k")
                    print(list(sorted(self.components.keys())))
                    print(list(xrange(self.n_components)))
                    exit()

                while r > total and k_new < self.n_components:
                    k_new += 1
                    total += p_zi[k_new]

                self.z[i] = k_new
                if k_new in self.components:
                    self.components[k_new].add_point(X[i])
                else:
                    new_kappa = np.random.gamma(2,10)
                    new_mean = sample_vmf_3d(X[i], new_kappa)
                    self.components[k_new] = vMFComponent(new_mean, new_kappa, [X[i]])

            m = self.means_
            if m.shape[0] == prev_means.shape[0] and abs(prev_means - m).sum() < eps:
                pm = prev_means
                print(abs(prev_means-m).sum())
                print("Converged!")
                break
            print("NC:", self.n_components)

        self._X = X

    def predict(self, X):
        if (X != self._X).any():
            self.fit(X)

        labels = np.unique(self.z)
        label_map = dict(zip(labels, xrange(len(labels))))
        return np.array([label_map[zi] for zi in self.z])

if __name__ == "__main__":
    np.random.seed(0)
    np.seterr(all="raise")
    N = 32

    c1 = sample_vmf_3d([1,0,0], 30, size=N)
    c2 = sample_vmf_3d([0,0,1], 30, size=N)

    points = np.vstack([c1, c2])

    d = DPMFMM(n_components=1, n_iter=100, alpha=10)
    d.fit(points)
    print("Means:\n", d.means_)
