import unittest

import numpy as np

from numpy.testing import assert_array_less, assert_allclose
from nose.tools import assert_equal, assert_greater

from sklearn.externals.six.moves import xrange
from sklearn.mixture import VBMFMM, sample_vmf_3d
from sklearn.metrics import v_measure_score

np.seterr(all='warn')
rng = np.random.RandomState(1)


def generate_symmetric_vmf_clusters(points_per_cluster, means, precs):
    """Generates a dataset composed of an equal number of samples from vMF
    distributions with the given parameters."""

    if isinstance(precs, (float, int)):
        precs = np.array([precs] * len(means), np.float32)

    X = np.vstack([ sample_vmf_3d(mean, prec, points_per_cluster, rng)
        for mean, prec in zip(means, precs) ])
    y = np.hstack([ [i] * points_per_cluster for i in xrange(len(means)) ])

    return X, y


def fit_get_active(points_per_cluster, means, precs):
    """Generates a synthetic dataset with the given means and precisions, fits
    a VMFMM model to the data, and returns the means and precisions that
    contribute to the prediction for that dataset."""
    X, y_true = generate_symmetric_vmf_clusters(points_per_cluster, means, precs)
    model = VBMFMM(random_state = rng)

    labels = model.fit_predict(X)
    active = np.unique(labels)
    active_means = model.means_[active]
    active_precs = model.precs_[active]

    return active_means, active_precs


def test_predict():
    """Test that varying the true vMF precisions and number of model components
    does not affect the model predictions."""

    points_per_cluster = 16
    cluster_means = np.vstack([np.eye(3), -np.eye(3)])
    cluster_precs = [20, 30, 50, 100]
    model_components = [8, 10, 12]

    for prec in cluster_precs:
        X, y_true = generate_symmetric_vmf_clusters(points_per_cluster, cluster_means, prec)

        for n_components in model_components:
            model = VBMFMM(n_components = n_components, random_state = rng)
            y_pred = model.fit_predict(X)

            # For spread-out data (precision < 30), the predictions will not be
            # 100% accurate for the synthetic datasets.
            assert_greater(v_measure_score(y_true, y_pred), 0.95)


def test_fit_means():
    """Test that the model's estimated cluster means are close to the true
    means."""

    points_per_cluster = 32
    true_means = np.vstack([np.eye(3), -np.eye(3)])
    true_precs = 20

    active_means, active_precs = fit_get_active(points_per_cluster, true_means, true_precs)

    dp = (active_means[np.newaxis] * true_means[:, np.newaxis]).sum(axis=-1)
    min_angles = np.arccos(dp).min(axis=1)
    assert_array_less(min_angles, 0.1)


def test_fit_precs():
    """Test that the model's estimated cluster precisions are roughly close to
    the true precisions."""

    points_per_cluster = 32
    true_means = np.vstack([np.eye(3), -np.eye(3)])
    true_precs = 20

    active_means, active_precs = fit_get_active(points_per_cluster, true_means, true_precs)

    # Estimated precisions are often only on the same order of magnitude as the
    # true precisions, hence the high rtol.
    assert_allclose(active_precs, true_precs, rtol=0.5)


if __name__ == '__main__':
    import nose
    nose.runmodule()
