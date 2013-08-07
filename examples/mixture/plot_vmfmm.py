"""
=======================================
Von-Mises Fisher Mixture Model Examples
=======================================

Fit a von-Mises Fisher mixture model to some test data using variational
inference.

Plots show the true and predicted labels, as well as the model log-likelihood
and cluster responsibilities.

The model only uses as many components as necessary to find a good fit for the
provided data.
"""
from colorsys import hsv_to_rgb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.mixture import VBMFMM, sample_vmf_3d

np.random.seed(0)
N = 128

# Generate a simple dataset with 3 vMF clusters.
cluster_pts = np.vstack([sample_vmf_3d([1,0,0], 30, N),
                 sample_vmf_3d([0,-1,0], 30, N),
                 sample_vmf_3d([0,0,1], 30, N)])
x, y, z = np.split(cluster_pts, 3, axis=1)
true_labels = np.hstack([ [i]*N for i in range(3) ])

# Fit a VMF Mixture Model to the data and predict labels.
clf = VBMFMM()
labels = clf.fit_predict(cluster_pts)

colors = np.array([hsv_to_rgb(c, 1, 1) for c in np.linspace(0,1,clf.n_components)])
fig = plt.figure(figsize=(16,16))

#
# Plot the test points colored by their true labels.
#
c = colors[true_labels]
ax = fig.add_subplot(221, projection='3d')
ax.scatter(x, y, z, c=c)
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
ax.set_title("True Labels")

#
# Plot the test points colored by their predicted labels.
#
c = colors[labels]
ax = fig.add_subplot(222, projection='3d')
ax.scatter(x, y, z, c=c)
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
ax.set_title("Predicted Labels")

#
# Plot a sphere surface colored by the data log likelihood under the model
# (scaled to lie within [0, 1]).
#
u = np.linspace(0, 2 * np.pi, N)
v = np.linspace(0, np.pi, N)
X = np.outer(np.cos(u), np.sin(v))
Y = np.outer(np.sin(u), np.sin(v))
Z = np.outer(np.ones(np.size(u)), np.cos(v))

sphere_pts = np.hstack([p.reshape(-1,1) for p in (X,Y,Z)])

lp, r = clf.score_samples(sphere_pts)
lp = (lp - lp.min()) / (lp.max() - lp.min())
logprob_c = np.array([hsv_to_rgb(2*(1-l)/3,1,1) for l in lp]).reshape(N,N,3)

ax = fig.add_subplot(223, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, facecolors=logprob_c, shade=False)
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
ax.set_title("Model Probability Map")

#
# Plot a sphere surface colored by the responsibility-weighted sum of label
# colors.
#
responsibility_c = (r[:,:,np.newaxis] * colors[np.newaxis]).sum(axis=1).reshape(N,N,3)

ax = fig.add_subplot(224, projection='3d')
ax.plot_surface(X, Y, Z, rstride=4, cstride=4, facecolors=responsibility_c, shade=False)
ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(-1,1)
ax.set_title("Cluster Responsibilities")
plt.show()
