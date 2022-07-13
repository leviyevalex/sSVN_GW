#%%
import numpy as np
import scipy as sc
import scipy.stats
from chainconsumer import ChainConsumer
import corner
import matplotlib.pyplot as plt
from scripts.plot_helper_functions import set_axes_equal
from opt_einsum import contract
import os
import logging
log = logging.getLogger(__name__)
root = os.path.dirname(os.path.abspath(__file__))

class vM_circle:
    def __init__(self, kappa, mu, id=''):
        self.id = id
        self.kappa = kappa
        self.mu = mu
        self.DoF = 1

        # Record evaluations
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0

    def getMinusLogLikelihood(self, theta):
        """
        Returns minus log of von Mises distribution
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns: (float) Density evaluation

        """
        return -1 * self.kappa * np.cos(theta - self.mu)

    def getGradientMinusLogLikelihood(self, theta):
        """
        Evaluates gradient of minus log of von Mises distribution
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns: (array) DoF shaped array

        """
        return self.kappa * np.sin(theta - self.mu)

    def getGNHessianMinusLogLikelihood(self, theta):
        """
        Calculate Gauss-Newton approximation of von Mises distribution
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns: (array) DoF x DoF shaped array of Gauss-Newton approximation at theta.

        """
        return self.kappa * np.cos(theta - self.mu)

    def _newDrawFromPrior(self, N):
        # Uniformly draw from [-pi, pi]
        return np.random.uniform(0, 2 * np.pi, (N, self.DoF))

    def sample_UnifHypersphere(self, npoints, ndim):
        vec = np.random.randn(ndim, npoints)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def sample_vMF(self, mu, kappa, num_samples):
        """Generate num_samples N-dimensional samples from von Mises Fisher
        distribution around center mu \in R^N with concentration kappa.
        """
        dim = len(mu)
        result = np.zeros((num_samples, dim))
        for nn in range(num_samples):
            # sample offset from center (on sphere) with spread kappa
            w = self._sample_weight(kappa, dim)

            # sample a point v on the unit sphere that's orthogonal to mu
            v = self._sample_orthonormal_to(mu)

            # compute new point
            result[nn, :] = v * np.sqrt(1. - w**2) + w * mu

        return result

    def _sample_weight(self, kappa, dim):
        """Rejection sampling scheme for sampling distance from center on
        surface of the sphere.
        """
        dim = dim - 1  # since S^{n-1}
        b = dim / (np.sqrt(4. * kappa**2 + dim**2) + 2 * kappa)
        x = (1. - b) / (1. + b)
        c = kappa * x + dim * np.log(1 - x**2)

        while True:
            z = np.random.beta(dim / 2., dim / 2.)
            w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
            u = np.random.uniform(low=0, high=1)
            if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
                return w

    def _sample_orthonormal_to(self, mu):
        """Sample point on sphere orthogonal to mu."""
        v = np.random.randn(mu.shape[0])
        proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
        orthto = v - proj_mu_v
        return orthto / np.linalg.norm(orthto)

    # Prior stuff
    def _getMinusLogPrior(self, theta):
        return 0

    def _getGradientMinusLogPrior(self, theta):
        return np.zeros(self.DoF)

    def _getHessianMinusLogPrior(self, theta):
        return np.zeros((self.DoF, self.DoF))

    # Posterior and vectorization
    # Args:
    #   thetas (array): N x DoF array, where N is number of samples
    # Returns: (array) N x 1, N x DoF, N x (DoF x DoF), N x (DoF x DoF) respectively

    def getMinusLogPosterior(self, theta):
        return self.getMinusLogLikelihood(theta) + self._getMinusLogPrior(theta)

    def getGradientMinusLogPosterior(self, theta):
        return self.getGradientMinusLogLikelihood(theta) + self._getGradientMinusLogPrior(theta)

    def getGNHessianMinusLogPosterior(self, theta):
        return self.getGNHessianMinusLogLikelihood(theta) + self._getHessianMinusLogPrior(theta)

    def getHessianMinusLogPosterior(self, theta):
        return self.getHessianMinusLogLikelihood(theta) + self._getHessianMinusLogPrior(theta)

    def getMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getMinusLogPosterior, 1, thetas)

    def getGradientMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getGradientMinusLogPosterior, 1, thetas)

    def getGNHessianMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getGNHessianMinusLogPosterior, 1, thetas)

    def getHessianMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getHessianMinusLogPosterior, 1, thetas)

#%%
# def sample_spherical(npoints, ndim=3):
#     vec = np.random.randn(ndim, npoints)
#     vec /= np.linalg.norm(vec, axis=0)
#     return vec
#
# phi = np.linspace(0, np.pi, 20)
# theta = np.linspace(0, 2 * np.pi, 40)
# x = np.outer(np.sin(theta), np.cos(phi))
# y = np.outer(np.sin(theta), np.sin(phi))
# z = np.outer(np.cos(theta), np.ones_like(phi))
#
# xi, yi, zi = sample_spherical(1000)
#
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
# ax.plot_wireframe(x, y, z, color='k', alpha=0.1, rstride=1, cstride=1)
# ax.scatter(xi, yi, zi, s=10, c='k', zorder=10)
# ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
# set_axes_equal(ax) # IMPORTANT - this is also required
# fig.show()
#
# #%%
# n = 100000
# xi, yi, zi = sample_spherical(n)
# positions = np.zeros((n, 3))
# positions[:,0] = xi
# positions[:,1] = yi
# positions[:,2] = zi
# c = ChainConsumer()
# c.add_chain(positions, parameters=["$x_1$", "$x_2$", "$x_3$"]).configure(statistics="max")
# fig2 = c.plotter.plot(filename="fig2.png", figsize="column")
#
# #%%
# xi, yi= sample_spherical(1000, ndim=2)
# fig, ax = plt.subplots(1, 1)
# ax.scatter(xi, yi)
# ax.set_box_aspect(1)
# fig.show()
#
# #%% von Mises-Fisher
# def sample_vMF(mu, kappa, num_samples):
#     """Generate num_samples N-dimensional samples from von Mises Fisher
#     distribution around center mu \in R^N with concentration kappa.
#     """
#     dim = len(mu)
#     result = np.zeros((num_samples, dim))
#     for nn in range(num_samples):
#         # sample offset from center (on sphere) with spread kappa
#         w = _sample_weight(kappa, dim)
#
#         # sample a point v on the unit sphere that's orthogonal to mu
#         v = _sample_orthonormal_to(mu)
#
#         # compute new point
#         result[nn, :] = v * np.sqrt(1. - w**2) + w * mu
#
#     return result
#
#
# def _sample_weight(kappa, dim):
#     """Rejection sampling scheme for sampling distance from center on
#     surface of the sphere.
#     """
#     dim = dim - 1  # since S^{n-1}
#     b = dim / (np.sqrt(4. * kappa**2 + dim**2) + 2 * kappa)
#     x = (1. - b) / (1. + b)
#     c = kappa * x + dim * np.log(1 - x**2)
#
#     while True:
#         z = np.random.beta(dim / 2., dim / 2.)
#         w = (1. - (1. + b) * z) / (1. - (1. - b) * z)
#         u = np.random.uniform(low=0, high=1)
#         if kappa * w + dim * np.log(1. - x * w) - c >= np.log(u):
#             return w
#
#
# def _sample_orthonormal_to(mu):
#     """Sample point on sphere orthogonal to mu."""
#     v = np.random.randn(mu.shape[0])
#     proj_mu_v = mu * np.dot(mu, v) / np.linalg.norm(mu)
#     orthto = v - proj_mu_v
#     return orthto / np.linalg.norm(orthto)
#
# #%%
# mu = np.array([0, 0, 1])
# samples = sample_vMF(mu, kappa=10, num_samples=1000)
#
# phi = np.linspace(0, np.pi, 20)
# theta = np.linspace(0, 2 * np.pi, 40)
# x = np.outer(np.sin(theta), np.cos(phi))
# y = np.outer(np.sin(theta), np.sin(phi))
# z = np.outer(np.cos(theta), np.ones_like(phi))
#
# fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
# ax.plot_wireframe(x, y, z, color='k', alpha=0.1, rstride=1, cstride=1)
# ax.scatter(samples[:,0], samples[:,1], samples[:,2], s=10, c='k', zorder=10)
# ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
# set_axes_equal(ax) # IMPORTANT - this is also required
# fig.show()
#
