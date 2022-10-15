"""
Multivariate gaussian toy problem
"""
#%%
import jax
import jax.numpy as jnp
from opt_einsum import contract
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
from functools import partial
from jax import random
#%%
# We assume the covariance is diagonal for simplicity


#%%
class multivariate_gaussian:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma 
        self.DoF = sigma.shape[0]
        self.Z = self.getNormalizationConstant(sigma)
    
    def getNormalizationConstant(self, sigma):
        return jnp.sqrt((2 * jnp.pi) ** self.DoF * jnp.prod(self.sigma))

    def getMinusLogLikelihood(self, theta):
        difference = theta - self.mu
        return jnp.log(self.Z) + jnp.dot(difference, difference / self.sigma) / 2
    
    def getGradientMinusLogLikelihood(self, theta):
        return (theta - self.mu) / self.sigma
    
    def getGNHessianMinusLogLikelihood(self, theta):
        return jnp.diag(1 / self.sigma)

    #####################################################################################
    # Prior methods
    #####################################################################################

    def _newDrawFromPrior(self, nParticles):
            """
            Return samples from a uniform prior.
            Included for convenience.
            Args:
                nParticles (int): Number of samples to draw.

            Returns: (array) nSamples x DoF array of representative samples

            """
            return np.random.uniform(low=-6, high=6, size=(nParticles, self.DoF))

    def _getMinusLogPrior(self, theta):
        return 0

    def _getGradientMinusLogPrior(self, theta):
        return np.zeros(self.DoF)

    def _getHessianMinusLogPrior(self, theta):
        return np.zeros((self.DoF, self.DoF))

    ##################################################################################
    # Form the posterior methods
    ##################################################################################
    def getMinusLogPosterior(self, theta):
        return self.getMinusLogLikelihood(theta) + self._getMinusLogPrior(theta)

    def getGradientMinusLogPosterior(self, theta):
        return self.getGradientMinusLogLikelihood(theta) + self._getGradientMinusLogPrior(theta)

    def getGNHessianMinusLogPosterior(self, theta):
        return self.getGNHessianMinusLogLikelihood(theta) + self._getHessianMinusLogPrior(theta)

    ######################################################################################
    # Get vectorized methods
    ######################################################################################
    def getGradientMinusLogPosterior_ensemble(self, thetas):
        return jax.vmap(self.getGradientMinusLogPosterior)(thetas)

    def getGNHessianMinusLogPosterior_ensemble(self, thetas):
        return jax.vmap(self.getGNHessianMinusLogPosterior)(thetas)

    @partial(jax.jit, static_argnums=(0,))
    def getDerivativesMinusLogPosterior_ensemble(self, thetas):
        gmlpt = self.getGradientMinusLogPosterior_ensemble(thetas)
        Hmlpt = self.getGNHessianMinusLogPosterior_ensemble(thetas)
        return (gmlpt, Hmlpt)
