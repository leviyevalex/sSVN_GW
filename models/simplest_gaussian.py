#%%
import sys
sys.path.append("..")
from opt_einsum import contract
import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np

class simplest_gaussian:
    def __init__(self, mu, cov):
        """
        Gaussian  class

        """
        self.mu = mu
        self.cov = cov
        self.DoF = len(self.mu)

        self.priorDict = None
        self.id = 'gauss_simple'

        # Record evaluations
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0
    
        # self.lower_bound = np.ones(self.DoF) * (0.9)
        # self.upper_bound = np.ones(self.DoF) * (1.2)

        np.random.seed(1)
        self.lower_bound = np.random.uniform(0.2, 1, self.DoF)
        self.upper_bound = np.random.uniform(2, 4.5, self.DoF)


    def potential(self, x):
        return -jnp.log(jax.scipy.stats.multivariate_normal.pdf(x, self.mu, self.cov))

    def grad_potential(self, x):
        return jax.grad(self.potential)(x)

    @partial(jax.jit, static_argnums=(0,))
    def getMinusLogPosterior_ensemble(self, X):
        return jax.vmap(self.potential)(X)

    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogPosterior_ensemble(self, X):
        return jax.vmap(self.grad_potential)(X)

    def getDerivativesMinusLogPosterior_ensemble(self, X):
        return (self.getGradientMinusLogPosterior_ensemble(X), self.cov[np.newaxis, ...])

    def _newDrawFromPrior(self, nSamples):
        # Returns a grid of particles in buffered hypercube
        samples = np.zeros((nSamples, self.DoF))
        for i in range(self.DoF):
            buffer = (self.upper_bound[i] - self.lower_bound[i]) / 8
            samples[:, i] = np.linspace(self.lower_bound[i] + buffer, self.upper_bound[i] - buffer, nSamples)
        return samples