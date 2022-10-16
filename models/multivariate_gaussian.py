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
        self.priorDict = None # hack
        self.id = 'multivariate_gaussian'
        # Record evaluations
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0
    
    @partial(jax.jit, static_argnums=(0,))
    def getNormalizationConstant(self, sigma):
        return jnp.sqrt((2 * jnp.pi) ** self.DoF * jnp.prod(self.sigma))

    @partial(jax.jit, static_argnums=(0,))
    def getMinusLogLikelihood(self, theta):
        difference = theta - self.mu
        return jnp.log(self.Z) + jnp.dot(difference, difference / self.sigma) / 2
    
    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogLikelihood(self, theta):
        return (theta - self.mu) / self.sigma
    
    @partial(jax.jit, static_argnums=(0,))
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

    def newDrawFromLikelihood(self, nParticles):
        return np.random.multivariate_normal(mean=self.mu, cov=np.diag(self.sigma), size=nParticles)
        raise NotImplementedError

    def _getMinusLogPrior(self, theta):
        return 0

    @partial(jax.jit, static_argnums=(0,))
    def _getGradientMinusLogPrior(self, theta):
        return jnp.zeros(self.DoF)

    @partial(jax.jit, static_argnums=(0,))
    def _getHessianMinusLogPrior(self, theta):
        return jnp.zeros((self.DoF, self.DoF))

    ##################################################################################
    # Form the posterior methods
    ##################################################################################
    @partial(jax.jit, static_argnums=(0,))
    def getMinusLogPosterior(self, theta):
        return self.getMinusLogLikelihood(theta) + self._getMinusLogPrior(theta)

    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogPosterior(self, theta):
        return self.getGradientMinusLogLikelihood(theta) + self._getGradientMinusLogPrior(theta)

    @partial(jax.jit, static_argnums=(0,))
    def getGNHessianMinusLogPosterior(self, theta):
        return self.getGNHessianMinusLogLikelihood(theta) + self._getHessianMinusLogPrior(theta)

    ######################################################################################
    # Get vectorized methods
    ######################################################################################
    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogPosterior_ensemble(self, thetas):
        return jax.vmap(self.getGradientMinusLogPosterior)(thetas)

    @partial(jax.jit, static_argnums=(0,))
    def getGNHessianMinusLogPosterior_ensemble(self, thetas):
        return jax.vmap(self.getGNHessianMinusLogPosterior)(thetas)

    @partial(jax.jit, static_argnums=(0,))
    def getDerivativesMinusLogPosterior_ensemble(self, thetas):
        gmlpt = self.getGradientMinusLogPosterior_ensemble(thetas)
        Hmlpt = self.getGNHessianMinusLogPosterior_ensemble(thetas)
        return (gmlpt, Hmlpt)

#%%
# from scipy.stats import truncnorm
# import numpy as np

# DoF = 10
# mu = np.zeros(DoF)
# sigma = np.random.uniform(low=1, high=10, size=DoF) # covariance matrix
# model = multivariate_gaussian(mu, sigma)

# samples = np.random.multivariate_normal(mean=mu, cov=np.diag(sigma), size=1000000)
# # %%
# np.logical_and
# np.where()

# #%%
# dist = tfd.TruncatedNormal(loc=[0., 1.], scale=1.,
#                            low=[-1., 0.],
#                            high=[1., 1.])

# # Evaluate the pdf of the distributions at 0.5 and 0.8 respectively returning
# # a 2-vector tensor.
# dist.prob([0.5, 0.8])

# # Get 3 samples, returning a 3 x 2 tensor.
# dist.sample([3])
#%%
import jax.numpy as jnp
import numpy as np

H = jnp.array(np.random.rand(5, 5, 2, 2))

res = H.swapaxes(1, 2).reshape(10, 10)
# %%