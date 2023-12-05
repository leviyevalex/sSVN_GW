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

class MoG:
    def __init__(self, weights, mus, covs):
        """
        Gaussian  class

        """
        self.mus = mus
        self.covs = covs
        self.weights = jax.nn.softmax(weights) # Makes sure that sum of entries = 1
        self.nComponents = len(weights)
        self.DoF = mus.shape[1]

        self.priorDict = None

        np.random.seed(1)

        self.lower_bound = jnp.array([-7, -7])
        self.upper_bound = jnp.array([7, 7])

    def posterior(self, x):
        posterior = 0
        for i in range(self.nComponents):
            posterior += self.weights[i] * jax.scipy.stats.multivariate_normal.pdf(x, self.mus[i], self.covs[i])
        return posterior

    def potential(self, x):
        return -jnp.log(self.posterior(x))

    def _newDrawFromPrior(self, nSamples):
        # Returns a grid of particles in buffered hypercube
        samples = np.zeros((nSamples, self.DoF))
        for i in range(self.DoF):
            buffer = (self.upper_bound[i] - self.lower_bound[i]) / 8
            # samples[:, i] = np.linspace(self.lower_bound[i] + buffer, self.upper_bound[i] - buffer, nSamples)
            samples[:, i] = np.random.uniform(self.lower_bound[i], self.upper_bound[i], nSamples)
        return samples

    def newDrawFromPosterior(self, N):
        bins = np.zeros(self.nComponents + 1)
        bins[1:] = np.cumsum(self.weights)
        r = np.random.uniform(size=N)
        counts, _ = np.histogram(r, bins=bins)
        samples = None
        for i in range(self.nComponents):
            n = counts[i]
            if n > 0:
                if samples is None:
                    samples = np.random.multivariate_normal(mean=self.mus[i], cov=self.covs[i], size=n)
                else:
                    samples = np.vstack((samples, np.random.multivariate_normal(mean=self.mus[i], cov=self.covs[i], size=n)))

        np.random.shuffle(samples)

        return samples




#%%
k = 3
d = 2
weights = jnp.array([2, 4, 2])

mus = jnp.zeros((k, d))
mus = mus.at[0].set(jnp.array([-5, 0]))
mus = mus.at[1].set(jnp.array([0, 0]))
mus = mus.at[2].set(jnp.array([5, 0]))

covs = jnp.zeros((k, d, d))
covs = covs.at[0].set(jnp.eye(d))
covs = covs.at[1].set(jnp.eye(d))
covs = covs.at[2].set(jnp.eye(d))

model = MoG(weights, mus, covs)
# # %%

# model.potential(jnp.array([1,1.]))

# # %%
# samples = model.newDrawFromPosterior(1000000)
# # %%
import corner 
corner.corner(model._newDrawFromPrior(10000), density=True)
# # %%
# model.weights
# # %%

# %%
