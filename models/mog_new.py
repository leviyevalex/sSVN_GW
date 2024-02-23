""" 
v0.2 - 1/23/24 - Updated potential to be more numerically stable

"""

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
    def __init__(self, weights, mus, covs, lower_bound, upper_bound):
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

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    # def posterior(self, x):
    #     posterior = 0.
    #     def body_fn(i, posterior):
    #         return posterior + self.weights[i] * jax.scipy.stats.multivariate_normal.pdf(x, self.mus[i], self.covs[i])
        
    #     posterior = jax.lax.fori_loop(0, self.nComponents, body_fn, posterior)
    #     return posterior

    # def posterior(self, x):
    #     # Numerically unstable
    #     posterior = 0
    #     for i in range(self.nComponents):
    #         posterior += self.weights[i] * jax.scipy.stats.multivariate_normal.pdf(x, self.mus[i], self.covs[i])
    #     return posterior

    # def potential(self, x):
    #     return -jnp.log(self.posterior(x))

    # def potential(self, x):
    #     # Still Unstable
    #     posterior = 0
    #     for i in range(self.nComponents):
    #         posterior += self.weights[i] * jnp.exp(-1 * jnp.dot(x - self.mus[i], x - self.mus[i]) / self.covs[i])
    #     return -jnp.log(posterior) 

    def potential(self, x):
        potentials = jnp.zeros(self.nComponents)
        for i in range(self.nComponents):
            # V_i = jnp.dot(x - self.mus[i], (x - self.mus[i]) / self.covs[i]) 
            V_i = -1 * jax.scipy.stats.multivariate_normal.logpdf(x, self.mus[i], jnp.diag(self.covs[i])) 
            potentials = potentials.at[i].set(V_i)
        return -jax.scipy.special.logsumexp(-potentials, b=self.weights)



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
                    samples = np.random.multivariate_normal(mean=self.mus[i], cov=np.diag(self.covs[i]), size=n)
                else:
                    samples = np.vstack((samples, np.random.multivariate_normal(mean=self.mus[i], cov=np.diag(self.covs[i]), size=n)))

        np.random.shuffle(samples)

        return samples




#%%
# k = 3
# d = 2
# weights = jnp.array([2, 4, 2])

# mus = jnp.zeros((k, d))
# mus = mus.at[0].set(jnp.array([-5, 0]))
# mus = mus.at[1].set(jnp.array([0, 0]))
# mus = mus.at[2].set(jnp.array([5, 0]))

# covs = jnp.zeros((k, d, d))
# covs = covs.at[0].set(jnp.eye(d))
# covs = covs.at[1].set(jnp.eye(d))
# covs = covs.at[2].set(jnp.eye(d))

# model = MoG(weights, mus, covs)
# # # %%

# # model.potential(jnp.array([1,1.]))

# # # %%
# # samples = model.newDrawFromPosterior(1000000)
# # # %%
# import corner 
# corner.corner(model._newDrawFromPrior(10000), density=True)
# # %%
# model.weights
# # %%

# %%
