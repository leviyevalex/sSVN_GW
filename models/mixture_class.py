""" 
Class which provides mixture functionality
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
# from models.JAXHRD import HRD

class Mixture:
    def __init__(self, components, mixture_weights, lower_bound, upper_bound, DoF):
        self.nComponents = len(components)
        assert self.nComponents == len(mixture_weights)
        self.components = components
        self.mixture_weights = mixture_weights 
        self.DoF = DoF
        self.id = 'mixture'

        # Record evaluations
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    @partial(jax.jit, static_argnums=(0,))
    def getMinusLogPosterior_ensemble(self, X):
        nParticles = X.shape[0]
        p = jnp.zeros(nParticles)
        for k in range(self.nComponents):
            V = self.components[k].getMinusLogPosterior_ensemble(X)
            p += jnp.exp(-V) * self.mixture_weights[k]
        return -jnp.log(p)
        
    @partial(jax.jit, static_argnums=(0,))
    def getDerivativesMinusLogPosterior_ensemble(self, X):
        nParticles = X.shape[0]
        gmlpt = jnp.zeros((nParticles, self.DoF))
        Hmlpt = jnp.zeros((nParticles, self.DoF, self.DoF))

        # Calculate posterior values for each component
        p = jnp.zeros((self.nComponents, nParticles)) 
        for i in range(self.nComponents):
            V = self.components[i].getMinusLogPosterior_ensemble(X)
            p = p.at[i].set(jnp.exp(-V))

        # Calculate derivative weights
        weights_ = self.mixture_weights[..., np.newaxis] * p
        weights = weights_ / np.sum(weights_, axis=0)

        for i in range(self.nComponents):
            gmlpt_, Hmlpt_ = self.components[i].getDerivativesMinusLogPosterior_ensemble(X)
            gmlpt += weights[i][..., np.newaxis] * gmlpt_  
            Hmlpt += weights[i][..., np.newaxis, np.newaxis] * Hmlpt_  

        return gmlpt, Hmlpt
    
    def newDrawFromPosterior(self, N):
        bins = np.zeros(self.nComponents + 1)
        bins[1:] = np.cumsum(self.mixture_weights)
        r = np.random.uniform(size=N)
        counts, _ = np.histogram(r, bins=bins)
        samples = None
        for i in range(self.nComponents):
            n = counts[i]
            if n > 0:
                if samples is None:
                    samples = self.components[i].newDrawFromPosterior(n)
                else:
                    samples = np.vstack((samples, self.components[i].newDrawFromPosterior(n)))

        np.random.shuffle(samples)

        return samples 

    def _newDrawFromPrior(self, nSamples):
        """
        Return samples from a uniform prior. Included for convenience.
        Args:
            nParticles (int): Number of samples to draw.

        Returns: 
            array: (nSamples, DoF) shaped array of samples from uniform prior

        """
        # return np.random.uniform(low=-6, high=6, size=(nSamples, self.DoF))
        draw = np.zeros((nSamples, self.DoF))
        for d in range(self.DoF):
            draw[:,d] = np.random.uniform(low=self.lower_bound[d], high=self.upper_bound[d], size=nSamples)

        return draw
# #%%
# import numpy as np
# # N = 5
# # DoF = 2

# weight_mixtures = np.array([0.1, 0.2, 0.5, 0.2])
# k = len(weight_mixtures)
# partition = np.zeros(k + 1)
# partition[1:] = np.cumsum(weight_mixtures)
# nSamples = 10
# r = np.random.uniform(size=nSamples)
# counts, _ = np.histogram(r, bins=partition)
# for i in k:
#     n = counts[i]
#     if n > 0:
#         if i == 0:
#             samples = components[i].newDrawFromPosterior(n)
#         else:
#             samples = np.vstack(samples, components[i].newDrawFromPosterior(n))

# #%%
# r = np.sort(np.random.uniform(low=0, high=1, size=N))

# intervals = np.zeros(len(weight_mixtures) + 1)
# intervals[1:] = weight_mixtures

# elements_per_bin = np.bincount(getBinIds(fgrid_dense, intervals)) # (ii)

# for i in range(nComponents):            
#     s_ = self.component[i].newDrawFromPosterior(50000)
#     samples[i * elements_per_bin[i]:(i+1) * elements_per_bin[i]] = s_

# return np.shuffle(samples)
# # %%
