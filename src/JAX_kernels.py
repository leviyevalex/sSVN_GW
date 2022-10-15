"""
JAX Kernels.
"""

import jax
import jax.numpy as jnp
from opt_einsum import contract
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
from functools import partial
from jax import random

class kernels:
    def __init__(self, nParticles, DoF, **kwargs):
        self.nParticles = nParticles 
        self.DoF = DoF
        self.kernel_type = kwargs['kernel_type']
        if self.kernel_type == 'RFG':
            self.nFeatures = kwargs['nFeatures']
            self.initRandomFeatures()
    
    def initRandomFeatures(self):
        key = random.PRNGKey(0)

        self.w0 = random.uniform(minval=0., maxval=2*jnp.pi, shape=(self.nFeatures,), key=key)
        self.w1 = random.normal(shape=(self.nFeatures, self.DoF), key=key)
        # self.w1 = jnp.random.normal(0, 1, (self.nFeatures, self.DoF), key=key)

    def getPairwiseDisplacement(self, X):
        return X[:,jnp.newaxis,:] - X[jnp.newaxis,:,:]

    @partial(jax.jit, static_argnums=(0,))
    def kernel_RBF(self, X, h, M):
        X = contract('ij, Nj -> Ni', jax.scipy.linalg.cholesky(M), X)
        pairwise_displacements = self.getPairwiseDisplacement(X)
        k = jnp.exp(-contract('mni, mni -> mn', pairwise_displacements, pairwise_displacements) / (2 * h))
        g1k = -1 * contract('mn, mni -> mni', k, pairwise_displacements) / h
        return (k, g1k) # Derivative returned on first slot

    @partial(jax.jit, static_argnums=(0,))
    def kernel_RFG(self, X, h, M, w0, w1, nFeatures):
        X = contract('ij, Nj -> Ni', jax.scipy.linalg.cholesky(M), X)
        arg = contract('ld, Nd -> Nl', w1, X) / jnp.sqrt(h) + w0
        phi = jnp.sqrt(2) * jnp.cos(arg)
        k = contract('ml, nl -> mn', phi, phi) / nFeatures
        gphi = -jnp.sqrt(2 / h) * contract('ml, li -> mli', jnp.sin(arg), w1)
        g1k = contract('mli, nl -> mni', gphi, phi) / nFeatures
        return (k, g1k)

    def getKernelWithDerivatives(self, X, h, M):
        if self.kernel_type == 'RFG':
            return self.kernel_RFG(X, h, M, self.w0, self.w1, self.nFeatures)
        elif self.kernel_type == 'RBF':
            return self.kernel_RBF(X, h, M)
        else:
            raise NotImplementedError()

# #%%
# from jax import random
# import jax.numpy as jnp
# key = random.PRNGKey(0)
# #%%
# random.uniform(minval=0., maxval=2*jnp.pi, shape=(2,), key=key)

# #%%
# random.normal(shape=(5, 3), key=key)



# #%%
# def test(a, **kwargs):
#     return kwargs['lala']

# kwargs = {'lala' : 5}
# test(1, **kwargs)

# #%%
# import numpy as np
# a = np.array([[3, 2, 1],
#               [4, 5, 6]])

# b = np.array([3, 2, 2])

# a / b
# # %%

# %%
