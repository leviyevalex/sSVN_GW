"""
JAX Kernels.
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

class kernels:
    def __init__(self, nParticles, DoF, kernel_type, nFeatures=None):
        self.nParticles = nParticles 
        self.DoF = DoF
        self.kernel_type = kernel_type
        if self.kernel_type == 'RFG':
            if nFeatures == None:
                self.initRandomFeatures(nFeatures=nParticles)
            else:
                self.initRandomFeatures(nFeatures=nFeatures)

    def initRandomFeatures(self, nFeatures):
        key = random.PRNGKey(0)
        self.w0 = random.uniform(minval=0., maxval=2*jnp.pi, shape=(nFeatures,), key=key)
        self.w1 = random.normal(shape=(nFeatures, self.DoF), key=key)

    ###########################################################
    # JAX high performance version
    # Note: Used in production for best performance
    # Note: Requires a bit more care to implement correctly.
    ###########################################################

    @partial(jax.jit, static_argnums=(0,))
    def getPairwiseDisplacement(self, X):
        return X[:,jnp.newaxis,:] - X[jnp.newaxis,:,:]

    @partial(jax.jit, static_argnums=(0,))
    def kernel_RBF(self, X, params):
        U = jax.scipy.linalg.cholesky(params['M'])
        X = contract('ij, Nj -> Ni', U, X)
        pairwise_displacements = self.getPairwiseDisplacement(X)
        k = jnp.exp(-contract('mni, mni -> mn', pairwise_displacements, pairwise_displacements) / (2 * params['h']))
        g1k = -1 * contract('mn, mni -> mni', k, pairwise_displacements) / params['h']
        g1k = contract('ij, mni -> mnj', U, g1k)
        return (k, g1k) # Derivative returned on first slot

    @partial(jax.jit, static_argnums=(0,))
    def kernel_Lp(self, X, params):
        U = jax.scipy.linalg.cholesky(params['M'])
        X = contract('ij, Nj -> Ni', jax.scipy.linalg.cholesky(params['M']), X)
        pairwise_displacements = self.getPairwiseDisplacement(X)
        k = jnp.exp(-jnp.sum(jnp.abs(pairwise_displacements) ** params['p'], axis=-1) / params['h'])
        g1k = (-params['p'] / params['h']) * contract('mn, mni -> mni', k, jnp.abs(pairwise_displacements) ** (params['p'] - 1)) * jnp.sign(pairwise_displacements)
        g1k = contract('ij, mni -> mnj', U, g1k)
        g1k = g1k.at[jnp.array(range(self.nParticles)), jnp.array(range(self.nParticles))].set(0)
        return (k, g1k) # Derivative returned on first slot

    @partial(jax.jit, static_argnums=(0,))
    def kernel_RFG(self, X, params, w0, w1):
        nFeatures = w0.size
        U = jax.scipy.linalg.cholesky(params['M'])
        X = contract('ij, Nj -> Ni', jax.scipy.linalg.cholesky(params['M']), X)
        arg = contract('ld, Nd -> Nl', w1, X) / jnp.sqrt(params['h']) + w0
        phi = jnp.sqrt(2) * jnp.cos(arg)
        k = contract('ml, nl -> mn', phi, phi) / nFeatures
        gphi = -jnp.sqrt(2 / params['h']) * contract('ml, li -> mli', jnp.sin(arg), w1)
        g1k = contract('mli, nl -> mni', gphi, phi) / nFeatures
        g1k = contract('ij, mni -> mnj', U, g1k)
        return (k, g1k)

    def getKernelWithDerivatives(self, X, params):
        if self.kernel_type == 'RFG':
            return self.kernel_RFG(X, params, self.w0, self.w1)
        elif self.kernel_type == 'RBF':
            return self.kernel_RBF(X, params)
        elif self.kernel_type == 'Lp':
            return self.kernel_Lp(X, params)
        else:
            raise NotImplementedError()

    ###########################################################
    # JAX vectorized versions
    # Note: Used to test kernels quickly, and for unit tests
    ###########################################################

    def k_hyper(self, y1, y2, params, k):
        f_inv = lambda y: (params['a'] + params['b'] * jnp.exp(y)) / (1 + jnp.exp(y))
        x1 = f_inv(y1)
        x2 = f_inv(y2)
        return k(x1, x2, params)

    @partial(jax.jit, static_argnums=(0,4,))
    def gram(self, x, y, params, func):
        return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1, params))(y))(x)

    @partial(jax.jit, static_argnums=(0,))
    def k_RBF(self, x, y, params):
        U = jax.scipy.linalg.cholesky(params['M'])
        x = U @ x 
        y = U @ y
        return jnp.exp(-jnp.dot(x - y, x - y) / (2 * params['h']))

    @partial(jax.jit, static_argnums=(0,))
    def k_RFG(self, x, y, params):
        U = jax.scipy.linalg.cholesky(params['M'])
        x = U @ x 
        y = U @ y
        phi_x = jnp.sqrt(2) * jnp.cos(self.w1 @ x / jnp.sqrt(params['h']) + self.w0) 
        phi_y = jnp.sqrt(2) * jnp.cos(self.w1 @ y / jnp.sqrt(params['h']) + self.w0)
        return jnp.mean(phi_x * phi_y)

    @partial(jax.jit, static_argnums=(0,))
    def k_Lp(self, x, y, params):
        U = jax.scipy.linalg.cholesky(params['M'])
        x = U @ x 
        y = U @ y
        return jnp.exp(-jnp.sum(jnp.abs(x-y) ** params['p']) / params['h'])

    @partial(jax.jit, static_argnums=(0,))
    def getKernelWithDerivatives_vectorized(self, particles, params):
        if self.kernel_type == 'RFG':
            k_func = self.k_RFG
        elif self.kernel_type == 'RBF':
            k_func = self.k_RBF
        elif self.kernel_type == 'Lp':
            k_func = self.k_Lp
        else:
            raise NotImplementedError()

        k = self.gram(particles, particles, params, k_func)
        g1k = self.gram(particles, particles, params, jax.jacobian(k_func))
        return (k, g1k)


#%% 
# UNIT TEST: Make sure hypercube kernel derivative is correctly defined



#%%
# UNIT TEST: Compare high performance RBF with vectorized RBF

# nParticles = 3
# DoF = 2
# kernel_class = kernels(nParticles=nParticles, DoF=DoF, kernel_type='RBF')
# particles = jnp.array(np.random.rand(nParticles, DoF))
# M = jnp.array([[10., 1], [1, 15]])
# # M = jnp.eye(DoF)
# h=2.
# params = {'M': M, 'h': h}
# k_1, g1k_1 = kernel_class.getKernelWithDerivatives(particles, params)
# k_2, g1k_2 = kernel_class.getKernelWithDerivatives_vectorized(particles, params)
# print(np.allclose(k_1, k_2))
# print(np.allclose(g1k_1, g1k_2))

# #%%
# # # UNIT TEST: Compare high performance RFG with vectorized RFG

# nParticles = 3
# DoF = 2
# kernel_class = kernels(nParticles=nParticles, DoF=DoF, kernel_type='RFG')
# particles = jnp.array(np.random.rand(nParticles, DoF))
# M = jnp.array([[10., 1], [1, 15]])
# # M = jnp.eye(DoF)
# h=2.
# params = {'M': M, 'h': h}
# k_1, g1k_1 = kernel_class.getKernelWithDerivatives(particles, params)
# k_2, g1k_2 = kernel_class.getKernelWithDerivatives_vectorized(particles, params)
# print(np.allclose(k_1, k_2))
# print(np.allclose(g1k_1, g1k_2))

# #%%
# # # UNIT TEST: Compare high performance Lp with vectorized Lp

# nParticles = 3
# DoF = 2
# kernel_class = kernels(nParticles=nParticles, DoF=DoF, kernel_type='Lp')
# particles = jnp.array(np.random.rand(nParticles, DoF))
# M = jnp.array([[10., 1], [1, 15]])
# # M = jnp.eye(DoF)
# h=2.
# params = {'M': M, 'h': h, 'p':2}
# k_1, g1k_1 = kernel_class.getKernelWithDerivatives(particles, params)
# k_2, g1k_2 = kernel_class.getKernelWithDerivatives_vectorized(particles, params)
# print(np.allclose(k_1, k_2))
# print(np.allclose(g1k_1, g1k_2))






# %%
