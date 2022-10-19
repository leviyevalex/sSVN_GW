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
    def __init__(self, nParticles, DoF, kernel_type):
        self.nParticles = nParticles 
        self.DoF = DoF
        self.kernel_type = kernel_type
        if self.kernel_type == 'RFG':
            self.initRandomFeatures(nFeatures=nParticles)

    def initRandomFeatures(self, nFeatures):
        key = random.PRNGKey(0)
        self.w0 = random.uniform(minval=0., maxval=2*jnp.pi, shape=(nFeatures,), key=key)
        self.w1 = random.normal(shape=(nFeatures, self.DoF), key=key)

    @partial(jax.jit, static_argnums=(0,4,))
    def gram(self, x, y, params, func):
        return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1, params))(y))(x)

    @partial(jax.jit, static_argnums=(0,))
    def k_RBF(self, x, y, kwargs):
        U = jax.scipy.linalg.cholesky(kwargs['M'])
        x = U @ x 
        y = U @ y
        return jnp.exp(-jnp.dot(x - y, x - y) / (2 * kwargs['h']))

    @partial(jax.jit, static_argnums=(0,))
    def k_RFG(self, x, y, kwargs):
        U = jax.scipy.linalg.cholesky(kwargs['M'])
        x = U @ x 
        y = U @ y
        phi_x = jnp.sqrt(2) * jnp.cos(self.w1 @ x / jnp.sqrt(kwargs['h']) + self.w0) 
        phi_y = jnp.sqrt(2) * jnp.cos(self.w1 @ y / jnp.sqrt(kwargs['h']) + self.w0)
        return jnp.mean(phi_x * phi_y)

    @partial(jax.jit, static_argnums=(0,))
    def k_Lp(self, x, y, kwargs):
        U = jax.scipy.linalg.cholesky(kwargs['M'])
        x = U @ x 
        y = U @ y
        return jnp.exp(-jnp.sum(jnp.abs(x-y) ** kwargs['p']) / kwargs['h'])

    @partial(jax.jit, static_argnums=(0,))
    def getKernelWithDerivatives_vectorized(self, particles, **params):
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


# #%%
# nParticles = 500
# DoF = 10
# kernel_params = {'h':2}            # RBF
# M = jnp.eye(DoF)
# # kernel_params = {'M':jnp.eye(DoF), 'sigma':2, 'p':2} # Lp
# kernel_class = kernels(nParticles, DoF, kernel_type='RBF')
# particles = jnp.array(np.random.rand(nParticles, DoF))

# # kernel_class.k_RBF(particles[0], particles[1], kernel_params)

# #%%
# k, g1k = kernel_class.getKernelWithDerivatives___(particles, M=M, **kernel_params)
# %%
