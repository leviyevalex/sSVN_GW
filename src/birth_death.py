""" 
v0.1

Remarks
-------
v0.1 - XLA compatible data structure with O(1) lookup, choice, and removal, with O(3N) memory usage added

"""
#%% v0.02 (extra data structure) 
import numpy as np
import jax.numpy as jnp
from functools import partial
import jax
from jax import tree_util

#%%
class ParticleSystem:
    """ 
    O(1) lookup, removal, and choice.
    """
    def __init__(self, N):
        self.N = N 
        self.array = jnp.zeros(shape=(N,2), dtype='int32')
        self.array = self.array.at[:,0].set(jnp.arange(N))
        self.array = self.array.at[:,1].set(jnp.ones(N, dtype='int32'))
        self.n_dead = 0
        self.pointer = jnp.arange(N)

    # @partial(jax.jit, static_argnums=(0,))
    def kill(self, i):
        """ 
        Kill particle i. This involves the following:
        (1) Setting boolean value held in `array` to 0
        (2) Shifting dead particle entry to the end of `array`
        (3) Updating `pointer` to keep track of where entries are in `array`
        """
        self.n_dead += 1
        j = self.pointer[i]

        # Update "lookup" table
        self.array = self.array.at[j,1].set(0)

        # Swap entries in array
        b = self.array[self.N - self.n_dead]                                     # Step 1
        self.array = self.array.at[self.N - self.n_dead].set(self.array[j])      # Step 2
        self.array = self.array.at[j].set(b)                                     # Step 3

        # Swap entries in pointer
        k = self.pointer[b[0]]                                                   # Step 1
        self.pointer = self.pointer.at[b[0]].set(self.pointer[i])                # Step 2
        self.pointer = self.pointer.at[i].set(k)                                 # Step 3

    def choose_alive(self, key):
        """ 
        Randomly choose a particle that is alive. 
        This is efficient because particles are organized according to alive/dead in `array`.
        """
        randint = jax.random.randint(key, (1,), 0, self.N - self.n_dead)[0]
        return self.array[randint, 0]
    
    def is_alive(self, i):
        """ 
        Returns binary value, (0,1) (dead, alive)
        """
        return self.array[self.pointer[i], 1]

#%% 

def flatten_ParticleSystem(obj):
    children = [obj.array, obj.n_dead, obj.pointer]
    aux_data = (obj.N)
    return children, aux_data

def unflatten_ParticleSystem(aux_data, children):
    obj = ParticleSystem(aux_data)
    obj.array = children[0]
    obj.n_dead = children[1]
    obj.pointer = children[2]
    return obj

jax.tree_util.register_pytree_node(ParticleSystem, flatten_ParticleSystem, unflatten_ParticleSystem)


#%% Jump logic methods

def excess_fun(i, j, jumps, particle_system):
    jumps = jumps.at[i].set(j)
    particle_system.kill(i)
    return jumps, particle_system

def deficit_fun(i, j, jumps, particle_system):
    jumps = jumps.at[j].set(i) 
    particle_system.kill(j)
    return jumps, particle_system

def scan_func(carry, x):
    key, jumps, Lambda, particle_system = carry 
    pred = Lambda[x] > 0
    key, subkey = jax.random.split(key)
    j = particle_system.choose_alive(subkey)
    jumps, particle_system = jax.lax.cond(x != -1, lambda: jax.lax.cond(pred, excess_fun, deficit_fun, *(x, j, jumps, particle_system)), lambda: (jumps, particle_system))
    return (key, jumps, Lambda, particle_system), jumps

import sys
sys.path.append("..")
from src.reparameterization import reparameterized_potential


# def indicator(x, a, b):
#     return jnp.prod(jnp.heaviside(x - a, 1) * jnp.heaviside(b - x, 1), axis=-1)

# # Univariate Gaussian CDF
# F = lambda arg, mu, sigma: 0.5 * (1 + jax.scipy.special.erf((arg - mu) / (sigma * jnp.sqrt(2)))) # Trivially extends to d > 1

# def trunc_gaussian(x, mu, sigma, a, b):
#     renormalization = jnp.prod(F(b, mu, sigma) - F(a, mu, sigma), axis=-1)
#     return indicator(x, a, b) * jax.scipy.stats.multivariate_normal.pdf(x, mu, jnp.diag(sigma ** 2)) / renormalization

# trunc_gaussian_batch = jax.vmap(trunc_gaussian, in_axes=(None, 0, None, None, None), out_axes=1)

# def von_mises(x, mu, k, f):
#     separation_vectors = x[:, None] - mu[None, ...]
#     # separation_vectors = x[:, jnp.newaxis, :] - mu[jnp.newaxis, :, :]
#     return jnp.prod(jnp.exp(k * jnp.cos(f * (separation_vectors))) * f / (2 * jnp.pi * jax.scipy.special.i0(k)), axis=-1)

# def custom_gw_kernel(X, Y, sigma, a, b):
#     periodic_coordinates = jnp.array([4, 6, 8])
#     bounded_coordinates = jnp.array([0, 1, 2, 3, 5, 7, 9, 10])
#     tmp1 = trunc_gaussian_batch(Y[:,bounded_coordinates], X[:,bounded_coordinates], sigma[bounded_coordinates], a[bounded_coordinates], b[bounded_coordinates])
#     tmp2 = von_mises(X[:,periodic_coordinates], Y[:,periodic_coordinates], 1 / sigma[periodic_coordinates], jnp.array([1, 2, 1]))
#     return tmp1 * tmp2

# def birth_death(key, X, potential_func, stepsize, bandwidth, p, stride, rate, a, b, gamma):
# from src.reparameterization import sigma, logistic_CDF, reparameterized_gradient
from src.reparameterization import logit, sigma_inv, logistic_PDF

from src.birth_death_kernels import gaussian_CDF

def birth_death(key, X, potential_func, rate, a, b, bounded_coordinates, periodic_coordinates, sigmas, gamma=1):

    """ 
    
    """
    nParticles = X.shape[0]
    key, subkey = jax.random.split(key)

    V_X = potential_func(X)

    arg = jnp.zeros((nParticles, nParticles))

    # Bounded coordinate contribution to KDE
    if len(bounded_coordinates) > 0:
        Y = logit(sigma_inv(X[:, bounded_coordinates], a[bounded_coordinates], b[bounded_coordinates]))
        separation_vectors_I = Y[:, None] - Y[None, ...]
        arg += -0.5 * jnp.sum((separation_vectors_I / sigmas[bounded_coordinates]) ** 2, axis=-1)

        # Potential pushforward
        delta = b[bounded_coordinates] - a[bounded_coordinates]
        f = logistic_PDF(Y)
        V_X += -jnp.sum(jnp.log(delta)) - jnp.sum(jnp.log(f), axis=1)
        
    # Periodic coordinate contribution to KDE
    if len(periodic_coordinates) > 0:
        separation_vectors_J = X[:, periodic_coordinates][:, None] - X[:, periodic_coordinates][None, ...]
        freq = 2 * jnp.pi / b[periodic_coordinates]
        arg += jnp.sum(jnp.cos(freq * separation_vectors_J) / (sigmas[periodic_coordinates] ** 2), axis=-1)
    
    arg += V_X[None, ...]

    beta = jax.scipy.special.logsumexp(arg, axis=1) 
    Lambda = beta - jnp.mean(beta)
    r = jax.random.uniform(minval=0, maxval=1, shape=Lambda.shape, key=subkey) # TODO Lambda.shape = nParticles (???) Clean it up?
    key, subkey = jax.random.split(key)
    threshold = r < 1 - jnp.exp(-jnp.abs(Lambda) * rate)
    idxs = jnp.argwhere(threshold, size=nParticles, fill_value=-1).squeeze()
    idxs = jax.random.permutation(subkey, idxs)
    key, subkey = jax.random.split(key)

    # Perform XLA compatible jump logic
    particle_system = ParticleSystem(nParticles)
    jumps = jnp.arange(nParticles)
    init = (key, jumps, Lambda, particle_system)
    jumps = jax.lax.scan(scan_func, init, idxs)

    return jumps[0][1]



    # renormalization = jnp.prod(gaussian_CDF(b[bounded_coordinates], X[:, bounded_coordinates], sigma[bounded_coordinates]) - gaussian_CDF(a[bounded_coordinates], X[:, bounded_coordinates], sigma[bounded_coordinates]), axis=-1)

    # gram = trunc_gaussian_batch(X, X, jnp.sqrt(bandwidth), a, b)
    
    # gram = custom_gw_kernel(X, X, jnp.sqrt(bandwidth), a, b)

    # beta = jnp.log(jnp.mean(gram, axis=1)) + V_X
    # Lambda = beta - jnp.mean(beta)

    # Calculate relevant quantities in dual space    
    # Y, V_Y = reparameterized_potential(X, potential_func, a, b, gamma)

    # Get particles with significant mass discrepancy in batch

    # Without logsumexp (standard)
    # kern_gram = k_lp(Y, h=bandwidth, p=p)
    # Lambda = jnp.log(jnp.mean(kern_gram, axis=1)) + V_Y
    # Lambda = Lambda - jnp.mean(Lambda) 

    # With logsumexp (standard)
    # separation_vectors = Y[:, jnp.newaxis, :] - Y[jnp.newaxis, :, :]
    # tmp = -jnp.sum(jnp.abs(separation_vectors) ** p, axis=-1) / (p * bandwidth) 
    # Lambda = jax.scipy.special.logsumexp(tmp, axis=1) + V_Y
    # Lambda = Lambda - jnp.mean(Lambda) 

    # Alternatively proposed estimate with logsumexp on dual space (UNCOMMENT THIS LATER)
    # Y, V_Y = reparameterized_potential(X, potential_func, a, b, gamma)
    # separation_vectors = Y[:, jnp.newaxis, :] - Y[jnp.newaxis, :, :]
    # tmp = -jnp.sum((jnp.abs(separation_vectors) ** p) / (p * bandwidth), axis=-1) # This should generalize to a vector of bandwidths
    # Lambda = jax.scipy.special.logsumexp(tmp + V_Y[None, ...], axis=1)
    # Lambda = Lambda - jnp.mean(Lambda) 

#%%
# import jax
# import jax.numpy as jnp



# def bounded_KDE(X, a, b, h=0.001):
#     """ 
#     https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
#     """
    
#     # Wrapper for scipy truncnorm function
#     a_, b_ = (a - X) / h, (b - X) / h


#     X[..., jnp.newaxis]
#     a_ = 
#     b_ = 
#     jax.scipy.stats.truncnorm.pdf(x a, b, X, h)


# Tests for the data structure
#%%
# nParticles = 6
# particle_system = ParticleSystem(nParticles)
# Lambda = jnp.array([1, -1, 1, -1, 1, 1])
# idxs = jnp.array([1, 2, 5, -1, -1, -1])

# @jax.jit
# def func():
#     key = jax.random.PRNGKey(1)
#     jumps = jnp.arange(nParticles)
#     init = (key, jumps, Lambda, particle_system)
#     jumps = jax.lax.scan(scan_func, init, idxs)[0][1]
#     return jumps

# #%%
# func()

#%%
#%%
# #%% Quick loop test
# def body_fun(i, val): 
#     val.kill(i)
#     return val

# @partial(jax.jit, static_argnums=(0,))
# def test_func(nParticles): 
#     a = ParticleSystem(nParticles)
#     return jax.lax.fori_loop(3, nParticles, body_fun, a)

# res = test_func(6)
# print(res.array)
# print(res.pointer)
# #%% First test
# a = ParticleSystem(6)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(0)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(5)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(4)
# print(a.array)
# print(a.pointer)
# #%%
# seed = 1200
# key = jax.random.PRNGKey(seed) 
# a.choose_alive(key)


# #%%
# a.kill(3)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(1)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(2)
# print(a.array)
# print(a.pointer)

# #%% Second test
# a = ParticleSystem(6)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(1)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(3)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(4)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(5)
# print(a.array)
# print(a.pointer)
# #%%
# a.kill(0)
# print(a.array)
# print(a.pointer)

# #%%
# a.kill(2)
# print(a.array)
# print(a.pointer)
# # %%











# %%
