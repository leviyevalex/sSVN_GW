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

#%% Putting it all together
import sys
sys.path.append("..")
from src.reparameterization import reparameterized_potential

def k_lp(X, p=2, h=0.001): 
    """ 
    Lp kernel implementation
    """
    # Get separation vectors
    separation_vectors = X[:, jnp.newaxis, :] - X[jnp.newaxis, :, :]
    
    # Calculate kernel
    k = jnp.exp(-jnp.sum(jnp.abs(separation_vectors) ** p, axis=-1) / (p * h))
    return k

def birth_death(key, X, potential_func, stepsize, bandwidth, p, stride, rate, a, b, gamma):
    """ 
    
    """
    nParticles = X.shape[0]
    key, subkey = jax.random.split(key)

    # Calculate relevant quantities in dual space    
    Y, V_Y = reparameterized_potential(X, potential_func, a, b, gamma)
    kern_gram = k_lp(Y, h=bandwidth, p=p)

    # Get particles with significant mass discrepancy in batch
    Lambda = jnp.log(jnp.mean(kern_gram, axis=1)) + V_Y
    Lambda = Lambda - jnp.mean(Lambda) 
    r = jax.random.uniform(minval=0, maxval=1, shape=Lambda.shape, key=subkey)
    threshold = r < 1 - jnp.exp(-jnp.abs(Lambda) * stepsize * stride * rate)
    idxs = jnp.argwhere(threshold, size=nParticles, fill_value=-1).squeeze()
    idxs = jax.random.permutation(key, idxs)

    # Perform XLA compatible jump logic
    particle_system = ParticleSystem(nParticles)
    jumps = jnp.arange(nParticles)
    init = (key, jumps, Lambda, particle_system)
    jumps = jax.lax.scan(scan_func, init, idxs)

    return jumps[0][1]




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
