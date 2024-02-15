""" 
v0.1

Remarks
-------
v0.1 - XLA compatible data structure with O(1) lookup, choice, and removal, with O(3N) memory usage added

"""

#%%
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

def true_fun(i, j, jumps):
    """ 
    Excess logic
    """
    jumps = jumps.at[0,i].set(j)
    jumps = jumps.at[1,i].set(0)
    return jumps

def false_fun(i, j, jumps):
    """ 
    Deficit logic
    """
    jumps = jumps.at[0,j].set(i) 
    jumps = jumps.at[1,j].set(0)
    return jumps

def scan_func(carry, x):
    """ 
    Operation to be performed on each element $x \in \xi$
    """
    key, jumps, Lambda, array, pointer = carry 
    pred = Lambda[x] > 0
    key, subkey = jax.random.split(key)


    j =  

    # Do nothing if entry is padded. This is an XLA artifact
    jumps = jax.lax.cond(x != -1, lambda: jax.lax.cond(pred, true_fun, false_fun, *(x, j, jumps)), lambda: jumps)
    return (key, jumps, Lambda), jumps



#%%
def batched_birth_death(key, X, potential_func, stepsize, bandwidth, stride, rate, a, b):
    """ 
    Remarks
    -------

    (1) 
    (2) 

    Notes
    -----

    
    """
    nParticles = X.shape[0]
    key, subkey = jax.random.split(key)

    # Calculate relevant quantities in dual space    
    V_X = potential_func(X)
    Y, V_Y = reparameterized_potential(X, potential_func, a, b)
    kern_gram = k_lp(Y, h=bandwidth)

    # Get particles with significant mass discrepancy in batch
    Lambda = jnp.log(jnp.mean(kern_gram, axis=1)) + V_Y
    Lambda = Lambda - jnp.mean(Lambda) 
    r = jax.random.uniform(minval=0, maxval=1, shape=Lambda.shape, key=key)
    threshold = r < 1 - jnp.exp(-jnp.abs(Lambda) * stepsize * stride * rate)
    idxs = jnp.argwhere(threshold, size=nParticles, fill_value=-1).squeeze()
    idxs = jax.random.permutation(key, idxs)

    # Perform XLA compatible jump logic
    jumps = jnp.arange(nParticles)
    init = (key, jumps, Lambda)
    jumps = jax.lax.scan(scan_func, init, idxs)


    return output


#%% v0.02 (extra data structure) 
import numpy as np
import jax.numpy as jnp
from functools import partial
import jax
#%%
class ParticleSystem:
    """ 
    O(1) lookup, removal, and choice.
    """
    def __init__(self, N):
        self.N = N 

        # Modified
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

        # Swap entries pointer
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



        

#%% First test
a = ParticleSystem(6)
print(a.array)
print(a.pointer)
#%%
a.kill(0)
print(a.array)
print(a.pointer)
#%%
a.kill(5)
print(a.array)
print(a.pointer)
#%%
a.kill(4)
print(a.array)
print(a.pointer)
#%%
seed = 1200
key = jax.random.PRNGKey(seed) 
a.choose_alive(key)


#%%
a.kill(3)
print(a.array)
print(a.pointer)
#%%
a.kill(1)
print(a.array)
print(a.pointer)
#%%
a.kill(2)
print(a.array)
print(a.pointer)

#%% Second test
a = ParticleSystem(6)
print(a.array)
print(a.pointer)
#%%
a.kill(1)
print(a.array)
print(a.pointer)
#%%
a.kill(3)
print(a.array)
print(a.pointer)
#%%
a.kill(4)
print(a.array)
print(a.pointer)
#%%
a.kill(5)
print(a.array)
print(a.pointer)
#%%
a.kill(0)
print(a.array)
print(a.pointer)

#%%
a.kill(2)
print(a.array)
print(a.pointer)
# %%



































#%%
# SCRATCHWORK
#%%
class ParticleSystem:
    """ 
    This was the first version.
    
    """
    def __init__(self, N):
        # Keep convention that top row contains information about particles indexed in lower row
        self.N = N 
        self.array = jnp.zeros(shape=(2,N), dtype='int32')
        self.array = self.array.at[0].set(jnp.ones(N, dtype='int32'))
        self.array = self.array.at[1].set(jnp.arange(N))
        self.n_dead = 0

        self.pointer = jnp.arange(N)

    # @partial(jax.jit, static_argnums=(0,))
    def kill(self, i):
        self.n_dead += 1
        j = self.pointer[i]

        # Update lookup table
        self.array = self.array.at[0,j].set(0)

        # Swap entries in array
        b = self.array[:,self.N - self.n_dead]                                     # Step 1
        self.array = self.array.at[:,self.N - self.n_dead].set(self.array[:,j])    # Step 2
        self.array = self.array.at[:,j].set(b)                                     # Step 3

        # Swap entries pointer
        k = self.pointer[b[1]]                                                     # Step 1
        self.pointer = self.pointer.at[b[1]].set(self.pointer[i])                  # Step 2
        self.pointer = self.pointer.at[i].set(k)                                   # Step 3

    def choose_alive(self, key):
        # Randomly choose a particle that is alive
        randint = jax.random.randint(key, (1,), 0, self.N - self.n_dead)[0]
        return self.array[1, randint]
    
    def is_alive(self, i):
        # Returns binary value, (0,1) (dead, alive)
        # TODO - this is actually broken as is! It needs to pointer before accessing!
        return self.array[0, i]


