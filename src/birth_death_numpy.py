"""
v0.1

Remarks
-------

v0.1 - Pure numpy/python implemention of birth death.

ListDict - Python data structure for O(1) lookup, selection, and removal

"""

import random
import jax 
import jax.numpy as jnp
from src.reparameterization import reparameterized_potential

class ListDict(object):
    """  
    Solution adapted from 
    https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
    Data structure with O(1) lookup, uniform random selection, and removal

    """
    def __init__(self, nParticles):
        self.item_to_position = {}
        self.items = []
        for n in range(nParticles):
            self.add_item(n)

    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove_item(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return random.choice(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

def k_lp(X, p=2, h=0.001): 
    """ 
    Lp kernel implementation
    """
    # Get separation vectors
    separation_vectors = X[:, jnp.newaxis, :] - X[jnp.newaxis, :, :]
    
    # Calculate kernel
    # k = jnp.exp(-jnp.sum(jnp.abs(separation_vectors) ** p, axis=-1) / (p * h))
    k = jnp.exp(-jnp.sum(jnp.abs(separation_vectors) ** p, axis=-1) / (p * h))
    return k

def birth_death_unaccelerated(key, X, potential_func, stepsize, bandwidth, p, stride, rate, a, b, gamma):
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
    idxs = jnp.argwhere(threshold, size=nParticles).squeeze()
    idxs = jax.random.permutation(key, idxs)

    alive = ListDict(nParticles)
    jumps = jnp.arange(nParticles)
    for i in idxs: 
        if i in alive:
            j = alive.choose_random_item()
            if Lambda[i] > 0:
                jumps = jumps.at[i].set(j)
                alive.remove_item(i)
            else:
                jumps = jumps.at[j].set(i)
                alive.remove_item(j)

    return jumps
