import sys
sys.path.append("..")

import jax, os, corner
import jax.numpy as jnp
from jax import grad, config
import matplotlib.pyplot as plt
import numpy as np
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
from functools import partial

# Load reparameterization methods
from src.reparameterization import sigma, logistic_CDF, reparameterized_gradient

# Load birth/death method
from src.birth_death import birth_death

def ula_kernel(key, X, potential, grad_potential, dt, iteration, lower, upper, stride, rate, bandwidth, bounded_coordinates, periodic_coordinates):
    """ 
    Remarks
    -------
    (1) A subkey is immediately used. The key is used to split
    (2) The periodic coordinates must begin at 0 for the modding to work nicely!!! Otherwise a shift in coordinates in necessary
    """
    N = X.shape[0]

    # Calculate gradients
    gmlpt_X = grad_potential(X)

    # Update bounded coordinates
    Y, gmlpt_Y = reparameterized_gradient(X[:, bounded_coordinates], gmlpt_X[:, bounded_coordinates], lower[bounded_coordinates], upper[bounded_coordinates])
    key, subkey = jax.random.split(key)
    Y = Y - gmlpt_Y * dt #+ jnp.sqrt(2 * dt) * jax.random.normal(key=subkey, shape=(N, len(bounded_coordinates)))
    X = X.at[:, bounded_coordinates].set(sigma(logistic_CDF(Y), lower[bounded_coordinates], upper[bounded_coordinates]))

    # Update periodic coordinates
    key, subkey = jax.random.split(key)    
    X = X.at[:, periodic_coordinates].add(-gmlpt_X[:, periodic_coordinates] * dt) #+ jnp.sqrt(2 * dt) * jax.random.normal(key=subkey, shape=(N, len(periodic_coordinates))))
    X = X.at[:, periodic_coordinates].set(jnp.mod(X[:, periodic_coordinates], upper[periodic_coordinates])) 

    # Perform jumps in primal space
    key, subkey = jax.random.split(key)
    # jumps = jax.lax.cond(jnp.mod(iteration, stride) == 0, lambda: birth_death(subkey, X, potential, bandwidth=bandwidth, rate=rate, a=lower, b=upper, bounded_coordinates=bounded_coordinates, periodic_coordinates=periodic_coordinates, sigma=standard_dev, f=f), lambda: jnp.arange(N))
    # X = X[jumps]

    iteration = iteration + 1
    return key, X, iteration

@partial(jax.jit, static_argnums=(1,2,3))
def ula_sampler_full_jax_jit(key, potential, grad_potential, n_iter, dt, x_0, lower, upper, stride, rate, bandwidth, b_idx, p_idx):

    # @progress_bar_scan(n_iter)
    # @scan_tqdm(1000)
    # @scan_tqdm(n_iter, print_rate=1, desc='progress bar', position=0, leave=False)
    def ula_step(carry, x):
        key, param, iteration = carry
        key, param, iteration = ula_kernel(key, param, potential, grad_potential, dt, iteration, lower, upper, stride, rate, bandwidth, b_idx, p_idx)
        return (key, param, iteration), param

    carry = (key, x_0, 0)
    _, samples = jax.lax.scan(ula_step, carry, None, n_iter)
    # _, samples = scan(ula_step, carry, None, n_iter)
    return samples