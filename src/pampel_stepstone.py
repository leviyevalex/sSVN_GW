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
from src.reparameterization import sigma, logistic_CDF, reparameterized_gradient, push_forward, pull_back

# Load birth/death method
from src.birth_death import birth_death
from src.helper import scan

from jax.scipy.linalg import solve_triangular, cholesky, cho_solve

def ula_kernel(key, X, potential, grad_potential, dt, iteration, lower, upper, stride, rate, sigmas, bounded_coordinates, periodic_coordinates):
    """ 
    Remarks
    -------
    (1) A subkey is immediately used. The key is used to split
    (2) The periodic coordinates must begin at 0 for the modding to work nicely!!! Otherwise a shift in coordinates in necessary
    """
    key, subkey = jax.random.split(key)
    N = X.shape[0]
    d = X.shape[1]

    # Calculate gradients
    gmlpt_X = grad_potential(X)

    if len(bounded_coordinates) > 0:
        Y, gmlpt_Y = reparameterized_gradient(X, gmlpt_X, lower, upper)
        X = X.at[:, bounded_coordinates].set(Y[:, bounded_coordinates])
        gmlpt_X = gmlpt_X.at[:, bounded_coordinates].set(gmlpt_Y[:, bounded_coordinates])

    Hmlpt += 0.01 * jnp.eye(d)[jnp.newaxis, ...] # Damping factor
    U = cholesky(Hmlpt, lower=False)
    X += -cho_solve((U, False), gmlpt_X) * dt  + jnp.sqrt(2 * dt) * solve_triangular(U, jax.random.normal(key=subkey, shape=(X.shape)), lower=False)

    # X += -gmlpt_X * dt + jnp.sqrt(2 * dt) * jax.random.normal(key=subkey, shape=X.shape) # Regular Langevin 
    key, subkey = jax.random.split(key)

    if len(bounded_coordinates) > 0:
        X = X.at[:, bounded_coordinates].set(pull_back(X[:, bounded_coordinates], lower[bounded_coordinates], upper[bounded_coordinates]))

    if len(periodic_coordinates) > 0:
        X = X.at[:, periodic_coordinates].set(jnp.mod(X[:, periodic_coordinates], upper[periodic_coordinates])) 

    # jumps = jax.lax.cond(jnp.mod(iteration, stride) == 0, lambda: birth_death(subkey, X, potential, rate=rate, a=lower, b=upper, bounded_coordinates=bounded_coordinates, periodic_coordinates=periodic_coordinates, sigmas=sigmas), lambda: jnp.arange(N))
    key, subkey = jax.random.split(key)

    # X = X[jumps]


    iteration = iteration + 1
    return key, X, iteration

@partial(jax.jit, static_argnums=(1,2,3))
def ula_sampler_full_jax_jit(key, potential, grad_potential, n_iter, dt, x_0, lower, upper, stride, rate, sigmas, b_idx, p_idx):

    # @progress_bar_scan(n_iter)
    # @scan_tqdm(1000)
    # @scan_tqdm(n_iter, print_rate=1, desc='progress bar', position=0, leave=False)
    def ula_step(carry, x):
        key, param, iteration = carry
        key, param, iteration = ula_kernel(key, param, potential, grad_potential, dt, iteration, lower, upper, stride, rate, sigmas, b_idx, p_idx)
        return (key, param, iteration), param

    carry = (key, x_0, 0)
    _, samples = jax.lax.scan(ula_step, carry, None, n_iter)
    # _, samples = scan(ula_step, carry, None, n_iter)
    return samples