"""
Module for transforming hypercube flows to flows over R^d.

Version: 0.1.0
"""


import jax.numpy as jnp

logit = lambda x: jnp.log(x / (1 - x))
logistic_CDF = lambda x: 1 / (1 + jnp.exp(-x))
logistic_PDF = lambda x: jnp.exp(-x) / (1 + jnp.exp(-x)) ** 2
sigma_inv = lambda x, a, b: (x - a) / (b - a)
sigma = lambda x, a, b: (b - a) * x + a

def reparameterization_full(X, V_X, grad_V_X, hess_V_X, a, b): 
    """ 
    Given potential in H^d, return transformed quantities in R^d
    """
    d = len(a)

    # Transform from hypercube to unit cube 
    delta = b - a
    V_Z = V_X - jnp.sum(jnp.log(delta))
    grad_V_Z = grad_V_X * delta[jnp.newaxis, ...]
    hess_V_Z = hess_V_X * jnp.outer(delta, delta)[jnp.newaxis, ...]

    # Transform from unit cube to R^d
    Y = logit(sigma_inv(X, a, b))
    f = logistic_PDF(Y)
    F = logistic_CDF(Y)

    V_Y = V_Z - jnp.sum(jnp.log(f), axis=1)
    
    grad_V_Y = grad_V_Z * f + 2 * F - 1
    
    hess_V_Y = hess_V_Z * (f[:, :, jnp.newaxis] * f[:, jnp.newaxis, :]) 
    # hess_V_Y = hess_V_Y.at[:, jnp.arange(d), jnp.arange(d)].add(2 * f)
    # hess_V_Y = hess_V_Y.at[:, jnp.arange(d), jnp.arange(d)].add(grad_V_Z * f * (1 - 2 * F))

    return Y, V_Y, grad_V_Y, hess_V_Y

def reparameterization(X, V_X, grad_V_X, a, b): 
    """ 
    Given potential in H^d, return transformed quantities in R^d
    """
    d = len(a)

    # Transform from hypercube to unit cube 
    delta = b - a
    V_Z = V_X - jnp.sum(jnp.log(delta))
    grad_V_Z = grad_V_X * delta[jnp.newaxis, ...]

    # Transform from unit cube to R^d
    Y = logit(sigma_inv(X, a, b))
    f = logistic_PDF(Y)
    F = logistic_CDF(Y)

    V_Y = V_Z - jnp.sum(jnp.log(f), axis=1)
    
    grad_V_Y = grad_V_Z * f + 2 * F - 1
    
    return Y, V_Y, grad_V_Y

def reparameterized_potential(X, potential_function, a, b):
    delta = b - a
    V_X = potential_function(X)

    V_Z = V_X - jnp.sum(jnp.log(delta))

    # Transform from unit cube to R^d
    Y = logit(sigma_inv(X, a, b))
    f = logistic_PDF(Y)

    V_Y = V_Z - jnp.sum(jnp.log(f), axis=1)
    
    return V_Y

def reparameterized_gradient(X, gradient_function, a, b):
    delta = b - a
    grad_V_X = gradient_function(X)
    grad_V_Z = grad_V_X * delta[jnp.newaxis, ...]

    # Transform from unit cube to R^d
    Y = logit(sigma_inv(X, a, b))
    f = logistic_PDF(Y)
    F = logistic_CDF(Y)
    
    grad_V_Y = grad_V_Z * f + 2 * F - 1
    
    return grad_V_Y

def reparameterized_gauss(X, gauss_function, a, b):
    d = len(a)
     # Transform from hypercube to unit cube 
    delta = b - a

    hess_V_X = gauss_function(X)
    hess_V_Z = hess_V_X * jnp.outer(delta, delta)[jnp.newaxis, ...]

    # Transform from unit cube to R^d
    Y = logit(sigma_inv(X, a, b))
    f = logistic_PDF(Y)
    F = logistic_CDF(Y)

    hess_V_Y = hess_V_Z * (f[:, :, jnp.newaxis] * f[:, jnp.newaxis, :]) 
    hess_V_Y = hess_V_Y.at[:, jnp.arange(d), jnp.arange(d)].add(2 * f)
    # hess_V_Y = hess_V_Y.at[:, jnp.arange(d), jnp.arange(d)].add(grad_V_Z * f * (1 - 2 * F))   
    return hess_V_Y
