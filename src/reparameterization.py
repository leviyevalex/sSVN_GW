"""
Module for transforming hypercube flows to flows over R^d.

Version: 0.1.0
"""


import jax.numpy as jnp
from jax.scipy.special import logit
from jax.scipy.stats import logistic # THIS CAN GIVE CDF AND PDF
logistic_CDF = lambda x: logistic.cdf(x)
logistic_PDF = lambda x: logistic.pdf(x)

# logit = lambda x: jnp.log(x / (1 - x))
# logit_prime = lambda x: 1 / (x - x ** 2)
# logistic_CDF = lambda x: 1 / (1 + jnp.exp(-x))
# logistic_PDF = lambda x: jnp.exp(-x) / (1 + jnp.exp(-x)) ** 2
# logistic_PDF = lambda x: 0.25 * jnp.cosh(x/2) ** (-2)



sigma_inv = lambda x, a, b: (x - a) / (b - a)
sigma = lambda x, a, b: (b - a) * x + a


push_forward = lambda X, a, b: logit(sigma_inv(X, a, b))

pull_back = lambda Y, a, b: sigma(logistic_CDF(Y), a, b)

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

def hessian_reparameterization(X, hess_V_X, a, b, unbounded_idxs):
    # Step 1
    d = len(a)

    delta = b - a

    if len(unbounded_idxs) > 0:
        delta = delta.at[unbounded_idxs].set(jnp.array([1])) # Pretend like its already on the unit. No transformation

    hess_V_Z = hess_V_X * jnp.outer(delta, delta)[jnp.newaxis, ...]

    # Step 2
    Y = logit(sigma_inv(X, a, b))
    # f = logistic_PDF(Y)

    f_first = logistic_PDF(Y)
    f_second = logistic_PDF(Y)

    if len(unbounded_idxs) > 0:
        f_first = f_first.at[:,unbounded_idxs].set(jnp.array([1]))
    hess_V_Y = hess_V_Z * (f_first[:, :, jnp.newaxis] * f_first[:, jnp.newaxis, :])

    if len(unbounded_idxs) > 0:
        f_second = f_second.at[:,unbounded_idxs].set(jnp.array([0]))

    hess_V_Y = hess_V_Y.at[:, jnp.arange(d), jnp.arange(d)].add(2 * f_second)

    return hess_V_Y
    # print('FIX REPARAMETERIZATION ON HESSIAN')
    # return hess_V_X


# def V_sharp(Y, V_func, a, b):
#     delta = b - a
#     V_Z = V_X - jnp.sum(jnp.log(delta))
#     grad_V_Z = grad_V_X * delta[jnp.newaxis, ...]

#     # Transform from unit cube to R^d
#     Y = logit(sigma_inv(X, a, b))
#     f = logistic_PDF(Y)
#     F = logistic_CDF(Y)

#     V_Y = V_Z - jnp.sum(jnp.log(f), axis=1)
    
#     grad_V_Y = grad_V_Z * f + 2 * F - 1
    
#     return Y, V_Y


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

def reparameterized_potential(X, potential_function, a, b, gamma=1):
    delta = b - a
    V_X = potential_function(X) * gamma

    V_Z = V_X - jnp.sum(jnp.log(delta))

    # Transform from unit cube to R^d
    Y = logit(sigma_inv(X, a, b))
    f = logistic_PDF(Y)

    V_Y = V_Z - jnp.sum(jnp.log(f), axis=1)
    
    return Y, V_Y # Modification made so as to be able to use this function in birth_death.py. change back if needed
    # return V_Y

# TODO MODIFICATION HERE. either accept gradient or gradient function
# def reparameterized_gradient(X, gradient_function, a, b, gamma=1):
def reparameterized_gradient(X, gradient, a, b, gamma=1):
    delta = b - a
    # grad_V_X = gradient_function(X) * gamma
    grad_V_X = gradient * gamma
    grad_V_Z = grad_V_X * delta[jnp.newaxis, ...]

    # Transform from unit cube to R^d
    Y = logit(sigma_inv(X, a, b))
    f = logistic_PDF(Y)
    F = logistic_CDF(Y)
    
    grad_V_Y = grad_V_Z * f + 2 * F - 1
    
    return Y, grad_V_Y # Modification made so as to be able to use this function in birth_death.py. change back if needed

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
