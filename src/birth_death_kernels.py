""" 
v0.3 

Changelog
-------
v0.1 - unnormalized Lp kernel implemented
v0.2 - implemented normalized uncorrelated multivariate Gaussian
v0.3 - implemented normalized von-Mises and uncorrelated, truncated multivariate Gaussian kernels

"""
import jax.numpy as jnp 
import jax

def k_lp(X, p=2, h=0.001): 
    """ 
    Lp kernel implementation
    """
    # Get separation vectors
    separation_vectors = X[:, None] - X[None, ...]
    
    # Calculate kernel
    k = jnp.exp(-jnp.sum(jnp.abs(separation_vectors) ** p, axis=-1) / (p * h))
    return k

def gaussian_CDF(x, mu, sigma):
    """ 
    CDF of univariate Gaussian. 
    """
    return 0.5 * (1 + jax.scipy.special.erf((x - mu) / (sigma * jnp.sqrt(2))))

def indicator(x, a, b):
    """ 
    Returns 1 if $x \in (a,b)$, 0 else.
    """
    return jnp.heaviside(x - a, 1) * jnp.heaviside(b - x, 1)

def multivariate_gaussian(x, mu, sigma):
    """ 
    Implements uncorrelated multivariate Gaussian
    """
    Z = jnp.sqrt(2 * jnp.pi) * sigma
    separation_vectors = x[:, None] - mu[None, ...]
    return jnp.prod(jnp.exp(-0.5 * (separation_vectors) ** 2 / sigma ** 2) / Z, axis=-1)

def truncated_multivariate_gaussian(x, mu, sigma, a, b):
    """ 
    A multivariate truncated Gaussian with diagonal covariance matrix.
    """
    I = jnp.prod(indicator(x, a, b), axis=-1)
    mvg = multivariate_gaussian(x, mu, sigma)
    renormalization = jnp.prod(gaussian_CDF(b, mu, sigma) - gaussian_CDF(a, mu, sigma), axis=-1)
    return I * mvg / renormalization

def von_mises(x, mu, k, f):
    """ 
    Multivariate uncorrelated von-Mises
    """
    separation_vectors = x[:, None] - mu[None, ...]
    return jnp.prod(jnp.exp(k * jnp.cos(f * (separation_vectors))) * f / (2 * jnp.pi * jax.scipy.special.i0(k)), axis=-1)
