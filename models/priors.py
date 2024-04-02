import jax.numpy as jnp
import numpy as np
from src.helper import rejection_sampling

def minusLogPrior(x):
    """ 
    Prior for gravitational wave parameter estimation

    """
    Mc      = x[0]
    eta     = x[1]
    dL      = x[2]
    theta   = x[3]
    phi     = x[4]
    iota    = x[5]
    psi     = x[6]
    tcoal   = x[7]
    Phicoal = x[8]
    chi1z   = x[9]
    chi2z   = x[10] 

    # Uniform in m_1, m_2 prior for Mc, eta 
    V_prior = -jnp.log(Mc)      
    V_prior += jnp.log(jnp.sqrt(1 - 4 * eta) * eta ** (6/5))

    return V_prior

def Mc_func(m1, m2):
    """ 
    Chirp mass in terms of m1, m2
    
    """
    return (m1 * m2) ** (3 / 5) / ((m1 + m2) ** (1/5))

def eta_func(m1, m2):
    """ 
    Symmetric mass ratio in terms of m1, m2
    
    """
    return (m1 * m2) / ((m1 + m2) ** (2))

def Mc_eta_uniform_masses_draw(N, a, b):
    """ 
    Draws (Mc, eta) samples that are uniform in m1, m2

    """
    m = 1000000
    m1_sams = np.random.uniform(low=10, high=80, size=m)
    m2_sams = np.random.uniform(low=10, high=80, size=m)

    draw = np.zeros((m, 2))   
    draw[:,0] = Mc_func(m1_sams, m2_sams)
    draw[:,1] = eta_func(m1_sams, m2_sams)

    return rejection_sampling(draw, a, b)[:N]