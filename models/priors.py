import jax.numpy as jnp
import numpy as np
from src.helper import rejection_sampling

MAX_ABS_CHI_PRIOR = 0.99

chi_prior = lambda chi: jnp.log(jnp.abs(MAX_ABS_CHI_PRIOR/chi))/(2.*MAX_ABS_CHI_PRIOR)

# TODO cos i uniform prior should be added later!!! 5/21/24
def minusLogPrior(x):
    """ 
    Prior for gravitational wave parameter estimation

    """
    Mc      = x[..., 0]
    #eta     = x[..., 1]
    q       = x[..., 1]
    dL      = x[..., 2]
    theta   = x[..., 3]
    phi     = x[..., 4]
    iota    = x[..., 5]
    psi     = x[..., 6]
    tcoal   = x[..., 7]
    Phicoal = x[..., 8]
    chi1z   = x[..., 9]
    chi2z   = x[..., 10] 

    # Uniform in m_1, m_2 prior for Mc, eta 
    V_prior = -jnp.log(Mc)      
    #V_prior += jnp.log(jnp.sqrt(1 - 4 * eta) * eta ** (6/5))
    # do q instead of eta
    V_prior += -jnp.log((1+q)**(2./5.) / q**(6./5.))

    # Power law in dL
    V_prior += -2 * jnp.log(dL)

    # sin priors
    V_prior += -jnp.log(jnp.sin(theta))
    V_prior += -jnp.log(jnp.sin(iota))

    # spin priors
    V_prior += -jnp.log(chi_prior(chi1z))
    V_prior += -jnp.log(chi_prior(chi2z))

    # V_prior += -jnp.log(jnp.log(jnp.abs(MAX_ABS_CHI_PRIOR/chi1z))/(2.*MAX_ABS_CHI_PRIOR))
    # V_prior += -jnp.log(jnp.log(jnp.abs(MAX_ABS_CHI_PRIOR/chi2z))/(2.*MAX_ABS_CHI_PRIOR))
    
    # Incorporate other priors here if desired
    # .
    # .
    # .

    return V_prior


cot = lambda x: jnp.cos(x) / jnp.sin(x)

def gradient_minusLogPrior(x): 
    """ 
    Gradient of prior
    
    """
    Mc      = x[..., 0]
    #eta     = x[..., 1]
    q       = x[..., 1]
    dL      = x[..., 2]
    theta   = x[..., 3]
    phi     = x[..., 4]
    iota    = x[..., 5]
    psi     = x[..., 6]
    tcoal   = x[..., 7]
    Phicoal = x[..., 8]
    chi1z   = x[..., 9]
    chi2z   = x[..., 10]

    # NOTE: This is updated as we add more priors

    # NOTE: Method assumes an N x d shaped array as input

    grad_V_prior = jnp.zeros((x.shape[0], 7))
    grad_V_prior = grad_V_prior.at[:, 0].set(-1 / Mc)
    # grad_V_prior = grad_V_prior.at[:, 1].set((-2 / (1 - 4 * eta) + 6 / (5 * eta))) # FIRST TRY
    #grad_V_prior = grad_V_prior.at[:, 1].set((6 - 34 * eta) / (5 * eta - 20 * eta ** 2)) # SECOND TRY


    #(WRONG)#### grad_V_prior = grad_V_prior.at[:, 1].set(-(6. + 8. * q) / (5. * q + 5. * q ** 2)) # THIRD TRY IN TERMS OF q (THIS IS WRONGGGGGGGGGGGGGGGGG)

    grad_V_prior = grad_V_prior.at[:, 1].set((6 + 4 * q) / (5 * q + 5 * q ** 2)) # THIRD TRY IN TERMS OF q

    
    # dL contribution to gradient of potential
    grad_V_prior = grad_V_prior.at[:, 2].set(-2 / dL)

    # sin prior contributions to the gradient of the potential

    grad_V_prior = grad_V_prior.at[:, 3].set(-cot(theta)) # theta
    grad_V_prior = grad_V_prior.at[:, 4].set(-cot(iota)) # iota

    # spin prior contributions to the gradient of the potential
    grad_V_prior = grad_V_prior.at[:, 5].set(MAX_ABS_CHI_PRIOR*jnp.sign(chi1z)/(chi1z**2 * jnp.abs(MAX_ABS_CHI_PRIOR/chi1z)*jnp.log(jnp.abs(MAX_ABS_CHI_PRIOR/chi1z)))) # chi1z
    grad_V_prior = grad_V_prior.at[:, 6].set(MAX_ABS_CHI_PRIOR*jnp.sign(chi2z)/(chi2z**2 * jnp.abs(MAX_ABS_CHI_PRIOR/chi2z)*jnp.log(jnp.abs(MAX_ABS_CHI_PRIOR/chi2z)))) # chi2z

    return grad_V_prior


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

def q_func(m1, m2):
    """ 
    Mass ratio in terms of m1, m2
    
    """
    return m2 / m1

# def Mc_eta_uniform_masses_draw(N, a, b):
#     """ 
#     Draws (Mc, eta) samples that are uniform in m1, m2

#     """
#     m = 1000000
#     m1_sams = np.random.uniform(low=10, high=80, size=m)
#     m2_sams = np.random.uniform(low=10, high=m1_sams) # NOTE: enforce m2 < m1
#     # m2_sams = np.random.uniform(low=10, high=80, size=m)

#     draw = np.zeros((m, 2))   
#     draw[:,0] = Mc_func(m1_sams, m2_sams)
#     #draw[:,1] = eta_func(m1_sams, m2_sams)
#     draw[:,1] = q_func(m1_sams, m2_sams)

#     return rejection_sampling(draw, a, b)[:N]

def Mc_q_uniform_masses_draw(N, a, b):
    """ 
    Draws (Mc, eta) samples that are uniform in m1, m2

    """
    m = 1000000
    m1_sams = np.random.uniform(low=10, high=80, size=m)
    m2_sams = np.random.uniform(low=10, high=m1_sams) # NOTE: enforce m2 < m1

    draw = np.zeros((m, 2))   
    draw[:,0] = Mc_func(m1_sams, m2_sams)
    draw[:,1] = q_func(m1_sams, m2_sams)

    return rejection_sampling(draw, a, b)[:N]



def dL_power_law_draw(N, a, b):
    """ 
    Draws r.v's with p(x) ~ x^2, a <= x <= b
    """
    u = np.random.uniform(size=N)
    return np.cbrt((a ** 3 + u * (b ** 3 - a ** 3)))

def inverse_cdf_sampling(fun, size, prange, res=1e5, **kwargs):
    """
    Sample from a probability distribution using inverse cdf sampling.

    :param function fun: PDF to sample from. It has to take an array of values as input and return an array containing their probabilities.
    :param int size: Size of the sample to generate.
    :param tuple prange: Range of the sample to generate.

    :return: Samples extracted from the desired PDF.
    :rtype: array
    """

    # generate uniform samples
    u = np.random.uniform(size=size)

    # compute inverse cdf
    x = np.linspace(prange[0],prange[1], int(res))
    y = fun(x, **kwargs)
    cdf_y = np.cumsum(y)
    cdf_y = cdf_y/cdf_y.max()
    res = np.interp(u, cdf_y,x)
    return res