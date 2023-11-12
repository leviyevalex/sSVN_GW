""" 
A toy TaylorF2 model for gradient and Hessian based parameter estimation

Version 0.2.1

Update History
--------------

0.1.0 - TaylorF2 method
0.1.1 - fixed units on chirp mass
0.2.0 - Cleaned up derivatives using phase expressions
0.2.1 - Used real assumptions in mathematica to further simplify expressions
"""


import jax.numpy as jnp
from jax import grad, config
import jax
import matplotlib.pyplot as plt
import os 
import numpy as np
from functools import partial
from opt_einsum import contract

config.update("jax_enable_x64", True)

class taylorf2:
    def __init__(self, injection, priorDict):
        self.injection = injection 
        self.priorDict = priorDict
        self.id = 'simple_model'
        self.DoF = 5 
        
        # Record evaluations
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0

        self.lower_bound = np.zeros(self.DoF)
        self.upper_bound = np.zeros(self.DoF)

        for i in range(self.DoF):
            self.lower_bound[i] = self.priorDict[i][0]
            self.upper_bound[i] = self.priorDict[i][1]
        self.lower_bound = jnp.array(self.lower_bound)
        self.upper_bound = jnp.array(self.upper_bound)
        self.index_label = [r'$t_c$', r'$\phi$', r'$\mathcal{M}_c$', r'$\eta$', r'$A$']

        ### Initializations ###

        # Defined fixed frequency grid
        self.fmin = 38
        self.fmax = 1024
        self.n_bins = 1000
        # self.n_bins = 6400
        self.frequency = jnp.linspace(self.fmin, self.fmax, num=self.n_bins+1)
        self.deltaf = (self.fmax - self.fmin) / self.n_bins

        self.m_sun_sec = 4.92549094830932e-6 # Conversion from solar masses to seconds (~5e-6)

        # Load noise data and store power spectral density over defined grid
        asd_path = os.path.join(os.getcwd(), 'aLIGO_O4_high_asd.txt')
        asd = np.loadtxt(asd_path, usecols=(0,1))
        freq_data = asd[:,0]
        psd_data = asd[:,1] ** 2
        self.PSD = jnp.interp(self.frequency, freq_data, psd_data, left=1., right=1.).squeeze() * 1e40 # NOTE TODO ##################### changed scale!!!!!!!

        # Precompute strain corresponding to injection 
        self.data = self.strain(self.injection, self.frequency)

        # Precompute signal to noise ratio
        self.SNR = self.signal_noise_ratio(self.injection)
        print('SNR is:', self.SNR)


        # Original strain
        # expr = (Amplitude * jnp.exp(-1j * (-(jnp.pi/4) + 2 * f * jnp.pi * time_coalescence + (
        #     3 * (1 + jnp.pi**(2/3) * ((f * Mc * m_sun_sec) / eta**(3/5))**(
        #         2/3) * (3715/756 + (55 * eta)/9))) / (
        #     128 * jnp.pi**(5/3) * ((f * Mc * m_sun_sec) / eta**(3/5))**(
        #         5/3) * eta) - phi))) / f**(7/6)

    def strain(self, x, f):
        tc = x[0]
        phi = x[1]
        Mc = x[2]
        eta = x[3]
        Amplitude = x[4]

        # New strain 
        expr = ((-1)**(1/4) * Amplitude * jnp.exp(-((3j)/(128 * (f * Mc * self.m_sun_sec)**(5/3) * jnp.pi**(5/3))) - 
                 2j * f * jnp.pi * tc - (5j * (743 + 924 * eta))/(32256 * f * Mc * self.m_sun_sec * jnp.pi * eta**(2/5)) + 1j * phi)) / f**(7/6)

        return expr

    def gradient_strain(self, x, f):
        Mc = x[2]
        eta = x[3]
        Amplitude = x[4]
        
        S = self.strain(x, f)
        
        expr1 = -2j * f * jnp.pi * S

        expr2 = 1j * S
        
        expr3 = (5j * f * self.m_sun_sec * S * (252 + jnp.pi**(2/3) * ((f * Mc * self.m_sun_sec) / eta**(3/5))**(2/3) * (743 + 924 * eta)))  \
        / (32256 * (f * Mc * self.m_sun_sec)**(8/3) * jnp.pi**(5/3))
        
        expr4 = -((1j * S * (-743 + 1386 * eta)) / (16128 * f * Mc * self.m_sun_sec * jnp.pi * eta**(7/5)))

        expr5 = S / Amplitude
        
        return jnp.array([expr1, expr2, expr3, expr4, expr5])

    # @partial(jax.jit, static_argnums=(0,))
    def potential_single(self, x):
        square_norm = lambda a, power_spectral_density, frequency_spacing: (4 * jnp.sum((a.real[..., :-1] ** 2 + a.imag[..., :-1] ** 2) / power_spectral_density[..., :-1] * frequency_spacing, axis=-1)).T

        residual = self.strain(x, self.frequency) - self.data

        return 0.5 * square_norm(residual, power_spectral_density=self.PSD, frequency_spacing=self.deltaf)

    def gradient_potential_single(self, x):
        overlap = lambda a, b, power_spectral_density, frequency_spacing: (4 * jnp.sum(a.conjugate()[..., :-1] * b[..., :-1] / power_spectral_density[..., :-1] * frequency_spacing, axis=-1)).T

        residual = self.strain(x, self.frequency) - self.data

        return overlap(self.gradient_strain(x, self.frequency), residual, power_spectral_density=self.PSD, frequency_spacing=self.deltaf).real

    def fisher_single(self, x):
        nabla_h = self.gradient_strain(x, self.frequency)
        inner_product = 4 * contract('if, jf, f -> ij', nabla_h.conjugate(), nabla_h, 1 / self.PSD) * self.deltaf
        return inner_product.real

    @partial(jax.jit, static_argnums=(0,))
    def fisher_ensemble(self, X):
        return jax.vmap(self.fisher_single)(X)

    # def potential(self, X):
    @partial(jax.jit, static_argnums=(0,))
    def getMinusLogPosterior_ensemble(self, X):
        return jax.vmap(self.potential_single)(X)

    # @partial(jax.jit, static_argnums=(0,))
    # def gradient_potential(self, X):
    #     # Direct derivative of potential
    #     # return jax.vmap(jax.jacrev(self.potential_single))(X)

    #     # Using cached Jacobian of strain
    #     return jax.vmap(self.gradient_potential_single)(X)

    def signal_noise_ratio(self, x):
        s = self.strain(x, self.frequency)
        square_norm = lambda a, power_spectral_density, frequency_spacing: (4 * jnp.sum((a.real[..., :-1] ** 2 + a.imag[..., :-1] ** 2) / power_spectral_density[..., :-1] * frequency_spacing, axis=-1)).T
        return np.sqrt(square_norm(s, self.PSD, self.deltaf))

    def _newDrawFromPrior(self, n):
        prior_draw = np.zeros((n, 5))
        for i in range(self.DoF): # Assuming uniform on all parameters
            low = self.priorDict[i][0]
            high = self.priorDict[i][1]
            buffer = (high-low) / 4
            # buffer = 0
            prior_draw[:, i] = np.random.uniform(low=low+buffer, high=high-buffer, size=n)
            # prior_draw[:, i] = np.random.uniform(low=self.true_params[i] - 1e-7, high=self.true_params[i] + 1e-7, size=n)
            # print('modified priors to be at mode!')
            print('buffer in prior: %f' % buffer)
        return jnp.array(prior_draw)
    
    def getDerivativesMinusLogPosterior_ensemble(self, X):
        n = X.shape[0]
        return self.gradient_potential(X), jnp.zeros((n, 5, 5))