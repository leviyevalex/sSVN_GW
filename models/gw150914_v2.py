import copy
import jax.numpy as jnp
import jax
import numpy as np
import os, sys

sys.path.append("...")
sys.path.append(".")
sys.path.append("..")
from   models.gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD
import models.gwfast.signal as signal
from   models.gwfast.network import DetNet, LV_DetNet
import models.gwfast.gwfastGlobals as glob
import models.gwfast.gwfastUtils as utils
from models.priors import minusLogPrior, gradient_minusLogPrior, Mc_eta_uniform_masses_draw


# from astropy.cosmology import Planck18
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

milliseconds_per_day = 3600. * 24 * 1000 # milliseconds per day
eta_rescaling = 100

class gwfast_LVGW150914(object):
    def __init__(self, wf_model=TaylorF2_RestrictedPN, nbins=1000, fmin=10., fmax=560.):

            # Define model, parameter order convention, and explicitly label periodic vs bounded coordinates
            self.wf_model = wf_model()
            self.DoF = 11
            self.gwfast_param_order = ['Mc', 'eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
            self.periodic_coordinates = jnp.array([4, 6, 8])
            self.bounded_coordinates = jnp.array([0, 1, 2, 3, 5, 7, 9, 10])

            # Define grid to evaluate frequency domain waveform
            self.fmin = fmin 
            self.fmax = fmax
            self.nbins = nbins
            self.fgrid = jnp.linspace(self.fmin, self.fmax, num=self.nbins + 1).squeeze()
            self.df = (self.fgrid[-1] - self.fgrid[0]) / self.nbins

            # Point to which detector characteristics we want
            asd_paths = {}
            asd_paths['L1']    = '/home/al44828/projects/sSVN_GW/notebooks/aLIGO_O4_high_asd.txt'
            asd_paths['H1']    = '/home/al44828/projects/sSVN_GW/notebooks/aLIGO_O4_high_asd.txt'
            asd_paths['Virgo'] = '/home/al44828/projects/sSVN_GW/notebooks/AdV_asd.txt'

            # Define network object
            self.Net = LV_DetNet(self.wf_model, fixed_fgrid=self.fgrid, verbose=True, ASDs=asd_paths)

            # Interpolate detector characteristics onto defined frequency grid
            self.PSDs = {}
            self.PSDs['L1']    = jnp.interp(self.fgrid, self.Net.signals['L1'].strainFreq, self.Net.signals['L1'].noiseCurve, left=1., right=1.).squeeze()
            self.PSDs['H1']    = jnp.interp(self.fgrid, self.Net.signals['H1'].strainFreq, self.Net.signals['H1'].noiseCurve, left=1., right=1.).squeeze()
            self.PSDs['Virgo'] = jnp.interp(self.fgrid, self.Net.signals['Virgo'].strainFreq, self.Net.signals['Virgo'].noiseCurve, left=1., right=1.).squeeze()

            # Injection parameters (GW150914)
            tGPS = np.array([1.1262594624e+09])
            tcoal = float(utils.GPSt_to_LMST(tGPS, lat=0., long=0.)) * milliseconds_per_day
            injParams = {}
            injParams['Mc']      = np.array([31.39])               # (0)   # [M_solar]      # Chirp mass
            injParams['eta']     = np.array([0.2485773]) * eta_rescaling           # (1)   # [Unitless]     # Symmetric mass ratio
            injParams['dL']      = np.array([0.43929])             # (2)   # [Gigaparsecs]  # Luminosity distance
            injParams['theta']   = np.array([2.78560281])          # (3)   # [Rad]          # Declination
            injParams['phi']     = np.array([1.67687425])          # (4)   # [Rad]          # Right ascention
            injParams['iota']    = np.array([2.67548653])          # (5)   # [Rad]          # Inclination
            injParams['psi']     = np.array([0.78539816])          # (6)   # [Rad]          # Polarization angle
            injParams['tcoal']   = np.array([tcoal])               # (7)   # [ms]           # Time of coalescence
            injParams['Phicoal'] = np.array([0.1])                 # (8)   # [Rad]          # Phase of coalescence
            injParams['chi1z']   = np.array([0.27210419])          # (9)   # [Unitless]     # Aligned spin 1
            injParams['chi2z']   = np.array([0.33355909])          # (10)  # [Unitless]     # Aligned spin 2
            self.injParams = injParams

            # Parameter bounds 
            bounds = {}
            bounds['Mc']      = [25., 35.]                      
            # bounds['eta']     = [0.20, 0.249]                 
            bounds['eta']     = [0.20 * eta_rescaling, 0.249 * eta_rescaling]         
            bounds['dL']      = [0.25, 2.]                     
            bounds['theta']   = [0., np.pi]                   
            bounds['phi']     = [0., 2 * np.pi]               
            bounds['iota']    = [0., np.pi]                   
            bounds['psi']     = [0., np.pi]                   
            bounds['tcoal']   = [tcoal - 1, tcoal + 1]        
            bounds['Phicoal'] = [0., 2 * np.pi]               
            bounds['chi1z']   = [-0.99, 0.99]                 
            bounds['chi2z']   = [-0.99, 0.99]                 
            self.bounds = bounds

            # Get mock data
            self.true_params = jnp.array([self.injParams[param].squeeze() for param in self.gwfast_param_order])
            # self.htrue = self.getSignal(self.true_params[None,:])
            self.htrue = self.getSignal(self.true_params.at[1].divide(eta_rescaling)[None,:])


            self.extraneous()

    def extraneous(self):
        """ 
        Print metadata and define variables needed to interface with sampler
        
        """

        print('fmin: ', self.fmin)
        print('fmax: ', self.fmax)
        print('df_standard: ', self.df)
        print('nbins: ', self.nbins)

        # Definitions for easy interfacing
        self.lower_bound = jnp.array([self.bounds[param][0] for param in self.gwfast_param_order]) 
        self.upper_bound = jnp.array([self.bounds[param][1] for param in self.gwfast_param_order]) 

        # Mock data signal-to-noise ratio
        self.snr  = self.square_norm(self.htrue['L1'], self.PSDs['L1'], self.df)
        self.snr += self.square_norm(self.htrue['H1'], self.PSDs['H1'], self.df)
        self.snr += self.square_norm(self.htrue['Virgo'], self.PSDs['Virgo'], self.df)

        print('SNR at true values: %.2f' % jnp.sqrt(self.snr)[0])

    def getSignal(self, X):
        """
        Wrapper for Francesco's GWStrain method

        Returns
        -------
        Dictonary of signals found in each detector

        """
        
        X_ = X.T.astype('complex128')
        signals = self.Net.GWstrain(Mc       = X_[0],
                                    eta      = X_[1],
                                    dL       = X_[2],
                                    theta    = X_[3],
                                    phi      = X_[4],
                                    iota     = X_[5],
                                    psi      = X_[6],
                                    tcoal    = X_[7] / milliseconds_per_day,
                                    Phicoal  = X_[8],
                                    chi1z    = X_[9],
                                    chi2z    = X_[10])
                            
        return signals 

    def _getJacobianSignal(self, X):
        """
        Wrapper for Francesco's SignalDerivatives method

        Returns
        -------
        A (d, N, f) shaped array 
        """

        X_ = X.T.astype('complex128')
        jacModel = self.Net._SignalDerivatives(Mc      = X_[0],
                                               eta     = X_[1],
                                               dL      = X_[2],
                                               theta   = X_[3],
                                               phi     = X_[4],
                                               iota    = X_[5],
                                               psi     = X_[6],
                                               tcoal   = X_[7] / milliseconds_per_day, # Correction 1
                                               Phicoal = X_[8],
                                               chi1z   = X_[9],
                                               chi2z   = X_[10]) 
        # Correction 2
        jacModel['L1'] = jacModel['L1'].at[9].divide(milliseconds_per_day) 
        jacModel['H1'] = jacModel['H1'].at[9].divide(milliseconds_per_day)
        jacModel['Virgo'] = jacModel['Virgo'].at[9].divide(milliseconds_per_day)

        # Switch parameter order (redundent in newer version of gwfast)
        jacModel['L1'] = jacModel['L1'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]
        jacModel['H1'] = jacModel['H1'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]
        jacModel['Virgo'] = jacModel['Virgo'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]

        return jacModel

    # NOTE: I include the final bin the sum to make the code more readable.

    # NOTE: The overlap and square norm methods are adapated to Francesco's (d,N,f) Jacobian convention

    def square_norm(self, a, PSD, deltaf):
        """ 
        Square norm for single detector estimated using (left) Riemann sum
        """
        square_norm = (4 * jnp.sum((a.real ** 2 + a.imag ** 2) / PSD * deltaf, axis=-1)).T
        return square_norm

    def overlap(self, a, b, PSD, deltaf):
        """ 
        Network overlap estimated using (left) Riemann sum
        """
        overlap = (4 * jnp.sum(a.conjugate() * b / PSD * deltaf, axis=-1)).T
        return overlap

    def potential(self, X):
        """ 
        Calculates potential V(x) = -ln(likelihood(x)) - ln(prior(x))

        """
        X = X.at[:, 1].divide(eta_rescaling)

        # Calculate residuals
        template = self.getSignal(X)
        residual = {}
        residual['L1'] = template['L1'] - self.htrue['L1']
        residual['H1'] = template['H1'] - self.htrue['H1']
        residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

        # Likelihood contribution to energy
        V  = 0.5 * self.square_norm(residual['L1'], self.PSDs['L1'], self.df)
        V += 0.5 * self.square_norm(residual['H1'], self.PSDs['H1'], self.df)
        V += 0.5 * self.square_norm(residual['Virgo'], self.PSDs['Virgo'], self.df)

        # Prior contribution to energy
        V += minusLogPrior(X)

        return V
    
    def gradient_potential(self, X): 
        """ 
        Calculates the gradient of the potential
        
        """

        X = X.at[:, 1].divide(eta_rescaling)

        # Calculate residuals
        template = self.getSignal(X)
        residual = {}
        residual['L1'] = template['L1'] - self.htrue['L1']
        residual['H1'] = template['H1'] - self.htrue['H1']
        residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

        # Gradient of likelihood 
        jacSignal = self._getJacobianSignal(X)
        grad_V = self.overlap(jacSignal['L1'], residual['L1'], self.PSDs['L1'], self.df).real
        grad_V += self.overlap(jacSignal['H1'], residual['H1'], self.PSDs['H1'], self.df).real
        grad_V += self.overlap(jacSignal['Virgo'], residual['Virgo'], self.PSDs['Virgo'], self.df).real

        # Gradient of prior
        grad_V = grad_V.at[:, jnp.array([0, 1])].add(gradient_minusLogPrior(X))

        # NOTE: As priors are added this will need to be updated as well!

        grad_V = grad_V.at[:, 1].divide(eta_rescaling)

        return grad_V

    def _newDrawFromPrior(self, n, seed=42):
        prior_samples = np.zeros((n, self.DoF))
        for i in range(self.DoF):
            prior_samples[:, i] = np.random.uniform(low=self.lower_bound[i], high=self.upper_bound[i], size=n)
        
        # eta rescaling adjustment
        a = jnp.copy(self.lower_bound).at[1].divide(eta_rescaling)[0:2]
        b = jnp.copy(self.upper_bound).at[1].divide(eta_rescaling)[0:2]
        prior_samples[:, 0:2] = Mc_eta_uniform_masses_draw(n, a, b)
        prior_samples[:,1] *= eta_rescaling

        return jnp.array(prior_samples)

    # def _newDrawFromPrior(self, n, seed=42):
    #     prior_draw = jnp.zeros((len(self.gwfast_param_order), n))
    #     key = jax.random.PRNGKey(seed)
    #     for i, param in enumerate(self.gwfast_param_order): # Assuming uniform on all parameters         
    #         buffer = 0
    #         prior_draw = prior_draw.at[i].set(jax.random.uniform(key, (n,), minval=self.priorDict[param][0]+buffer, maxval=self.priorDict[param][1]-buffer))
    #         key, subkey = jax.random.split(key)
    #     if self.verbose:
    #         print('buffer in prior: %f' % buffer)
    #     return prior_draw.T