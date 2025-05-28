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
from models.priors import minusLogPrior, gradient_minusLogPrior, Mc_q_uniform_masses_draw, dL_power_law_draw, inverse_cdf_sampling, chi_prior


# from astropy.cosmology import Planck18
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

# Stuff for real data
from gwpy.timeseries import TimeSeries
from scipy.signal.windows import tukey

# milliseconds_per_day = 3600. * 24 * 100 # milliseconds per day (FUDGED!!!)
milliseconds_per_day = 3600. * 24 * 1000 # milliseconds per day
eta_rescaling = 100
q_rescaling = 1
dL_rescaling = 1 # DONT CHANGE THIS

def eta_from_q(q):
    return q/(1+q)**2

def jac_q_to_eta(q):
    
    return (1.-q) / (1.+q)**3

class gwfast_LVGW150914(object):
    # def __init__(self, wf_model=TaylorF2_RestrictedPN, nbins=1000, fmin=10., fmax=560.):
    def __init__(self, wf_model=TaylorF2_RestrictedPN, fmin=20., fmax=512.):

            # Define model, parameter order convention, and explicitly label periodic vs bounded coordinates
            self.wf_model = wf_model()
            self.DoF = 11
            #self.gwfast_param_order = ['Mc', 'eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
            self.gwfast_param_order = ['Mc', 'q', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
            self.periodic_coordinates = jnp.array([4, 6, 8])
            self.bounded_coordinates = jnp.array([0, 1, 2, 3, 5, 7, 9, 10])

            # Define grid to evaluate frequency domain waveform
            self.fmin = fmin 
            self.fmax = fmax
            tGPS = np.round(np.array([1126259462.419288]))
            
            self.htrue = {}
            # Settings as in https://git.ligo.org/lscsoft/bilby/blob/master/examples/gw_examples/data_examples/GW150914.py 
            # NOTE: DO NOT MODIFY
            self.fgrid, self.htrue['L1'] = self.get_data_from_gwpy(ifo='L1', gps_time=tGPS[0], duration=4, post_trigger_duration=2, roll_off=0.4)
            _, self.htrue['H1']          = self.get_data_from_gwpy(ifo='H1', gps_time=tGPS[0], duration=4, post_trigger_duration=2, roll_off=0.4)
            
            self.htrue['L1'] = self.htrue['L1'][jnp.newaxis,:]
            self.htrue['H1'] = self.htrue['H1'][jnp.newaxis,:]
            
            self.nbins = len(self.fgrid)
            self.df = np.array(self.fgrid)[1] - np.array(self.fgrid)[0]

            # Point to which detector characteristics we want
            asd_paths = {}
            
            asd_paths['L1']    = '/home/al44828/projects/sSVN_GW/notebooks/LIGO_L_ASD_GW150914.txt'
            asd_paths['H1']    = '/home/al44828/projects/sSVN_GW/notebooks/LIGO_H_ASD_GW150914.txt'
            
            # Define network object
            self.Net = LV_DetNet(self.wf_model, fixed_fgrid=self.fgrid, verbose=True, ASDs=asd_paths)

            # Interpolate detector characteristics onto defined frequency grid
            self.PSDs = {}
            self.PSDs['L1']    = jnp.interp(self.fgrid, self.Net.signals['L1'].strainFreq, self.Net.signals['L1'].noiseCurve, left=1., right=1.).squeeze()
            self.PSDs['H1']    = jnp.interp(self.fgrid, self.Net.signals['H1'].strainFreq, self.Net.signals['H1'].noiseCurve, left=1., right=1.).squeeze()
            
            # LATEST CATELOG MEDIANS ( THESE ARE THE CORRECT ONES, TODO CHANGE LATER!!! )
            tcoal = float(utils.GPSt_to_LMST(tGPS, lat=0., long=0.)) * milliseconds_per_day
            injParams = {}
            injParams['Mc']      = np.array([30.68716026])                        # (0)   # [M_solar]      # Chirp mass
            #injParams['eta']     = np.array([0.2488933]) * eta_rescaling          # (1)   # [Unitless]     # Symmetric mass ratio
            injParams['q']       = np.array([0.8752328774395706]) * q_rescaling   # (1)   # [Unitless]     # Mass ratio
            injParams['dL']      = np.array([0.46752133]) * dL_rescaling          # (2)   # [Gigaparsecs]  # Luminosity distance
            injParams['theta']   = np.array([2.76406998])                         # (3)   # [Rad]          # Declination
            injParams['phi']     = np.array([2.0343809])                          # (4)   # [Rad]          # Right ascention
            injParams['iota']    = np.array([2.69895795])                         # (5)   # [Rad]          # Inclination
            injParams['psi']     = np.array([1.45278442])                         # (6)   # [Rad]          # Polarization angle
            injParams['tcoal']   = np.array([tcoal])                              # (7)   # [ms]           # Time of coalescence
            injParams['Phicoal'] = np.array([0.1])                                # (8)   # [Rad]          # Phase of coalescence
            injParams['chi1z']   = np.array([-0.0496784])                         # (9)   # [Unitless]     # Aligned spin 1
            injParams['chi2z']   = np.array([-0.00661958])                        # (10)  # [Unitless]     # Aligned spin 2

            self.injParams = injParams

            # Parameter bounds 
            bounds = {}
            bounds['Mc']      = [25., 35.]                      
            # bounds['eta']     = [0.20, 0.249]                 
            # bounds['eta']     = [0.20 * eta_rescaling, 0.249 * eta_rescaling]         
            #bounds['eta']     = [0.20 * eta_rescaling, 0.25 * eta_rescaling] 
            bounds['q']       = [0.38 * q_rescaling, 1. * q_rescaling]         
            bounds['dL']      = [0.05 * dL_rescaling, 2. * dL_rescaling]                     
            bounds['theta']   = [0., np.pi]                   
            bounds['phi']     = [0., 2 * np.pi]               
            bounds['iota']    = [0., np.pi]                   
            bounds['psi']     = [0., np.pi]                   
            # bounds['tcoal']   = [tcoal - 1, tcoal + 1]   # NOTE: This will need to be increased eventually     
            bounds['tcoal']   = [tcoal - 100, tcoal + 100]
            bounds['Phicoal'] = [0., 2 * np.pi]               
            bounds['chi1z']   = [-0.99, 0.99]                 
            bounds['chi2z']   = [-0.99, 0.99]                 
            self.bounds = bounds

            self.extraneous()

            self.true_params = jnp.array([self.injParams[param].squeeze() for param in self.gwfast_param_order])


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
    
    def get_data_from_gwpy(self, ifo, gps_time, duration=4, post_trigger_duration=2, roll_off=0.4):
        
        end_time = gps_time + post_trigger_duration
        start_time = end_time - duration
        
        # The roll-off (in seconds) used in the Tukey window corresponds to alpha * duration / 2 for scipy tukey window.
        tukey_alpha = 2 * roll_off / duration
        
        data_time = TimeSeries.fetch_open_data(ifo,
                                                start_time,
                                                end_time, cache=True)

        n = len(data_time)
        delta_t = data_time.dt.value  
        data_freq = jnp.fft.rfft(jnp.array(data_time.value) * tukey(n, tukey_alpha)) * delta_t
        freq = jnp.fft.rfftfreq(n, delta_t)
        
        frequencies = freq[(freq > self.fmin) & (freq < self.fmax)]
        data = data_freq[(freq > self.fmin) & (freq < self.fmax)]
        
        return frequencies, data

    def getSignal(self, X):
        """
        Wrapper for Francesco's GWStrain method

        Returns
        -------
        Dictonary of signals found in each detector

        """
        
        X_ = X.T.astype('complex128')
        signals = self.Net.GWstrain(Mc       = X_[0],
                                    #eta      = X_[1],
                                    eta      = eta_from_q(X_[1]), # NOTE: eta is derived from q now
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
                                               #eta     = X_[1],
                                               eta      = eta_from_q(X_[1]), # NOTE: eta is derived from q now
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
        # jacModel['Virgo'] = jacModel['Virgo'].at[9].divide(milliseconds_per_day)
        
        # NOTE: now the derivative is with respect to q, not eta, so we need to plug in the last jacobian element from eta to q
        jacModel['L1'] = jacModel['L1'].at[1].multiply(jac_q_to_eta(X_[1])[...,None]) 
        jacModel['H1'] = jacModel['H1'].at[1].multiply(jac_q_to_eta(X_[1])[...,None])

        # Switch parameter order (redundent in newer version of gwfast)
        jacModel['L1'] = jacModel['L1'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]
        jacModel['H1'] = jacModel['H1'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]
        # jacModel['Virgo'] = jacModel['Virgo'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]

        return jacModel

    # NOTE: I include the final bin the sum to make the code more readable.

    # NOTE: The overlap and square norm methods are adapated to Francesco's (d,N,f) Jacobian convention

    def square_norm(self, a, PSD, deltaf):
        """ 
        Square norm for single detector estimated using (left) Riemann sum
        """
        square_norm = (4 * deltaf * jnp.sum((a.real ** 2 + a.imag ** 2) / PSD, axis=-1)).T
        return square_norm

    def overlap(self, a, b, PSD, deltaf):
        """ 
        Network overlap estimated using (left) Riemann sum
        """
        overlap = (4 * deltaf * jnp.sum((a.conjugate() * b) / PSD, axis=-1)).T
        return overlap

    def overlap_GN(self, a, b, PSD, deltaf):
        tmp = 4 * deltaf * jnp.sum((a.conjugate()[None, ...] * b[:, None, ...]) / PSD, axis=-1)
        return jnp.swapaxes(tmp, 0, 2)

    def potential(self, X):
        """ 
        Calculates potential V(x) = -ln(likelihood(x)) - ln(prior(x))

        """
        # TODO: DOUBLE CHECK THAT THIS ISNT OVERWRITING PARTICLES OUTSIDE OF THIS METHOD


        #X = X.at[:, 1].divide(eta_rescaling)
        X = X.at[:, 1].divide(q_rescaling)
        X = X.at[:, 2].divide(dL_rescaling)

        # Calculate residuals
        template = self.getSignal(X)
        residual = {}
        residual['L1'] = template['L1'] - self.htrue['L1']
        residual['H1'] = template['H1'] - self.htrue['H1']
        # residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

        # Likelihood contribution to energy
        V  = 0.5 * self.square_norm(residual['L1'], self.PSDs['L1'], self.df)
        V += 0.5 * self.square_norm(residual['H1'], self.PSDs['H1'], self.df)
        # V += 0.5 * self.square_norm(residual['Virgo'], self.PSDs['Virgo'], self.df)

        # Prior contribution to energy
        V += minusLogPrior(X)

        return V
    
    def gradient_potential(self, X): 
        """ 
        Calculates the gradient of the potential
        
        """

        #X = X.at[:, 1].divide(eta_rescaling)
        X = X.at[:, 1].divide(q_rescaling)
        X = X.at[:, 2].divide(dL_rescaling)

        # Calculate residuals
        template = self.getSignal(X)
        residual = {}
        residual['L1'] = template['L1'] - self.htrue['L1']
        residual['H1'] = template['H1'] - self.htrue['H1']
        # residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

        # Gradient of likelihood 
        jacSignal = self._getJacobianSignal(X)
        grad_V = self.overlap(jacSignal['L1'], residual['L1'], self.PSDs['L1'], self.df).real
        grad_V += self.overlap(jacSignal['H1'], residual['H1'], self.PSDs['H1'], self.df).real
        # grad_V += self.overlap(jacSignal['Virgo'], residual['Virgo'], self.PSDs['Virgo'], self.df).real

        # Gradient of prior
        grad_V = grad_V.at[:, jnp.array([0, 1, 2, 3, 5, 9, 10])].add(gradient_minusLogPrior(X))

        # NOTE: As priors are added this will need to be updated as well!

        #grad_V = grad_V.at[:, 1].divide(eta_rescaling)
        grad_V = grad_V.at[:, 1].divide(q_rescaling)
        grad_V = grad_V.at[:, 2].divide(dL_rescaling)

        return grad_V

    def gradient_potential_GAUSS(self, X): 
        """ 
        Calculates the gradient of the potential
        
        """

        X = X.at[:, 1].divide(eta_rescaling)
        X = X.at[:, 2].divide(dL_rescaling)

        # Calculate residuals
        template = self.getSignal(X)
        residual = {}
        residual['L1'] = template['L1'] - self.htrue['L1']
        residual['H1'] = template['H1'] - self.htrue['H1']
        # residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

        # Gradient of likelihood 
        jacSignal = self._getJacobianSignal(X)
        grad_V = self.overlap(jacSignal['L1'], residual['L1'], self.PSDs['L1'], self.df).real
        grad_V += self.overlap(jacSignal['H1'], residual['H1'], self.PSDs['H1'], self.df).real
        # grad_V += self.overlap(jacSignal['Virgo'], residual['Virgo'], self.PSDs['Virgo'], self.df).real

        # Gradient of prior
        grad_V = grad_V.at[:, jnp.array([0, 1])].add(gradient_minusLogPrior(X))

        # NOTE: As priors are added this will need to be updated as well!

        grad_V = grad_V.at[:, 1].divide(eta_rescaling)
        grad_V = grad_V.at[:, 2].divide(dL_rescaling)

        #######################################
        # Gauss-Newton approximation to Hessian
        #######################################

        gauss_newton = self.overlap_GN(jacSignal['L1'], jacSignal['L1'], self.PSDs['L1'], self.df).real
        gauss_newton += self.overlap_GN(jacSignal['H1'], jacSignal['H1'], self.PSDs['H1'], self.df).real
        # gauss_newton += self.overlap_GN(jacSignal['Virgo'], jacSignal['Virgo'], self.PSDs['Virgo'], self.df).real

        gauss_newton = gauss_newton.at[:, :, 1].divide(eta_rescaling)
        gauss_newton = gauss_newton.at[:, :, 2].divide(dL_rescaling)

        gauss_newton = gauss_newton.at[:, 1, :].divide(eta_rescaling)
        gauss_newton = gauss_newton.at[:, 2, :].divide(dL_rescaling)

        # gauss_newton = gauss_newton.at[:, 0, 0].add(X[:,0] ** 2) # Hessian of prior on Mc



        return grad_V, gauss_newton


    # def _newDrawFromPrior(self, n, seed=42):
    #     prior_samples = np.zeros((n, self.DoF))
    #     for i in range(self.DoF):
    #         prior_samples[:, i] = np.random.uniform(low=self.lower_bound[i], high=self.upper_bound[i], size=n)

    #     # Draw samples from prior law
    #     prior_samples[:, 2] = dL_power_law_draw(n, self.lower_bound[2], self.upper_bound[2])
        
    #     # eta rescaling adjustment + uniform in m1, m2
    #     # a = jnp.copy(self.lower_bound).at[1].divide(eta_rescaling)[0:2]
    #     # b = jnp.copy(self.upper_bound).at[1].divide(eta_rescaling)[0:2]

    #     # q rescaling
    #     a = jnp.copy(self.lower_bound).at[1].divide(q_rescaling)[0:2]
    #     b = jnp.copy(self.upper_bound).at[1].divide(q_rescaling)[0:2]


    #     prior_samples[:, 0:2] = Mc_eta_uniform_masses_draw(n, a, b)
    #     # prior_samples[:,1] *= eta_rescaling

    #     prior_samples[:,1] *= q_rescaling

    #     # uniform in sin samples

    #     return jnp.array(prior_samples)


    def generate_noise_from_asd(self, asd_freq, asd_val, freqs, seed=None):
        '''
        Generate frequency domain noise from a given ASD curve at the input frequencies. NOTE: It assumes linear spacing of the frequencies.

        :param array asd_freq: The frequencies at which the ASD is provided.
        :param array asd_val: The ASD values at ``asd_freq``.
        :param array freqs: The frequencies at which the noise is to be generated.
        :param int seed: The seed for the random number generator.
        
        :return: The generated noise in the frequency domain.
        :rtype: array
        '''
        if seed is not None:
            np.random.seed(seed)

        strainGrids = np.interp(freqs, asd_freq, asd_val, left=1., right=1.)
        scale = 0.5 * strainGrids/np.sqrt(freqs[1] - freqs[0])
        
        nre = np.random.normal(0., scale)
        nco = np.random.normal(0., scale)

        return nre + 1j*nco

    def _newDrawFromPrior(self, n, seed=42):
        prior_samples = np.zeros((n, self.DoF))
        for i in range(self.DoF):
            prior_samples[:, i] = np.random.uniform(low=self.lower_bound[i], high=self.upper_bound[i], size=n)

        # Draw samples from prior law
        # TODO NOTE : UNCOMMENT THIS OUT LATER!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # prior_samples[:, 2] = dL_power_law_draw(n, self.lower_bound[2], self.upper_bound[2])
        
        # eta rescaling adjustment + uniform in m1, m2
        # a = jnp.copy(self.lower_bound).at[1].divide(eta_rescaling)[0:2]
        # b = jnp.copy(self.upper_bound).at[1].divide(eta_rescaling)[0:2]

        # q rescaling
        a = jnp.copy(self.lower_bound).at[1].divide(q_rescaling)[0:2]
        b = jnp.copy(self.upper_bound).at[1].divide(q_rescaling)[0:2]


        prior_samples[:, 0:2] = Mc_q_uniform_masses_draw(n, a, b)
        # prior_samples[:,1] *= eta_rescaling

        prior_samples[:,1] *= q_rescaling

        # uniform in cos samples
        prior_samples[:, np.array([3, 5])] = np.arccos(np.random.uniform(low=-1, high=1, size=(n,2)))

        # spin prior samples
        prior_samples[:,9] = inverse_cdf_sampling(chi_prior, n, (self.lower_bound[9], self.upper_bound[9]))
        prior_samples[:,10] = inverse_cdf_sampling(chi_prior, n, (self.lower_bound[10], self.upper_bound[10]))


        return jnp.array(prior_samples)