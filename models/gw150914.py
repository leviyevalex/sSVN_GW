#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%
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

# from astropy.cosmology import Planck18
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

# TODO CHANGED THIS!!!


class gwfast_LVGW150914(object):

    def __init__(self, wf_model='TaylorF2', verbose=False, nbins=1000, fmin=10.):
        """
        Args:
            wf_model (WaveFormModel): waveform model to use
            verbose (bool): print verbose output
            nbins (int): number of frequency bins to use
        """
        self.time_fudge_factor = 1000
        self.eta_fudge_factor = 1
        self.seconds_per_day = 3600.*24. * self.time_fudge_factor


        # sSVN_GW related attributes
        self.verbose = verbose
        self.id = 'gwfast_model'
        if wf_model == 'TaylorF2':
            self.wf_model = TaylorF2_RestrictedPN()
            if verbose:
                print('Using waveform model: TaylorF2')
        elif wf_model == 'IMRPhenomD':
            self.wf_model = IMRPhenomD()
            if verbose:
                print('Using waveform model: IMRPhenomD')
        else:
            raise ValueError('Waveform model not implemented yet.')
        
        self._initParams()

        # Parameter order convention
        self.gwfast_param_order = ['Mc','eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
        self.DoF = len(self.gwfast_param_order)

        self.periodic_coordinates = jnp.array([4, 6, 8])
        self.bounded_coordinates = jnp.array([0, 1, 2, 3, 5, 7, 9, 10])

        # Definitions for easy interfacing
        self.true_params = jnp.array([self.injParams[param].squeeze() for param in self.gwfast_param_order])
        self.lower_bound = jnp.array([self.priorDict[param][0] for param in self.gwfast_param_order]) 
        self.upper_bound = jnp.array([self.priorDict[param][1] for param in self.gwfast_param_order]) 

        self.nbins = nbins
        self.fmin = fmin
        self._initFrequencyGrid()
        self._initDetectors()

        self.htrue = self.getSignal(self.true_params[None,:])
        if verbose:
            snr = self.square_norm(self.htrue['L1'], self.PSDs['L1'], self.df)
            snr = snr + self.square_norm(self.htrue['H1'], self.PSDs['H1'], self.df)
            snr = snr + self.square_norm(self.htrue['Virgo'], self.PSDs['Virgo'], self.df)
            print('SNR at true values: %.2f'%jnp.sqrt(snr)[0])

    def _initParams(self):
        """ 
        Remarks:
        (i)   `tcoal` is accepted in units GMST fraction of a day
        (ii)  GPSt_to_LMST returns GMST in units of fraction of day (GMST is LMST computed at long = 0°)
        (iv)  Use [tcoal - 3e-7, tcoal + 3e-7] prior when in units of days

        """
        injParams = {}
        priorDict = {}

        # GW150914
        tGPS = np.array([1.1262594624e+09])
        tcoal = float(utils.GPSt_to_LMST(tGPS, lat=0., long=0.)) * self.seconds_per_day # [0, 1] 
        injParams['Mc']      = np.array([31.39])               # (1)   # (0)   # [M_solar]      # Chirp mass
        injParams['eta']     = np.array([0.2485773]) * self.eta_fudge_factor           # (2)   # (1)   # [Unitless]     # Symmetric mass ratio
        # injParams['eta']     = np.array([0.2485773])           # (2)   # (1)   # [Unitless]     # Symmetric mass ratio
        injParams['dL']      = np.array([0.43929])             # (3)   # (2)   # [Gigaparsecs]  # Luminosity distance
        injParams['theta']   = np.array([2.78560281])          # (4)   # (3)   # [Rad]          # Declination
        injParams['phi']     = np.array([1.67687425])          # (5)   # (4)   # [Rad]          # Right ascention
        injParams['iota']    = np.array([2.67548653])          # (6)   # (5)   # [Rad]          # Inclination
        injParams['psi']     = np.array([0.78539816])          # (7)   # (6)   # [Rad]
        injParams['tcoal']   = np.array([tcoal])               # (8)   # (7)   # [sec]
        injParams['Phicoal'] = np.array([0.1])                 # (9)   # (8)   # [Rad]
        injParams['chi1z']   = np.array([0.27210419])          # (10)  # (9)   # [Unitless]
        injParams['chi2z']   = np.array([0.33355909])          # (11)  # (10)  # [Unitless]

        # TODO sample in cos iota (uniform prior on cosi) 
        # cos(\iota)  ~ Unif[-1,1]
        # cos(\theta) ~ Unif[-1,1]

        priorDict = {}
        # Use these for testing
        priorDict['Mc']      = [25, 35]                            # [M_solar]     
        priorDict['eta']     = [0.20 * self.eta_fudge_factor, 0.249 * self.eta_fudge_factor]                         # [Unitless]
        # priorDict['dL']      = [0.05, 2]                           # [GPC]
        priorDict['dL']      = [0.25, 2]                           # [GPC]
        priorDict['theta']   = [0., np.pi]                         # [Rad]
        priorDict['phi']     = [0., 2 * np.pi]                     # [Rad]
        priorDict['iota']    = [0., np.pi]                         # [Rad] # Note: Maybe use cos i variable?
        priorDict['psi']     = [0., np.pi]                         # [Rad]
        delta_t = 0.001 * self.time_fudge_factor
        priorDict['tcoal']   = [tcoal - delta_t, tcoal + delta_t]          # [sec]
        priorDict['Phicoal'] = [0., 2 * np.pi]                     # [Rad]
        priorDict['chi1z']   = [-0.99, 0.99]                       # [Unitless]
        priorDict['chi2z']   = [-0.99, 0.99]                       # [Unitless]

        self.priorDict = priorDict
        self.injParams = injParams

    def _initFrequencyGrid(self): # Checks: X
        """
        Setup frequency grids that will be used

        """
        #TODO

        # self.fmax = self.wf_model.fcut(**self.injParams)[0] 
        self.fmax = 560.


        self.fgrid = jnp.linspace(self.fmin, self.fmax, num=self.nbins + 1).squeeze()
        self.df = (self.fgrid[-1] - self.fgrid[0]) / self.nbins

        if self.verbose:
            print('fmin: ', self.fmin)
            print('fmax: ', self.fmax)
            print('df_standard: ', self.df)
            print('nbins: ', self.nbins)

    def _initDetectors(self): 
        """Initialize detectors and store PSD interpolated over defined frequency grid

        """

        self.Net = LV_DetNet(self.wf_model, fixed_fgrid=self.fgrid, verbose=self.verbose)

        self.PSDs = {}
        self.PSDs['L1'] = jnp.interp(self.fgrid, self.Net.signals['L1'].strainFreq, self.Net.signals['L1'].noiseCurve, left=1., right=1.).squeeze()
        self.PSDs['H1'] = jnp.interp(self.fgrid, self.Net.signals['H1'].strainFreq, self.Net.signals['H1'].noiseCurve, left=1., right=1.).squeeze()
        self.PSDs['Virgo'] = jnp.interp(self.fgrid, self.Net.signals['Virgo'].strainFreq, self.Net.signals['Virgo'].noiseCurve, left=1., right=1.).squeeze()
    
    def getSignal(self, X):
        """Method to calculate signal for each X[i] over f_grid in detector det

        """
        
        X_ = X.T.astype('complex128')
        signals = self.Net.GWstrain(Mc       = X_[0],
                                    # eta      = X_[1],
                                    eta      = X_[1] / self.eta_fudge_factor,
                                    dL       = X_[2],
                                    theta    = X_[3],
                                    phi      = X_[4],
                                    iota     = X_[5],
                                    psi      = X_[6],
                                    tcoal    = X_[7] / self.seconds_per_day,
                                    Phicoal  = X_[8],
                                    chi1z    = X_[9],
                                    chi2z    = X_[10])
                            
        return signals 

    def _getJacobianSignal(self, X):
        """A vectorized method which computes the Jacobian of the signal model

        Parameters
        ----------
        X : array
            (N, d) shaped array of particle positions

        Returns
        -------
        array
            gwfast returns a (d, N, f) shaped array 
        """
        X_ = X.T.astype('complex128')
        jacModel = self.Net._SignalDerivatives(Mc      = X_[0],
                                            #    eta     = X_[1],
                                               eta     = X_[1] / self.eta_fudge_factor,
                                               dL      = X_[2],
                                               theta   = X_[3],
                                               phi     = X_[4],
                                               iota    = X_[5],
                                               psi     = X_[6],
                                               tcoal   = X_[7] / self.seconds_per_day, # Correction 1
                                               Phicoal = X_[8],
                                               chi1z   = X_[9],
                                               chi2z   = X_[10]) 

        # jacModel['L1'] = jacModel['L1'].at[7].divide(seconds_per_day) # Correction 2
        # jacModel['H1'] = jacModel['H1'].at[7].divide(seconds_per_day)
        # jacModel['Virgo'] = jacModel['Virgo'].at[7].divide(seconds_per_day)
        # return jacModel

        # NOTE: Francesco's suggested fix
        jacModel['L1'] = jacModel['L1'].at[9].divide(self.seconds_per_day) # Correction 2
        jacModel['H1'] = jacModel['H1'].at[9].divide(self.seconds_per_day)
        jacModel['Virgo'] = jacModel['Virgo'].at[9].divide(self.seconds_per_day)

        jacModel['L1'] = jacModel['L1'].at[1].divide(self.eta_fudge_factor) # Correction 2
        jacModel['H1'] = jacModel['H1'].at[1].divide(self.eta_fudge_factor)
        jacModel['Virgo'] = jacModel['Virgo'].at[1].divide(self.eta_fudge_factor)

        jacModel['L1'] = jacModel['L1'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]
        jacModel['H1'] = jacModel['H1'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]
        jacModel['Virgo'] = jacModel['Virgo'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]

        return jacModel
    
    def square_norm(self, a, PSD, deltaf):
        """ 
        Square norm for single detector estimated using left Riemann sum
        """
        square_norm = (4 * jnp.sum((a.real[..., :-1] ** 2 + a.imag[..., :-1] ** 2) / PSD[..., :-1] * deltaf, axis=-1)).T
        return square_norm

    def overlap(self, a, b, PSD, deltaf):
        """ 
        Network overlap estimated using left Riemann sum
        """
        overlap = (4 * jnp.sum(a.conjugate()[..., :-1] * b[..., :-1] / PSD[..., :-1] * deltaf, axis=-1)).T
        return overlap

    def _newDrawFromPrior(self, n, seed=42):
        prior_draw = jnp.zeros((len(self.gwfast_param_order), n))
        key = jax.random.PRNGKey(seed)
        for i, param in enumerate(self.gwfast_param_order): # Assuming uniform on all parameters         
            buffer = 0
            prior_draw = prior_draw.at[i].set(jax.random.uniform(key, (n,), minval=self.priorDict[param][0]+buffer, maxval=self.priorDict[param][1]-buffer))
            key, subkey = jax.random.split(key)
        if self.verbose:
            print('buffer in prior: %f' % buffer)
        return prior_draw.T
    
    # @partial(jax.jit, static_argnums=(0,))
    def minusLogLikelihood(self, X): 

        template = self.getSignal(X)
        residual = {}
        residual['L1'] = template['L1'] - self.htrue['L1']
        residual['H1'] = template['H1'] - self.htrue['H1']
        residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

        log_likelihood = 0.5 * self.square_norm(residual['L1'], self.PSDs['L1'], self.df)
        log_likelihood = log_likelihood + 0.5 * self.square_norm(residual['H1'], self.PSDs['H1'], self.df)
        log_likelihood = log_likelihood + 0.5 * self.square_norm(residual['Virgo'], self.PSDs['Virgo'], self.df)

        # New prior contributions TODO: REMOVE AFTER?

        eta = jnp.copy(X[:,1] / self.eta_fudge_factor)
        # eta = X[:,1] / eta_fudge_factor


        # TODO: Multiply by number of detectors???
        # log_likelihood = log_likelihood - jnp.log(X[:,0]) + jnp.log(jnp.sqrt(1 - 4 * X[:, 1]) * X[:, 1] ** (6/5))
        log_likelihood = log_likelihood - jnp.log(X[:,0]) + jnp.log(jnp.sqrt(1 - 4 * eta) * eta ** (6/5)) #+ jnp.log(eta_fudge_factor)


        # return log_likelihood, residual
        return log_likelihood
    
    # @partial(jax.jit, static_argnums=(0,))
    def gradient_minusLogLikelihood(self, X, residual=None): # Checks: XX
        # Remarks:
        # (i) Jacobian is (d, N, f) shaped. sum over final axis gives (d, N), then transpose to give (N, d)

        if residual is None:
            template = self.getSignal(X)
            residual = {}
            residual['L1'] = template['L1'] - self.htrue['L1']
            residual['H1'] = template['H1'] - self.htrue['H1']
            residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

        jacSignal = self._getJacobianSignal(X)
        grad_log_like = self.overlap(jacSignal['L1'], residual['L1'], self.PSDs['L1'], self.df).real
        grad_log_like = grad_log_like + self.overlap(jacSignal['H1'], residual['H1'], self.PSDs['H1'], self.df).real
        grad_log_like = grad_log_like + self.overlap(jacSignal['Virgo'], residual['Virgo'], self.PSDs['Virgo'], self.df).real

        # New prior contributions TODO: REMOVE AFTER?
        # grad_log_like = grad_log_like.at[:, 0].add(-1 / X[:,0])
        # grad_log_like = grad_log_like.at[:, 1].add(-2 / (1 - 4 * X[:,1]) + 6 / (5 * X[:,1]))

        grad_log_like = grad_log_like.at[:, 0].add(-1 / X[:,0])
        grad_log_like = grad_log_like.at[:, 1].add((-2 / (1 - 4 * X[:,1]) + 6 / (5 * X[:,1])) / self.eta_fudge_factor)
        # grad_log_like = grad_log_like.at[:, 1].add(-2 / (1 - 4 * X[:,1]) + 6 / (5 * X[:,1]))


        return grad_log_like

# %%
# model = gwfast_LVGW150914()

# #%%

# X = model._newDrawFromPrior(500)
# # %%
# %%timeit
# model.minusLogLikelihood(X)
# #%%
# %%timeit
# model.gradient_minusLogLikelihood(X).block_until_ready()
# # %%
# Xg
# %%
