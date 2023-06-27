#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
import copy
from random import uniform
import jax.numpy as jnp
import jax
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD
import gwfast.signal as signal
from gwfast.network import DetNet
import gwfast.gwfastGlobals as glob
import gwfast.gwfastUtils as utils
from astropy.cosmology import Planck18
from opt_einsum import contract
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

#%%
class gwfast_class(object):
    
    # def __init__(self, NetDict, WaveForm, injParams, priorDict):
    def __init__(self, chi, eps, mode='TaylorF2', freeze_indicies=[]):
        """
        Args:
            NetDict (dict): dictionary containing the specifications of the detectors in the network
            WaveForm (WaveFormModel): waveform model to use
            injParams (dict): injection parameters
            priorDict (dict): Provides (min,max) range for each coordinate

        Remarks:
            (1) Example of NetDict: {"H1": {"lat": 46.455, "long": -119.408, "xax": 170.99924234706103, "shape": "L", "psd_path":"path/to/psd"}}

        """

        # sSVN_GW related attributes
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0
        self.id = 'gwfast_model'
        self.mode = mode
        self._initParams()
        # self.priorDict  = priorDict

        # gw_fast related attributes
        self.EarthMotion = False

        # Parameter order convention
        self.gwfast_param_order = ['Mc','eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
        self.gwfast_params_neglected = ['chi1x', 'chi2x', 'chi1y', 'chi2y', 'LambdaTilde', 'deltaLambda', 'ecc']

        # Sample over subset of parameters
        # self.DoF = 11
        self.freeze_indicies = freeze_indicies
        self.active_indicies = np.array(list(set(np.arange(0, 11)) - set(self.freeze_indicies)))
        self.DoF = len(self.active_indicies)
        self.DoF_total = 11

        # Definitions for easy interfacing
        self.true_params = jnp.array([self.injParams[param].squeeze() for param in self.gwfast_param_order])
        self.lower_bound = np.array([self.priorDict[param][0] for param in self.gwfast_param_order])[self.active_indicies] # Mod 1
        self.upper_bound = np.array([self.priorDict[param][1] for param in self.gwfast_param_order])[self.active_indicies] # Mod 2

        self._initFrequencyGrid()
        self._initDetectors()

        # NEW
        self.h0_standard = {}
        self.h0_dense = {}
        for det in self.detsInNet.keys():
            self.h0_standard[det] = self.getSignal(self.true_params[None,:], self.fgrid_standard, det).squeeze()
            self.h0_dense[det] = self.getSignal(self.true_params[None,:], self.fgrid_dense, det).squeeze()

        # OLD
        # self.h0_standard = self._getInjectedSignals(self.injParams, self.fgrid_standard) # Fiducial signal
        # self.h0_dense = self._getInjectedSignals(self.injParams, self.fgrid_dense)       # Fiducial signal

        # Form data used in injection
        self.d_dense = {}
        self.d_standard = {}
        for det in self.detsInNet.keys():
            self.d_dense[det] = self.h0_dense[det] 
            self.d_standard[det] = self.h0_standard[det]

        # Heterodyned strategy
        self.d_d = self._precomputeDataInnerProduct()
        # TODO reenable when using heterodyne
        self._reinitialize(chi=chi, eps=eps)

        # Debugging (ignore)
        self.hj0 = None

    def _initParams(self):
        """ 
        Remarks:
        (i)   `tcoal` is accepted in units GMST fraction of a day
        (ii)  GPSt_to_LMST returns GMST in units of fraction of day (GMST is LMST computed at long = 0°)
        (iv)  Use [tcoal - 3e-7, tcoal + 3e-7] prior when in units of days

        """
        self.seconds_per_day = 86400. 
        injParams = {}
        priorDict = {}

        # GW170817
        # z = np.array([0.00980])
        # tGPS = np.array([1187008882.4])
        # # tcoal = utils.GPSt_to_LMST(tGPS, lat=0., long=0.)[0]
        # injParams['Mc']      = np.array([1.1859])*(1.+z) 
        # injParams['eta']     = np.array([0.24786618323504223]) 
        # injParams['dL']      = Planck18.luminosity_distance(z).value/1000. 
        # injParams['theta']   = np.array([np.pi/2. + 0.4080839999999999]) 
        # injParams['phi']     = np.array([3.4461599999999994])
        # injParams['iota']    = np.array([2.545065595974997]) 
        # injParams['psi']     = np.array([0.1]) # 0
        # injParams['tcoal']   = np.array(utils.GPSt_to_LMST(tGPS, lat=0., long=0.)) * self.seconds_per_day 
        # injParams['Phicoal'] = np.array([0.1]) # 0 
        # injParams['chi1z']   = np.array([0.005136138323169717]) 
        # injParams['chi2z']   = np.array([0.003235146993487445]) 
        # tcoal = injParams['tcoal']


        # priorDict['Mc']      = [1.19750182, 1.19754182]        # [M_solar]     
        # priorDict['eta']     = [0.24, 0.25]                    # [Unitless]
        # priorDict['dL']      = [0.04, 0.05]                    # [GPC]
        # priorDict['theta']   = [0., np.pi]                     # [Rad]
        # priorDict['phi']     = [0., 2 * np.pi]                 # [Rad]
        # priorDict['iota']    = [0., np.pi]                     # [Rad]
        # priorDict['psi']     = [0., np.pi]                     # [Rad]
        # # priorDict['tcoal']   = [tcoal - 0.001, tcoal + 0.001]  # [sec]
        # priorDict['tcoal']   = [tcoal[0] - 0.01, tcoal[0] + 0.01]  # [sec]
        # priorDict['Phicoal'] = [0., 2 * np.pi]                 # [Rad]
        # priorDict['chi1z']   = [-0.99, 0.99]                       # [Unitless]
        # priorDict['chi2z']   = [-0.99, 0.99]                       # [Unitless]


        # GW150914
        tGPS = np.array([1.1262594624e+09])
        tcoal = float(utils.GPSt_to_LMST(tGPS, lat=0., long=0.)) * self.seconds_per_day # [0, 1] 
        injParams['Mc']      = np.array([31.39])               # (1)   # (0)   # [M_solar]      # Chirp mass
        injParams['eta']     = np.array([0.2485773])           # (2)   # (1)   # [Unitless]     # Symmetric mass ratio
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
        priorDict['eta']     = [0.20, 0.25]                         # [Unitless]
        priorDict['dL']      = [0.05, 2]                           # [GPC]
        priorDict['theta']   = [0., np.pi]                         # [Rad]
        priorDict['phi']     = [0., 2 * np.pi]                     # [Rad]
        priorDict['iota']    = [0., np.pi]                         # [Rad] # Note: Maybe use cos i variable?
        priorDict['psi']     = [0., np.pi]                         # [Rad]
        priorDict['tcoal']   = [tcoal - 0.01, tcoal + 0.01]          # [sec]
        priorDict['Phicoal'] = [0., 2 * np.pi]                     # [Rad]
        priorDict['chi1z']   = [-0.99, 0.99]                       # [Unitless]
        priorDict['chi2z']   = [-0.99, 0.99]                       # [Unitless]

        # Use these for the full problem
        # priorDict['Mc']      = [10, 80]                            # [M_solar]     
        # priorDict['eta']     = [0.1, 0.25]                         # [Unitless]
        # priorDict['dL']      = [0.05, 2]                           # [GPC]
        # priorDict['theta']   = [0., np.pi]                         # [Rad]
        # priorDict['phi']     = [0., 2 * np.pi]                     # [Rad]
        # priorDict['iota']    = [0., np.pi]                         # [Rad] # Note: Maybe use cos i variable?
        # priorDict['psi']     = [0., np.pi]                         # [Rad]
        # priorDict['tcoal']   = [tcoal - 0.1, tcoal + 0.1]          # [sec]
        # priorDict['Phicoal'] = [0., 2 * np.pi]                     # [Rad]
        # priorDict['chi1z']   = [-0.99, 0.99]                       # [Unitless]
        # priorDict['chi2z']   = [-0.99, 0.99]                       # [Unitless]

        # for param in ['Phicoal', 'psi', 'iota', 'theta', 'phi']:
        # for param in ['Phicoal', 'psi', 'phi']:
        #     x = injParams[param][0]
        #     delta = x - (priorDict[param][1] + priorDict[param][0]) / 2
        #     priorDict[param][0] += delta
        #     priorDict[param][1] += delta

        self.priorDict = priorDict
        self.injParams = injParams

        ################################################################
        # Notes:
        # (i)   Geometry of every available detector
        # (ii)  Extract only LIGO/Virgo detectors
        # (iii) Providing ASD path to psd_path with flag "is_ASD = True"
        # (iv)  Add paths to detector sensitivities
        # (v)   Choice of waveform
        ################################################################

        all_detectors = copy.deepcopy(glob.detectors) # (i)
        dets = ['L1', 'H1', 'Virgo']
        # dets = ['Virgo']
        print('Using detectors', dets)
        LV_detectors = {det:all_detectors[det] for det in dets} # (ii) 
        detector_ASD = dict() # (iii)

        # O2 PSD
        # detector_ASD['L1']    = '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt'
        # detector_ASD['H1']    = '2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt'
        # detector_ASD['Virgo'] = 'Hrec_hoft_V1O2Repro2A_16384Hz.txt'

        # O3 PSD
        # detector_ASD['L1']    = 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
        # detector_ASD['H1']    = 'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'
        # detector_ASD['Virgo'] = 'O3-V1_sensitivity_strain_asd.txt'

        # Uses GWfast PSDs. Make sure to change this back if using GWFAST!
        # LV_detectors['L1']['psd_path']    = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['L1']) # (iv) 
        # LV_detectors['H1']['psd_path']    = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['H1'])
        # LV_detectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['Virgo'])

        # O4 PSD

        detector_ASD['L1']    = '/mnt/c/Users/alex/Documents/sSVN_GW/notebooks/aLIGO_O4_high_asd.txt'
        detector_ASD['H1']    = '/mnt/c/Users/alex/Documents/sSVN_GW/notebooks/aLIGO_O4_high_asd.txt'
        detector_ASD['Virgo'] = '/mnt/c/Users/alex/Documents/sSVN_GW/notebooks/AdV_ASD.txt'

        # detector_ASD['L1']    = '/home/alex/anaconda3/envs/myenv/lib/python3.10/site-packages/bilby/gw/detector/noise_curves/aLIGO_O4_high_asd.txt'
        # detector_ASD['H1']    = '/home/alex/anaconda3/envs/myenv/lib/python3.10/site-packages/bilby/gw/detector/noise_curves/aLIGO_O4_high_asd.txt'
        # detector_ASD['Virgo'] = '/mnt/c/sSVN_GW/notebooks/AdV_ASD.txt'

        LV_detectors['L1']['psd_path']    = detector_ASD['L1'] # (iv) 
        LV_detectors['H1']['psd_path']    = detector_ASD['H1']
        LV_detectors['Virgo']['psd_path'] = detector_ASD['Virgo']

        self.NetDict = LV_detectors

        # (v) 
        waveform_model = self.mode
        if waveform_model == 'TaylorF2':
            self.wf_model = TaylorF2_RestrictedPN(apply_fcut=False) 
        elif waveform_model == 'IMRPhenomD':
            self.wf_model = IMRPhenomD(apply_fcut=False) 
        print('Using waveform model: %s' % waveform_model)

    def _initFrequencyGrid(self, fmin=20): # Checks: X
        """
        Setup frequency grids that will be used
        Remarks:                                           
        (i)    Setup [f_min, f_max] interval
        (ii)   gwfast sets h(fcut; theta_0) to 0, causing division by zero errors.
               This ensures that we do not evaluate the fiducial signal at fcut.
               Note: This is a hack, and would better be fixed in gwfast. 
        """
        # (i)
        self.fmin = fmin  # 10
        self.fmax = self.wf_model.fcut(**self.injParams)[0] - 1e-7 # (ii)

        self.nbins_standard = 1000 # 2000
        self.fgrid_standard = np.linspace(self.fmin, self.fmax, num=self.nbins_standard + 1).squeeze()
        self.df_standard = (self.fgrid_standard[-1] - self.fgrid_standard[0]) / self.nbins_standard

        self.nbins_dense = 10000
        self.fgrid_dense = np.linspace(self.fmin, self.fmax, num=self.nbins_dense + 1).squeeze()
        self.df_dense = (self.fgrid_dense[-1] - self.fgrid_dense[0]) / self.nbins_dense

        print('nbins_standard=%i' % self.nbins_standard)
        print('nbins_dense=%i' % self.nbins_dense)

    def _initDetectors(self): 
        """Initialize detectors and store PSD interpolated over defined frequency grid

        Remarks
        -------

        (1) This returns $S_n(f)$ for the desired frequency grid (self.fgrid)
        """

        self.detsInNet = {}
        self.PSD_standard = {}
        self.PSD_dense = {}
        for det in self.NetDict.keys():
            self.detsInNet[det] = signal.GWSignal(self.wf_model,
                                                  psd_path       = self.NetDict[det]['psd_path'],
                                                  detector_shape = self.NetDict[det]['shape'],
                                                  det_lat        = self.NetDict[det]['lat'],
                                                  det_long       = self.NetDict[det]['long'],
                                                  det_xax        = self.NetDict[det]['xax'],
                                                  useEarthMotion = self.EarthMotion,
                                                  fmin           = self.fmin, 
                                                  fmax           = self.fmax,
                                                  verbose        = False,
                                                  is_ASD         = True)

            self.PSD_standard[det] = jnp.interp(self.fgrid_standard, self.detsInNet[det].strainFreq, self.detsInNet[det].noiseCurve, left=1., right=1.).squeeze()
            self.PSD_dense[det] = jnp.interp(self.fgrid_dense, self.detsInNet[det].strainFreq, self.detsInNet[det].noiseCurve, left=1., right=1.).squeeze()


    def getSignal(self, X, f_grid, det):
        """Method to calculate signal for each X[i] over f_grid in detector det

        Remarks
        -------

        (i)   Same frequency grid is currently being used for each particle
        (ii)  gwfast.GWstrain expects an f x N matrix
        (iii) gwfast.GWstrain expects `tcoal` in units of seconds
        (iv)  Transpose is included to return an N x f matrix
        (v)   X must be (N x d) shaped, for one sample is must be (1 x d) shaped
        """
        nParticles = X.shape[0]
        dict_params_neglected = self._getDictParamsNeglected(nParticles)
        fgrids = jnp.repeat(f_grid[...,np.newaxis], nParticles, axis=1) # (i)
        X_ = X.T.astype('complex128')
        signal = (self.detsInNet[det].GWstrain(fgrids, # (ii)                        
                                               Mc      = X_[0],
                                               eta     = X_[1],
                                               dL      = X_[2],
                                               theta   = X_[3],
                                               phi     = X_[4],
                                               iota    = X_[5],
                                               psi     = X_[6],
                                               tcoal   = X_[7] / self.seconds_per_day, # (iii)
                                               Phicoal = X_[8],
                                               chiS    = X_[9],
                                               chiA    = X_[10],
                                               is_chi1chi2 = 'True',
                                               **dict_params_neglected)).T # (iv) 
                            
        return signal 

    # @partial(jax.jit, static_argnums=(0,))
    def _getJacobianSignal(self, X, f_grid, det):
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
        nParticles = X.shape[0]
        dict_params_neglected = self._getDictParamsNeglected(nParticles)
        fgrids = jnp.repeat(f_grid[...,np.newaxis], nParticles, axis=1)
        X_ = X.T.astype('complex128')
        jacModel = self.detsInNet[det]._SignalDerivatives_use(fgrids, 
                                                              Mc      = X_[0],
                                                              eta     = X_[1],
                                                              dL      = X_[2],
                                                              theta   = X_[3],
                                                              phi     = X_[4],
                                                              iota    = X_[5],
                                                              psi     = X_[6],
                                                              tcoal   = X_[7] / self.seconds_per_day, # Correction 1
                                                              Phicoal = X_[8],
                                                              chiS    = X_[9],
                                                              chiA    = X_[10],
                                                              use_chi1chi2 = True,
                                                              **dict_params_neglected) 

        jacModel = jacModel.at[7].divide(self.seconds_per_day) # Correction 2

        return jacModel
            
    # @partial(jax.jit, static_argnums=(0,))
    def square_norm(self, a, PSD, deltaf):
        """ 
        Square norm for single detector estimated using left Riemann sum
        """
        square_norm = (4 * jnp.sum((a.real[..., :-1] ** 2 + a.imag[..., :-1] ** 2) / PSD[..., :-1] * deltaf, axis=-1)).T
        return square_norm

    # @partial(jax.jit, static_argnums=(0,))
    def overlap(self, a, b, PSD, deltaf):
        """ 
        Network overlap estimated using left Riemann sum
        """
        overlap = (4 * jnp.sum(a.conjugate()[..., :-1] * b[..., :-1] / PSD[..., :-1] * deltaf, axis=-1)).T
        return overlap

    # @partial(jax.jit, static_argnums=(0,))
    def overlap_trap(self, a, b, PSD, fgrid):
        """ 
        Overlap estimated using trapezoid method
        """
        integrand = a.conjugate() * b / PSD
        overlap = (4 * jnp.trapz(integrand, fgrid)).T
        return overlap
    
    def _precomputeDataInnerProduct(self): # Checks
        print('Precomputing squared SNR for likelihood')
        # TODO: This can be replaced with the overlap method later!
        # Remarks:
        # (i) The snr is actually sqrt(<d,d>). We are calculating the square! Hence SNR2
        SNR2 = {}
        SNR = 0
        for det in self.detsInNet.keys():
            res = self.square_norm(self.d_dense[det], self.PSD_dense[det], self.df_dense)
            # res = 4 * np.sum((self.d_dense[det].real ** 2 + self.d_dense[det].imag ** 2) / self.PSD_dense[det]) * self.df_dense
            SNR2[det] = res
            SNR += res
        print('SNR: %f' % np.sqrt(SNR))
        return SNR2

    # def standard_minusLogLikelihood(self, X): # Checks: XX
    # @partial(jax.jit, static_argnums=(0,))
    # def getMinusLogPosterior_ensemble(self, X): # Checks: XX
    #     """ 
    #     """
    #     nParticles = X.shape[0]
    #     log_likelihood = jnp.zeros(nParticles)
    #     for det in self.detsInNet.keys():
    #         template = self.getSignal(X, self.fgrid_standard, det) # signal template
    #         residual = template - self.d_standard[det][np.newaxis, ...]
    #         log_likelihood += 0.5 * self.square_norm(residual, self.PSD_standard[det], self.df_standard) 
    #     return log_likelihood

    @partial(jax.jit, static_argnums=(0,))
    def standard_gradientMinusLogLikelihood(self, X): # Checks: XX
        # Remarks:
        # (i) Jacobian is (d, N, f) shaped. sum over final axis gives (d, N), then transpose to give (N, d)
        nParticles = X.shape[0]
        grad_log_like = jnp.zeros((nParticles, self.DoF))
        for det in self.detsInNet.keys():
            template  = self.getSignal(X, self.fgrid_standard, det)
            jacSignal = self._getJacobianSignal(X, self.fgrid_standard, det)
            residual  = template - self.d_standard[det][np.newaxis, ...]
            grad_log_like += self.overlap(jacSignal, residual, self.PSD_standard[det], self.df_standard).real
        return grad_log_like
    
    @partial(jax.jit, static_argnums=(0,))
    def standard_GNHessianMinusLogLikelihood(self, X): # Checks: XX
        nParticles = X.shape[0]
        GN = jnp.zeros((nParticles, self.DoF, self.DoF))
        for det in self.detsInNet.keys():
            jacSignal = self._getJacobianSignal(X, self.fgrid_standard, det)
            inner_product = 4 * contract('iNf, jNf, f -> Nij', jacSignal.conjugate(), jacSignal, 1 / self.PSD_standard[det]) * self.df_standard
            GN += inner_product.real
        return GN

    # @partial(jax.jit, static_argnums=(0,))
    # def getDerivativesMinusLogPosterior_ensemble(self, X_reduced):
    #     """ 
    #     Returns SUBSET of derivatives using "standard" discretization
    #     """
    #     nParticles = X_reduced.shape[0]
    #     X = jnp.zeros((nParticles, self.DoF_total))
    #     if len(self.freeze_indicies) > 0:
    #         X = X.at[:, self.freeze_indicies].set(jnp.tile(self.true_params[self.freeze_indicies], nParticles).reshape(nParticles, len(self.freeze_indicies)))
    #     X = X.at[:, self.active_indicies].set(X_reduced)

    #     grad_log_like = jnp.zeros((nParticles, self.DoF_total))
    #     GN = jnp.zeros((nParticles, self.DoF_total, self.DoF_total))
    #     for det in self.detsInNet.keys():
    #         template  = self.getSignal(X, self.fgrid_standard, det)
    #         jacSignal = self._getJacobianSignal(X, self.fgrid_standard, det)
    #         residual  = template - self.d_standard[det][np.newaxis, ...]
    #         grad_log_like += self.overlap(jacSignal, residual, self.PSD_standard[det], self.df_standard).real
    #         inner_product = 4 * contract('iNf, jNf, f -> Nij', jacSignal.conjugate(), jacSignal, 1 / self.PSD_standard[det]) * self.df_standard
    #         GN += inner_product.real
    #     return grad_log_like[:, self.active_indicies], GN[:, self.active_indicies][:, :, self.active_indicies]





#########################################################################
# HETERODYNE METHODS
#########################################################################

    def _reinitialize(self, chi, eps):
        self.chi = chi 
        self.eps = eps
        print('Forming sparse subgrid')
        self.getHeterodyneBins(chi=chi, eps=eps)
        print('Completed')
        print('Calculating summary data')
        self.A0, self.A1, self.B0, self.B1, self.C0, self.C1, self.B2 = self.getSummary_data()
        print('Completed')

    def getHeterodyneBins(self, chi, eps): # Checks X
        # print('Getting heterodyned bins')
        # Remarks:
        # (i)   0.5 is a dummy variable for x==0 case (which we dont care for)
        # (ii)  Alternatively, we may add frequencies then recover the indicies using np.searchsorted(A,B):
        # https://stackoverflow.com/questions/33678543/finding-indices-of-matches-of-one-array-in-another-array
        # (iii) Maximum accumulation over a single bin should not exceep eps. If this is th case, pick
        # a denser grid, or pick a larger eps. 

        gamma = np.array([-5/3, -2/3, 1, 5/3, 7/3])
        f_star = self.fmax * np.heaviside(gamma, 0.5) + self.fmin * np.heaviside(-gamma, 0.5) # (i) 
        delta = lambda f_minus, f_plus: 2 * np.pi * chi * np.sum(np.abs((f_plus[:, np.newaxis] / f_star) ** gamma - (f_minus[:, np.newaxis] / f_star) ** gamma), axis=-1)
        delta_single = np.max(delta(self.fgrid_dense[:-1], self.fgrid_dense[1:]))
        assert delta_single < eps # (iii)
        subindex = [0] # (ii)
        index_f_minus = 0
        j = 1
        while j <= self.nbins_dense:
            if j == self.nbins_dense:
                subindex.append(j)
                break
            d = delta(self.fgrid_dense[index_f_minus, np.newaxis], self.fgrid_dense[j, np.newaxis])[0]
            if d >= eps:
                subindex.append(j - 1)
                index_f_minus = j - 1
                continue
            j += 1

        self.indicies_kept = np.array(subindex)
        self.bin_edges = self.fgrid_dense[self.indicies_kept]
        self.nbins = len(subindex) - 1
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        print('Heterodyne binning scheme: %i' % self.nbins)

    def getSummary_data(self):
        """ 
        Calculate the summary data for heterodyne likelihood
        Remarks:
        (i)   Label which frequencies belong to which bin (from 0 to nbins - 1)
            (ia)   np.digitize labels the first bin with a 1
            (ib)  We subtract 1 to change the convention in (ia) to begin with 0
            (ic) The last bin includes the right endpoint in our binning convention
        (ii)  np.bincount begins tallying from 0. This is why convention (iia) is convenient   
        (iii) To avoid out of bound error for first bin
        (iv)  Included to keep (iiia) valid
        (v)   Indicies have been shifted to the right by previous step
        """
        def sumBins(array, bin_indicies):
            """
            Given an `array`, and `bin_indicies` which defines how to partition `array`, return
            sum in each partition
            """
            tmp = np.zeros(len(array) + 1).astype(array.dtype)
            tmp[1:] = np.cumsum(array) # (iii)
            tmp[-2] = tmp[-1] # (iv) 
            return tmp[bin_indicies[1:]] - tmp[bin_indicies[:-1]] # (v) 

        def getBinIds(grid, bins):
            """ 
            Given bins, returns an array labeling which bin each point in grid belongs to.
            Bins are labeled beginning from 0 to nbins - 1!
            """
            bin_ids = (np.digitize(grid, bins)) - 1 # (ia), (ib)
            bin_ids[-1] = len(bins) - 2 # (ic)
            return bin_ids

        A0, A1, B0, B1, C0, C1, B2 = {}, {}, {}, {}, {}, {}, {}

        elements_per_bin = np.bincount(getBinIds(self.fgrid_dense, self.bin_edges)) # (ii)

        deltaf_in_bin = self.fgrid_dense - np.repeat(self.bin_edges[:-1], elements_per_bin)

        for det in self.detsInNet.keys():
            A0_integrand = 4 * self.h0_dense[det].conjugate() * self.d_dense[det] / self.PSD_dense[det] * self.df_dense 
            A1_integrand = A0_integrand * deltaf_in_bin
            B0_integrand = 4 * (self.h0_dense[det].real ** 2 + self.h0_dense[det].imag ** 2) / self.PSD_dense[det] * self.df_dense
            B1_integrand = B0_integrand * deltaf_in_bin
            C0_integrand = 4 * self.h0_dense[det].conjugate() * (self.h0_dense[det] - self.d_dense[det]) / self.PSD_dense[det] * self.df_dense 
            C1_integrand = C0_integrand * deltaf_in_bin
            B2_integrand = B0_integrand * deltaf_in_bin ** 2

            for data, integrand in zip([A0, A1, B0, B1, C0, C1, B2], [A0_integrand, A1_integrand, B0_integrand, B1_integrand, C0_integrand, C1_integrand, B2_integrand]):
                data[det] = sumBins(integrand, self.indicies_kept)

        return A0, A1, B0, B1, C0, C1, B2

    def getFirstSplineData(self, X, det):
        """ 
        Return N x b matrix for spline of r := h/h0 in a particular detector
        """
        # Remarks:
        # (i)   r is the heterodyne
        # (ii)  These are the y-intercepts for each bin (N x b)
        # (iii) These are the slopes for each bin (N x b)
        h = self.getSignal(X, self.bin_edges, det)
        r = h / self.h0_dense[det][self.indicies_kept] # (i)
        r0 = r[:, :-1] # (ii)
        r1 = (r[:, 1:] - r[:, :-1]) / self.bin_widths # (iii)
        return r0, r1

    def getSecondSplineData(self, X, det):
        """ 
        Return matrix of shape (d, N, b) for spline of r_{,j} := h_{,j} / h0
        Note: Identical as first spline data for first order.
        """
        hj = self._getJacobianSignal(X, self.bin_edges, det)
        rj = hj / self.h0_dense[det][self.indicies_kept]
        rj0 = rj[..., :-1]
        rj1 = (rj[..., 1:] - rj[..., :-1]) / self.bin_widths
        return rj0, rj1


    # def heterodyne_minusLogLikelihood(self, X_reduced): # Checks X

    @partial(jax.jit, static_argnums=(0,))
    def getMinusLogPosterior_ensemble(self, X_reduced): # Checks X

        """ 
        Remarks:
        (i) Summary data has shape (b,)
        """
        nParticles = X_reduced.shape[0]
        X = jnp.zeros((nParticles, self.DoF_total))
        if len(self.freeze_indicies) > 0:
            X = X.at[:, self.freeze_indicies].set(jnp.tile(self.true_params[self.freeze_indicies], nParticles).reshape(nParticles, len(self.freeze_indicies)))
        X = X.at[:, self.active_indicies].set(X_reduced)

        nParticles = X.shape[0]
        log_like = jnp.zeros(nParticles)
        for det in self.detsInNet.keys():
            r0, r1 = self.getFirstSplineData(X, det)
            h_d = jnp.sum(self.A0[det][jnp.newaxis] * r0.conjugate() + self.A1[det][jnp.newaxis] * r1.conjugate(), axis=1)
            h_h = jnp.sum(self.B0[det][jnp.newaxis] * jnp.abs(r0) ** 2 + 2 * self.B1[det][jnp.newaxis] * (r0.conjugate() * r1).real, axis=1)
            log_like += 0.5 * h_h - h_d.real + 0.5 * self.d_d[det]
        return log_like

    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogPosterior_ensemble(self, X):
        # Remarks:
        # (i)   second spline data is (d, N, b) shaped
        nParticles = X.shape[0]
        grad_log_like = jnp.zeros((nParticles, self.DoF))
        for det in self.detsInNet.keys():
            # r0, r1 = self.getFirstSplineData(X, det)
            # rj0, rj1 = self.getSecondSplineData(X, det)

            # hj_d = contract('jNb -> Nj', self.A0[det] * rj0[det].conjugate() \
            #                            + self.A1[det] * rj1[det].conjugate(), backend='jax')

            # hj_h = contract('jNb -> Nj', self.B0[det] * rj0[det].conjugate() * r0[det][np.newaxis] 
            #                           +  self.B1[det] *(rj0[det].conjugate() * r1[det][np.newaxis] + rj1[det].conjugate() * r0[det][np.newaxis]), backend='jax')

            # grad_log_like += hj_h.real - hj_d.real

            r0, r1 = self.getFirstSplineData(X, det)
            r0j, r1j = self.getSecondSplineData(X, det)
            grad_log_like += \
            jnp.sum((self.B0[det] * r0j.conjugate() * (r0-1)) + (self.B1[det] * (r0j.conjugate() * r1 + r1j.conjugate() * (r0-1))), axis=-1).T.real 

        return grad_log_like

    # @partial(jax.jit, static_argnums=(0,))
    # def getGNHessianMinusLogPosterior_ensemble(self, X):
    #     nParticles = X.shape[0]
    #     rj0, rj1 = self.getSecondSplineData(X)
    #     GN = jnp.zeros((nParticles, self.DoF, self.DoF))
    #     for det in self.detsInNet.keys():
    #         term1 = contract('b, jNb, kNb -> Njk', self.B0[det], rj0[det].conjugate(), rj0[det], backend='jax')
    #         term2 = contract('b, jNb, kNb -> Njk', self.B1[det], rj0[det].conjugate(), rj1[det], backend='jax')
    #         term3 = contract('Nkj -> Njk', term2.conjugate(), backend='jax')
    #         GN += term1.real + term2.real + term3.real
    #         # GN += term1.real 
    #     return GN

    # @partial(jax.jit, static_argnums=(0,))  
    # def getDerivativesMinusLogPosterior_ensemble(self, X):
    #     """ 
    #     Returns ENTIRE set of derivatives using sparse grid (heterodyning)
    #     """
    #     nParticles = X.shape[0]
    #     grad_log_like = jnp.zeros((nParticles, self.DoF))
    #     GN = jnp.zeros((nParticles, self.DoF, self.DoF))
    #     for det in self.detsInNet.keys():
    #         r0, r1 = self.getFirstSplineData(X, det)
    #         r0j, r1j = self.getSecondSplineData(X, det)
    #         grad_log_like += \
    #         jnp.sum((self.B0[det] * r0j.conjugate() * (r0-1)) + (self.B1[det] * (r0j.conjugate() * r1 + r1j.conjugate() * (r0-1))), axis=-1).T.real 
    #         term1 = contract('b, jNb, kNb -> Njk', self.B0[det], r0j.conjugate(), r0j, backend='jax')
    #         GN += term1.real 
    #     return grad_log_like, GN

################################################################
    @partial(jax.jit, static_argnums=(0,))
    def getDerivativesMinusLogPosterior_ensemble(self, X_reduced):
        """ 
        Returns subset of derivatives using sparse heterodyned method!
        """
        nParticles = X_reduced.shape[0]
        X = jnp.zeros((nParticles, self.DoF_total))
        if len(self.freeze_indicies) > 0:
            X = X.at[:, self.freeze_indicies].set(jnp.tile(self.true_params[self.freeze_indicies], nParticles).reshape(nParticles, len(self.freeze_indicies)))
        X = X.at[:, self.active_indicies].set(X_reduced)
        grad_log_like = jnp.zeros((nParticles, self.DoF_total))
        GN = jnp.zeros((nParticles, self.DoF_total, self.DoF_total))
        for det in self.detsInNet.keys():
            r0, r1 = self.getFirstSplineData(X, det)
            r0j, r1j = self.getSecondSplineData(X, det)
            grad_log_like += \
            jnp.sum((self.B0[det] * r0j.conjugate() * (r0-1)) + (self.B1[det] * (r0j.conjugate() * r1 + r1j.conjugate() * (r0-1))), axis=-1).T.real 

            # term1 = contract('b, jNb, kNb -> Njk', self.B0[det], r0j.conjugate(), r0j, backend='jax')
            # term2 = contract('b, jNb, kNb -> Njk', self.B1[det], r0j.conjugate(), r1j, backend='jax')
            # term3 = contract('Nkj -> Njk', term2.conjugate(), backend='jax')
            # GN += term1.real + term2.real + term3.real

            term1 = contract('b, jNb, kNb -> Njk', self.B0[det], r0j.conjugate(), r0j, backend='jax')
            GN += term1.real 

        return grad_log_like[:, self.active_indicies], GN[:, self.active_indicies][:, :, self.active_indicies]

    def _newDrawFromPrior(self, n):
        prior_draw = np.zeros((n, 11))
        for i, param in enumerate(self.gwfast_param_order): # Assuming uniform on all parameters
            low = self.priorDict[param][0]
            high = self.priorDict[param][1]
            # buffer = (high-low) / 5
            buffer = 0
            prior_draw[:, i] = np.random.uniform(low=low+buffer, high=high-buffer, size=n)
            # prior_draw[:, i] = np.random.uniform(low=self.true_params[i] - 1e-7, high=self.true_params[i] + 1e-7, size=n)
            # print('modified priors to be at mode!')
        print('buffer in prior: %f' % buffer)
        return prior_draw[:, self.active_indicies]

    # def fill(self, X_reduced):
    #     """ 
    #     Return corresponding points in total parameter space
    #     Note: Other indicies are held constant
    #     """
    #     nParticles = X_reduced.shape[0]
    #     X = jnp.zeros((nParticles, self.DoF_total))
    #     X = X.at[:, self.freeze_indicies].set(np.tile(self.true_params[self.freeze_indicies], nParticles).reshape(nParticles, len(self.freeze_indicies)))
    #     X = X.at[:, self.active_indicies].set(X_reduced)
    #     return X


    def getGrad_heterodyne(self, X):
        func = jax.jacrev(self.heterodyne_minusLogLikelihood)
        return jax.vmap(func)(X)

    # def getHess_heterodyne(self, X)
    #     func = jax.jacrev(jax.jacrev(self.heterodyne_minusLogLikelihood))
    #     return jax.vmap(func)(X)



        test1 = func(x)
        test2 = model.getGradientMinusLogPosterior_ensemble(x)
        np.allclose(test1, test2)



















################################################################################


#####################################################
# Methods helpful for debugging integrals evaluations
#####################################################
    def r_heterodyne(self, X, fgrid):
        """ 
        Calculate proposed heterodyne r(theta) := h(theta) / h0 
        """
        r = {}
        signal = self.getSignal(X, fgrid)
        for det in self.detsInNet.keys():
            h0 = np.interp(fgrid, self.fgrid_dense, self.h0_dense[det]).squeeze()
            r[det] = signal[det] / h0
        return r

    def derivative_heterodyne(self, X, fgrid):
        jac_r = {}
        if self.hj0 == None:
            self.hj0 = self._getJacobianSignal(self.true_params, fgrid)
        jacSignal = self._getJacobianSignal(X, fgrid)
        for det in self.detsInNet.keys():
            jac_r[det] = jacSignal[det] / self.hj0[det]
        return jac_r








#############
    # Not important for logic, keep farther away
    def _getDictParamsNeglected(self, N):
        return {neglected_params: jnp.zeros(N).astype('complex128') for neglected_params in self.gwfast_params_neglected}




################################################################
# Other methods. Clean up later!!!
################################################################

    # def _newDrawFromPrior(self, nParticles):
        """ 
        Return prior for entire subset
        """
    #     prior_draw = np.zeros((nParticles, self.DoF))
    #     for i, param in enumerate(self.gwfast_param_order): # Assuming uniform on all parameters
    #         low = self.priorDict[param][0]
    #         high = self.priorDict[param][1]
    #         prior_draw[:, i] = np.random.uniform(low=low, high=high, size=nParticles)
                   
    #     return prior_draw

    def getCrossSection(self, a, b, func, ngrid):
        # a, b are the parameters for which we want the marginals:
        x = np.linspace(self.priorDict[a][0], self.priorDict[a][1], ngrid)
        y = np.linspace(self.priorDict[b][0], self.priorDict[b][1], ngrid)
        X, Y = np.meshgrid(x, y)
        particle_grid = np.zeros((ngrid ** 2, self.DoF))
        index1 = self.gwfast_param_order.index(a)
        index2 = self.gwfast_param_order.index(b)
        parameter_mesh = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T
        particle_grid[:, index1] = parameter_mesh[:, 0]
        particle_grid[:, index2] = parameter_mesh[:, 1]
        for i in range(self.DoF): # Fix all other parameters
            if i != index1 and i != index2:
                particle_grid[:, i] = np.ones(ngrid ** 2) * self.true_params[i]
        # Z = np.exp(-1 * func(particle_grid).reshape(ngrid,ngrid))
        Z = func(particle_grid).reshape(ngrid,ngrid)
        # s = self.getGradientMinusLogPosterior_ensemble(particle_grid)
        # Z = np.linalg.norm(s,axis=1).reshape(ngrid,ngrid)
        fig, ax = plt.subplots(figsize = (5, 5))
        cp = ax.contourf(X, Y, Z)
        # cbar = fig.colorbar(cp)
        plt.colorbar(cp)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title('Likelihood cross section')
        filename = a + b + '.png'
        path = os.path.join('marginals', filename)
        fig.savefig(path)

    def _warmup_potential(self, warmup):
        if warmup is True:
            print('Warming up potential')
            self.getDerivativesMinusLogPosterior_ensemble(self._newDrawFromPrior(self.nParticles))

    def _warmup_potential_derivative(self, warmup):
        if warmup is True:
            print('Warming up derivatives')
            self.getDerivativesMinusLogPosterior_ensemble(self._newDrawFromPrior(self.nParticles))



    def r(self, X):
        """ 
        Calculate the ratio of signal model with fiducial signal
        """
        r = {}
        signal = self.getSignal(X, self.bin_edges)
        for det in self.detsInNet.keys():
            r[det] = signal[det] / self.h0_dense[det][self.indicies_kept]
        return r

    def jac_r(self, X):
        jac_r = {}
        jacSignal = self._getJacobianSignal(X, self.fgrid_standard)
        for det in self.detsInNet.keys():
            jac_r[det] = jacSignal[det] / self.h0_standard[det]
        return jac_r




    # def getSummaryData(self):
    #     """ 
    #     Calculate summary data
    #     """
    #     print('Calculating summary data')
    #     # Init dicts
    #     self.A0 = {}
    #     self.A1 = {}
    #     self.B0 = {}
    #     self.B1 = {}
    #     bin_index = (np.digitize(self.fgrid_dense, self.bin_edges)) - 1 # To index bins from 0 to nbins - 1
    #     bin_index[-1] = self.nbins - 1 # Make sure the right endpoint is inclusive!

    #     for det in self.detsInNet.keys():
    #         self.A0[det] = np.zeros((self.nbins)).astype('complex128')
    #         self.A1[det] = np.zeros((self.nbins)).astype('complex128')
    #         self.B0[det] = np.zeros((self.nbins))
    #         self.B1[det] = np.zeros((self.nbins))
    #         for b in range(self.nbins):
    #             indicies = np.where(bin_index == b)
    #             tmp1 = 4 * self.h0_dense[det][indicies].conjugate() * self.d_dense[det][indicies] / self.PSD_dense[det][indicies] * self.df_dense
    #             tmp2 = 4 * (self.h0_dense[det][indicies].real ** 2 + self.h0_dense[det][indicies].imag ** 2) / self.PSD_dense[det][indicies] * self.df_dense
    #             self.A0[det][b] = np.sum(tmp1)
    #             self.A1[det][b] = np.sum(tmp1 * (self.fgrid_dense[indicies] - self.bin_edges[b]))
    #             self.B0[det][b] = np.sum(tmp2)
    #             self.B1[det][b] = np.sum(tmp2 * (self.fgrid_dense[indicies] - self.bin_edges[b]))
    #     print('Summary data calculation completed')










    # def getSplineData(self, X):
    #     """ 
    #     Return N x b matrix
    #     """
    #     r = self.r(X)
    #     r0 = {}
    #     r1 = {}
    #     for det in self.detsInNet.keys():
    #         r0[det] = r[det][:, :-1] # y intercept
    #         r1[det] = (r[det][:, 1:] - r[det][:, :-1]) / self.bin_widths[np.newaxis, ...] # slopes
    #     return r0, r1





    # def getJacSplineData(self, X):
    #     jac_r = self.jac_r(X)
    #     jac_r0 = {}
    #     jac_r1 = {}
    #     for det in self.detsInNet.keys():
    #         jac_r0[det] = jac_r[det][..., :-1]
    #         jac_r1[det] = (jac_r[det][..., 1:] - jac_r[det][..., :-1]) / self.bin_widths
    #     return jac_r0, jac_r1


                
        # (ii)
        # self.nbins_standard = 2000
        # self.df_standard = (self.fmax - self.fmin) / self.nbins_standard # (iii)
        # print('Standard binning scheme: % i bins' % self.nbins_standard)
        # self.fgrid_standard = np.linspace(self.fmin, self.fmax, num=self.nbins_standard + 1).squeeze()

        # # (iv)
        # self.nbins_dense = 10000 # 2000
        # self.df_dense = (self.fmax - self.fmin) / self.nbins_dense
        # print('Dense bins: % i bins' % self.nbins_dense)
        # self.fgrid_dense = np.linspace(self.fmin, self.fmax, num=self.nbins_dense + 1).squeeze()




    # def _getInjectedSignals(self, injParams, fgrid):
    #     """
    #     Fiducial signals over dense grid (one for each detector)
    #     Note: See remarks on self.getSignal
    #     Remarks:
    #     (i)   Squeeze to return (f,) shaped array
    #     (ii)  Signal returns a 0 for the maximum frequency. This is a hack which fixes this issue
    #     (iii) `tcoal` in GWstrain must be in units of days.
    #     """
    #     dict_params_neglected = self._getDictParamsNeglected(1)
    #     h0 = {}
    #     for det in self.detsInNet.keys():
    #         h0[det] = self.detsInNet[det].GWstrain(fgrid, 
    #                                                Mc      = injParams['Mc'].astype('complex128'),
    #                                                eta     = injParams['eta'].astype('complex128'),
    #                                                dL      = injParams['dL'].astype('complex128'),
    #                                                theta   = injParams['theta'].astype('complex128'),
    #                                                phi     = injParams['phi'].astype('complex128'),
    #                                                iota    = injParams['iota'].astype('complex128'),
    #                                                psi     = injParams['psi'].astype('complex128'),
    #                                                tcoal   = injParams['tcoal'].astype('complex128') / self.seconds_per_day, # (iii)
    #                                                Phicoal = injParams['Phicoal'].astype('complex128'),
    #                                                chiS    = injParams['chi1z'].astype('complex128'),
    #                                                chiA    = injParams['chi2z'].astype('complex128'),
    #                                                is_chi1chi2 = 'True',
    #                                                **dict_params_neglected).squeeze() # (i)

    #     return h0


    # See if explicitly not using unpacking dict gives different results?
    # def old_getSignal(self, X, f_grid, det):
    #     """ 
    #     Method to calculate signal for each X[i] over f_grid in detector det.
    #     Remarks:
    #     (i)   Same frequency grid is currently being used for each particle
    #     (ii)  gwfast.GWstrain expects an f x N matrix
    #     (iii) gwfast.GWstrain expects `tcoal` in units of seconds
    #     (iv)  Transpose is included to return an N x f matrix
    #     (v)   X must be (N x f) shaped, for one sample is must be (1 x f) shaped!!!
    #     """
    #     nParticles = X.shape[0]
    #     dict_params_neglected = self._getDictParamsNeglected(nParticles)
    #     fgrids = jnp.repeat(f_grid[...,np.newaxis], nParticles, axis=1) # (i)
    #     X_ = X.T.astype('complex128')
    #     signal = (self.detsInNet[det].GWstrain(fgrids, # (ii)                        
    #                                            Mc      = X_[0],
    #                                            eta     = X_[1],
    #                                            dL      = X_[2],
    #                                            theta   = X_[3],
    #                                            phi     = X_[4],
    #                                            iota    = X_[5],
    #                                            psi     = X_[6],
    #                                            tcoal   = X_[7] / self.seconds_per_day, # (iii)
    #                                            Phicoal = X_[8],
    #                                            chiS    = X_[9],
    #                                            chiA    = X_[10],
    #                                            is_chi1chi2 = 'True',
    #                                            **dict_params_neglected)).T # (iv) 
                            
    #     return signal 


