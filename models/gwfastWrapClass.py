#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
from argparse import ArgumentDefaultsHelpFormatter
from random import uniform
from tkinter import N
import jax.numpy as jnp

import numpy as np
import sys
import os
import time
import copy
import time
import matplotlib.pyplot as plt

import gwfast.gwfastUtils as utils
import gwfast.gwfastGlobals as glob
import gwfast.signal as signal

from gwfast.network import DetNet
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD
# from astropy.cosmology import Planck18
from opt_einsum import contract

# nParticles = 1
#%%
class gwfast_class(object):
    
    def __init__(self, NetDict, WaveForm, injParams, priorDict, df, nParticles, fmax=None, fmin=10, EarthMotion=False, customseed=None):
        """
        Args:
            NetDict (dict): dictionary containing the specifications of the detectors in the network
            WaveForm (WaveFormModel): waveform model to use
            injParams (dict): injection parameters
            df (float): inverse of sampling frequency - linear spacing
            fmax (float): maximum frequency at which to the frequency grid
            fmin (float): minimum frequency of the frequency grid
            EarthMotion (bool): include or not the effect of Earth motion. Default is False, meaning motion is not included
            customseed (float): input seed for the noise generation in the data

        E.g NetDict: {"H1": {"lat": 46.455, "long": -119.408, "xax": 170.99924234706103, "shape": "L", "psd_path":"path/to/psd"}}
        """

        # sSVN_GW related attributes
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0
        self.id = 'gwfast_model'
        self.priorDict = priorDict
        self.N = nParticles

        # gw_fast related attributes
        self.wf_model = WaveForm
        self.NetDict = NetDict
        self.fmin = fmin
        self.fmax = fmax
        self.EarthMotion = EarthMotion
        self.injParams = injParams
        self.df = df
        if customseed is not None:
            self.seed = customseed
        else:
            self.seed = None

        # Parameter order convention set here
        # self.names = ['Mc','eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']

        self._initDetectorSignals()
        self._initFrequencyGrid()
        self._initInterpolatedPSD()
        self._initParamNames()
        self._initInjectedSignal(method='sim', add_noise=False)

        self.DoF = len(self.names_active)  

        self.indicies_of_inactive_params = []
        for param in self.injParams.keys():
            if param in self.names_inactive:
                self.indicies_of_inactive_params.append(list(self.injParams.keys()).index(param))

    def _initParamNames(self):
        self.param_names = {'Mc': '$\mathcal{M}_c$',                      # Chirp mass
                            'eta': '$\eta$',                              # Symmetric mass ratio 
                            'dL': '$d_L$',                                # Luminosity distance
                            'theta': '$\theta$',                          # pi/2 - declination angle
                            'phi': '$\phi$',                              # Right ascention
                            'iota': '$\iota$',                            # Inclination angle
                            'psi': '$\psi$',                              # Polarization angle
                            'tcoal': '$t_c$',                             # Coalescence time
                            'Phicoal': '$\phi_c$',                        # Coalescence phase
                            'chi1z': '$\chi_{1z}$',                       # Unitless spin z-component of object 1 
                            'chi2z': '$\chi_{2z}$',                       # Unitless spin z-component of object 2
                            'chi1x': '$\chi_{1x}$',                       # Unitless spin x-component of object 1
                            'chi2x': '$\chi_{2x}$',                       # Unitless spin x-component of object 2
                            'chi1y': '$\chi_{1y}$',                       # Unitless spin y-component of object 1
                            'chi2y': '$\chi_{2y}$',                       # Unitless spin y-component of object 2
                            'LambdaTilde': '$\tilde{\Lambda}$',           # ? TODO
                            'deltaLambda': '$\Delta \tilde{\Lambda}$',    # ? TODO
                            'ecc': '$\eps$',                              # Eccentricity
                            'chiS': '$\chi_{1z}$',                        # Flag ischi1chi2=True. This is just for convenience
                            'chiA': '$\chi_{2z}$'}                              

        # self.names_active_modified = copy.deepcopy(self.names_active)

        self.names_prior_order = list(self.priorDict.keys()) # Defined order
        self.names_neglected = ['chi1x', 'chi2x', 'chi1y', 'chi2y', 'LambdaTilde', 'deltaLambda', 'ecc']
        self.names_inactive = [param for param in self.priorDict.keys() if type(self.priorDict[param]) != list]
        self.names_active = [param for param in self.priorDict.keys() if param not in self.names_inactive]
        # self.names_active = list(self.priorDict.keys()) # Actual parameters
        self.dict_params_neglected = self.arrayToDict(np.zeros((self.N, len(self.names_neglected))).astype('complex128'), self.names_neglected)
        self.params_inactive = np.array([self.priorDict[param] for param in self.names_inactive])
        self.dict_params_inactive = {param: values for param, values in [(self.names_inactive[i], (np.ones(self.N) * self.params_inactive[i]).astype('complex128'))  for i in range(len(self.names_inactive))]}
        self.true_params = np.array([self.injParams[x].squeeze() for x in self.names_active])

        self.list_active_indicies = []
        for param in self.names_prior_order:
            if param in self.names_active:
                self.list_active_indicies.append(self.names_prior_order.index(param))
                
    def _initDetectorSignals(self): 
        # Initialise the signal objects
        self.detsInNet = {}
        for d in self.NetDict.keys():
            self.detsInNet[d] = signal.GWSignal(self.wf_model,
                                                psd_path=self.NetDict[d]['psd_path'],
                                                detector_shape = self.NetDict[d]['shape'],
                                                det_lat= self.NetDict[d]['lat'],
                                                det_long=self.NetDict[d]['long'],
                                                det_xax=self.NetDict[d]['xax'],
                                                verbose=True,
                                                useEarthMotion = self.EarthMotion,
                                                fmin=self.fmin, fmax=self.fmax,
                                                is_ASD=True)

        self.signalDerivativeKwargs = dict()
        self.signalDerivativeKwargs['rot'] = 0.
        self.signalDerivativeKwargs['use_m1m2'] = False
        self.signalDerivativeKwargs['use_chi1chi2'] = True
        self.signalDerivativeKwargs['use_prec_ang'] = False
        self.signalDerivativeKwargs['computeAnalyticalDeriv'] = True

        self.signalKwargs = dict()
        self.signalKwargs['rot'] = 0.
        self.signalKwargs['is_m1m2'] = False
        self.signalKwargs['is_chi1chi2'] = True
        self.signalKwargs['is_prec_ang'] = False

    def _initFrequencyGrid(self):
        # Setup linear frequency grid
        fcut = self.wf_model.fcut(**self.injParams)
        if self.fmax is None:
            self.fmax = fcut
        else:
            fcut = jnp.where(fcut > self.fmax, self.fmax, fcut)
        self.grid_resolution = jnp.floor(jnp.real((1 + (fcut - self.fmin) / self.df)))
        self.fgrid = jnp.linspace(self.fmin, fcut, num=int(self.grid_resolution))
        self.fgrids = jnp.repeat(self.fgrid, self.N, axis=1)

    def _initInterpolatedPSD(self):
        # Remark: This returns $S_n(f)$ for the desired frequency grid (self.fgrid)
        self.strainGrid = {}
        for det in self.detsInNet.keys():
            # Remark: Padded with 1's to ensure damping of frequency bins outside PSD range.
            self.strainGrid[det] = jnp.interp(self.fgrid, self.detsInNet[det].strainFreq, self.detsInNet[det].noiseCurve, left=1., right=1.).squeeze()
    
    def _initInjectedSignal(self, method='sim', add_noise=False, **kwargs):
        """
        Function to choose the input signal data. I.e, the detectorwise responses to gravitational radiation
        
        Args:
            method (str): input method,
                          - for the moment only 'sim' is implemented, meaning that the signal is generated from input parameters
                            In this case the input parameters have to be provided in a dictionary.
            add_noise (bool): bool to choose whether or not to add noise to the simulated signal
            
        """
        if method=='sim':
            self._simulate_signal(add_noise)
        else:
            raise ValueError('Method not yet implemented.')

    def _simulate_signal(self, add_noise=False):
        """
        Function to simulate a mock signal in each considered detector given its parameters.
        The result is set as a class attribute, as well as the frequency grid.
        Args:
            add_noise (bool): bool to choose whether or not to add noise to the simulated signal
            df (float): spacing of the frequency grid
            injParams (dict): dictionary containing the input parameters, built as
                injParams = {'Mc':np.array([...]), 'eta':np.array([...]), 'dL':np.array([...]),
                             'theta':np.array([...]), 'phi':np.array([...]), 'iota':np.array([...]),
                             'psi':np.array([...]), 'tcoal':np.array([...]), 'Phicoal':np.array([...]),
                             'chi1z':np.array([...]), 'chi2z':np.array([...]),
                             'Lambda1':np.array([...]), 'Lambda2':np.array([...])}

        """
        # Set the seed for reproducibility
        np.random.seed(self.seed)

        dict_params_neglected = {param: values for param, values in [(self.names_neglected[i], np.array([0.]).astype('complex128'))  for i in range(len(self.names_neglected))]}
        dict_params_inactive  = {param: values for param, values in [(self.names_inactive[i],  np.array([self.params_inactive[i]]).astype('complex128'))  for i in range(len(self.names_inactive))]}
        dict_params_active    = {param: values for param, values in [(self.names_active[i],    np.array([self.true_params[i]]).astype('complex128'))  for i in range(len(self.names_active))]}
        self.signal_data = {}
        
        # Compute the signal as seen in each detector and store the result
        for det in self.detsInNet.keys():
            self.signal_data[det] = self.detsInNet[det].GWstrain(self.fgrid, 
                                                                 **dict_params_active, 
                                                                 **dict_params_inactive, 
                                                                 **dict_params_neglected, 
                                                                 **self.signalKwargs)

            if add_noise:
                # Add Gaussian noise with std given by the detector ASD if needed
                self.signal_data[det] = self.signal_data[det] + np.random.normal(loc=0.,scale=self.strainGrid)
        
    def _getResidual_Vec(self, X):
    
        """
        Function to compute the difference between the data signal and the template (with parameters X).
        
        Args:
         X (nd.array): (d, Nev) shaped array, with d being the size of the parameter space and Nev the nuber of events to simulate.
             Remark: Represents point in d-dimensional parameter space $\chi$
                     The order is Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z (, LambdaTilde, deltaLambda)

         Returns dict of (F, Nev) shaped (nd.array) with F being the size of the frequency grid
            Remark: The keys of the dictionary represent the detectors.
                    The arrays represent the residuals for each frequency bin, up to bin $F$.
        """
        dict_params_active = self.arrayToDict(X.astype('complex128'), self.names_active)
        residual = {}
        for det in self.detsInNet.keys():
            residual[det] = self.detsInNet[det].GWstrain(self.fgrids, 
                                                         **dict_params_active, 
                                                         **self.dict_params_inactive, 
                                                         **self.dict_params_neglected, 
                                                         **self.signalKwargs) - self.signal_data[det]
        return residual
        
    def _getJacobianResidual_Vec(self, X):
    
        """
        Function to compute the derivatives of the template with parameters theta.
        
        Args:
        X (nd.array): (N x d) now (UPDATED)

         X (nd.array): (d, Nev) shaped array, with d being the size of the parameter space and Nev the nuber of events to simulate.
         Remark: Represents point in d-dimensional parameter space $\chi$
                 The order is Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z (, LambdaTilde, deltaLambda)
         
         Returns dict of (d, F, Nev) shaped (nd.array) with F being the size of the frequency grid
            Remark: The keys of the dictionary represent the detectors.
                    The arrays represent Jacobian of residual evaluated at theta.
        """
        # REMARK: Returns (d, Nev, F) shaped array instead of a (d, F, Nev) shaped array, where F is the size of the frequency grid
        # TODO Ask Francesco if this is a bug or expected.

        dict_params_active = self.arrayToDict(X.astype('complex128'), self.names_active)
        residualJac = {}
        
        # This is needed to change units in tc and variable from iota to cos(iota)
        # tcelem   = self.wf_model.ParNums['tcoal'] # GMST accounts for geometry of earth spinning
        # print(tcelem)
        # iotaelem = self.wf_model.ParNums['iota']


        for det in self.detsInNet.keys():
            residualJac[det] = self.detsInNet[det]._SignalDerivatives_use(self.fgrids, 
                                                                          **dict_params_active, 
                                                                          **self.dict_params_inactive, 
                                                                          **self.dict_params_neglected, 
                                                                          **self.signalDerivativeKwargs)
            # residualJac[det] = residualJac[det].at[tcelem,:,:].divide(3600.*24.)
        return residualJac

    def arrayToDict(self, thetas, names):
        D = len(names)
        return {param: values for param, values in [(names[i], thetas.T[i]) for i in range(D)]}

    # def dictToArray(self):
    #     pass

    def getMinusLogLikelihood_ensemble(self, thetas):
        """ 
        thetas = N x DoF
        """
        residual_dict = self._getResidual_Vec(thetas) # Remark: gwfast_class uses reversed convention for particles
        log_like = jnp.zeros(thetas.shape[0]).astype('complex128')
        for det in self.detsInNet.keys():
            norm = jnp.abs(residual_dict[det]) ** 2
            log_like += contract('fm, f -> m', norm, 1 / self.strainGrid[det])
        return 2 * log_like.real * self.df

    def getGradientMinusLogPosterior_ensemble(self, thetas):
        # REMARK: Returns (d, Nev, F) shaped array instead of a (d, F, Nev) shaped array, where F is the size of the frequency grid
        # TODO Ask Francesco if this is a bug or expected.
        """ 
        thetas = N x DoF
        """
        residual_dict = self._getResidual_Vec(thetas) # Input is reversed here
        jacResidual_dict = self._getJacobianResidual_Vec(thetas)
        grad_log_like = np.zeros(thetas.shape).astype('complex128')
        for det in self.detsInNet.keys():
            grad_log_like += contract('dNf, fN, f -> Nd', jacResidual_dict[det].conjugate(), residual_dict[det], 1 / self.strainGrid[det])[:, self.list_active_indicies]
        return (4 * grad_log_like.real * self.df)

    def getGNHessianMinusLogPosterior_ensemble(self, thetas):
        jacResidual_dict = self._getJacobianResidual_Vec(thetas)
        GN = np.zeros((self.N, self.DoF, self.DoF)).astype('complex128')
        for det in self.detsInNet.keys():
            GN += contract('dNf, bNf, f -> Ndb', jacResidual_dict[det].conjugate(), jacResidual_dict[det], 1 / self.strainGrid[det])
        return 4 * self.df * GN.real

    def getDerivativesMinusLogPosterior_ensemble(self, thetas):
        residual_dict = self._getResidual_Vec(thetas) 
        jacResidual_dict = self._getJacobianResidual_Vec(thetas)
        grad_log_like = np.zeros(thetas.shape).astype('complex128')
        GN = np.zeros((self.N, self.DoF, self.DoF)).astype('complex128')
        for det in self.detsInNet.keys():
            grad_log_like += contract('dNf, fN, f -> Nd', jacResidual_dict[det].conjugate(), residual_dict[det], 1 / self.strainGrid[det])
            GN += contract('dNf, bNf, f -> Ndb', jacResidual_dict[det].conjugate(), jacResidual_dict[det], 1 / self.strainGrid[det])

        return (4 * grad_log_like.real * self.df, 4 * self.df * GN.real)

    def _newDrawFromPrior(self, nParticles):
        """
        Return samples from a uniform prior.
        Included for convenience.
        Args:
            nParticles (int): Number of samples to draw.

        Returns: (array) nSamples x DoF array of representative samples

        """
        prior_draw = np.zeros((nParticles, self.DoF))

        for i, param in enumerate(self.names_active): # Assuming uniform on all parameters
            prior_draw[:, i] = np.random.uniform(low=self.priorDict[param][0], high=self.priorDict[param][1], size=nParticles)
        
        return prior_draw










        # Mc_prior = np.random.uniform(low=32, high=36, size=nParticles)

        # eta_prior = np.random.uniform(low=0.1, high=0.25, size=nParticles)

        # dL_prior = np.random.uniform(low=0.2, high=0.5, size=nParticles)
        
        # theta_prior = np.random.uniform(low=0., high=np.pi, size=nParticles)
        
        # phi_prior = np.random.uniform(low=0., high=2*np.pi, size=nParticles)
        
        # cos_iota_prior = np.random.uniform(low=-1., high=1., size=nParticles)
        
        # psi_prior = np.random.uniform(low=0., high=np.pi, size=nParticles)
        
        # tcoal_prior = np.random.uniform(low=0, high=0.001, size=nParticles)
        
        # Phicoal_prior = np.random.uniform(low=0., high=0.001, size=nParticles)
        
        # chi1z_prior = np.random.uniform(low=-1., high=1., size=nParticles)
        
        # chi2z_prior = np.random.uniform(low=-1., high=1., size=nParticles)

        # prior_draw[:,0] = Mc_prior
        # prior_draw[:,1] = eta_prior
        # prior_draw[:,2] = dL_prior
        # prior_draw[:,3] = theta_prior
        # prior_draw[:,4] = phi_prior
        # prior_draw[:,5] = cos_iota_prior
        # prior_draw[:,6] = psi_prior
        # prior_draw[:,7] = tcoal_prior
        # prior_draw[:,8] = Phicoal_prior
        # prior_draw[:,9] = chi1z_prior
        # prior_draw[:,10] = chi2z_prior

        # return prior_draw

        # return np.random.multivariate_normal(mean=self.true_params.squeeze(), cov=0.01 * np.eye(self.DoF), size=nParticles) + 0.1

        # return np.random.uniform(low=0.2, high=0.25, size=(nParticles, self.DoF))








################################################
# UNIT TESTS
################################################

# # %% Define injected signal parameters (GW170817)
# z = np.array([0.00980])
# tGPS = np.array([1187008882.4])

# Mc = np.array([1.1859])*(1.+z)
# dL = Planck18.luminosity_distance(z).value/1000
# theta = np.array([np.pi/2. + 0.4080839999999999]) # shifted declination
# phi = np.array([3.4461599999999994]) # right ascention
# iota = np.array([2.545065595974997])
# psi = np.array([0.])
# # tcoal = utils.GPSt_to_LMST(tGPS, lat=0., long=0.) # GMST is LMST computed at long = 0°
# tcoal = np.array([0.])
# eta = np.array([0.24786618323504223])
# Phicoal = np.array([0.])
# chi1z = np.array([0.005136138323169717])
# chi2z = np.array([0.003235146993487445])
# injParams = {'Mc': Mc, 'eta': eta, 'dL': dL, 'theta': theta, 'phi': phi, 'iota': iota, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal, 'chi1z': chi1z, 'chi2z': chi2z}

# #%% Setup network class and precalculate injected signal
# alldetectors = copy.deepcopy(glob.detectors)
# LVdetectors = {det:alldetectors[det] for det in ['L1', 'H1', 'Virgo']}
# print('Using detectors '+str(list(LVdetectors.keys())))
# LVdetectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt')
# LVdetectors['H1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt')
# LVdetectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', 'Hrec_hoft_V1O2Repro2A_16384Hz.txt')
# waveform = TaylorF2_RestrictedPN()
# fmin = 32
# fmax = 512
# df = 1./10
# # Remark : sampling rate should be equal to 2*f_max (Nyquist theorem)
# model = gwfast_class(LVdetectors, waveform, fmin=fmin, fmax=fmax)
# model.get_signal(method='sim', add_noise=False, df=df, **injParams)




# #%%

# grid = model.fgrid.squeeze()
# #%%

# args = np.argwhere(np.logical_and(grid > 100, grid < 112))
# #%% 
# # plt.plot(model.fgrid.squeeze()[args], np.abs(model.signal_data['H1'].real.squeeze()[args]))
# plt.plot(model.fgrid.squeeze(), (model.signal_data['H1'].real.squeeze()))
# # #%%
# # TODO REMARK: Why is the coalescence time effecting the # of oscillations?
# #%% Create test particles
# N = 2
# DoF = 11
# thetaTrue = np.array([Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z]) 
# # jitter = np.random.uniform(low=0.1, high=0.25, size=(DoF, N))
# jitter = np.random.uniform(low=0.1, high=0.25, size=(DoF, nParticles))
# testParticles = thetaTrue + jitter

# #%%
# x = testParticles[:,0]
# test = model.getMinusLogLikelihood_single(x)


# #%%
# import numdifftools as nd
# test_a = model.getGradientMinusLogPosterior_ensemble(testParticles.T)
# #%%
# test_b = nd.Gradient(model.getMinusLogLikelihood_ensemble)(testParticles.T)
# #%%
# assert np.allclose(test_a, test_b)


# #%%
# a = model.getMinusLogLikelihood_ensemble(testParticles.T)

# #%% Get residuals
# r = model._getResidual_Vec(testParticles)

# #%% Get Jacobian of residuals
# Jr = model._getJacobianResidual_Vec(testParticles)

# #%%


# # strainGrid = jnp.interp(self.fgrid, self.detsInNet[det].strainFreq, self.detsInNet[det].noiseCurve, left=1., right=1.) # What does left/right=1 mean physically?
# # strainGrid = jnp.interp(model.fgrid, model.detsInNet[det].strainFreq, model.detsInNet[det].noiseCurve, left=1., right=1.) # What does left/right=1 mean physically?
# #%%

# # det = 'H1'
# # plt.plot(jnp.log(model.detsInNet[det].noiseCurve[0:12000]))


# # %%

# %%
