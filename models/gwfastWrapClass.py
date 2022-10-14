#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
from argparse import ArgumentDefaultsHelpFormatter
from random import uniform
# from tkinter import N
import jax.numpy as jnp
import jax

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

from functools import partial

# nParticles = 1
#%%
class gwfast_class(object):
    
    def __init__(self, NetDict, WaveForm, injParams, priorDict, grid_type='geometric', nParticles=1, EarthMotion=False):
        """
        Args:
            NetDict (dict): dictionary containing the specifications of the detectors in the network
            WaveForm (WaveFormModel): waveform model to use
            injParams (dict): injection parameters
            priorDict (dict): Provides (min,max) range for each coordinate
            grid_type (str): Option to use predefined frequency grid. Options available: 'linear', 'geometric'.
            EarthMotion (bool): include or not the effect of Earth motion. Default is False, meaning motion is not included

        Remarks:
            (1) Example of NetDict: {"H1": {"lat": 46.455, "long": -119.408, "xax": 170.99924234706103, "shape": "L", "psd_path":"path/to/psd"}}
            
            (2) Parameter order as follows: ['Mc','eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
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
        self.grid_type = grid_type
        self.EarthMotion = EarthMotion
        self.injParams = injParams
        self.fmin = 10
        self.fmax = 325


        self.time_scale = 3600. * 24. # Multiply to transform from units of days to units of seconds

        self._initDetectorSignals()
        self._initFrequencyGrid()
        self._initInterpolatedPSD()
        self._initParamNames()
        self._initInjectedSignal(method='sim', add_noise=False)

        self.warmup_derivative(False) # Warmup for JIT compile

    def warmup_derivative(self, warmup):
        self.getDerivativesMinusLogPosterior_ensemble(self._newDrawFromPrior(self.N))

    def _initParamNames(self):
        """
        Predefines useful parameter quantities that may be used in the code.
        """

        self.param_names = {'Mc'         : '$\mathcal{M}_c$',                      # Chirp mass
                            'eta'        : '$\eta$',                               # Symmetric mass ratio 
                            'dL'         : '$d_L$',                                # Luminosity distance
                            'theta'      : '$\theta$',                             # pi/2 - declination angle
                            'phi'        : '$\phi$',                               # Right ascention
                            'iota'       : '$\iota$',                              # Inclination angle
                            'psi'        : '$\psi$',                               # Polarization angle
                            'tcoal'      : '$t_c$',                                # Coalescence time
                            'Phicoal'    : '$\phi_c$',                             # Coalescence phase
                            'chi1z'      : '$\chi_{1z}$',                          # Unitless spin z-component of object 1 
                            'chi2z'      : '$\chi_{2z}$',                          # Unitless spin z-component of object 2
                            'chi1x'      : '$\chi_{1x}$',                          # Unitless spin x-component of object 1
                            'chi2x'      : '$\chi_{2x}$',                          # Unitless spin x-component of object 2
                            'chi1y'      : '$\chi_{1y}$',                          # Unitless spin y-component of object 1
                            'chi2y'      : '$\chi_{2y}$',                          # Unitless spin y-component of object 2
                            'LambdaTilde': '$\tilde{\Lambda}$',                    # ? TODO
                            'deltaLambda': '$\Delta \tilde{\Lambda}$',             # ? TODO
                            'ecc'        : '$\eps$',                               # Eccentricity
                            'chiS'       : '$\chi_S$',                             # Symmetric dimensionless spin
                            'chiA'       : '$\chi_A$'}                             # Antisymmetric dimensionless spin

        self.names_prior_order = list(self.priorDict.keys()) # Defined order
        self.names_neglected = ['chi1x', 'chi2x', 'chi1y', 'chi2y', 'LambdaTilde', 'deltaLambda', 'ecc']
        self.names_inactive = [param for param in self.priorDict.keys() if type(self.priorDict[param]) != list]
        self.names_active = [param for param in self.priorDict.keys() if param not in self.names_inactive]

        self.dict_params_neglected = self.arrayToDict(np.zeros((self.N, len(self.names_neglected))).astype('complex128'), self.names_neglected)
        self.params_inactive = np.array([self.priorDict[param] for param in self.names_inactive])
        self.dict_params_inactive = {param: values for param, values in [(self.names_inactive[i], (np.ones(self.N) * self.params_inactive[i]).astype('complex128'))  for i in range(len(self.names_inactive))]}
        self.true_params = np.array([self.injParams[x].squeeze() for x in self.names_active])

        self.list_active_indicies = []
        for param in self.names_prior_order:
            if param in self.names_active:
                self.list_active_indicies.append(self.names_prior_order.index(param))
                
        self.lower_bound = np.array([self.priorDict[param][0] for param in self.names_active])
        self.upper_bound = np.array([self.priorDict[param][1] for param in self.names_active])
        self.bound_tol = 1e-9

        self.DoF = len(self.names_active)  

        self.indicies_of_inactive_params = []
        for param in self.injParams.keys():
            if param in self.names_inactive:
                self.indicies_of_inactive_params.append(list(self.injParams.keys()).index(param))

    def _initStaticVariablesIntegration(self):
        self.bins = self.fgrid[1:] - self.fgrid[:-1]
        self.bins_over_PSD = {}
        for det in self.detsInNet.keys():
            self.bins_over_PSD = self.bins / self.strainGrid[det][:-1]

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
                                                verbose=False,
                                                useEarthMotion = self.EarthMotion,
                                                fmin=self.fmin, fmax=self.fmax,
                                                is_ASD=True)

        self.signalDerivativeKwargs = dict()
        self.signalDerivativeKwargs['rot'] = 0.

        # Remark: If we want to use chiS, chiA vs chi1z, chi2, set appropriate flags to True vs False

        self.signalDerivativeKwargs['use_m1m2'] = False

        self.signalDerivativeKwargs['use_chi1chi2'] = True
        # self.signalDerivativeKwargs['use_chi1chi2'] = False
        
        self.signalDerivativeKwargs['use_prec_ang'] = False
        self.signalDerivativeKwargs['computeAnalyticalDeriv'] = True

        self.signalKwargs = dict()
        self.signalKwargs['rot'] = 0.
        self.signalKwargs['is_m1m2'] = False

        self.signalKwargs['is_chi1chi2'] = True
        # self.signalKwargs['is_chi1chi2'] = False
        
        self.signalKwargs['is_prec_ang'] = False

    def _initFrequencyGrid(self):
        # Setup frequency grid
        fcut = self.wf_model.fcut(**self.injParams)
        if self.fmax is None:
            self.fmax = fcut
        else:
            fcut = jnp.where(fcut > self.fmax, self.fmax, fcut)

        self.grid_to_use = 'geometric'
        # self.grid_to_use = 'linear'

        if self.grid_to_use == 'geometric':
            self.grid_resolution = int(100)
            self.fgrid = jnp.geomspace(self.fmin, fcut, num=self.grid_resolution)
        elif self.grid_to_use == 'linear':
            self.df = 1./5
            self.grid_resolution = int(jnp.floor(jnp.real((1 + (fcut - self.fmin) / self.df))))
            self.fgrid = jnp.linspace(self.fmin, fcut, num=self.grid_resolution)

        self.fgrids = jnp.repeat(self.fgrid, self.N, axis=1)

        # self.fgrid = self.fgrid # So we can use it elsewhere normally

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
        np.random.seed(None)

        dict_params_neglected = {param: values for param, values in [(self.names_neglected[i], np.array([0.]).astype('complex128'))  for i in range(len(self.names_neglected))]}
        dict_params_inactive  = {param: values for param, values in [(self.names_inactive[i],  np.array([self.params_inactive[i]]).astype('complex128'))  for i in range(len(self.names_inactive))]}
        dict_params_active    = {param: values for param, values in [(self.names_active[i],    np.array([self.true_params[i]]).astype('complex128'))  for i in range(len(self.names_active))]}
        self.signal_data = {}

        dict_params_active['chiS'] = dict_params_active.pop('chi1z')
        dict_params_active['chiA'] = dict_params_active.pop('chi2z')



        
        dict_params_active['tcoal'] /= self.time_scale # Remark: Rexpress in days for input into signal.GWstrain. Input given in seconds.

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
        
    
    def arrayToDict(self, thetas, names):
        # Remark: We use D here so that the method works generally. 
        D = len(names)
        return {param: values for param, values in [(names[i], thetas.T[i]) for i in range(D)]}

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
            
            !!!REMARK!!! : CODE HAS BEEN CHANGED TO RETURN N x F matrix!!!!
        """
        # Remark: Express tcoal in days for input into signal.GWstrain
        # tcelem = self.wf_model.ParNums['tcoal'] # GMST accounts for geometry of earth spinning

        X = X.astype('complex128')
        Mc      = X[:,0]           
        eta     = X[:,1]
        dL      = X[:,2]
        theta   = X[:,3]
        phi     = X[:,4]
        iota    = X[:,5]
        psi     = X[:,6]
        tcoal   = X[:,7] / self.time_scale
        Phicoal = X[:,8]
        chiS    = X[:,9]
        chiA    = X[:,10]

        # dict_params_active = self.arrayToDict(X.astype('complex128'), self.names_active)

        # dict_params_active['tcoal'] = dict_params_active['tcoal'] / self.time_scale

        residual = {}


        
        for det in self.detsInNet.keys():
            residual[det] = (self.detsInNet[det].GWstrain(self.fgrids, 
                                                          Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA,
                                                        #  **dict_params_active, 
                                                         **self.dict_params_inactive, 
                                                         **self.dict_params_neglected, 
                                                         **self.signalKwargs) - self.signal_data[det]).T # Return a N x f matrix
                         
        return residual 
        # signal data is f x 1 array
        # output of gw strain is 
        #  
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

        # !!!REMARK!!!: Returns (d, Nev, F) shaped array instead of a (d, F, Nev) shaped array, where F is the size of the frequency grid
        # TODO Ask Francesco if this is a bug or expected.
        # TODO Ask Francesco if d follows the order defined above.
        """

        # dict_params_active = self.arrayToDict(X.astype('complex128'), self.names_active)

        # Remark: There are two modifications that need to be performed here to correctly accept t_c in seconds!
        # dict_params_active['tcoal'] = dict_params_active['tcoal'] / self.time_scale # Correction 1

        X = X.astype('complex128')
        Mc      = X[:,0]           
        eta     = X[:,1]
        dL      = X[:,2]
        theta   = X[:,3]
        phi     = X[:,4]
        iota    = X[:,5]
        psi     = X[:,6]
        tcoal   = X[:,7] / self.time_scale # Correction #1
        Phicoal = X[:,8]
        chiS    = X[:,9]
        chiA    = X[:,10]






        residualJac = {}
        
        # This is needed to change units in tc and variable from iota to cos(iota)
        # print(tcelem)
        # iotaelem = self.wf_model.ParNums['iota']
        tcelem = self.wf_model.ParNums['tcoal'] # GMST accounts for geometry of earth spinning




        for det in self.detsInNet.keys():
            residualJac[det] = self.detsInNet[det]._SignalDerivatives_use(self.fgrids, 
                                                                          Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chiS, chiA,
                                                                        #   **dict_params_active, 
                                                                          **self.dict_params_inactive, 
                                                                          **self.dict_params_neglected, 
                                                                          **self.signalDerivativeKwargs)#.at[tcelem,:,:].divide(3600.*24.)

            residualJac[det] = residualJac[det].at[tcelem].divide(self.time_scale) # Correction 2

            # Remark: signal._SignalDerivatvies_use eats t_c in units of fraction of the day. Convert sec to days.
        return residualJac
            
    def _riemannSum(self, integrand, grid, axis=-1):
        return jnp.sum(integrand[..., :-1] * (grid[1:] - grid[:-1]), axis=axis)

    def _signal_inner_product(self, a, b, det, mode):
        quadrature_rule = 'trapezoid' 
        # quadrature_rule = 'riemann'

        if mode == 'l':
            integrand = contract('Nf,  Nf,  f -> Nf',   a.conjugate(), b, 1 / self.strainGrid[det])
        elif mode == 'g':
            integrand = contract('dNf, Nf,  f -> Ndf',  a.conjugate(), b, 1 / self.strainGrid[det])
        elif mode == 'h':
            integrand = contract('dNf, bNf, f -> Ndbf', a.conjugate(), b, 1 / self.strainGrid[det])

        if quadrature_rule == 'riemann':
            return 4 * self._riemannSum(integrand.real, self.fgrid.squeeze())
        elif quadrature_rule == 'trapezoid':
            return 4 * jnp.trapz(integrand.real, self.fgrid.squeeze())

    # def signal_inner_product_like(self, a, b, det):
    #     integrand = contract('Nf,  Nf,  f -> Nf',   a.conjugate(), b, 1 / self.strainGrid[det])
    #     return 4 * jnp.trapz(integrand.real, self.fgrid.squeeze())



    # @partial(jax.jit, static_argnums=(0,))
    def getMinusLogPosterior___(self, thetas):
        """_summary_

        Args:
            thetas (_type_): _description_

        Returns:
            _type_: _description_

        # See arxiv:1809.02293 Eq 42.
        """
        residual_dict = self._getResidual_Vec(thetas) 
        log_like = jnp.zeros(self.N)
        for det in self.detsInNet.keys():
            log_like = log_like + self._signal_inner_product(residual_dict[det], residual_dict[det], det, 'l')
        return log_like / 2

    # @partial(jax.jit, static_argnums=(0,))
    def getDerivativesMinusLogPosterior_ensemble(self, thetas):
        residual_dict = self._getResidual_Vec(thetas) 
        jacResidual_dict = self._getJacobianResidual_Vec(thetas)
        grad_log_like = jnp.zeros((self.N, self.DoF))
        GN = jnp.zeros((self.N, self.DoF, self.DoF))
        for det in self.detsInNet.keys():
            grad_log_like = grad_log_like + self._signal_inner_product(jacResidual_dict[det], residual_dict[det], det, 'g')#[:, self.list_active_indicies]
            GN = GN + self._signal_inner_product(jacResidual_dict[det], jacResidual_dict[det], det, 'h')#[:, self.list_active_indicies][..., self.list_active_indicies]
        return (grad_log_like, GN)

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
            low = self.priorDict[param][0]
            high = self.priorDict[param][1]
            # padding = (high-low)/10
            # prior_draw[:, i] = np.random.uniform(low=low+padding, high=high-padding, size=nParticles)
            prior_draw[:, i] = np.random.uniform(low=low, high=high, size=nParticles)
        
        return prior_draw

    def getCrossSection(self, a, b):
        ngrid = int(np.sqrt(self.N))
        # a, b are the parameters for which we want the marginals:
        x = np.linspace(self.priorDict[a][0], self.priorDict[a][1], ngrid)
        y = np.linspace(self.priorDict[b][0], self.priorDict[b][1], ngrid)
        X, Y = np.meshgrid(x, y)
        particle_grid = np.zeros((ngrid ** 2, self.DoF))
        index1 = self.names_active.index(a)
        index2 = self.names_active.index(b)
        parameter_mesh = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T
        particle_grid[:, index1] = parameter_mesh[:, 0]
        particle_grid[:, index2] = parameter_mesh[:, 1]
        for i in range(self.DoF): # Fix all other parameters
            if i != index1 and i != index2:
                particle_grid[:, i] = np.ones(ngrid ** 2) * self.true_params[i]
        Z = np.exp(-1 * self.getMinusLogPosterior_ensemble(particle_grid).reshape(ngrid,ngrid))
        fig, ax = plt.subplots(figsize = (5, 5))
        cp = ax.contourf(X, Y, Z)
        # cbar = fig.colorbar(cp)
        plt.colorbar(cp)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title('Analytically calculated marginal')
        filename = a + b + '.png'
        path = os.path.join('marginals', filename)
        fig.savefig(path)

    def _inBounds(self, X, lower_bound, upper_bound, tol=1e-9):
        # Pad particle behavior near boundaries
        Y = copy.deepcopy(X)

        nParticles = Y.shape[0]
        DoF = Y.shape[1]

        lower_bound = np.tile(lower_bound, nParticles).reshape(nParticles, DoF)
        upper_bound = np.tile(upper_bound, nParticles).reshape(nParticles, DoF)

        below = Y <= lower_bound 
        Y[below] = lower_bound[below] + tol

        above = Y >= upper_bound
        Y[above] = upper_bound[above] - tol

        return Y




    # def dictToArray(self):
    #     pass

    # @jax.jit
    # @partial(jax.jit, static_argnums=(0,))
    # def getMinusLogPosterior_ensemble___(self, thetas):
    #     """ 
    #     thetas = N x DoF
    #     See arxiv:1809.02293 Eq 42.
    #     """
    #     residual_dict = self._getResidual_Vec(thetas) 
    #     log_like = jnp.zeros(thetas.shape[0])
    #     quadrature = 'trapezoid' # 'riemann'  
    #     if quadrature == 'riemann':
    #         for det in self.detsInNet.keys():
    #             tmp = (self.fgrid[1:] - self.fgrid[:-1]) / self.strainGrid[det][:-1]
    #             norm = jnp.abs(residual_dict[det]) ** 2
    #             log_like += jnp.sum(norm[:-1]) * tmp
    #         return 2 * log_like 
    #     elif quadrature == 'trapezoid':
    #         tmp1 = (self.fgrid[1:] - self.fgrid[:-1]).squeeze()
    #         for det in self.detsInNet.keys():
    #             # integrand = contract('fm, f -> mf', jnp.abs(residual_dict[det]) ** 2, 1 / self.strainGrid[det]) # OLD
    #             integrand = contract('mf, f -> mf', jnp.abs(residual_dict[det]) ** 2, 1 / self.strainGrid[det]) # MODIFIED FOR MF RESIDUAL
    #             # log_like += 2 * jnp.trapz(integrand, self.fgrid.squeeze())
    #             log_like += jnp.sum((integrand[:, 1:] + integrand[:, :-1]) * tmp1)
    #         return log_like

    # @jax.jit
    # def getGradientMinusLogPosterior_ensemble(self, thetas):
    #     # REMARK: Returns (d, Nev, F) shaped array instead of a (d, F, Nev) shaped array, where F is the size of the frequency grid
    #     # TODO Ask Francesco if this is a bug or expected.
    #     """ 
    #     thetas = N x DoF
    #     """
    #     residual_dict = self._getResidual_Vec(thetas) # Input is reversed here
    #     jacResidual_dict = self._getJacobianResidual_Vec(thetas)
    #     grad_log_like = jnp.zeros(thetas.shape).astype('complex128')
    #     for det in self.detsInNet.keys():
    #         grad_log_like += contract('dNf, fN, f -> Nd', jacResidual_dict[det].conjugate(), residual_dict[det], 1 / self.strainGrid[det])[:, self.list_active_indicies]
    #     return (4 * grad_log_like.real * self.df)

    # # @jax.jit
    # def getGNHessianMinusLogPosterior_ensemble(self, thetas):
    #     jacResidual_dict = self._getJacobianResidual_Vec(thetas)
    #     GN = jnp.zeros((self.N, self.DoF, self.DoF)).astype('complex128')
    #     for det in self.detsInNet.keys():
    #         GN += contract('dNf, bNf, f -> Ndb', jacResidual_dict[det].conjugate(), jacResidual_dict[det], 1 / self.strainGrid[det])[:, self.list_active_indicies, self.list_active_indicies]
    #     return 4 * self.df * GN.real

    # @partial(jax.jit, static_argnums=(0,))
    # def getDerivativesMinusLogPosterior_ensemble__(self, thetas):
    #     residual_dict = self._getResidual_Vec(thetas) 
    #     jacResidual_dict = self._getJacobianResidual_Vec(thetas)
    #     grad_log_like = jnp.zeros(thetas.shape)
    #     GN = jnp.zeros((self.N, self.DoF, self.DoF))
    #     # GN_test = jnp.zeros((self.N, self.DoF, self.DoF))

    #     if self.grid_to_use == 'linear':
    #         for det in self.detsInNet.keys():
    #             grad_log_like += contract('dNf, fN, f -> Nd', jacResidual_dict[det].conjugate(), residual_dict[det], 1 / self.strainGrid[det])[:, self.list_active_indicies].real
    #             GN += contract('dNf, bNf, f -> Ndb', jacResidual_dict[det].conjugate(), jacResidual_dict[det], 1 / self.strainGrid[det])[:, self.list_active_indicies][:, ..., self.list_active_indicies].real
    #         return (4 * grad_log_like.real * self.df, 4 * self.df * GN.real)
    #     elif self.grid_to_use == 'geometric':
    #         tmp1 = (self.fgrid[1:] - self.fgrid[:-1]).squeeze()
    #         for det in self.detsInNet.keys():
    #             integrand_grad = contract('dNf, fN -> Ndf', jacResidual_dict[det].conjugate(), residual_dict[det])[:, self.list_active_indicies].real / self.strainGrid[det]
    #             grad_log_like += 2 * jnp.sum((integrand_grad[...,1:] + integrand_grad[...,:-1]) * tmp1, axis=-1)

    #             # DEBUG: Compare to jnp.trapz method
    #             # grad_log_like += 4 * jnp.trapz(integrand_grad, self.fgrid.squeeze())

    #             GN += contract('dNf, bNf, f -> Ndb', jacResidual_dict[det].conjugate()[..., 1:], jacResidual_dict[det][..., 1:], tmp1 / self.strainGrid[det][1:])[:, self.list_active_indicies][..., self.list_active_indicies].real
    #             GN += contract('dNf, bNf, f -> Ndb', jacResidual_dict[det].conjugate()[..., :-1], jacResidual_dict[det][..., :-1], tmp1 / self.strainGrid[det][:-1])[:, self.list_active_indicies][..., self.list_active_indicies].real
    #             GN *= 2

                # integrand_GN = contract('dNf, bNf -> Ndbf', jacResidual_dict[det].conjugate(), jacResidual_dict[det]).real[:, self.list_active_indicies][:, :, self.list_active_indicies] / self.strainGrid[det]
                # GN_test += 4 * jnp.trapz(integrand_GN, self.fgrid.squeeze())

            # return (grad_log_like, GN)



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
