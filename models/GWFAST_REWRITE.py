#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#%%
from random import uniform
import jax.numpy as jnp
import jax
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import gwfast.signal as signal
from gwfast.network import DetNet
from opt_einsum import contract
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

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

        """

        # sSVN_GW related attributes
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0
        self.id = 'gwfast_model'
        self.priorDict  = priorDict
        self.nParticles = nParticles

        # gw_fast related attributes
        self.wf_model = WaveForm
        self.NetDict = NetDict
        self.grid_type = grid_type
        self.EarthMotion = EarthMotion
        self.injParams = injParams
        self.DoF = 11

        self.gwfast_param_order = ['Mc','eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
        self.gwfast_params_neglected = ['chi1x', 'chi2x', 'chi1y', 'chi2y', 'LambdaTilde', 'deltaLambda', 'ecc']
        self.dict_params_neglected_1 = {neglected_params: jnp.array([0.]).astype('complex128') for neglected_params in self.gwfast_params_neglected}
        self.dict_params_neglected_N = {neglected_params: jnp.zeros(self.nParticles).astype('complex128') for neglected_params in self.gwfast_params_neglected}

        self.true_params = jnp.array([self.injParams[param].squeeze() for param in self.gwfast_param_order])
        self.lower_bound = np.array([self.priorDict[param][0] for param in self.gwfast_param_order])
        self.upper_bound = np.array([self.priorDict[param][1] for param in self.gwfast_param_order])

        self.time_scale = 3600. * 24. # Multiply to transform from units of days to units of seconds

        self.grid_type = 'linear' # 'geometric'
        self._initFrequencyGrid(self.grid_type)
        self._initDetectors()
        self._initStrainData(method='sim', add_noise=False)

        # Warmup for JIT compile
        self._warmup_potential(True)
        self._warmup_potential_derivative(True) 

    def _warmup_potential(self, warmup):
        if warmup is True:
            print('Warming up potential')
            self.getDerivativesMinusLogPosterior_ensemble(self._newDrawFromPrior(self.nParticles))

    def _warmup_potential_derivative(self, warmup):
        if warmup is True:
            print('Warming up derivatives')
            self.getDerivativesMinusLogPosterior_ensemble(self._newDrawFromPrior(self.nParticles))

    def _initFrequencyGrid(self, grid_to_use):
        self.fmin = 10
        self.fmax = 325
        fcut = self.wf_model.fcut(**self.injParams)
        if self.fmax is None:
            self.fmax = fcut
        else:
            fcut = jnp.where(fcut > self.fmax, self.fmax, fcut)

        if grid_to_use == 'geometric':
            self.grid_resolution = int(100)
            self.fgrid = jnp.geomspace(self.fmin, fcut, num=self.grid_resolution)
        elif grid_to_use == 'linear':
            self.df = 1 / 2. # Sampling rate in Hz 
            self.grid_resolution = int(jnp.floor(jnp.real((1 + (fcut - self.fmin) / self.df))))
            print('Using % i bins' % self..grid_resolution)
            self.fgrid = jnp.linspace(self.fmin, fcut, num=self.grid_resolution)

        self.fgrids = jnp.repeat(self.fgrid, self.nParticles, axis=1)

    def _initDetectors(self): 
        """Initialize detectors and store PSD interpolated over defined frequency grid

        Remarks
        -------

        (1) This returns $S_n(f)$ for the desired frequency grid (self.fgrid)
        """

        self.detsInNet = {}
        self.PSD_dict = {}
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

            self.PSD_dict[det] = jnp.interp(self.fgrid, self.detsInNet[det].strainFreq, self.detsInNet[det].noiseCurve, left=1., right=1.).squeeze()

    def _initStrainData(self, method='sim', add_noise=False):
        """Initializes strain data by various means

        Parameters
        ----------
        method : str, optional
            Different ways to get strain data, by default 'sim'
            'sim' = Simulate signal
        add_noise : bool, optional
            Option to make add noise to injected signal, by default False

        """
        if method=='sim':
            self.strain_data = self._simulateSignal()
            if add_noise is True:
                # Add Gaussian noise with std given by the detector ASD if needed
                for det in self.detsInNet.keys():
                    self.signal_data[det] = self.signal_data[det] + np.random.normal(loc=0., scale=self.PSD_dict)
        else:
            raise NotImplementedError

    def _simulateSignal(self):
        """Calculates (clean) mock signal in each detector given injected parameters

        Parameters
        ----------
        add_noise : bool, optional
            Adds noise to the simulated signal, by default False
        """

        # Set the seed for reproducibility
        np.random.seed(None)

        # Compute the signal as seen in each detector and store the result
        strain_data = {}
        for det in self.detsInNet.keys():
            strain_data[det] = self.detsInNet[det].GWstrain(self.fgrid, 
                                                            Mc      = self.injParams['Mc'].astype('complex128'),
                                                            eta     = self.injParams['eta'].astype('complex128'),
                                                            dL      = self.injParams['dL'].astype('complex128'),
                                                            theta   = self.injParams['theta'].astype('complex128'),
                                                            phi     = self.injParams['phi'].astype('complex128'),
                                                            iota    = self.injParams['iota'].astype('complex128'),
                                                            psi     = self.injParams['psi'].astype('complex128'),
                                                            tcoal   = self.injParams['tcoal'].astype('complex128') / self.time_scale, # Change units to seconds
                                                            Phicoal = self.injParams['Phicoal'].astype('complex128'),
                                                            chiS    = self.injParams['chi1z'].astype('complex128'),
                                                            chiA    = self.injParams['chi2z'].astype('complex128'),
                                                            is_chi1chi2 = 'True',
                                                            **self.dict_params_neglected_1)

        return strain_data

    def _getResidual_Vec(self, X):
        """A vectorized method which calculates the residual between the the template with batch parameters 'X' and signal 

        Parameters
        ----------
        X : array
            (N, d) shaped array of particle positions

        Returns
        -------
        array
            (N, f) shaped array of residuals r(params) = template(params) - data. Note that f='number of frequency bins' 
        
        Remarks
        -------
        (1) gwfast.signal.GWstrain takes 'tcoal' in units of days
        """
    
        residual = {}
        X_ = X.T.astype('complex128')
        for det in self.detsInNet.keys():
            residual[det] = (self.detsInNet[det].GWstrain(self.fgrids, 
                                                          Mc      = X_[0],
                                                          eta     = X_[1],
                                                          dL      = X_[2],
                                                          theta   = X_[3],
                                                          phi     = X_[4],
                                                          iota    = X_[5],
                                                          psi     = X_[6],
                                                          tcoal   = X_[7] / self.time_scale, # Change units to seconds
                                                          Phicoal = X_[8],
                                                          chiS    = X_[9],
                                                          chiA    = X_[10],
                                                          is_chi1chi2 = 'True',
                                                          **self.dict_params_neglected_N) - self.strain_data[det]).T # Return a N x f matrix
                         
        return residual 

    def _getJacobianResidual_Vec(self, X):
        """A vectorized method which computes the Jacobian of the residual

        Parameters
        ----------
        X : array
            (N, d) shaped array of particle positions

        Returns
        -------
        array
            (d, N, f) shaped array of derivatives
        """

        residualJac = {}
        X_ = X.T.astype('complex128')
        
        for det in self.detsInNet.keys():
            residualJac[det] = self.detsInNet[det]._SignalDerivatives_use(self.fgrids, 
                                                                          Mc      = X_[0],
                                                                          eta     = X_[1],
                                                                          dL      = X_[2],
                                                                          theta   = X_[3],
                                                                          phi     = X_[4],
                                                                          iota    = X_[5],
                                                                          psi     = X_[6],
                                                                          tcoal   = X_[7] / self.time_scale, # Correction 1
                                                                          Phicoal = X_[8],
                                                                          chiS    = X_[9],
                                                                          chiA    = X_[10],
                                                                          use_chi1chi2 = True,
                                                                          **self.dict_params_neglected_N) 

            residualJac[det] = residualJac[det].at[7].divide(self.time_scale) # Correction 2

        return residualJac
            
    @partial(jax.jit, static_argnums=(0,))
    def getMinusLogPosterior_ensemble(self, thetas):
        """Calculates the potential

        Parameters
        ----------
        thetas : array
            (N, d) shaped array representing particle positions

        Returns
        -------
        array
            (N,) shaped array of potential evaluations
        
        References
        ----------
        (1) arxiv:1809.02293 Eq 42.

        """
        residual_dict = self._getResidual_Vec(thetas) 
        log_like = jnp.zeros(self.nParticles)
        for det in self.detsInNet:
            log_like = log_like + contract('Nf, Nf, f -> N', residual_dict[det].conjugate(), residual_dict[det], 1 / self.PSD_dict[det]).real
        return self.df * log_like / 2
    
    @partial(jax.jit, static_argnums=(0,))
    def getDerivativesMinusLogPosterior_ensemble(self, thetas):
        """Method calculating both the gradient of the potential and the Fisher information matrix

        Parameters
        ----------
        thetas : array
            (N, d) shaped array representing particle positions

        Returns
        -------
        tuple
            returns (N, d), (N, d, d) shaped arrays representing the gradient of the potential and the Fisher matrix respectively
        """
        residual_dict = self._getResidual_Vec(thetas) 
        jacResidual_dict = self._getJacobianResidual_Vec(thetas)
        grad_log_like = jnp.zeros((self.nParticles, self.DoF))
        GN = jnp.zeros((self.nParticles, self.DoF, self.DoF))
        for det in self.detsInNet:
            grad_log_like = grad_log_like + contract('dNf, Nf, f -> Nd', jacResidual_dict[det].conjugate(), residual_dict[det], 1 / self.PSD_dict[det]).real 
            GN = GN + contract('dNf, bNf, f -> Ndb', jacResidual_dict[det].conjugate(), jacResidual_dict[det], 1 / self.PSD_dict[det]).real
        return (self.df * grad_log_like, self.df * GN)





    def _riemannSum(self, integrand, grid, axis=-1):
        """Approximate integral using Riemann sum definition

        Parameters
        ----------
        integrand : array
            (..., f) shaped array by default representing the integrand
        grid : array
            (f,) shaped array representing the grid
            
        axis : int, optional
            axis of integration, by default -1

        Returns
        -------
        array
            Array with one fewer axis representing the integral
        """
        return jnp.sum(integrand[..., :-1] * (grid[1:] - grid[:-1]), axis=axis)

    # def _signal_inner_product(self, a, b, det, mode):
    #     """Evaluate noise-weighted inner product

    #     Parameters
    #     ----------
    #     a : array
    #         First element in inner product
    #     b : array
    #         Second element in inner product
    #     det : str
    #         String representing which detector noise characteristic to use in weighing procedure. Options = 'L1', 'H1', 'Virgo'
    #     mode : str
    #         String representing what kind of tensor is in first slot of inner product.
    #         'l' = Needed in calculating the likelihood
    #         'g' = Needed in calculating the gradient of the likelihood
    #         'h' = Needed in calculating the Fisher information

    #     Returns
    #     -------
    #     array
    #         Returns the noise weighted inner product
    #     """
    #     quadrature_rule = 'trapezoid' 
    #     # quadrature_rule = 'riemann'

    #     if mode == 'l':
    #         integrand = contract('Nf,  Nf,  f -> Nf',   a.conjugate(), b, 1 / self.PSD_dict[det])
    #     elif mode == 'g':
    #         integrand = contract('dNf, Nf,  f -> Ndf',  a.conjugate(), b, 1 / self.PSD_dict[det])
    #     elif mode == 'h':
    #         integrand = contract('dNf, bNf, f -> Ndbf', a.conjugate(), b, 1 / self.PSD_dict[det])

    #     if quadrature_rule == 'riemann':
    #         return 4 * self._riemannSum(integrand.real, self.fgrid.squeeze())
    #     elif quadrature_rule == 'trapezoid':
    #         return 4 * jnp.trapz(integrand.real, self.fgrid.squeeze())


    # # @partial(jax.jit, static_argnums=(0,))
    # def getMinusLogPosterior_ensemble(self, thetas):
    #     """Calculates the potential

    #     Parameters
    #     ----------
    #     thetas : array
    #         (N, d) shaped array representing particle positions

    #     Returns
    #     -------
    #     array
    #         (N,) shaped array of potential evaluations
        
    #     References
    #     ----------
    #     (1) arxiv:1809.02293 Eq 42.

    #     """

    #     residual_dict = self._getResidual_Vec(thetas) 
    #     log_like = jnp.zeros(self.nParticles)
    #     for det in self.detsInNet.keys():
    #         log_like = log_like + self._signal_inner_product(residual_dict[det], residual_dict[det], det, 'l')
    #     return log_like / 2

    # # @partial(jax.jit, static_argnums=(0,))
    # def getDerivativesMinusLogPosterior_ensemble(self, thetas):
    #     """Method calculating both the gradient of the potential and the Fisher information matrix

    #     Parameters
    #     ----------
    #     thetas : array
    #         (N, d) shaped array representing particle positions

    #     Returns
    #     -------
    #     tuple
    #         returns (N, d), (N, d, d) shaped arrays representing the gradient of the potential and the Fisher matrix respectively
    #     """
    #     residual_dict = self._getResidual_Vec(thetas) 
    #     jacResidual_dict = self._getJacobianResidual_Vec(thetas)
    #     grad_log_like = jnp.zeros((self.nParticles, self.DoF))
    #     GN = jnp.zeros((self.nParticles, self.DoF, self.DoF))
    #     for det in self.detsInNet.keys():
    #         grad_log_like = grad_log_like + self._signal_inner_product(jacResidual_dict[det], residual_dict[det], det, 'g')
    #         GN = GN + self._signal_inner_product(jacResidual_dict[det], jacResidual_dict[det], det, 'h')
    #     return (grad_log_like, GN)

    def _newDrawFromPrior(self, nSamples):
        prior_draw = np.zeros((nSamples, self.DoF))

        for i, param in enumerate(self.gwfast_param_order): # Assuming uniform on all parameters
            low = self.priorDict[param][0]
            high = self.priorDict[param][1]
            prior_draw[:, i] = np.random.uniform(low=low, high=high, size=nSamples)
                   
        return prior_draw

    def getCrossSection(self, a, b):
        ngrid = int(np.sqrt(self.nParticles))
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

