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
    
    def __init__(self, NetDict, WaveForm, injParams, priorDict, nParticles=1):
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
        self.priorDict  = priorDict
        self.nParticles = nParticles
        self.DoF = 11

        # gw_fast related attributes
        self.wf_model = WaveForm
        self.NetDict = NetDict
        self.EarthMotion = False
        self.injParams = injParams
        self.time_scale = 3600 * 24. # Multiply to transform from units of days to units of seconds

        # Parameter order convention
        self.gwfast_param_order = ['Mc','eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
        self.gwfast_params_neglected = ['chi1x', 'chi2x', 'chi1y', 'chi2y', 'LambdaTilde', 'deltaLambda', 'ecc']

        # Define dictionary of neglected parameters to keep signal evaluation methods clean
        self.dict_params_neglected_1 = {neglected_params: jnp.array([0.]).astype('complex128') for neglected_params in self.gwfast_params_neglected}
        self.dict_params_neglected_N = {neglected_params: jnp.zeros(self.nParticles).astype('complex128') for neglected_params in self.gwfast_params_neglected}

        # Definitions for easy interfacing
        self.true_params = jnp.array([self.injParams[param].squeeze() for param in self.gwfast_param_order])
        self.lower_bound = np.array([self.priorDict[param][0] for param in self.gwfast_param_order])
        self.upper_bound = np.array([self.priorDict[param][1] for param in self.gwfast_param_order])


        self._initFrequencyGrid()
        self._initDetectors()
        self.h0_standard = self._getInjectedSignals(injParams, self.fgrid_standard)
        self.h0_dense = self._getInjectedSignals(injParams, self.fgrid_dense)
        # self._initStrainData(method='sim', add_noise=False)

        # self.d_d = self._precomputeDataInnerProduct()


        # self.getHeterodyneBins()
        # self._h0() # Setup fiducial signal over dense grid

        # self.getSummaryData()
        # Warmup for JIT compile
        # self._warmup_potential(True)
        # self._warmup_potential_derivative(True) 


    def _initFrequencyGrid(self, fmin=20, fmax=None): # Checks: X
        """
        Setup frequency grids that will be used
        """
        self.fmin = fmin  # 10
        self.fmax = fmax  # 325
        fcut = self.wf_model.fcut(**self.injParams)
        if self.fmax is None:
            self.fmax = fcut
        else:
            fcut = jnp.where(fcut > self.fmax, self.fmax, fcut)
        self.fcut = float(fcut)

        ###############################
        # Standard frequency grid setup
        ###############################
        # Remarks
        # (i) Once nbins_standard is calculated, df_standard must be updated!
        signal_duration = 4. # [s]
        self.df_standard = 1 / signal_duration
        self.nbins_standard = int(jnp.ceil(((self.fmax - self.fmin) / self.df_standard)))
        self.df_standard = (self.fmax - self.fmin) / self.nbins_standard # (i)
        print('Standard binning scheme: % i bins' % self.nbins_standard)
        self.fgrid_standard = jnp.linspace(self.fmin, fcut, num=self.nbins_standard + 1).squeeze()

        ########################################
        # Dense frequency setup for heterodyning
        ########################################
        self.nbins_dense = 1000 # TODO: Make this more dense once not testing
        self.df_dense = (self.fmax - self.fmin) / self.nbins_dense
        print('Dense bins: % i bins' % self.nbins_dense)
        self.fgrid_dense = jnp.linspace(self.fmin, fcut, num=self.nbins_dense + 1).squeeze()

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

    def _getInjectedSignals(self, injParams, fgrid):
        """
        Fiducial signals over dense grid (one for each detector)
        Note: See remarks on self.getSignal
        Remarks:
        (i) Squeeze to return (f,) shaped array
        """
        h0 = {}
        for det in self.detsInNet.keys():
            h0[det] = self.detsInNet[det].GWstrain(fgrid, 
                                                   Mc      = injParams['Mc'].astype('complex128'),
                                                   eta     = injParams['eta'].astype('complex128'),
                                                   dL      = injParams['dL'].astype('complex128'),
                                                   theta   = injParams['theta'].astype('complex128'),
                                                   phi     = injParams['phi'].astype('complex128'),
                                                   iota    = injParams['iota'].astype('complex128'),
                                                   psi     = injParams['psi'].astype('complex128'),
                                                   tcoal   = injParams['tcoal'].astype('complex128') / self.time_scale, # Change units to seconds
                                                   Phicoal = injParams['Phicoal'].astype('complex128'),
                                                   chiS    = injParams['chi1z'].astype('complex128'),
                                                   chiA    = injParams['chi2z'].astype('complex128'),
                                                   is_chi1chi2 = 'True',
                                                   **self.dict_params_neglected_1).squeeze() # (i)

        return h0

    def getSignal(self, X, f_grid):
        """ 
        Method to calculate signal for each X[i] over f_grid. 
        Remarks:
        (i)   Same frequency grid is currently being used for each particle
        (ii)  gwfast.GWstrain expects an f x N matrix
        (iii) gwfast.GWstrain expects `tcoal` in units of seconds
        (iv)  Transpose is included to return an N x f matrix
        """
        fgrids = jnp.repeat(f_grid[...,np.newaxis], self.nParticles, axis=1) # (i)
        signal = {}
        X_ = X.T.astype('complex128')
        for det in self.detsInNet.keys():
            signal[det] = (self.detsInNet[det].GWstrain(fgrids, # (ii)                        
                                                        Mc      = X_[0],
                                                        eta     = X_[1],
                                                        dL      = X_[2],
                                                        theta   = X_[3],
                                                        phi     = X_[4],
                                                        iota    = X_[5],
                                                        psi     = X_[6],
                                                        tcoal   = X_[7] / self.time_scale, # (iii)
                                                        Phicoal = X_[8],
                                                        chiS    = X_[9],
                                                        chiA    = X_[10],
                                                        is_chi1chi2 = 'True',
                                                        **self.dict_params_neglected_N)).T # (iv) 
                            
        return signal 

    def _getJacobianSignal(self, X, f_grid):
        """A vectorized method which computes the Jacobian of the signal model

        Parameters
        ----------
        X : array
            (N, d) shaped array of particle positions

        Returns
        -------
        array
            (d, N, f) shaped array of derivatives
        """
        fgrids = jnp.repeat(f_grid[...,np.newaxis], self.nParticles, axis=1)
        jacModel = {}
        X_ = X.T.astype('complex128')
        for det in self.detsInNet.keys():
            jacModel[det] = self.detsInNet[det]._SignalDerivatives_use(fgrids, 
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

            jacModel[det] = jacModel[det].at[7].divide(self.time_scale) # Correction 2

        return jacModel
            
    def standard_minusLogLikelihood(self, X): # Checks: X
        """ 
        """
        log_likelihood = np.zeros(self.nParticles)
        signal = self.getSignal(X, self.fgrid_standard)
        for det in self.detsInNet.keys():
            residual = signal[det] - self.h0_standard[det][np.newaxis, ...]
            inner_product = 4 * np.sum((residual.real ** 2 + residual.imag ** 2) / self.PSD_standard[det][np.newaxis, ...], axis=-1) * self.df_standard
            log_likelihood += 0.5 * inner_product
        return log_likelihood




#########################################################################
# HETERODYNE METHODS
#########################################################################

        # inner_product = lambda x, y: 4 * np.sum(x * y.conjugate()[np.newaxis, ...] / self.PSD_standard[np.newaxis, ...], axis=-1) * self.df_standard
    def getHeterodyneBins(self, chi=0.1, eps=0.5):
        """ 
        Get sparse grid for heterodyned scheme
        """
        f_max = float(self.fcut) #512
        f_min = float(self.fmin) #20
        gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3])
        # num_grid_ticks = 100000
        num_grid_ticks = self.n_dense + 1
        # f_grid = np.linspace(f_min, f_max, num_grid_ticks) # Begin with dense grid
        f_grid = self.f_grid_dense
        bound = lambda f_minus, f_plus: 2 * np.pi * chi * np.sum((1 - (f_minus / f_plus) ** np.abs(gamma)))
        bin_edges = [f_min]
        indicies_kept = [0]
        i = 0 # grid index
        j = 0 # bin edge index
        while i < num_grid_ticks:
            while i < num_grid_ticks and bound(bin_edges[j], f_grid[i]) < eps:
                i += 1
            bin_edges.append(f_grid[i - 1])
            indicies_kept.append(i - 1)
            j += 1

        self.indicies_kept = np.array(indicies_kept)
        self.bin_edges = np.array(bin_edges)
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
        self.nbins = self.bin_edges.shape[0] - 1

    def r(self, X):
        """ 
        Calculate the ratio of signal model with fiducial signal
        """
        r = {}
        signal = self.getSignal(X, self.bin_edges)
        for det in self.detsInNet.keys():
            r[det] = signal[det] / self.h0[det][self.indicies_kept].T
        return r

    def getSplineData(self, X):
        r = self.r(X)
        r0 = {}
        r1 = {}
        for det in self.detsInNet.keys():
            tmp1 = r[det][:, 1:] - r[det][:, :-1]
            r0[det] = tmp1 / 2               # y intercept
            r1[det] = tmp1 / self.bin_widths[np.newaxis, ...] # slopes
        return r0, r1

    def getSummaryData(self):
        """ 
        Calculate summary data
        """
        # Init dicts
        self.A0 = {}
        self.A1 = {}
        self.B0 = {}
        self.B1 = {}
        for det in self.detsInNet.keys():
            self.A0[det] = np.zeros((self.nbins)).astype('complex128')
            self.A1[det] = np.zeros((self.nbins)).astype('complex128')
            self.B0[det] = np.zeros((self.nbins))
            self.B1[det] = np.zeros((self.nbins))
        
        bin_index = (np.digitize(self.f_grid_dense, self.bin_edges))[:,0] - 1 # To index bins from 0 to nbins - 1
        bin_index[-1] = self.nbins - 1 # Make sure the right endpoint is inclusive!
        for i in range(len(self.f_grid_dense)):
            b = bin_index[i]
            # d * h0, in this case d = h0!
            tmp1 = 4 * self.h0[det][i] * self.h0[det][i].conjugate() / self.PSD_dict_dense[det][i] * self.df
            tmp2 = 4 * np.abs(self.h0[det][i]) ** 2 / self.PSD_dict_dense[det][i] * self.df 
            self.A0[det][b] += tmp1
            self.A1[det][b] += tmp1 * (self.f_grid_dense[i] - self.bin_centers[b])
            self.B0[det][b] += tmp2
            self.B1[det][b] += tmp2 * (self.f_grid_dense[i] - self.bin_centers[b])

    def likelihood_heterodyne(self, X):
        r0, r1 = self.getSplineData(X)
        log_like = np.zeros(self.nParticles)
        for det in self.detsInNet.keys():
            h_d = np.sum(self.A0[det].conjugate()[np.newaxis] * r0[det] + self.A1[det].conjugate()[np.newaxis] * r1[det], axis=1)
            h_h = np.sum(self.B0[det][np.newaxis] * np.abs(r0[det]) ** 2 + 2 * self.B1[det][np.newaxis] * (r0[det].conjugate() * r1[det]).real, axis=1)
            log_like += h_h - 2 * h_d.real + self.d_d[det]  
        return log_like





    def _precomputeDataInnerProduct(self):
        inner_product = {}
        for det in self.detsInNet.keys():
            inner_product[det] = 4 * np.sum((self.h0.real ** 2 + self.h0.imag ** 2) / self.PSD_dense) * self.df_dense
        return inner_product


################################################################
# Other methods. Clean up later!!!
################################################################

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

    def _warmup_potential(self, warmup):
        if warmup is True:
            print('Warming up potential')
            self.getDerivativesMinusLogPosterior_ensemble(self._newDrawFromPrior(self.nParticles))

    def _warmup_potential_derivative(self, warmup):
        if warmup is True:
            print('Warming up derivatives')
            self.getDerivativesMinusLogPosterior_ensemble(self._newDrawFromPrior(self.nParticles))