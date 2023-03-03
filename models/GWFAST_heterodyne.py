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
    
    def __init__(self, NetDict, WaveForm, injParams, priorDict):
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
        self.DoF = 11

        # gw_fast related attributes
        self.wf_model = WaveForm
        self.NetDict = NetDict
        self.EarthMotion = False
        self.injParams = injParams
        self.seconds_per_day = 86400. 


        # Parameter order convention
        self.gwfast_param_order = ['Mc','eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
        self.gwfast_params_neglected = ['chi1x', 'chi2x', 'chi1y', 'chi2y', 'LambdaTilde', 'deltaLambda', 'ecc']

        # Define dictionary of neglected parameters to keep signal evaluation methods clean
        # self.dict_params_neglected_1 = {neglected_params: jnp.array([0.]).astype('complex128') for neglected_params in self.gwfast_params_neglected}
        # self.dict_params_neglected_N = {neglected_params: jnp.zeros(self.nParticles).astype('complex128') for neglected_params in self.gwfast_params_neglected}

        # Definitions for easy interfacing
        self.true_params = jnp.array([self.injParams[param].squeeze() for param in self.gwfast_param_order])
        self.lower_bound = np.array([self.priorDict[param][0] for param in self.gwfast_param_order])
        self.upper_bound = np.array([self.priorDict[param][1] for param in self.gwfast_param_order])

        self._initFrequencyGrid()
        self._initDetectors()
        self.h0_standard = self._getInjectedSignals(injParams, self.fgrid_standard)
        self.h0_dense = self._getInjectedSignals(injParams, self.fgrid_dense)

        # Heterodyned strategy
        self.d_d = self._precomputeDataInnerProduct()
        self.getHeterodyneBins(chi=1, eps=0.1)
        # self.getHeterodyneBins(chi=1, eps=0.1)
        self.getSummaryData()

        # Warmup for JIT compile
        # self._warmup_potential(True)
        # self._warmup_potential_derivative(True) 

    def _getDictParamsNeglected(self, N):
        return {neglected_params: jnp.zeros(N).astype('complex128') for neglected_params in self.gwfast_params_neglected}

    def _initFrequencyGrid(self, fmin=20, fmax=None): # Checks: X
        """
        Setup frequency grids that will be used
        """
        self.fmin = fmin  # 10
        self.fmax = fmax  # 325
        fcut = self.wf_model.fcut(**self.injParams)[0]
        if self.fmax is None:
            self.fmax = fcut
        else:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)
        self.fcut = fcut

        ###############################
        # Standard frequency grid setup
        ###############################
        # Remarks
        # (i) Once nbins_standard is calculated, df_standard must be updated!
        signal_duration = 4. # [s]
        self.df_standard = 1 / signal_duration
        self.nbins_standard = int(np.ceil(((self.fmax - self.fmin) / self.df_standard)))
        # self.nbins_standard = 9000
        self.df_standard = (self.fmax - self.fmin) / self.nbins_standard # (i)
        print('Standard binning scheme: % i bins' % self.nbins_standard)
        self.fgrid_standard = np.linspace(self.fmin, fcut, num=self.nbins_standard + 1).squeeze()

        ########################################
        # Dense frequency setup for heterodyning
        ########################################
        self.nbins_dense = 10000 
        self.df_dense = (self.fmax - self.fmin) / self.nbins_dense
        print('Dense bins: % i bins' % self.nbins_dense)
        self.fgrid_dense = np.linspace(self.fmin, fcut, num=self.nbins_dense + 1).squeeze()

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
        (i)   Squeeze to return (f,) shaped array
        (ii)  Signal returns a 0 for the maximum frequency. This is a hack which fixes this issue
        (iii) `tcoal` in GWstrain must be in units of days.
        """
        dict_params_neglected = self._getDictParamsNeglected(1)
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
                                                   tcoal   = injParams['tcoal'].astype('complex128'), #/ self.seconds_per_day, # (iii)
                                                   Phicoal = injParams['Phicoal'].astype('complex128'),
                                                   chiS    = injParams['chi1z'].astype('complex128'),
                                                   chiA    = injParams['chi2z'].astype('complex128'),
                                                   is_chi1chi2 = 'True',
                                                   **dict_params_neglected).squeeze() # (i)

            h0[det] = h0[det].at[-1].set(h0[det][-2]) # (ii)

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
        nParticles = X.shape[0]
        dict_params_neglected = self._getDictParamsNeglected(nParticles)
        fgrids = jnp.repeat(f_grid[...,np.newaxis], nParticles, axis=1) # (i)
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
                                                        tcoal   = X_[7], #/ self.seconds_per_day, # (iii)
                                                        Phicoal = X_[8],
                                                        chiS    = X_[9],
                                                        chiA    = X_[10],
                                                        is_chi1chi2 = 'True',
                                                        **dict_params_neglected)).T # (iv) 
                            
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
            gwfast returns a (d, N, f) shaped array 
        """
        nParticles = X.shape[0]
        dict_params_neglected = self._getDictParamsNeglected(nParticles)
        fgrids = jnp.repeat(f_grid[...,np.newaxis], nParticles, axis=1)
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
                                                                       tcoal   = X_[7], #/ self.seconds_per_day, # Correction 1
                                                                       Phicoal = X_[8],
                                                                       chiS    = X_[9],
                                                                       chiA    = X_[10],
                                                                       use_chi1chi2 = True,
                                                                       **dict_params_neglected) 

            #jacModel[det] = jacModel[det].at[7].divide(self.seconds_per_day) # Correction 2

        return jacModel
            
    @partial(jax.jit, static_argnums=(0,))
    def standard_minusLogLikelihood(self, X): # Checks: X
        """ 
        """
        nParticles = X.shape[0]
        log_likelihood = jnp.zeros(nParticles)
        signal = self.getSignal(X, self.fgrid_standard)
        for det in self.detsInNet.keys():
            residual = signal[det] - self.h0_standard[det][np.newaxis, ...]
            inner_product = 4 * jnp.sum((residual.real ** 2 + residual.imag ** 2) / self.PSD_standard[det][np.newaxis, ...], axis=-1) * self.df_standard
            log_likelihood += 0.5 * inner_product
        return log_likelihood

    def standard_gradientMinusLogLikelihood(self, X): # Checks: X
        nParticles = X.shape[0]
        grad_log_like = jnp.zeros((nParticles, self.DoF))
        signal = self.getSignal(X, self.fgrid_standard)
        jacSignal = self._getJacobianSignal(X, self.fgrid_standard)
        for det in self.detsInNet.keys():
            residual = signal[det] - self.h0_standard[det][np.newaxis, ...]
            inner_product = (4 * jnp.sum(jacSignal[det].conjugate() * residual[np.newaxis, ...] / self.PSD_standard[det], axis=-1) * self.df_standard).T
            grad_log_like += inner_product.real
        return grad_log_like
    
    def standard_GNHessianMinusLogLikelihood(self, X): # Checks: X
        nParticles = X.shape[0]
        GN = jnp.zeros((nParticles, self.DoF, self.DoF))
        jacSignal = self._getJacobianSignal(X, self.fgrid_standard)
        for det in self.detsInNet.keys():
            inner_product = 4 * contract('iNf, jNf, f -> Nij', jacSignal[det].conjugate(), jacSignal[det], 1 / self.PSD_standard[det]) * self.df_standard
            GN += inner_product.real
        return GN




#########################################################################
# HETERODYNE METHODS
#########################################################################

    def _precomputeDataInnerProduct(self):
        print('Precomputing inner product')
        inner_product = {}
        for det in self.detsInNet.keys():
            inner_product[det] = 4 * np.sum((self.h0_dense[det].real ** 2 + self.h0_dense[det].imag ** 2) / self.PSD_dense[det]) * self.df_dense
        return inner_product

    def getHeterodyneBins(self, chi, eps):
        print('Getting heterodyned bins')
        # Remarks:
        # (i)  0.5 is a dummy variable for x==0 case (which we dont care for)
        gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3])
        f_star = self.fmax * np.heaviside(gamma, 0.5) + self.fmin * np.heaviside(-gamma, 0.5) # (i) 
        delta = lambda f_minus, f_plus: 2 * np.pi * chi * np.sum(np.abs((f_plus / f_star) ** gamma - (f_minus/f_star) ** gamma))

        delta0 = delta(self.fgrid_dense[0], self.fgrid_dense[1])
        print('delta0 = %f' % delta0)
        if eps < delta0:
            print('First bin cannot satisfy bound. Changing epsilon from %f to %f' % (eps, delta0))
            eps = delta0

        subindex = []
        index_f_minus = 0
        j = 0
        while j <= self.nbins_dense:
            if j == 0 or j == self.nbins_dense:
                subindex.append(j)
            else:
                d = delta(self.fgrid_dense[index_f_minus], self.fgrid_dense[j])
                if d > eps:
                    j -= 1
                    subindex.append(j)
                    index_f_minus = j
            j += 1

        self.indicies_kept = np.array(subindex)
        self.bin_edges = self.fgrid_dense[self.indicies_kept]
        self.nbins = len(subindex) - 1
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        print('Heterodyne binning scheme: %i' % self.nbins)


    def r(self, X):
        """ 
        Calculate the ratio of signal model with fiducial signal
        """
        r = {}
        signal = self.getSignal(X, self.bin_edges)
        for det in self.detsInNet.keys():
            r[det] = signal[det] / self.h0_dense[det][self.indicies_kept]
        return r

    def getSplineData(self, X):
        """ 
        Return N x b matrix
        """
        r = self.r(X)
        r0 = {}
        r1 = {}
        for det in self.detsInNet.keys():
            r0[det] = r[det][:, :-1] # y intercept
            r1[det] = (r[det][:, 1:] - r[det][:, :-1]) / self.bin_widths[np.newaxis, ...] # slopes
        return r0, r1

    def jac_r(self, X):
        jac_r = {}
        jacSignal = self._getJacobianSignal(X, self.bin_edges)
        for det in self.detsInNet.keys():
            jac_r[det] = jacSignal[det] / self.h0_dense[det][self.indicies_kept]
        return jac_r

    def getJacSplineData(self, X):
        jac_r = self.jac_r(X)
        jac_r0 = {}
        jac_r1 = {}
        for det in self.detsInNet.keys():
            jac_r0[det] = jac_r[det][..., :-1]
            jac_r1[det] = (jac_r[det][..., 1:] - jac_r[det][..., :-1]) / self.bin_widths
        return jac_r0, jac_r1

    def getSummaryData(self):
        """ 
        Calculate summary data
        """
        print('Calculating summary data')
        # Init dicts
        self.A0 = {}
        self.A1 = {}
        self.B0 = {}
        self.B1 = {}
        bin_index = (np.digitize(self.fgrid_dense, self.bin_edges)) - 1 # To index bins from 0 to nbins - 1
        bin_index[-1] = self.nbins - 1 # Make sure the right endpoint is inclusive!

        for det in self.detsInNet.keys():
            self.A0[det] = np.zeros((self.nbins)).astype('complex128')
            self.A1[det] = np.zeros((self.nbins)).astype('complex128')
            self.B0[det] = np.zeros((self.nbins))
            self.B1[det] = np.zeros((self.nbins))
            for b in range(self.nbins):
                indicies = np.where(bin_index == b)
                tmp1 = 4 * self.h0_dense[det][indicies] * self.h0_dense[det][indicies].conjugate() / self.PSD_dense[det][indicies] * self.df_dense
                tmp2 = 4 * (self.h0_dense[det][indicies].real ** 2 + self.h0_dense[det][indicies].imag ** 2) / self.PSD_dense[det][indicies] * self.df_dense
                self.A0[det][b] = np.sum(tmp1)
                self.A1[det][b] = np.sum(tmp1 * (self.fgrid_dense[indicies] - self.bin_edges[b]))
                self.B0[det][b] = np.sum(tmp2)
                self.B1[det][b] = np.sum(tmp2 * (self.fgrid_dense[indicies] - self.bin_edges[b]))
        print('Summary data calculation completed')
    
    @partial(jax.jit, static_argnums=(0,))
    def heterodyne_minusLogLikelihood(self, X):
        nParticles = X.shape[0]
        r0, r1 = self.getSplineData(X)
        log_like = jnp.zeros(nParticles)
        for det in self.detsInNet.keys():
            h_d = jnp.sum(self.A0[det][jnp.newaxis] * r0[det].conjugate() + self.A1[det][jnp.newaxis] * r1[det].conjugate(), axis=1)
            h_h = jnp.sum(self.B0[det][jnp.newaxis] * jnp.abs(r0[det]) ** 2 + 2 * self.B1[det][jnp.newaxis] * (r0[det].conjugate() * r1[det]).real, axis=1)
            log_like += 0.5 * h_h - h_d.real + 0.5 * self.d_d[det]
        return log_like

    # def heterodyne_gradientMinusLogLikelihood(self, X):
    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogPosterior_ensemble(self, X):
        nParticles = X.shape[0]
        r0, r1 = self.getSplineData(X)
        jac_r0, jac_r1 = self.getJacSplineData(X)
        grad_log_like = np.zeros((nParticles, self.DoF))
        for det in self.detsInNet.keys():

            jh_d = contract('b, jNb -> Nj', self.A0[det], jac_r0[det].conjugate(), backend='jax') \
                 + contract('b, jNb -> Nj', self.A1[det], jac_r1[det].conjugate(), backend='jax')

            jh_h = contract('b, jNb, Nb -> Nj', self.B0[det], jac_r0[det].conjugate(), r0[det], backend='jax') \
                 + contract('b, jNb, Nb -> Nj', self.B1[det], jac_r0[det].conjugate(), r1[det], backend='jax') \
                 + contract('b, jNb, Nb -> Nj', self.B1[det], jac_r1[det].conjugate(), r0[det], backend='jax')

            grad_log_like += jh_h.real - jh_d.real

        return grad_log_like

    # def heterodyne_GNHessianMinusLogLikelihood(self, X):
    @partial(jax.jit, static_argnums=(0,))
    def getGNHessianMinusLogPosterior_ensemble(self, X):
        nParticles = X.shape[0]
        jac_r0, jac_r1 = self.getJacSplineData(X)
        GN = jnp.zeros((nParticles, self.DoF, self.DoF))
        for det in self.detsInNet.keys():
            jh_jh = contract('b, jNb, kNb -> Njk', self.B0[det], jac_r0[det].conjugate(), jac_r0[det], backend='jax') \
                  + contract('b, jNb, kNb -> Njk', self.B1[det], jac_r0[det].conjugate(), jac_r1[det], backend='jax') \
                  + contract('b, jNb, kNb -> Njk', self.B1[det], jac_r1[det].conjugate(), jac_r0[det], backend='jax') 
                            
            GN += jh_jh.real

        return GN

    @partial(jax.jit, static_argnums=(0,))  
    def getDerivativesMinusLogPosterior_ensemble(self, X):
        nParticles = X.shape[0]
        r0, r1 = self.getSplineData(X)
        jac_r0, jac_r1 = self.getJacSplineData(X)
        grad_log_like = jnp.zeros((nParticles, self.DoF))
        GN = jnp.zeros((nParticles, self.DoF, self.DoF))
        for det in self.detsInNet.keys():

            jh_d = contract('b, jNb -> Nj', self.A0[det], jac_r0[det].conjugate(), backend='jax') \
                 + contract('b, jNb -> Nj', self.A1[det], jac_r1[det].conjugate(), backend='jax')

            jh_h = contract('b, jNb, Nb -> Nj', self.B0[det], jac_r0[det].conjugate(), r0[det], backend='jax') \
                 + contract('b, jNb, Nb -> Nj', self.B1[det], jac_r0[det].conjugate(), r1[det], backend='jax') \
                 + contract('b, jNb, Nb -> Nj', self.B1[det], jac_r1[det].conjugate(), r0[det], backend='jax')

            grad_log_like += jh_h.real - jh_d.real

            jh_jh = contract('b, jNb, kNb -> Njk', self.B0[det], jac_r0[det].conjugate(), jac_r0[det], backend='jax') \
                  + contract('b, jNb, kNb -> Njk', self.B1[det], jac_r0[det].conjugate(), jac_r1[det], backend='jax') \
                  + contract('b, jNb, kNb -> Njk', self.B1[det], jac_r1[det].conjugate(), jac_r0[det], backend='jax') 
                            
            GN += jh_jh.real

        return grad_log_like, GN


################################################################
# Other methods. Clean up later!!!
################################################################

    def _newDrawFromPrior(self, nParticles):
        prior_draw = np.zeros((nParticles, self.DoF))
        for i, param in enumerate(self.gwfast_param_order): # Assuming uniform on all parameters
            low = self.priorDict[param][0]
            high = self.priorDict[param][1]
            prior_draw[:, i] = np.random.uniform(low=low, high=high, size=nParticles)
                   
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




    # def getHeterodyneBins_new(self, chi, eps):
    #     print('Getting heterodyned bins')

    #     gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3])
    #     delta = lambda f_minus, f_plus: 2 * np.pi * chi * np.sum((1 - (f_minus / f_plus) ** np.abs(gamma)))
    #     delta0 = delta(self.fgrid_dense[0], self.fgrid_dense[1])
    #     if eps < delta0:
    #         print('First bin cannot satisfy bound. Changing epsilon from %f to %f' % (eps, delta0))
    #         eps = delta0

    #     bin_edges = [self.fgrid_dense[0]]
    #     indicies_kept = [0]
    #     idx_fminus = 0
    #     for i in np.arange(1, self.nbins_dense - 1):
    #         next_bin_over_bound = delta(self.fgrid_dense[idx_fminus], self.fgrid_dense[i + 1]) > eps
    #         if next_bin_over_bound:
    #             bin_edges.append(self.fgrid_dense[i])
    #             idx_fminus = i
    #             indicies_kept.append(i)
    #     bin_edges.append(self.fgrid_dense[-1])
    #     indicies_kept.append(self.nbins_dense)

    #     # Store information
    #     self.nbins = len(bin_edges) - 1
    #     self.indicies_kept = np.array(indicies_kept)
    #     self.bin_edges = np.array(bin_edges)
    #     self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
    #     self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
    #     print('Heterodyne binning scheme: %i' % self.nbins)


    # def getHeterodyneBins(self, chi=5, eps=0.1):
    #     """ 
    #     Get sparse grid for heterodyned scheme
    #     """
    #     print('Getting heterodyne bins')
    #     gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3])
    #     bound = lambda f_minus, f_plus: 2 * np.pi * chi * np.sum((1 - (f_minus / f_plus) ** np.abs(gamma)))
    #     bin_edges = [self.fmin]
    #     indicies_kept = [0]
    #     n_ticks_dense = self.nbins_dense + 1
    #     i = 0 # grid index
    #     j = 0 # bin edge index
    #     while i < n_ticks_dense:
    #         while i < n_ticks_dense and bound(bin_edges[j], self.fgrid_dense[i]) < eps:
    #             i += 1
    #         bin_edges.append(self.fgrid_dense[i - 1])
    #         indicies_kept.append(i - 1)
    #         j += 1

    #     self.indicies_kept = np.array(indicies_kept)
    #     self.bin_edges = np.array(bin_edges)
    #     self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
    #     self.bin_centers = (self.bin_edges[1:] + self.bin_edges[:-1]) / 2
    #     self.nbins = self.bin_edges.shape[0] - 1
    #     print('Heterodyne binning scheme: %i' % self.nbins)


        # inner_product = lambda x, y: 4 * np.sum(x * y.conjugate()[np.newaxis, ...] / self.PSD_standard[np.newaxis, ...], axis=-1) * self.df_standard

        # deltas = 2 * np.pi * chi * np.sum(((1 - (self.fgrid_dense[:-1] / self.fgrid_dense[1:])[...,np.newaxis] ** np.abs(gamma))), axis=-1)


















