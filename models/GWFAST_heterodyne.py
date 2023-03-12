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
from opt_einsum import contract
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

#%%
class gwfast_class(object):
    
    # def __init__(self, NetDict, WaveForm, injParams, priorDict):
    def __init__(self, chi, eps):
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
        self._initParams()
        # self.priorDict  = priorDict

        # gw_fast related attributes
        self.EarthMotion = False

        # Parameter order convention
        self.gwfast_param_order = ['Mc','eta', 'dL', 'theta', 'phi', 'iota', 'psi', 'tcoal', 'Phicoal', 'chi1z', 'chi2z']
        self.gwfast_params_neglected = ['chi1x', 'chi2x', 'chi1y', 'chi2y', 'LambdaTilde', 'deltaLambda', 'ecc']
        self.DoF = 11

        # Definitions for easy interfacing
        self.true_params = jnp.array([self.injParams[param].squeeze() for param in self.gwfast_param_order])
        self.lower_bound = np.array([self.priorDict[param][0] for param in self.gwfast_param_order])
        self.upper_bound = np.array([self.priorDict[param][1] for param in self.gwfast_param_order])

        self._initFrequencyGrid()
        self._initDetectors()
        self.h0_standard = self._getInjectedSignals(self.injParams, self.fgrid_standard) # Fiducial signal
        self.h0_dense = self._getInjectedSignals(self.injParams, self.fgrid_dense)       # Fiducial signal

        # Form data used in injection
        self.d_dense = {}
        self.d_standard = {}
        np.random.seed(0)
        for det in self.detsInNet.keys():
            # corruption_dense = np.random.normal(size=self.nbins_dense + 1)
            # corruption_dense = 0
            # self.d_dense[det] = self.h0_dense[det] + corruption_dense
            # self.d_standard[det] = np.interp(self.fgrid_standard, self.fgrid_dense, self.d_dense[det]).squeeze()
            self.d_dense[det] = self.h0_dense[det] 
            self.d_standard[det] = self.h0_standard[det]


        # Heterodyned strategy
        self.d_d = self._precomputeDataInnerProduct()

        self._reinitialize(chi=chi, eps=eps)

        # self.chi = chi
        # self.eps = eps
        # self.getHeterodyneBins(chi=chi, eps=eps)
        # self.getSummaryData()

        # Debugging
        self.hj0 = None
        # Warmup for JIT compile
        # self._warmup_potential(True)
        # self._warmup_potential_derivative(True) 

    def _initParams(self):
        # Remarks:
        # (i)   `tcoal` is accepted in units GMST fraction of a day
        # (ii)  GPSt_to_LMST returns GMST in units of fraction of day (GMST is LMST computed at long = 0°)
        # (iii) (GW150914) like parameters
        # (iv)  Use [tcoal - 3e-7, tcoal + 3e-7] prior when in units of days
        self.seconds_per_day = 86400. 
        tGPS = np.array([1.12625946e+09])
        tcoal = float(utils.GPSt_to_LMST(tGPS, lat=0., long=0.)) * self.seconds_per_day # [0, 1] 
        injParams = dict()
        injParams['Mc']      = np.array([34.3089283])          # (1)   # (0)               # [M_solar]
        injParams['eta']     = np.array([0.2485773])           # (2)   # (1)               # [Unitless]
        injParams['dL']      = np.array([1.5])               # (3)   # (2)               # [Gigaparsecs]  # [2.634]
        injParams['theta']   = np.array([2.78560281])          # (4)   # (3)               # [Rad]
        injParams['phi']     = np.array([1.67687425])          # (5)   # (4)               # [Rad]
        injParams['iota']    = np.array([2.67548653])          # (6)   # (5)               # [Rad]
        injParams['psi']     = np.array([0.78539816])          # (7)   # (6)               # [Rad]
        injParams['tcoal']   = np.array([tcoal])               # (8)   # (7)               # []
        injParams['Phicoal'] = np.array([0.])                  # (9)   # (8)               # [Rad]
        injParams['chi1z']   = np.array([0.27210419])          # (10)  # (9)               # [Unitless]
        injParams['chi2z']   = np.array([0.33355909])          # (11)  # (10)              # [Unitless]
        self.injParams = injParams

        priorDict = {}
        priorDict['Mc']      = [33., 36.]                      # [M_solar]      # [29., 39.]
        priorDict['eta']     = [0.23, 0.25]                    # [Unitless]
        priorDict['dL']      = [1., 3.]                        # [GPC]  [1., 4.]
        priorDict['theta']   = [0., np.pi]                     # [Rad]
        priorDict['phi']     = [0., 2 * np.pi]                 # [Rad]
        priorDict['iota']    = [0., np.pi]                     # [Rad]
        priorDict['psi']     = [0., np.pi]                     # [Rad]
        priorDict['tcoal']   = [tcoal - 0.01, tcoal + 0.01]    # []
        priorDict['Phicoal'] = [0., 2 * np.pi]                 # [Rad]
        priorDict['chi1z']   = [-1., 1.]                       # [Unitless]
        priorDict['chi2z']   = [-1., 1.]                       # [Unitless]
        self.priorDict = priorDict

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
        print('Using detectors', dets)
        LV_detectors = {det:all_detectors[det] for det in dets} # (ii) # LV_detectors = {det:all_detectors[det] for det in ['L1']}
        detector_ASD = dict() # (iii)
        detector_ASD['L1']    = 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
        detector_ASD['H1']    = 'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'
        detector_ASD['Virgo'] = 'O3-V1_sensitivity_strain_asd.txt'

        LV_detectors['L1']['psd_path']    = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['L1']) # (iv) 
        LV_detectors['H1']['psd_path']    = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['H1'])
        LV_detectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['Virgo'])
        self.NetDict = LV_detectors

        # (v) 
        waveform_model = 'TaylorF2'
        if waveform_model == 'TaylorF2':
            self.wf_model = TaylorF2_RestrictedPN() 
        elif waveform_model == 'IMRPhenomD':
            self.wf_model = IMRPhenomD() 
        print('Using waveform model: %s' % waveform_model)

    def _initFrequencyGrid(self, fmin=20, fmax=None): # Checks: X
        """
        Setup frequency grids that will be used
        Remarks:                                           
        (i)   Setup [f_min, f_max] interval
        (ii)  Standard frequency grid setup
        (iii) Once nbins_standard is calculated, df_standard must be updated
        (iv)  Dense frequency setup for heterodyning
        """
        # (i)
        self.fmin = fmin  # 10
        self.fmax = fmax  # 325
        fcut = self.wf_model.fcut(**self.injParams)[0]
        if self.fmax is None:
            self.fmax = fcut
        else:
            fcut = np.where(fcut > self.fmax, self.fmax, fcut)
        self.fcut = fcut
        
        # (ii)
        signal_duration = 4. # 4 [s]
        self.df_standard = 1 / signal_duration
        # self.nbins_standard = int(np.ceil(((self.fmax - self.fmin) / self.df_standard))) # 9000
        self.nbins_standard = 2000
        self.df_standard = (self.fmax - self.fmin) / self.nbins_standard # (iii)
        print('Standard binning scheme: % i bins' % self.nbins_standard)
        self.fgrid_standard = np.linspace(self.fmin, fcut, num=self.nbins_standard + 1).squeeze()

        # (iv)
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
                                                   tcoal   = injParams['tcoal'].astype('complex128') / self.seconds_per_day, # (iii)
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
                                                        tcoal   = X_[7] / self.seconds_per_day, # (iii)
                                                        Phicoal = X_[8],
                                                        chiS    = X_[9],
                                                        chiA    = X_[10],
                                                        is_chi1chi2 = 'True',
                                                        **dict_params_neglected)).T # (iv) 
                            
        return signal 

    # @partial(jax.jit, static_argnums=(0,))
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
                                                                       tcoal   = X_[7] / self.seconds_per_day, # Correction 1
                                                                       Phicoal = X_[8],
                                                                       chiS    = X_[9],
                                                                       chiA    = X_[10],
                                                                       use_chi1chi2 = True,
                                                                       **dict_params_neglected) 

            jacModel[det] = jacModel[det].at[7].divide(self.seconds_per_day) # Correction 2

        return jacModel
            
    # @partial(jax.jit, static_argnums=(0,))
    def square_norm(self, a, PSD, deltaf):
        """ 
        Square norm estimated using left Riemann sum
        """
        res = {}
        for det in self.detsInNet.keys():
            res[det] = (4 * jnp.sum((a[det].real[..., :-1] ** 2 + a[det].imag[..., :-1] ** 2) / PSD[det][..., :-1] * deltaf, axis=-1)).T
        return res

    # @partial(jax.jit, static_argnums=(0,))
    def overlap(self, a, b, PSD, deltaf):
        """ 
        Overlap estimated using left Riemann sum
        """
        res = {}
        for det in self.detsInNet.keys():
            res[det] = (4 * jnp.sum(a[det].conjugate()[..., :-1] * b[det][..., :-1] / PSD[det][..., :-1] * deltaf, axis=-1)).T
        return res

    def overlap_trap(self, a, b, PSD, fgrid):
        """ 
        Overlap estimated using left Riemann sum
        """
        res = {}
        for det in self.detsInNet.keys():
            integrand = a[det].conjugate() * b[det] / PSD[det]
            res[det] = (4 * jnp.trapz(integrand, fgrid)).T
            # res[det] = (4 *  * deltaf, axis=-1)).T
        return res
    
    def _precomputeDataInnerProduct(self): # Checks
        print('Precomputing squared SNR for likelihood')
        # Remarks:
        # (i) The snr is actually sqrt(<d,d>). We are calculating the square! Hence SNR2
        SNR2 = {}
        SNR = 0
        for det in self.detsInNet.keys():
            res = 4 * np.sum((self.d_dense[det].real ** 2 + self.d_dense[det].imag ** 2) / self.PSD_dense[det]) * self.df_dense
            SNR2[det] = res
            SNR += res
        print('SNR: %f' % np.sqrt(SNR))
        return SNR2

    @partial(jax.jit, static_argnums=(0,))
    def standard_minusLogLikelihood(self, X): # Checks: XX
        """ 
        """
        nParticles = X.shape[0]
        log_likelihood = jnp.zeros(nParticles)
        template = self.getSignal(X, self.fgrid_standard) # signal template
        for det in self.detsInNet.keys():
            residual = template[det] - self.d_standard[det][np.newaxis, ...]
            inner_product = 4 * jnp.sum((residual.real ** 2 + residual.imag ** 2) / self.PSD_standard[det][np.newaxis, ...], axis=-1) * self.df_standard
            log_likelihood += 0.5 * inner_product
        return log_likelihood

    @partial(jax.jit, static_argnums=(0,))
    def standard_gradientMinusLogLikelihood(self, X): # Checks: XX
        # Remarks:
        # (i) Jacobian is (d, N, f) shaped. sum over final axis gives (d, N), then transpose to give (N, d)
        nParticles = X.shape[0]
        grad_log_like = jnp.zeros((nParticles, self.DoF))
        template = self.getSignal(X, self.fgrid_standard)
        jacSignal = self._getJacobianSignal(X, self.fgrid_standard)
        for det in self.detsInNet.keys():
            residual = template[det] - self.d_standard[det][np.newaxis, ...]
            inner_product = (4 * jnp.sum(jacSignal[det].conjugate() * residual[np.newaxis, ...] / self.PSD_standard[det], axis=-1) * self.df_standard).T # (i)
            # Confirm that these two give the same result.
            # inner_product = (4 * contract('dNf, Nf, f -> Nd', jacSignal[det].conjugate(), residual, 1 / self.PSD_standard[det]) * self.df_standard
            grad_log_like += inner_product.real
        return grad_log_like
    
    @partial(jax.jit, static_argnums=(0,))
    def standard_GNHessianMinusLogLikelihood(self, X): # Checks: XX
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

    def _reinitialize(self, chi, eps):
        self.chi = chi 
        self.eps = eps
        print('Forming sparse subgrid')
        self.getHeterodyneBins(chi=chi, eps=eps)
        print('Completed')
        print('Calculating summary data')
        self.A0, self.A1, self.B0, self.B1 = self.getSummary_data()
        print('Completed')

    def getHeterodyneBins(self, chi, eps): # Checks X
        print('Getting heterodyned bins')
        # Remarks:
        # (i)   0.5 is a dummy variable for x==0 case (which we dont care for)
        # (ii)  Alternatively, we may add frequencies then recover the indicies using np.searchsorted(A,B):
        # https://stackoverflow.com/questions/33678543/finding-indices-of-matches-of-one-array-in-another-array

        gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3])
        f_star = self.fmax * np.heaviside(gamma, 0.5) + self.fmin * np.heaviside(-gamma, 0.5) # (i) 
        delta = lambda f_minus, f_plus: 2 * np.pi * chi * np.sum(np.abs((f_plus / f_star) ** gamma - (f_minus/f_star) ** gamma))
        delta_new = lambda f_minus, f_plus: 2 * np.pi * chi * np.sum(np.abs((f_plus[:, np.newaxis] / f_star) ** gamma - (f_minus[:, np.newaxis] / f_star) ** gamma), axis=-1)
        delta0 = np.min(delta_new(self.fgrid_dense[:-1], self.fgrid_dense[1:]))
        if eps < delta0:
            print('Changing epsilon from %f to %f' % (eps, delta0))
            eps = delta0

        subindex = [] # (ii)
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
            Given an array over a dense grid, and the indicies of the dense grid which define
            the subgrid (and by extension the bins), return the sum over elements in each bin.
            """
            tmp = np.zeros(len(array) + 1).astype(array.dtype)
            tmp[1:] = np.cumsum(array) # (iii)
            tmp[-2] = tmp[-1] # (iv) 
            return tmp[bin_indicies[1:]] - tmp[bin_indicies[:-1]] # (v) 

        A0, A1, B0, B1 = {}, {}, {}, {}

        bin_id = (np.digitize(self.fgrid_dense, self.bin_edges)) - 1 # (ia), (ib)
        bin_id[-1] = self.nbins - 1 # (ic)
        elements_per_bin = np.bincount(bin_id) # (ii)

        deltaf_in_bin = self.fgrid_dense - np.repeat(self.bin_edges[:-1], elements_per_bin)

        for det in self.detsInNet.keys():
            A0_integrand = 4 * self.h0_dense[det].conjugate() * self.d_dense[det] / self.PSD_dense[det] * self.df_dense
            A1_integrand = A0_integrand * deltaf_in_bin
            B0_integrand = 4 * (self.h0_dense[det].real ** 2 + self.h0_dense[det].imag ** 2) / self.PSD_dense[det] * self.df_dense
            B1_integrand = B0_integrand * deltaf_in_bin
            for data, integrand in zip([A0, A1, B0, B1], [A0_integrand, A1_integrand, B0_integrand, B1_integrand]):
                data[det] = sumBins(integrand, self.indicies_kept)

        return A0, A1, B0, B1




    def getFirstSplineData(self, X):
        """ 
        Return N x b matrix for spline of r := h/h0
        """
        # Remarks:
        # (i)   r is the heterodyne
        # (ii)  These are the y-intercepts for each bin (N x b)
        # (iii) These are the slopes for each bin (N x b)
        r0 = {}
        r1 = {}
        h = self.getSignal(X, self.bin_edges)
        for det in self.detsInNet.keys():
            r = h[det] / self.h0_dense[det][self.indicies_kept] # (i)
            r0[det] = r[:, :-1] # (ii)
            r1[det] = (r[:, 1:] - r[:, :-1]) / self.bin_widths # (iii)
        return r0, r1

    def getSecondSplineData(self, X):
        """ 
        Return matrix of shape (d, N, b) for spline of r_{,j} := h_{,j} / h0
        Note: Identical as first spline data for first order.
        """
        rj0 = {}
        rj1 = {}
        hj = self._getJacobianSignal(X, self.bin_edges)
        for det in self.detsInNet.keys():
            rj = hj[det] / self.h0_dense[det][self.indicies_kept]
            rj0[det] = rj[..., :-1]
            rj1[det] = (rj[..., 1:] - rj[..., :-1]) / self.bin_widths
        return rj0, rj1

    @partial(jax.jit, static_argnums=(0,))
    def heterodyne_minusLogLikelihood(self, X): # Checks X
        # Remarks:
        # (i) Summary data has shape (b,)
        nParticles = X.shape[0]
        r0, r1 = self.getFirstSplineData(X)
        log_like = jnp.zeros(nParticles)
        for det in self.detsInNet.keys():
            h_d = jnp.sum(self.A0[det][jnp.newaxis] * r0[det].conjugate() + self.A1[det][jnp.newaxis] * r1[det].conjugate(), axis=1)
            h_h = jnp.sum(self.B0[det][jnp.newaxis] * jnp.abs(r0[det]) ** 2 + 2 * self.B1[det][jnp.newaxis] * (r0[det].conjugate() * r1[det]).real, axis=1)
            log_like += 0.5 * h_h - h_d.real + 0.5 * self.d_d[det]
        return log_like

    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogPosterior_ensemble(self, X):
        # Remarks:
        # (i)   second spline data is (d, N, b) shaped

        nParticles = X.shape[0]
        r0, r1 = self.getFirstSplineData(X)
        rj0, rj1 = self.getSecondSplineData(X)
        grad_log_like = np.zeros((nParticles, self.DoF))
        for det in self.detsInNet.keys():
            hj_d = contract('jNb -> Nj', self.A0[det] * rj0[det].conjugate() \
                                       + self.A1[det] * rj1[det].conjugate(), backend='jax')

            hj_h = contract('jNb -> Nj', self.B0[det] * rj0[det].conjugate() * r0[det][np.newaxis] 
                                      +  self.B1[det] *(rj0[det].conjugate() * r1[det][np.newaxis] + rj1[det].conjugate() * r0[det][np.newaxis]), backend='jax')
            grad_log_like += hj_h.real - hj_d.real
        return grad_log_like

    # @partial(jax.jit, static_argnums=(0,))
    def getGNHessianMinusLogPosterior_ensemble(self, X):
        nParticles = X.shape[0]
        rj0, rj1 = self.getSecondSplineData(X)
        GN = jnp.zeros((nParticles, self.DoF, self.DoF))
        for det in self.detsInNet.keys():
            term1 = contract('b, jNb, kNb -> Njk', self.B0[det], rj0[det].conjugate(), rj0[det], backend='jax')
            term2 = contract('b, jNb, kNb -> Njk', self.B1[det], rj0[det].conjugate(), rj1[det], backend='jax')
            term3 = contract('Nkj -> Njk', term2.conjugate(), backend='jax')
            GN += term1.real + term2.real + term3.real
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

    def _newDrawFromPrior(self, nParticles):
        prior_draw = np.zeros((nParticles, self.DoF))
        for i, param in enumerate(self.gwfast_param_order): # Assuming uniform on all parameters
            low = self.priorDict[param][0]
            high = self.priorDict[param][1]
            prior_draw[:, i] = np.random.uniform(low=low, high=high, size=nParticles)
                   
        return prior_draw

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
        Z = np.exp(-1 * func(particle_grid).reshape(ngrid,ngrid))
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



    # def jac_r(self, X):
    #     jac_r = {}
    #     jacSignal = self._getJacobianSignal(X, self.bin_edges)
    #     for det in self.detsInNet.keys():
    #         jac_r[det] = jacSignal[det] / self.h0_dense[det][self.indicies_kept]
    #     return jac_r

    # def getJacSplineData(self, X):
    #     jac_r = self.jac_r(X)
    #     jac_r0 = {}
    #     jac_r1 = {}
    #     for det in self.detsInNet.keys():
    #         jac_r0[det] = jac_r[det][..., :-1]
    #         jac_r1[det] = (jac_r[det][..., 1:] - jac_r[det][..., :-1]) / self.bin_widths
    #     return jac_r0, jac_r1