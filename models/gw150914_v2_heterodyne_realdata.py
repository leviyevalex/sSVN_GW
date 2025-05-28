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
from scipy.interpolate import interp1d

# from astropy.cosmology import Planck18
from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)

# from jaxopt import Bisection

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


    # bin_ends[-1] = len(f_grid) - 1
    # arr = np.array(arr)
    # f_grid = np.array(f_grid)
    # sparse_grid = np.array(sparse_grid)
    # cumsum[1:] = jnp.cumsum(arr)
def sum_in_bins(arr, f_grid, sparse_grid):
    """ 
    Sum values of `arr` binned according to `f_grid` within ranges defined by `sparse_grid`.
    Eliminates last element in sparse_grid to avoid bin edge placement at boundary.
    Assumes f_grid and sparse_grid are sorted and have no repeated elements.
    Pads cumulative sum array to handle edge case where i=0.
    """

    # Find bin starts using the left edge of each bin
    bin_starts = jnp.searchsorted(f_grid, sparse_grid[:-1], side='left')
    
    # Correct calculation for bin ends
    bin_ends = jnp.searchsorted(f_grid, sparse_grid[1:], side='left') - 1
    
    # Handle out-of-bounds case for the last bin
    bin_ends = bin_ends.at[-1].set(len(f_grid) - 1)

    # Compute cumulative sum array for fast range sums
    cumsum = jnp.zeros(len(arr) + 1, dtype=arr.dtype)
    cumsum = cumsum.at[1:].set(jnp.cumsum(arr))

    # Return the sum in each bin by difference of cumulative sums
    return cumsum[bin_ends + 1] - cumsum[bin_starts]


# def getBinIds(grid, bins):
#     """ 
#     Given bins, returns an array labeling which bin each point in grid belongs to.
#     Bins are labeled beginning from 0 to nbins - 1!
#     """
#     bin_ids = np.digitize(grid, bins) - 1 # (ia), (ib)
#     bin_ids[-1] = len(bins) - 2 # (ic)
#     return bin_ids


class gwfast_LVGW150914(object):
    # def __init__(self, wf_model=TaylorF2_RestrictedPN, nbins=1000, fmin=10., fmax=560.):
    def __init__(self, wf_model=TaylorF2_RestrictedPN, nbins=1968, fmin=20., fmax=512., chi=0.5, eps=0.5):

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
            # self.nbins = nbins
            # self.fgrid = jnp.linspace(self.fmin, self.fmax, num=self.nbins + 1).squeeze()
            # self.df = (self.fgrid[-1] - self.fgrid[0]) / self.nbins

            # Settings as in https://git.ligo.org/lscsoft/bilby/blob/master/examples/gw_examples/data_examples/GW150914.py 
            tGPS = np.array([1126259462.4])
            post_trigger_duration=2
            duration=4
            start_time = tGPS + post_trigger_duration - duration
            self.reftime = utils.GPSt_to_LMST(start_time, 0., 0.)[0]
            
            self.data = {}
            asd_compute = {}
            
            self.fgrid, self.data['L1'], asd_compute['L1'] = self.get_data_from_gwpy(ifo='L1', gps_time=tGPS[0], duration=duration, post_trigger_duration=post_trigger_duration, roll_off=0.2, roll_off_psd=0.4)
            _, self.data['H1'], asd_compute['H1']          = self.get_data_from_gwpy(ifo='H1', gps_time=tGPS[0], duration=duration, post_trigger_duration=post_trigger_duration, roll_off=0.2, roll_off_psd=0.4)
            
            self.data['L1'] = self.data['L1'][jnp.newaxis,:]
            self.data['H1'] = self.data['H1'][jnp.newaxis,:]
            # self.nbins = len(self.fgrid)
            self.nbins = len(self.fgrid - 1)
            self.df = np.array(self.fgrid)[1] - np.array(self.fgrid)[0]

            # Point to which detector characteristics we want
            asd_paths = {}
            asd_paths['L1']    = '/home/al44828/projects/sSVN_GW/notebooks/LIGO_L_ASD_GW150914.txt'
            asd_paths['H1']    = '/home/al44828/projects/sSVN_GW/notebooks/LIGO_H_ASD_GW150914.txt'
            # asd_paths['Virgo'] = '/home/al44828/projects/sSVN_GW/notebooks/AdV_asd.txt'

            # Define network object
            self.Net = LV_DetNet(self.wf_model, fixed_fgrid=self.fgrid, verbose=False, ASDs=asd_paths)
            
            self.Net.signals['L1'].strainFreq = self.fgrid
            self.Net.signals['H1'].strainFreq = self.fgrid
            self.Net.signals['L1'].noiseCurve = asd_compute['L1']**2
            self.Net.signals['H1'].noiseCurve = asd_compute['H1']**2

            # Interpolate detector characteristics onto defined frequency grid
            self.PSDs = {}
            self.PSDs['L1']    = jnp.interp(self.fgrid, self.Net.signals['L1'].strainFreq, self.Net.signals['L1'].noiseCurve, left=1., right=1.).squeeze()
            self.PSDs['H1']    = jnp.interp(self.fgrid, self.Net.signals['H1'].strainFreq, self.Net.signals['H1'].noiseCurve, left=1., right=1.).squeeze()

            # LATEST CATELOG MEDIANS ( THESE ARE THE CORRECT ONES, TODO CHANGE LATER!!! )
            # tGPS = np.array([1126259462.419288])
            dtcoal = (tGPS - start_time)[0]/3600/24#float(np.array(utils.GPSt_to_LMST(tGPS, lat=0., long=0.))) * milliseconds_per_day
            tcoal = (self.reftime + dtcoal) * milliseconds_per_day
            injParams = {}
            injParams['Mc']      = np.array([30.68716026])                        # (0)   # [M_solar]      # Chirp mass
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
            bounds['q']       = [0.5 * q_rescaling, 1. * q_rescaling]         
            bounds['dL']      = [0.05 * dL_rescaling, 2. * dL_rescaling]                     
            bounds['theta']   = [0., np.pi]                   
            bounds['phi']     = [0., 2 * np.pi]               
            bounds['iota']    = [0., np.pi]                   
            bounds['psi']     = [0., np.pi]                   
            # bounds['tcoal']   = [tcoal - 100, tcoal + 100]
            bounds['tcoal']   = [tcoal - 10, tcoal + 10]
            bounds['Phicoal'] = [0., 2 * np.pi]               
            bounds['chi1z']   = [-0.99, 0.99]                 
            bounds['chi2z']   = [-0.99, 0.99]                 
            self.bounds = bounds

            # Get mock data
            # NOTE: Rescaling of t_c is handled in `getSignal` method separately
            self.true_params = jnp.array([self.injParams[param].squeeze() for param in self.gwfast_param_order])
            self.htrue = self.getSignal(self.true_params.at[jnp.array([1,2])].divide(jnp.array([q_rescaling, dL_rescaling]))[None,:])
            self.htrue['H1'] = self.htrue['H1'].squeeze()
            self.htrue['L1'] = self.htrue['L1'].squeeze()

            simulated_data = False
            if simulated_data:
                self.data = copy.copy(self.htrue)
                self.noise = {}
                np.random.seed(42)
                for det in self.PSDs.keys(): # NOTE: Number of
                    self.noise[det] = self.generate_noise_from_asd(self.Net.signals[det].strainFreq, np.sqrt(self.Net.signals[det].noiseCurve), self.fgrid)
                    # self.noise[det] = 0 # NOTE: COMMENT THIS OUT IF YOU WANT NOISY INJECTION!!!
                    self.data[det] += self.noise[det]

            self.extraneous()

            # New heterodyning stuff
            self.chi = chi 
            self.eps = eps
            self.sparse_grid = self.get_heterodyne_grid(chi, eps, fmin, fmax)
            self.bin_widths = self.sparse_grid[1:] - self.sparse_grid[:-1]

            # Define new net to evaluate at sparse grid
            self.Net_sparse = LV_DetNet(self.wf_model, fixed_fgrid=self.sparse_grid, verbose=True, ASDs=asd_paths)

            # Cache h0
            self.h0 = self.getSignal_sparse(self.true_params.at[jnp.array([1,2])].divide(jnp.array([q_rescaling, dL_rescaling]))[None,:])
            self.h0['H1'] = self.h0['H1'].squeeze()
            self.h0['L1'] = self.h0['L1'].squeeze()

            # Cache data norm
            self.d_d = {}
            self.d_d['L1'] = self.square_norm(self.data['L1'], self.PSDs['L1'], self.df)
            self.d_d['H1'] = self.square_norm(self.data['H1'], self.PSDs['H1'], self.df)

            self.A0, self.A1, self.B0, self.B1 = self.get_summary_data()


    # g_inverse = interp1d(jax.vmap(g)(f_dense), f_dense, kind='linear') 

    def get_heterodyne_grid(self, chi, eps, f_min, f_max): # NOTE: We use jax here for the convenient vmap on `g`
        gammas = jnp.array([-5/3, -2/3, 1, 5/3, 7/3])

        f_star = f_max * jnp.heaviside(gammas, 1) + f_min * (1 - jnp.heaviside(gammas, 1))
        
        g = lambda f: 2 * jnp.pi * chi * jnp.sum(jnp.sign(gammas) * ((f / f_star) ** gammas))

        self.num_bins = jnp.floor((g(f_max) - g(f_min)) / eps) + 1

        eps_prime = (g(f_max) - g(f_min)) / self.num_bins

        assert(eps_prime < eps)

        print('n_bins = %i' % self.num_bins)

        g_grid = g(f_min) + jnp.arange(0, self.num_bins + 1) * eps_prime

        # Interpolation step
        f_dense = jnp.linspace(f_min, f_max, 10000) 
        g_dense = jax.vmap(g)(f_dense)        
        # g_inverse = jnp.interp1d(g_grid, g_dense, f_dense) 

        # sparse_grid = g_inverse(g_grid)
        sparse_grid = jnp.interp(g_grid, g_dense, f_dense) 
        # sparse_grid[0] = f_min
        # sparse_grid[-1] = f_max

        assert jnp.allclose(sparse_grid, jnp.sort(sparse_grid))

        return sparse_grid

    # elements_per_bin = np.bincount(getBinIds(self.fgrid, self.sparse_grid)) # (ii)
    # B0_integrand = 4 * (self.htrue[det].real ** 2 + self.htrue[det].imag ** 2) / self.PSDs[det] * self.df
    # bin_ids[-1] = len(self.sparse_grid) - 2
    def get_summary_data(self): 
        A0, A1, B0, B1 = {}, {}, {}, {}

        # Find which frequencies belong to which bin
        bin_ids = jnp.digitize(self.fgrid, self.sparse_grid) - 1
        bin_ids = bin_ids.at[-1].set(len(self.sparse_grid) - 2)

        elements_per_bin = jnp.bincount(bin_ids)

        f_left = self.sparse_grid[:-1]
        # f_mid = self.sparse_grid[:-1] + self.bin_widths / 2
        deltaf_in_bin = self.fgrid - np.repeat(f_left, elements_per_bin)

        # assert(np.any(elements_per_bin == 0) == False)

        for det in self.PSDs.keys():
            A0_integrand = 4 * (self.htrue[det].conjugate() * self.data[det]) / self.PSDs[det] * self.df
            A1_integrand = A0_integrand * deltaf_in_bin
            B0_integrand = 4 * jnp.abs(self.htrue[det]) ** 2 / self.PSDs[det] * self.df
            B1_integrand = B0_integrand * deltaf_in_bin
            for summary_data, integrand in zip([A0, A1, B0, B1], [A0_integrand, A1_integrand, B0_integrand, B1_integrand]):
                summary_data[det] = sum_in_bins(integrand.squeeze(), self.fgrid, self.sparse_grid)
        return A0, A1, B0, B1

    def potential(self, X): # Checks X

        """ 
        Remarks:
        (i) Summary data has shape (b,)
        """

        nParticles = X.shape[0]
        V = jnp.zeros(nParticles)

        X = X.at[:, 1].divide(q_rescaling)
        X = X.at[:, 2].divide(dL_rescaling)

        h = self.getSignal_sparse(X)

        for det in self.PSDs.keys():
            r = h[det] / self.h0[det]
            r0 = r[:, :-1] # Left points (y-intercepts)
            # r0 = r[:, -1] + (r[:, 1:] - r[:, :-1]) / 2 # center points (y-intercepts)
            r1 = (r[:, 1:] - r[:, :-1]) / self.bin_widths # Slopes

            h_d = jnp.sum(self.A0[det] * r0.conjugate() + self.A1[det] * r1.conjugate(), axis=1)
            h_h = jnp.sum(self.B0[det] * jnp.abs(r0) ** 2 + 2 * self.B1[det] * (r0.conjugate() * r1).real, axis=1)

            V += 0.5 * h_h - h_d.real + 0.5 * self.d_d[det]

        # Prior contribution to energy
        V += minusLogPrior(X)

        return V


    def gradient_potential(self, X):
        nParticles = X.shape[0]
        grad_V = jnp.zeros((nParticles, self.DoF))

        X = X.at[:, 1].divide(q_rescaling)
        X = X.at[:, 2].divide(dL_rescaling)

        h = self.getSignal_sparse(X)
        hj = self._getJacobianSignal_sparse(X)

        for det in self.PSDs.keys():
            r = h[det] / self.h0[det]
            r0 = r[:, :-1] # Left points (y-intercepts)
            r1 = (r[:, 1:] - r[:, :-1]) / self.bin_widths # Slopes

            rj = hj[det] / self.h0[det]
            rj0 = rj[..., :-1]
            rj1 = (rj[..., 1:] - rj[..., :-1]) / self.bin_widths

            hj_d = jnp.sum(self.A0[det] * rj0.conjugate() + self.A1[det] * rj1.conjugate(), axis=-1).T
            hj_h = jnp.sum(self.B0[det] * rj0.conjugate() * r0[np.newaxis] + self.B1[det] * (rj0.conjugate() * r1 + rj1.conjugate() * r0), axis=-1).T

            grad_V += hj_h.real - hj_d.real

        
        # Gradient of prior
        grad_V = grad_V.at[:, jnp.array([0, 1, 2, 3, 5, 9, 10])].add(gradient_minusLogPrior(X))

        # NOTE: As priors are added this will need to be updated as well!



        #grad_V = grad_V.at[:, 1].divide(eta_rescaling)
        grad_V = grad_V.at[:, 1].divide(q_rescaling)
        grad_V = grad_V.at[:, 2].divide(dL_rescaling)

        return grad_V






    # def getFirstSplineData(self, X):
    #     """ 
    #     Return N x b matrix for spline of r := h/h0 in a particular detector
    #     """
    #     # Remarks:
    #     # (i)   r is the heterodyne
    #     # (ii)  These are the y-intercepts for each bin (N x b)
    #     # (iii) These are the slopes for each bin (N x b)
    #     h = self.getSignal_sparse(X)
    #     r0, r1 = {}, {}

    #     for det in self.PSDs.keys():
    #         r = h[det] / self.h0[det] # (i)
    #         r0[det]  = r[:, :-1] # (ii)
    #         r1[det] = (r[:, 1:] - r[:, :-1]) / self.bin_widths # (iii)

    #     return r0, r1

    # def getSecondSplineData(self, X, det):
    #     """ 
    #     Return matrix of shape (d, N, b) for spline of r_{,j} := h_{,j} / h0
    #     Note: Identical as first spline data for first order.
    #     """
    #     hj = self._getJacobianSignal_sparse(X, self.sparse_grid, det)
    #     rj = hj / self.h0[det]
    #     rj0 = rj[..., :-1]
    #     rj1 = (rj[..., 1:] - rj[..., :-1]) / self.bin_widths
    #     return rj0, rj1







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
        # self.snr += self.square_norm(self.htrue['Virgo'], self.PSDs['Virgo'], self.df)


        print('SNR at true values: %.2f' % jnp.sqrt(self.snr))

    def getSignal(self, X):
        """
        Wrapper for Francesco's GWStrain method

        Returns
        -------
        Dictonary of signals found in each detector with shape N, f

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

    def getSignal_sparse(self, X):
        """
        Wrapper for Francesco's GWStrain method

        Returns
        -------
        Dictonary of signals found in each detector with shape N, f

        """
        
        X_ = X.T.astype('complex128')
        signals = self.Net_sparse.GWstrain(Mc       = X_[0],
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

    def _getJacobianSignal_sparse(self, X):
        """
        Wrapper for Francesco's SignalDerivatives method

        Returns
        -------
        A (d, N, f) shaped array 
        """

        X_ = X.T.astype('complex128')
        jacModel = self.Net_sparse._SignalDerivatives(Mc      = X_[0],
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
        jacModel['L1'] = jacModel['L1'].at[7].divide(milliseconds_per_day) 
        jacModel['H1'] = jacModel['H1'].at[7].divide(milliseconds_per_day)
        # jacModel['Virgo'] = jacModel['Virgo'].at[7].divide(milliseconds_per_day)
        
        # NOTE: now the derivative is with respect to q, not eta, so we need to plug in the last jacobian element from eta to q
        jacModel['L1'] = jacModel['L1'].at[1].multiply(jac_q_to_eta(X_[1])[...,None]) 
        jacModel['H1'] = jacModel['H1'].at[1].multiply(jac_q_to_eta(X_[1])[...,None])

        # Switch parameter order (redundent in newer version of gwfast)
        #jacModel['L1'] = jacModel['L1'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]
        #jacModel['H1'] = jacModel['H1'][jnp.array([0, 1, 4, 5, 6, 7, 8, 9, 10, 2, 3])]
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

    # def potential(self, X):
    #     """ 
    #     Calculates potential V(x) = -ln(likelihood(x)) - ln(prior(x))

    #     """
    #     # TODO: DOUBLE CHECK THAT THIS ISNT OVERWRITING PARTICLES OUTSIDE OF THIS METHOD


    #     #X = X.at[:, 1].divide(eta_rescaling)
    #     X = X.at[:, 1].divide(q_rescaling)
    #     X = X.at[:, 2].divide(dL_rescaling)

    #     # Calculate residuals
    #     template = self.getSignal(X)
    #     residual = {}
    #     residual['L1'] = template['L1'] - self.data['L1']
    #     residual['H1'] = template['H1'] - self.data['H1']
    #     # residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

    #     # Likelihood contribution to energy
    #     V  = 0.5 * self.square_norm(residual['L1'], self.PSDs['L1'], self.df)
    #     V += 0.5 * self.square_norm(residual['H1'], self.PSDs['H1'], self.df)
    #     # V += 0.5 * self.square_norm(residual['Virgo'], self.PSDs['Virgo'], self.df)

    #     # Prior contribution to energy
    #     V += minusLogPrior(X)

    #     return V
    
    # def gradient_potential(self, X): 
    #     """ 
    #     Calculates the gradient of the potential
        
    #     """

    #     #X = X.at[:, 1].divide(eta_rescaling)
    #     X = X.at[:, 1].divide(q_rescaling)
    #     X = X.at[:, 2].divide(dL_rescaling)

    #     # Calculate residuals
    #     template = self.getSignal(X)
    #     residual = {}
    #     residual['L1'] = template['L1'] - self.data['L1']
    #     residual['H1'] = template['H1'] - self.data['H1']
    #     # residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

    #     # Gradient of likelihood 
    #     jacSignal = self._getJacobianSignal(X)
    #     grad_V = self.overlap(jacSignal['L1'], residual['L1'], self.PSDs['L1'], self.df).real
    #     grad_V += self.overlap(jacSignal['H1'], residual['H1'], self.PSDs['H1'], self.df).real
    #     # grad_V += self.overlap(jacSignal['Virgo'], residual['Virgo'], self.PSDs['Virgo'], self.df).real

    #     # Gradient of prior
    #     grad_V = grad_V.at[:, jnp.array([0, 1, 2, 3, 5, 9, 10])].add(gradient_minusLogPrior(X))

    #     # NOTE: As priors are added this will need to be updated as well!

    #     #grad_V = grad_V.at[:, 1].divide(eta_rescaling)
    #     grad_V = grad_V.at[:, 1].divide(q_rescaling)
    #     grad_V = grad_V.at[:, 2].divide(dL_rescaling)

        # return grad_V



    # def gradient_potential_GAUSS(self, X): 
    #     """ 
    #     Calculates the gradient of the potential
        
    #     """

    #     X = X.at[:, 1].divide(eta_rescaling)
    #     X = X.at[:, 2].divide(dL_rescaling)

    #     # Calculate residuals
    #     template = self.getSignal(X)
    #     residual = {}
    #     residual['L1'] = template['L1'] - self.htrue['L1']
    #     residual['H1'] = template['H1'] - self.htrue['H1']
    #     # residual['Virgo'] = template['Virgo'] - self.htrue['Virgo']

    #     # Gradient of likelihood 
    #     jacSignal = self._getJacobianSignal(X)
    #     grad_V = self.overlap(jacSignal['L1'], residual['L1'], self.PSDs['L1'], self.df).real
    #     grad_V += self.overlap(jacSignal['H1'], residual['H1'], self.PSDs['H1'], self.df).real
    #     # grad_V += self.overlap(jacSignal['Virgo'], residual['Virgo'], self.PSDs['Virgo'], self.df).real

    #     # Gradient of prior
    #     grad_V = grad_V.at[:, jnp.array([0, 1])].add(gradient_minusLogPrior(X))

    #     # NOTE: As priors are added this will need to be updated as well!

    #     grad_V = grad_V.at[:, 1].divide(eta_rescaling)
    #     grad_V = grad_V.at[:, 2].divide(dL_rescaling)

    #     #######################################
    #     # Gauss-Newton approximation to Hessian
    #     #######################################

    #     gauss_newton = self.overlap_GN(jacSignal['L1'], jacSignal['L1'], self.PSDs['L1'], self.df).real
    #     gauss_newton += self.overlap_GN(jacSignal['H1'], jacSignal['H1'], self.PSDs['H1'], self.df).real
    #     # gauss_newton += self.overlap_GN(jacSignal['Virgo'], jacSignal['Virgo'], self.PSDs['Virgo'], self.df).real

    #     gauss_newton = gauss_newton.at[:, :, 1].divide(eta_rescaling)
    #     gauss_newton = gauss_newton.at[:, :, 2].divide(dL_rescaling)

    #     gauss_newton = gauss_newton.at[:, 1, :].divide(eta_rescaling)
    #     gauss_newton = gauss_newton.at[:, 2, :].divide(dL_rescaling)

    #     # gauss_newton = gauss_newton.at[:, 0, 0].add(X[:,0] ** 2) # Hessian of prior on Mc

        # return grad_V, gauss_newton
    

        # # Root solving code
        # def F(f, factor):
        #     return g(f) - factor

        # def root(factor):
        #     bisec = Bisection(optimality_fun=F, lower=f_min - 1, upper=f_max + 1)
        #     return bisec.run(factor=factor).params

        # sparse_grid = jnp.zeros(int(num_bins + 1))

        # # TODO: Do this using vmap
        # print('Root solving')
        # for i in range(len(g_grid)):
        #     # print(i)
        #     entry = root(g_grid[i])
        #     sparse_grid = sparse_grid.at[i].set(entry.squeeze())

        # TODO We dont need the preimage of g on the first and last elements. We can fix this later
        # sparse_grid = sparse_grid.at[0].set(f_min)
        # sparse_grid = sparse_grid.at[-1].set(f_max)




    def _newDrawFromPrior(self, n, seed=42):
        prior_samples = np.zeros((n, self.DoF))
        for i in range(self.DoF):
            prior_samples[:, i] = np.random.uniform(low=self.lower_bound[i], high=self.upper_bound[i], size=n)

        # Draw samples from prior law
        prior_samples[:, 2] = dL_power_law_draw(n, self.lower_bound[2], self.upper_bound[2])
        
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

    def get_data_from_gwpy(self, ifo, gps_time, duration=4, post_trigger_duration=2, roll_off=0.2, roll_off_psd=0.4, psd_extra_duration=32):
        
        end_time = gps_time + post_trigger_duration
        start_time = end_time - duration
        
        # The roll-off (in seconds) used in the Tukey window corresponds to alpha * duration / 2 for scipy tukey window.
        tukey_alpha = 2 * roll_off / duration
        tukey_alpha_psd = 2 * roll_off_psd / duration
        
        data_time = TimeSeries.fetch_open_data(ifo,
                                                start_time,
                                                end_time, cache=True)

        n = len(data_time)
        delta_t = data_time.dt.value  
        data_freq = jnp.fft.rfft(jnp.array(data_time.value) * tukey(n, tukey_alpha)) * delta_t
        freq = jnp.fft.rfftfreq(n, delta_t)
        
        data_time_asd = TimeSeries.fetch_open_data(ifo,
                                                   start_time - duration*psd_extra_duration,
                                                   start_time, cache=True)
        asd = data_time_asd.asd(fftlength=duration, overlap=0, window=("tukey", tukey_alpha_psd), method="median").value
        
        frequencies = freq[(freq > self.fmin) & (freq < self.fmax)]
        data = data_freq[(freq > self.fmin) & (freq < self.fmax)] * np.exp(-1j * 2 * np.pi * frequencies * self.reftime * 3600. * 24.)
        asd = asd[(freq > self.fmin) & (freq < self.fmax)]
        
        return frequencies, data, asd  

# def sum_in_bins(arr, f_grid, sparse_grid):
#     """ 
#     # NOTE: Eliminate last element in sparse_grid to avoid bin edge placement at boundary
#     # NOTE: Assumes f_grid and sparse_grid are sorted and have no repeated elements
#     # NOTE: Pad cumsum array to handle edge case where i=0
#     """
#     bin_starts = np.searchsorted(f_grid, sparse_grid[:-1], side='left')

#     bin_ends = np.zeros_like(bin_starts)
#     bin_ends[:-1] = bin_starts[1:] - 1
#     bin_ends[-1] = len(f_grid) - 1

#     cumsum = np.zeros(len(arr) + 1, dtype=arr.dtype)
#     cumsum[1:] = np.cumsum(arr)
#     return cumsum[bin_ends + 1] - cumsum[bin_starts]