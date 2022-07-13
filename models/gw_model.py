import numpy as np
import os
import pathlib
from scipy import interpolate
import jax
import jax.numpy as jnp
from models.ripple import ms_to_Mc_eta
from models.ripple.waveforms import IMRPhenomD
#%% Get noise interpolation
# data = np.loadtxt(os.path.join(ROOT, 'models', 'ripple', 'noise_resources', 'aLIGO.dat'))
# interpolated_psd = interpolate.interp1d(data[:, 0], data[:, 1])
# #%%
# # Now we need to generate the frequency grid
# f_l = 24
# f_u = 512
# del_f = 0.01
# fs = jnp.arange(f_l, f_u, del_f)
# #%%
# waveform = lambda theta: IMRPhenomD.gen_IMRPhenomD(fs, theta).real
# #%%
# noise_injection = np.random.normal(0, 1, fs.shape[0])
# #%%
# likelihood = lambda theta: np.sum(((np.absolute(waveform(theta) - noise_injection)) ** 2) / interpolated_psd(fs))
# #%%
# m1_msun = 20.0 # In solar masses
# m2_msun = 19.0
# chi1 = 0.5 # Dimensionless spin
# chi2 = -0.5
# tc = 0.0 # Time of coalescence in seconds
# phic = 0.0 # Time of coalescence
# dist_mpc = 440 # Distance to source in Mpc
#
# Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))
# theta_ripple_h0 = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic])
# #%%
# likelihood(theta_ripple_h0)


#%%
class gw_model:
    def __init__(self, id='gw_model'):
        # Record evaluations
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0
        self.DoF = 7

        # Frequency range and PSD
        f_l = 24
        f_u = 512
        self.del_f = 0.01
        self.fs = jnp.arange(f_l, f_u, self.del_f)
        self.PSD = self._interpolatePSD()
        # self.noise_injection = self._n()

        # True signal
        m1_msun = 20.0 # Mass 1 (In solar masses)
        m2_msun = 19.0 # Mass 2 (In solar masses)
        chi1 = 0.5  # Dimensionless spin 1
        chi2 = -0.5 # Dimensionless spin 2
        tc = 0.0  # Time of coalescence in seconds
        phic = 0.0  # Phase of coalescence
        dist_mpc = 440  # Distance to source in Mpc (amplitude)
        Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun])) # Chirp mass, symmetric mass ratio

        ###

        self.theta_true = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic])

        ###

        self.data = self._h(self.theta_true)

        ###

    def _h(self, theta):
        # Generate waveform given parameters
        return IMRPhenomD.gen_IMRPhenomD(self.fs, theta).real

    def _interpolatePSD(self):
        models_folder = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(models_folder, 'ripple', 'noise_resources', 'aLIGO.dat')
        data = np.loadtxt(data_path)
        return jnp.array(interpolate.interp1d(data[:, 0], data[:, 1])(self.fs))

    # def _n(self):
    #     return jnp.array(np.random.normal(0, 1, self.fs.shape[0]))

    # @jax.jit

    def getResidual(self, theta):
        return 2 * jnp.sqrt(self.del_f) * jnp.absolute(self._h(theta) - self.data) / jnp.sqrt(self.PSD)

    def getJacobianResidual(self, theta):
        return jax.jacfwd(self.getResidual)(theta)

    def getMinusLogLikelihood(self, r):
        return 0.5 * jnp.dot(r, r)

    def getGradientMinusLogLikelihood(self, r, J):
        return jnp.dot(J.T, r)

    def getGNHessianMinusLogLikelihood(self, J):
        return jnp.matmul(J.T, J)

    def _newDrawFromPrior(self, nParticles):
        """
        Return samples from a uniform prior.
        Included for convenience.
        Args:
            nParticles (int): Number of samples to draw.

        Returns: (array) nSamples x DoF array of representative samples

        """
        return np.random.uniform(low=-6, high=6, size=(nParticles, self.DoF))

    def _getMinusLogPrior(self, theta):
        return 0

    def _getGradientMinusLogPrior(self, theta):
        return np.zeros(self.DoF)

    def _getHessianMinusLogPrior(self, theta):
        return np.zeros((self.DoF, self.DoF))




####################################################################################################################

    def getMinusLogPosterior(self, theta):
        return self.getMinusLogLikelihood(theta) + self._getMinusLogPrior(theta)

    def getGradientMinusLogPosterior(self, theta):
        return self.getGradientMinusLogLikelihood(theta) + self._getGradientMinusLogPrior(theta)

    def getGNHessianMinusLogPosterior(self, theta):
        return self.getGNHessianMinusLogLikelihood(theta) + self._getHessianMinusLogPrior(theta)

    # Vectorized versions of previous methods
    # Args:
    #   thetas (array): N x DoF array, where N is number of samples
    # Returns: (array) N x 1, N x DoF, N x (DoF x DoF), N x (DoF x DoF) respectively

    def getMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getMinusLogPosterior, 1, thetas)

    def getGradientMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getGradientMinusLogPosterior, 1, thetas)

    def getGNHessianMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getGNHessianMinusLogPosterior, 1, thetas)

    def getHessianMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getHessianMinusLogPosterior, 1, thetas)

def main():
    model = gw_model()

    # m1_msun = 20.0 # In solar masses
    # m2_msun = 19.0
    # chi1 = 0.5 # Dimensionless spin
    # chi2 = -0.5
    # tc = 0.0 # Time of coalescence in seconds
    # phic = 0.0 # Time of coalescence
    # dist_mpc = 440 # Distance to source in Mpc
    # Mc, eta = ms_to_Mc_eta(jnp.array([m1_msun, m2_msun]))
    # theta_ripple_h0 = jnp.array([Mc, eta, chi1, chi2, dist_mpc, tc, phic])

    noise1 = np.random.normal(0, 1, model.theta_true.size)
    noise2 = np.random.normal(0, 1, model.theta_true.size)
    r = model.getResidual(model.theta_true + 0.01 * noise)
    J = model.getJacobianResidual(model.theta_true)

    gmlpt = model.getGradientMinusLogLikelihood(r, J)
    Hmlpt = model.getGNHessianMinusLogLikelihood(J)
    pass


if __name__ is '__main__':
    main()