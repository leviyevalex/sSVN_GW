#%%
# Export PATH
import sys
sys.path.append("..")
# 
from models.gwfastWrapClass import gwfast_class
from src.samplers import samplers
from scripts.plot_helper_functions import collect_samples
import numpy as np
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import os
import time
import copy
import time
import gwfast.gwfastUtils as utils
import gwfast.gwfastGlobals as glob
import gwfast.signal as signal
from gwfast.network import DetNet
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD
# from astropy.cosmology import Planck18
from numdifftools import Jacobian
import matplotlib.pyplot as plt
##### (GW170817)
# z = np.array([0.00980])
# tGPS = np.array([1187008882.4])
# # Eleven parameters in total!
# Mc = np.array([1.1859])*(1.+z) # Chirp mass
# eta = np.array([0.24786618323504223]) # Symmetric mass ratio
# dL = Planck18.luminosity_distance(z).value/1000 # Luminosity distance
# theta = np.array([np.pi/2. + 0.4080839999999999]) # (shifted) declination
# phi = np.array([3.4461599999999994]) # Right ascention
# iota = np.array([2.545065595974997]) # Inclination
# psi = np.array([0.]) # 
# tcoal = utils.GPSt_to_LMST(tGPS, lat=0., long=0.) # Coalescence time (GMST is LMST computed at long = 0°)
# Phicoal = np.array([0.]) # Coalescence phase
# chi1z = np.array([0.005136138323169717]) # Spin 1
# chi2z = np.array([0.003235146993487445]) # Spin 2
# # Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, LambdaTilde, deltaLambda
# true_params = np.array([Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z])
# injParams = {'Mc': Mc, 'eta': eta, 'dL': dL, 'theta': theta, 'phi': phi, 'iota': iota, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal, 'chi1z': chi1z, 'chi2z': chi2z}

###### (GW150914)
tGPS = np.array([1.12625946e+09])
# Eleven parameters in total!
Mc = np.array([34.3089283]) # (1)
eta = np.array([0.2485773]) # (2)
dL = np.array([0.43929891]) # (3) # TODO: In gigaparsecs? true: proportional to dl^2 (for now use unif)
theta = np.array([2.78560281]) # (4) unif in cos
phi = np.array([1.67687425]) # (5) unif
iota = np.array([2.67548653]) # (6) unif in cos
psi = np.array([0.78539816]) # (7) 
# tcoal = utils.GPSt_to_LMST(tGPS, lat=0., long=0.) # Coalescence time (GMST is LMST computed at long = 0°)
tcoal = np.array([0.]) # (8) Unif super narrow
Phicoal = np.array([0.]) # (9) Unif
chi1z = np.array([0.27210419]) # (10) Unif
chi2z = np.array([0.33355909]) # (11) Unif

injParams = {'Mc': Mc, 'eta': eta, 'dL': dL, 'theta': theta, 'phi': phi, 'iota': iota, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal, 'chi1z': chi1z, 'chi2z': chi2z}


#%% Setup network class and precalculate injected signal
alldetectors = copy.deepcopy(glob.detectors)
LVdetectors = {det:alldetectors[det] for det in ['L1', 'H1', 'Virgo']}
print('Using detectors '+str(list(LVdetectors.keys())))
LVdetectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt')
LVdetectors['H1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt')
LVdetectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', 'Hrec_hoft_V1O2Repro2A_16384Hz.txt')
waveform = TaylorF2_RestrictedPN()
# waveform = IMRPhenomD()
fmin = 20.
fmax = 325.
df = 1./5
# model = gwfast_class(LVdetectors, waveform, injParams, fmin=fmin, fmax=fmax)
model = gwfast_class(LVdetectors, waveform, injParams, fmin=fmin, fmax=fmax)
model.get_signal(method='sim', add_noise=False, df=df) # Remark: Not needed to calculate the Fisher. All we need is derivative of template.
print('Using %i bins' % model.res)


particle = np.array([Mc[0], eta[0], dL[0], theta[0], phi[0], iota[0], psi[0], tcoal[0], Phicoal[0], chi1z[0], chi2z[0]])[np.newaxis,...] + 0.01
#%%
from numdifftools import Jacobian, Gradient

Jacobian(model.getMinusLogLikelihood_ensemble)(particle)

#%%

model.getGradientMinusLogPosterior_ensemble(particle)


#%%
det = 'H1'
# To plot the noise curve
# plt.plot(np.log(model.detsInNet[det].noiseCurve))
# To plot the amplitude
grid = model.fgrid.squeeze()
args = np.argwhere(np.logical_and(grid > 400, grid < 500))

# plt.plot(model.fgrid.squeeze()[args], (model.signal_data['H1'].real.squeeze()[args]))
plt.plot(model.fgrid.squeeze(), (model.signal_data['H1'].real.squeeze()))

#%% 
# UNIT TEST: GN approximation equal to Fisher information
net = DetNet(model.detsInNet)
snr = net.SNR(injParams)
print('SNR is %f' % snr)

#%%
sampler1 = samplers(model=model, nIterations=100, nParticles=100, profile=False)
sampler1.apply(method='sSVGD', eps=0.01)
# %%
from corner import corner
X1 = collect_samples(sampler1.history_path)
a = corner(X1, smooth=1.)

#%%
import deepdish as dd
dd.io.save('test2.h5', X1)

#%%
import deepdish as dd
from chainconsumer import ChainConsumer
from scripts.plot_helper_functions import collect_samples
X1 = dd.io.load('svn_phenomd_n50_niter100.h5')
params = [r"$\mathcal{M}_c$", r"$\eta$", r"$d_L$", r"$\theta$", r"$\phi$", r"$\cos(\iota)$", r"$\psi$", r"$t_c$", r"$\phi_c$", r"$\chi_1$", r"$\chi_2$"]
c = ChainConsumer().add_chain(X1, parameters=params)
true_params = np.array([Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z])
summary = c.analysis.get_summary()
fig = c.plotter.plot_distributions(truth=true_params)

#%%
1 * True
# %%
