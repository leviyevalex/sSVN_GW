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
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD_NRTidalv2
from astropy.cosmology import Planck18

# %% Define injected signal parameters (GW170817)
z = np.array([0.00980])
tGPS = np.array([1187008882.4])
# Eleven parameters in total!
Mc = np.array([1.1859])*(1.+z) # Chirp mass
eta = np.array([0.24786618323504223]) # Symmetric mass ratio
dL = Planck18.luminosity_distance(z).value/1000 # Luminosity distance
theta = np.array([np.pi/2. + 0.4080839999999999]) # (shifted) declination
phi = np.array([3.4461599999999994]) # Right ascention
iota = np.array([2.545065595974997]) # Inclination
psi = np.array([0.]) # 
tcoal = utils.GPSt_to_LMST(tGPS, lat=0., long=0.) # Coalescence time (GMST is LMST computed at long = 0°)
Phicoal = np.array([0.]) # Coalescence phase
chi1z = np.array([0.005136138323169717]) # Spin 1
chi2z = np.array([0.003235146993487445]) # Spin 2
# Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, LambdaTilde, deltaLambda
true_params = np.array([Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z])
injParams = {'Mc': Mc, 'eta': eta, 'dL': dL, 'theta': theta, 'phi': phi, 'iota': iota, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal, 'chi1z': chi1z, 'chi2z': chi2z}

#%% Setup network class and precalculate injected signal
alldetectors = copy.deepcopy(glob.detectors)
LVdetectors = {det:alldetectors[det] for det in ['L1', 'H1', 'Virgo']}
print('Using detectors '+str(list(LVdetectors.keys())))
LVdetectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt')
LVdetectors['H1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt')
LVdetectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', 'Hrec_hoft_V1O2Repro2A_16384Hz.txt')
waveform = TaylorF2_RestrictedPN()
fmin = 32
fmax = 512
df = 1./1
model = gwfast_class(LVdetectors, waveform, fmin=fmin, fmax=fmax)
model.get_signal(method='sim', add_noise=False, df=df, **injParams) # Remark: Not needed to calculate the Fisher. All we need is derivative of template.



#%% 
# UNIT TEST: GN approximation equal to Fisher information
nParticles = 1
net = DetNet(model.detsInNet)
fisher_mine = model.getGNHessianMinusLogLikelihood(true_params.T)[0]
fisher_his = net.FisherMatr(injParams, df=df, res=None)[:, :, 0]
np.allclose(fisher_mine, fisher_his)


#%%
net = DetNet(model.detsInNet)

jitter = np.random.uniform(low=0.1, high=0.25, size=(nParticles, model.DoF))

testParticles = true_params.T + jitter



#%%
ngrid = 500

x = np.linspace(0.1, 2, ngrid)
y = np.linspace(0.1, 0.25, ngrid)
X, Y = np.meshgrid(x, y)
Z = np.exp(-1 * model.getMinusLogPosterior_ensemble(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T).reshape(ngrid,ngrid))



model.getMinusLogLikelihood_ensemble







#%%
sampler1 = samplers(model=model, nIterations=100, nParticles=50, profile=False)
sampler1.apply(method='SVGD', eps=0.01)
# %%
from corner import corner
X1 = collect_samples(sampler1.history_path)
a = corner(X1)

#%%
###############################################################
# WINDOWS POST PROCESSING
###############################################################
import numpy as np
import sys
sys.path.append("..")
# from astropy.cosmology import Planck18
# Redefine parameters
z = np.array([0.00980])
tGPS = np.array([1187008882.4])

Mc = np.array([1.1859])*(1.+z) # Chirp mass
eta = np.array([0.24786618323504223]) # Symmetric mass ratio
dL = np.array([0.04374755])
# dL = Planck18.luminosity_distance(z).value/1000 # Luminosity distance
theta = np.array([np.pi/2. + 0.4080839999999999]) # (shifted) declination
phi = np.array([3.4461599999999994]) # Right ascention
iota = np.array([2.545065595974997]) # Inclination
psi = np.array([0.]) # 
tcoal = np.array([0.43432288])
# tcoal = utils.GPSt_to_LMST(tGPS, lat=0., long=0.) # Coalescence time (GMST is LMST computed at long = 0°)
Phicoal = np.array([0.]) # Coalescence phase
chi1z = np.array([0.005136138323169717]) # Spin 1
chi2z = np.array([0.003235146993487445]) # Spin 2
# Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, LambdaTilde, deltaLambda
true_params = np.array([Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z])
from chainconsumer import ChainConsumer
from scripts.plot_helper_functions import collect_samples
path = r'C:\sSVN_GW\outdir\1662472415\output_data.h5'
X1 = collect_samples(path)
# Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, LambdaTilde, deltaLambda

params = [r"$\mathcal{M}_c$", r"$\eta$", r"$d_L$", r"$\theta$", r"$\phi$", r"$\iota$", r"$\psi$", r"$t_c$", r"$\phi_c$", r"$\chi_1$", r"$\chi_2$"]
c = ChainConsumer().add_chain(X1, parameters=params)
true_params = np.array([Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z])
summary = c.analysis.get_summary()
fig = c.plotter.plot_distributions(truth=true_params)
# %%
