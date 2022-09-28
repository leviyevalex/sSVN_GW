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
from collections import OrderedDict
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
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
from numdifftools import Jacobian, Gradient
from jax import jacobian
from corner import corner

##########################
####### (GW150914) #######
##########################
tGPS = np.array([1.12625946e+09])
injParams = dict()
# injParams['tcoal'] = utils.GPSt_to_LMST(tGPS, lat=0., long=0.) # Coalescence time, in units of fraction of day (GMST is LMST computed at long = 0°) 
injParams['tcoal']   = np.array([0.])                # Units: fraction of days
injParams['Mc']      = np.array([34.3089283])        # Units: M_sun
injParams['eta']     = np.array([0.2485773])         # Units: Unitless
# injParams['dL']      = np.array([0.43929891 * 3])    # Units: Gigaparsecs 
injParams['dL']      = np.array([1.31789673])
injParams['theta']   = np.array([2.78560281])        # Units: Rad
injParams['phi']     = np.array([1.67687425])        # Units: Rad
injParams['iota']    = np.array([2.67548653])        # Units: Rad
injParams['psi']     = np.array([0.78539816])        # Units: Rad
injParams['Phicoal'] = np.array([0.])                # Units: Rad
injParams['chiS']    = np.array([0.27210419])        # Units: Unitless
injParams['chiA']    = np.array([0.33355909])        # Units: Unitless

#%%  Setup gravitational wave network problem

all_detectors = copy.deepcopy(glob.detectors) # Geometry of every available detector

# LV_detectors = {det:all_detectors[det] for det in ['L1', 'H1', 'Virgo']} # Extract only LIGO/Virgo detectors
LV_detectors = {det:all_detectors[det] for det in ['L1']}

print('Using detectors ' + str(list(LV_detectors.keys())))

detector_ASD = dict() # Remark: Providing ASD path to psd_path with flag "is_ASD = True" in
detector_ASD['L1']    = 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
detector_ASD['H1']    = 'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'
detector_ASD['Virgo'] = 'O3-V1_sensitivity_strain_asd.txt'

LV_detectors['L1']['psd_path']    = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['L1']) # Add paths to detector sensitivities
# LV_detectors['H1']['psd_path']    = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['H1'])
# LV_detectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['Virgo'])

waveform = TaylorF2_RestrictedPN() # Choice of waveform
# waveform = IMRPhenomD()

fgrid_dict = {'fmin': 20, 'fmax': 325, 'df': 1./5} # All parameters related to frequency grid.

priorDict = OrderedDict()
priorDict['Mc']      = [30., 40.]          # (1)
priorDict['eta']     = [0.23, 0.25]        # (2)
priorDict['dL']      = [0.25, 2.]          # (3)
priorDict['theta']   = [2.5, 2.9]          # (4)
priorDict['phi']     = [1.2, 2.2]          # (5)
priorDict['iota']    = [1.5, np.pi]        # (6)
priorDict['psi']     = [0., 2.]            # (7)
priorDict['tcoal']   = [0, 0.00000001]     # (8)
priorDict['Phicoal'] = [0., 1.]            # (9)
priorDict['chiS']    = [-0.5, 0.6]         # (10)
priorDict['chiA']    = [-0.1, 1.]          # (11)
# Remark: Flag is chi1chi2=True. Parameter names will be transformed in gwfast to (chi1z, chi2z)
# Keep these constant

# priorDict['Mc']      = injParams['Mc']          # (1) 
# priorDict['eta']     = injParams['eta']         # (2)
# priorDict['dL']      = injParams['dL']          # (3)
priorDict['theta']   = injParams['theta']       # (4)
priorDict['phi']     = injParams['phi']         # (5)
priorDict['iota']    = injParams['iota']        # (6)
priorDict['psi']     = injParams['psi']         # (7)
priorDict['tcoal']   = injParams['tcoal']       # (8)
priorDict['Phicoal'] = injParams['Phicoal']     # (9)
# priorDict['chiS']    = injParams['chiS']        # (10)
# priorDict['chiA']    = injParams['chiA']        # (11)

nParticles = 300 ** 2
model = gwfast_class(LV_detectors, waveform, injParams, priorDict, nParticles=nParticles, **fgrid_dict)
print('Using % i bins' % model.grid_resolution)

#%%
# Diagnostics

injParams_original = copy.deepcopy(injParams)
chi1z, chi2z = (injParams_original.pop('chiS'), injParams_original.pop('chiA'))
injParams_original['chi1z'] = chi1z
injParams_original['chi2z'] = chi2z
net = DetNet(model.detsInNet)
snr = net.SNR(injParams_original)
print('SNR is  ', snr)
H1_response = model.signal_data['H1']
plt.plot(model.fgrid, H1_response)

#%% RUN SAMPLER
sampler1 = samplers(model=model, nIterations=200, nParticles=nParticles, profile=False)
sampler1.apply(method='sSVN', eps=0.1)
# %% PLOT SUMMARY
X1 = collect_samples(sampler1.history_path)
a = corner(X1, smooth=0.5, labels=model.names_active)

#%%
model.getMarginal('chiA', 'chiS')

#%%
# Transformation methods
import numpy as np
a = np.array([1, 1])    # Lower bounds
b = np.array([2, 2])   # Upper bounds
def F_inv(Y, a, b):
    return (a + b * np.exp(Y)) / (1 + np.exp(Y))

def F(X, a, b):
    return np.log((X - a) / (b - X))

def dF_inv(Y, a, b):
    return b * np.exp(Y) / (1 + np.exp(Y)) - np.exp(Y) * (a + b * np.exp(Y)) / (1 + np.exp(Y)) ** 2

def diagHessF_inv(Y, a, b):
    return - 2 * b * np.exp(2 * Y) / (1 + np.exp(Y)) ** 2 
           + b * np.exp(Y) / (1 + np.exp(Y))
           + 2 * np.exp(2 * Y) * (a + b * (a + b * np.exp(Y))) / (1 + np.exp(Y)) ** 2
           - np.exp(Y) * (a + b * np.exp(Y)) / (1 + np.exp(Y)) ** 2


particle = np.array([1.5, 1.5])

F_inv(F(particle, a, b), a, b)








#%%
from itertools import combinations
pairs = list(combinations(model.names_active, 2))
for pair in pairs:
    print(pair)
    model.getMarginal(pair[0], pair[1])
#%%
def convert(m1, m2):
    Mc = (m1 * m2) ** (3/5) / (m1 + m2) ** (1/5)

    eta = m1 * m2 / (m1 + m2) ** 2 
    return (Mc, eta)

convert(60, 10)

#%%

# def _inBounds(X):

#     lower_bound = np.tile(self.lower_bound, self.N).reshape(self.N, self.DoF)
#     upper_bound = np.tile(self.upper_bound, self.N).reshape(self.N, self.DoF)

#     below = X <= lower_bound 
#     above = X >= upper_bound

#     X[below] = lower_bound[below] + self.bound_tol
#     X[above] = upper_bound[above] - self.bound_tol

        # return X



#%%



lower = np.array([0.11, 0.12, 0.13])
upper = np.array([0.13, 0.15, 0.17])


particle = np.array([[0.10, 0.13, 0.14],
                     [0.8, 0.3, 0.5]])




#%%






#%% SANITY CHECK: Compare hard coded gradient with numerical, and JAX derivatives

particle = copy.deepcopy(model.true_params[np.newaxis, ...]) + 0.1
# particle[:,7] -= 0.1
# grad1 = model.getGradientMinusLogPosterior_ensemble(particle)
grad1, Fisher1 = model.getDerivativesMinusLogPosterior_ensemble(particle)
grad2 = Gradient(model.getMinusLogLikelihood_ensemble, method='central', step=0.0001)(particle)
grad3 = jacobian(model.getMinusLogLikelihood_ensemble)(particle)[0,0]
print(grad1.squeeze())
print(grad2[0])
print(grad3)
print(np.allclose(grad1.squeeze(), grad3))

#%%
test = np.array([[1, 2, 3],
                 [1, 2 ,3],
                 [1, 2 ,3]])


test[:, [0,2]]

#%% SANITY CHECK: Compare Gauss-Newton approximation to Fisher information calculated by gwfast

particle_true = copy.deepcopy(model.true_params[np.newaxis, ...])
GN = model.getGNHessianMinusLogPosterior_ensemble(particle_true)[0]
Fisher = net.FisherMatr(injParams_original, 
                        res=float(model.grid_resolution), 
                        # res=None, 
                        # df=fgrid_dict['df'], 
                        df=None, 
                        spacing='lin', 
                        use_prec_ang=False)

difference = (1 - GN / Fisher) * 100



#%%


#%%
# Record extents here
# bounds = {'Mc': [33.75, 34.75], 
#           'eta': [0.245, 0.25], 
#           'dL': [0.4, 0.47], 
#           'theta': [2.7, 2.9], 
#           'phi': [1.5, 1.8], 
#           'iota': [2.5, 4.1], 
#           'psi': [0.7, 0.9], 
#           'tcoal': [0, 0.00000001], 
#           'Phicoal': [0., 0.2], 
#           'chi1z': [0.26, 0.285], 
#           'chi2z': [0.31, 0.35]}

#%%
# UNIT TEST: What do all the 2D marginals look like?

    # fig.show()





#%%
ngrid = 500
x = np.linspace(33.75, 34.75, ngrid)
y = np.linspace(0.245, 0.25, ngrid)
X, Y = np.meshgrid(x, y)
particle_grid = np.zeros((ngrid ** 2, 11))
particle_grid[:, 0:2] = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T
for i in range(2, 11):
    particle_grid[:, i] = np.ones(ngrid ** 2) * model.true_params[i]

#%%
Z = np.exp(-1 * model.getMinusLogLikelihood_ensemble(particle_grid).reshape(ngrid,ngrid))

#%%
# Setup static figure
fig, ax = plt.subplots(figsize = (5, 5))
# plt.axis('off')
cp = ax.contourf(X, Y, Z)
ax.set_xlabel('$M_c$')
ax.set_ylabel('$\eta$')






















#%%
res = (1-GN/Fisher)

#%%
# print(np.allclose(grad1[0], grad2.squeeze()))

#%%





#%%



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

psds = model.strainGrid
# %%
psd_L1 = psds['L1']
psd_H1 = psds['H1']
psd_Virgo = psds['Virgo']
fig, ax = plt.subplots()
ax.plot(model.detsInNet['L1'].strainFreq, psd_L1, label='L1')
ax.plot(model.detsInNet['H1'].strainFreq, psd_H1, label='H1')
# ax.plot(psd_Virgo, label='Virgo')
ax.legend()

#%%
fig1, ax1 = plt.subplots()
for det in model.detsInNet:
    ax1.plot(model.fgrid, np.log(model.signal_data[det]), label=det+' signal')
    ax1.plot(model.fgrid, np.log(model.strainGrid[det] ** (0.5)), label=det+' noise')
ax1.legend()


# %%
L1 = model.detsInNet['L1']
H1 = model.detsInNet['H1']
Virgo = model.detsInNet['Virgo']
#%%
L1_noise = L1.noiseCurve

# %%
plt.loglog(L1_noise)
# %%
plt.loglog(model.signal_data['H1'])
# %%


# %%
test_tGPS = np.array([1187008882.4])

Test_tcoal = utils.GPSt_to_LMST(tGPS, lat=0., long=0.) # GMST is LMST computed at long = 0° 

#%%
#%%
def f(a, b, c):
    return a + b + c

dic = {'a':2, 'b':3}

f(c=1, **dic)
# %%
import numpy as np
M = np.array([[1, 1],
              [1, 1]])

a = np.array([1, 2])


# %%
def f(a):
    return a + 1

f(a, c='lol')