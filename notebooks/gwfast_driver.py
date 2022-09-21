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
injParams['dL']      = np.array([0.43929891])        # Units: Gigaparsecs 
injParams['theta']   = np.array([2.78560281])        # Units: Rad
injParams['phi']     = np.array([1.67687425])        # Units: Rad
injParams['iota']    = np.array([2.67548653])        # Units: Rad
injParams['psi']     = np.array([0.78539816])        # Units: Rad
injParams['Phicoal'] = np.array([0.])                # Units: Rad
injParams['chiS']    = np.array([0.27210419])        # Units: Unitless
injParams['chiA']    = np.array([0.33355909])        # Units: Unitless



# injParams['chi1z']   = np.array([0.27210419])        
# injParams['chi2z']   = np.array([0.33355909])        

#%%  Setup gravitational wave network problem

all_detectors = copy.deepcopy(glob.detectors) # Geometry of every available detector

LV_detectors = {det:all_detectors[det] for det in ['L1', 'H1', 'Virgo']} # Extract only LIGO/Virgo detectors

print('Using detectors ' + str(list(LV_detectors.keys())))

detector_ASD = dict() # Remark: Providing ASD path to psd_path with flag "is_ASD = True" in
detector_ASD['L1']    = 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
detector_ASD['H1']    = 'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'
detector_ASD['Virgo'] = 'O3-V1_sensitivity_strain_asd.txt'

LV_detectors['L1']['psd_path']    = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['L1']) # Add paths to detector sensitivities
LV_detectors['H1']['psd_path']    = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['H1'])
LV_detectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['Virgo'])

waveform = TaylorF2_RestrictedPN() # Choice of waveform
# waveform = IMRPhenomD()

fgrid_dict = {'fmin': 20, 'fmax': 325, 'df': 1./5} # All parameters related to frequency grid.

priorDict = OrderedDict()
priorDict['Mc']      = [33.75, 34.75]         # (1)
priorDict['eta']     = [0.245, 0.25]          # (2)
priorDict['dL']      = [0.4, 0.47]            # (3)
priorDict['theta']   = [2.7, 2.9]             # (4)
priorDict['phi']     = [1.5, 1.8]             # (5)
priorDict['iota']    = [2.5, 4.1]             # (6)
priorDict['psi']     = [0.7, 0.9]             # (7)
priorDict['tcoal']   = [0, 0.00000001]        # (8)
# priorDict['tcoal']   = injParams['tcoal']
priorDict['Phicoal'] = [0., 0.2]              # (9)
priorDict['chiS']    = [0.26, 0.285]          # (10)
priorDict['chiA']    = [0.31, 0.35]           # (11)
# Remark: Flag is chi1chi2=True. Parameter names will be transformed in gwfast to (chi1z, chi2z)

nParticles = 1
model = gwfast_class(LV_detectors, waveform, injParams, priorDict, nParticles=nParticles, **fgrid_dict)
print('Using % i bins' % model.grid_resolution)

# Get SNR

injParams_original = copy.deepcopy(injParams)
chi1z, chi2z = (injParams_original.pop('chiS'), injParams_original.pop('chiA'))
injParams_original['chi1z'] = chi1z
injParams_original['chi2z'] = chi2z
net = DetNet(model.detsInNet)
snr = net.SNR(injParams_original)
print('SNR is %f' % snr)

#%% RUN SAMPLER
sampler1 = samplers(model=model, nIterations=100, nParticles=nParticles, profile=False)
sampler1.apply(method='SVGD', eps=0.005)
# %% PLOT SUMMARY
X1 = collect_samples(sampler1.history_path)
a = corner(X1, smooth=0.5)

#%% SANITY CHECK: Compare hard coded gradient with numerical, and JAX derivatives

particle = copy.deepcopy(model.true_params[np.newaxis, ...]) + 0.1
particle[7] -= 0.1
likelihood = model.getMinusLogLikelihood_ensemble(particle)
grad1 = Gradient(model.getMinusLogLikelihood_ensemble, method='central', step=0.0001)(particle)
grad2 = model.getGradientMinusLogPosterior_ensemble(particle)
grad3 = jacobian(model.getMinusLogLikelihood_ensemble)(particle)[0,0]
print(np.allclose(grad1[0], grad2.squeeze()))
a = grad1[0][7]
b = grad2.squeeze()[7]
# Only t_c comes out wrong
print('t_c numerical = %f, t_c jax = %f' % (a,b))
print(grad1[0])
print(grad2.squeeze())
print(grad3)
print(np.allclose(grad2.squeeze(), grad3))

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
bounds = {'Mc': [33.75, 34.75], 
          'eta': [0.245, 0.25], 
          'dL': [0.4, 0.47], 
          'theta': [2.7, 2.9], 
          'phi': [1.5, 1.8], 
          'iota': [2.5, 4.1], 
          'psi': [0.7, 0.9], 
          'tcoal': [0, 0.00000001], 
          'Phicoal': [0., 0.2], 
          'chi1z': [0.26, 0.285], 
          'chi2z': [0.31, 0.35]}

#%%
# UNIT TEST: What do all the 2D marginals look like?
def getMarginal(a, b):
    # a, b are the parameters for which we want the marginals:
    ngrid = 100
    x = np.linspace(bounds[a][0], bounds[a][1], ngrid)
    y = np.linspace(bounds[b][0], bounds[b][1], ngrid)
    X, Y = np.meshgrid(x, y)
    particle_grid = np.zeros((ngrid ** 2, 11))
    index1 = model.names.index(a)
    index2 = model.names.index(b)
    parameter_mesh = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T
    particle_grid[:, index1] = parameter_mesh[:, 0]
    particle_grid[:, index2] = parameter_mesh[:, 1]
    for i in range(11):
        if i != index1 and i!= index2:
            particle_grid[:, i] = np.ones(ngrid ** 2) * model.true_params[i]
    Z = np.exp(-1 * model.getMinusLogLikelihood_ensemble(particle_grid).reshape(ngrid,ngrid))
    fig, ax = plt.subplots(figsize = (5, 5))
    cp = ax.contourf(X, Y, Z)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title('Analytically calculated marginal')
    filename = a + b + '.png'
    path = os.path.join('marginals', filename)
    fig.savefig(path)
    # fig.show()

getMarginal('Mc', 'chi2z')



#%%
from itertools import combinations
pairs = list(combinations(model.names, 2))
for pair in pairs:
    getMarginal(pair[0], pair[1])



#%%
ngrid = 100
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