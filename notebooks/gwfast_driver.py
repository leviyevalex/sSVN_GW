#%%
# Export PATH
import sys
sys.path.append("..")
# 
# from models.gwfastWrapClass import gwfast_class
from models.GWFAST_REWRITE import gwfast_class
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
from numdifftools import Jacobian, Gradient, Hessdiag
from jax import jacobian, grad
from corner import corner
import jax
import jax.numpy as jnp

#%%
#############################
# (GW150914) like parameters
#############################

tGPS = np.array([1.12625946e+09])
injParams = dict()
injParams['Mc']      = np.array([34.3089283])        # Units: M_sun
injParams['eta']     = np.array([0.2485773])         # Units: Unitless
injParams['dL']      = np.array([2.634])             # Units: Gigaparsecs 
injParams['theta']   = np.array([2.78560281])        # Units: Rad
injParams['phi']     = np.array([1.67687425])        # Units: Rad
injParams['iota']    = np.array([2.67548653])        # Units: Rad
injParams['psi']     = np.array([0.78539816])        # Units: Rad
injParams['tcoal']   = np.array([0.])                # Units: Fraction of day 
# injParams['tcoal'] = np.array(utils.GPSt_to_LMST(tGPS, lat=0., long=0.) * (3600 * 24)) # Coalescence time, in units of fraction of day (GMST is LMST computed at long = 0°) 
injParams['Phicoal'] = np.array([0.])                # Units: Rad
injParams['chi1z']   = np.array([0.27210419])        # Units: Unitless
injParams['chi2z']   = np.array([0.33355909])        # Units: Unitless

#%%
#  Setup gravitational wave network problem

all_detectors = copy.deepcopy(glob.detectors) # Geometry of every available detector

# LV_detectors = {det:all_detectors[det] for det in ['L1', 'H1', 'Virgo']} # Extract only LIGO/Virgo detectors
LV_detectors = {det:all_detectors[det] for det in ['L1', 'H1']}

print('Using detectors ' + str(list(LV_detectors.keys())))

detector_ASD = dict() # Remark: Providing ASD path to psd_path with flag "is_ASD = True" in
detector_ASD['L1']    = 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
detector_ASD['H1']    = 'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'
detector_ASD['Virgo'] = 'O3-V1_sensitivity_strain_asd.txt'

LV_detectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['L1']) # Add paths to detector sensitivities
LV_detectors['H1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['H1'])
# LV_detectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['Virgo'])

# waveform = TaylorF2_RestrictedPN() # Choice of waveform
waveform = IMRPhenomD()


priorDict = {}

priorDict['Mc']      = [29., 39.]          # (1)   # (0)
priorDict['eta']     = [0.22, 0.25]        # (2)   # (1)
priorDict['dL']      = [0.1, 4.]           # (3)   # (2)
priorDict['theta']   = [0., np.pi]         # (4)   # (3)
priorDict['phi']     = [0., 2*np.pi]       # (5)   # (4)
priorDict['iota']    = [0., np.pi]         # (6)   # (5)
priorDict['psi']     = [0., np.pi]         # (7)   # (6)
priorDict['tcoal']   = [injParams['tcoal'][0] - 0.001, injParams['tcoal'][0] + 0.001]           # (8)   # (7)
priorDict['Phicoal'] = [0., 0.1]           # (9)   # (8)
# priorDict['chiS']    = [-1., 1.]           # (10)  # (9)
# priorDict['chiA']    = [-1., 1.]           # (11)  # (10)
priorDict['chi1z']    = [-1., 1.]           # (10)  # (9)
priorDict['chi2z']    = [-1., 1.]           # (11)  # (10)

#%%
nParticles = 200 ** 2
model = gwfast_class(LV_detectors, waveform, injParams, priorDict, nParticles=nParticles)
print('Using % i bins' % model.grid_resolution)

#%%
from itertools import combinations
pairs = list(combinations(model.names_active, 2))
for pair in pairs:
    print(pair)
    model.getCrossSection(pair[0], pair[1])




#%%

model.getCrossSection('dL', 'iota')

#%%
###################################################
# Diagnostics
###################################################

net = DetNet(model.detsInNet)
snr = net.SNR(copy.deepcopy(injParams))
print('SNR is  ', snr)
L1_response = model.strain_data['L1']
power_spectral_density = model.PSD_dict['L1']
fig, axs = plt.subplots(1, 2)
# axs[0].plot(model.fgrid, L1_response.squeeze())
axs[0].plot(model.fgrid, (L1_response.squeeze() ** 2 / power_spectral_density))
axs[1].plot(model.fgrid, (L1_response.squeeze()))

#%% 
###################################################
# UNIT TEST: Do derivatives agree?
###################################################

particles = jnp.array(model._newDrawFromPrior(nParticles))
grad1, Fisher1 = model.getDerivativesMinusLogPosterior_ensemble(particles)
grad2 = jacobian(model.getMinusLogPosterior___)(particles)[0,0]
print(grad1[0])
print(grad2)


#%%
# TEST: Keep things deterministic. Calculation in sec
true_particle_days = jnp.array(model.true_params)[np.newaxis,...] + 0.01
true_particle_sec = true_particle_days.at[:, 7].multiply(model.time_scale)

# true_particle_sec[:, 7] *= model.time_scale
grad1, Fisher1 = model.getDerivativesMinusLogPosterior_ensemble(true_particle_sec)
grad2 = jacobian(model.getMinusLogPosterior___)(true_particle_sec)
print(grad1[0])
print(grad2)



#%%
from numdifftools import Jacobian
grad3 = Jacobian(model.getMinusLogPosterior___)(np.array(true_particle_sec))




#%%
particles = model._newDrawFromPrior(nParticles)
# grad1, Fisher1 = model.getDerivativesMinusLogPosterior_ensemble(particles)


#%%
###################################################
# UNIT TEST: Compare Fisher matricies to each other
###################################################

# gwfast methods internally replace chiS and chiA
injParams_copy = copy.deepcopy(injParams)
# injParams_copy.pop('chiS')
# injParams_copy.pop('chiA')
# injParams_copy['chi1z'] = chi1z
# injParams_copy['chi2z'] = chi2z

# Evaluate at true parameters
true_particle_sec  = jnp.array(model.true_params[np.newaxis])
# true_particle_days = copy.deepcopy(true_particle_sec)
grad1, Fisher1    = model.getDerivativesMinusLogPosterior_ensemble(true_particle_sec)
fisher_mine       = np.array(Fisher1[0])
# fisher_mine[7]   /= 3600 * 24  # Days to seconds transformation (?)
# fisher_mine.T[7] /= 3600 * 24
injParams_copy['tcoal'] /= model.time_scale # Put back into days
fisher_gwfast     = net.FisherMatr(copy.deepcopy(injParams_copy), 
                                   res=float(model.grid_resolution), 
                                   # res=None, 
                                   # df=fgrid_dict['df'], 
                                   df=None, 
                                   spacing='geom', 
                                   use_prec_ang=False)#[model.list_active_indicies][:, model.list_active_indicies]

np.allclose(fisher_gwfast, fisher_mine)
#%%


#%%
error_fisher = (1 - (fisher_mine / fisher_gwfast)) * 100
#%%
fisher_mine = np.array(Fisher1[0])
# fisher_gwfast = net.FisherMatr(injParams_original)


#%%
gmlpt, hmlpt = model.getDerivativesMinusLogPosterior_ensemble(particles)








#%% RUN SAMPLER
sampler1 = samplers(model=model, nIterations=100, nParticles=nParticles, profile=False)
sampler1.apply(method='sSVN', eps=0.1, h=2*model.DoF)




################################################
# PLOT SUMMARY
################################################
X1 = collect_samples(sampler1.history_path)
a = corner(X1, smooth=0.5, labels=model.names_active)

#%%
from scripts.create_animation import animate_driver
from scripts.create_contour import create_contour

#%%
contour_file_path1 = create_contour(sampler1, model.lower_bound, model.upper_bound)

#%%
path = '/mnt/c/sSVN_GW/outdir/1664469465/output_data.h5' 
animation_path1 = animate_driver(contour_file_path1, sampler1)



#%%


#%%

#%%
from numdifftools import Gradient, Hessdiag
import numpy as np
import matplotlib.pyplot as plt

def F_inv(Y, a, b):
    return (a + b * np.exp(Y)) / (1 + np.exp(Y))

def F(X, a, b):
    return np.log((X - a) / (b - X))

def dF_inv(Y, a, b):
    return b * np.exp(Y) / (1 + np.exp(Y)) - np.exp(Y) * (a + b * np.exp(Y)) / (1 + np.exp(Y)) ** 2

def dF(X, a, b):
    return (b - a) / ((X - a) * (b - X))

def diagHessF_inv(Y, a, b):
    return - 2 * b * np.exp(2 * Y) / (1 + np.exp(Y)) ** 2 \
           + b * np.exp(Y) / (1 + np.exp(Y)) \
           + 2 * np.exp(2 * Y) * (a + b * np.exp(Y)) / (1 + np.exp(Y)) ** 3 \
           - np.exp(Y) * (a + b * np.exp(Y)) / (1 + np.exp(Y)) ** 2

#%%
# Test 1 - Plot for several dimensions

# Define hypercube [1, 2] X [3, 4]
a = np.array([1, 3])    # Lower bounds
b = np.array([2, 4])    # Upper bounds

# Draw particles uniformly in hypercube
particle_x =  np.random.uniform(low=a[0], high=b[0], size=100)
particle_y =  np.random.uniform(low=a[1], high=b[1], size=100)
particles = np.zeros((100, 2))
particles[:, 0] = particle_x
particles[:, 1] = particle_y

# Plot original and transformed particles on top of each other
y = F(particles, a, b)
fig, ax = plt.subplots()
ax.scatter(particles[:, 0], particles[:, 1], c='b', label='Original')
ax.scatter(y[:, 0], y[:, 1], c='r', label='Transformed')
ax.legend()

# Visualize relationship between (x, x'), (y, y')
fig, ax = plt.subplots()
ax.scatter(particles[:,0], y[:,0], c='b', label='x')
ax.scatter(particles[:,1], y[:,1], c='r', label='y')
ax.legend()

# Check that batch calculation is stored properly
i = 5
particle = particles[i]
transformed_particle = F(particle, a, b)
print(np.allclose(transformed_particle, y[i]))

#%%
# Test 2 - Test that F_inv is defined properly

# Define hypercube [1, 2] X [3, 4]
a = np.array([1, 3])    # Lower bounds
b = np.array([2, 4])    # Upper bounds

# Draw particles uniformly in hypercube
particle_x =  np.random.uniform(low=a[0], high=b[0], size=100)
particle_y =  np.random.uniform(low=a[1], high=b[1], size=100)
particles = np.zeros((100, 2))
particles[:, 0] = particle_x
particles[:, 1] = particle_y

print(np.allclose(particles, F_inv(F(particles, a, b), a, b)))

#%%
# Test 3 - Compare numerical and analytic derivatives

# Define hypercube [1, 2] X [3, 4]
a = np.array([1, 3])    # Lower bounds
b = np.array([2, 4])    # Upper bounds

# Draw particles uniformly in hypercube
n = 3
particle_x =  np.random.uniform(low=a[0], high=b[0], size=n)
particle_y =  np.random.uniform(low=a[1], high=b[1], size=n)
particles = np.zeros((n, 2))
particles[:, 0] = particle_x
particles[:, 1] = particle_y

test_1a = dF(particles, a, b)
test_2a = dF_inv(particles, a, b)
test_3a = diagHessF_inv(particles, a, b)
for i in range(n):
    # Calculate derivative numerically 
    test_1b = Gradient(F)(particles[i], a, b)[range(2), range(2)]
    test_2b = Gradient(F_inv)(particles[i], a, b)[range(2), range(2)]
    test_3b = Hessdiag(F_inv)(particles[i], a, b)[range(2), range(2)]

    assert np.allclose(test_1a[i], test_1b)
    assert np.allclose(test_2a[i], test_2b)
    assert np.allclose(test_3a[i], test_3b)

#%%
# Test 4 - Numerically confirm f'(x) = 1 / f'(y)

# Define hypercube [1, 2] X [3, 4]
a = np.array([1, 3])    # Lower bounds
b = np.array([2, 4])    # Upper bounds

# Draw particles uniformly in hypercube
n = 3
particle_x =  np.random.uniform(low=a[0], high=b[0], size=n)
particle_y =  np.random.uniform(low=a[1], high=b[1], size=n)
particles = np.zeros((n, 2))
particles[:, 0] = particle_x
particles[:, 1] = particle_y

Y = F(particles, a, b)

test_a = dF(particles, a, b)
test_b = 1 / dF_inv(Y, a, b)

assert np.allclose(test_a, test_b)


#%%

a = np.array([1])    # Lower bounds
b = np.array([2])   # Upper bounds
# Test that the derivative values are correct
particle = np.array([1.5])

test_1 = 
print(np.allclose(test_1, test_2))

test_a = diagHessF_inv(particle, a, b)
test_b = Hessdiag(F_inv)(particle, a, b)
test_c = Hessian(F_inv)(particle, a, b)
print(np.allclose(test_a, test_b))

#%% 
# Test that batch functionality works correctly
particle1 = np.array([1.5])
particle2 = np.array([1.7])

particles = (np.array([1.5, 1.7]))

print(F_inv(particle1, a, b))
print(F_inv(particle2, a, b))
print(F_inv(particles, a, b))

#%% 
# Test that vector functionality works correctly
a = np.array([1, 1])
b = np.array([2, 2])
particle1 = np.array([1.5, 1.7])
print(F_inv(particle1, a, b))

#%%
# Test that vector batch functionality works correctly
a = np.array([1, 1])
b = np.array([2, 2])
particle1 = np.array([[1.5, 1.7],
                      [1.5, 1.8]])

print(diagHessF_inv(particle1, a, b))







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

particle = copy.deepcopy(model.true_params[np.newaxis, ...]) + 0.01
likelihood = model.getMinusLogPosterior_ensemble(particle)
grad1, Fisher1 = model.getDerivativesMinusLogPosterior_ensemble(particle)
grad2 = Gradient(model.getMinusLogPosterior_ensemble, method='central', step=0.0001)(particle)
grad3 = jacobian(model.getMinusLogPosterior_ensemble)(particle)[0,0]
print(grad1.squeeze())
# print(grad2[0])
print(grad3)
print(np.allclose(grad1.squeeze(), grad3))

#%%
grad_, Fisher_ = model.getDerivativesMinusLogPosterior_ensemble(particle)



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


#%%
# Trapezoid rule test

grid = np.geomspace(1, 2, 50)
def f(x):
    return x ** 2
plt.plot(grid, f(grid))
# %%
# %%
def trapz(y, x):
    return 0.5*((x[1:]-x[:-1])*(y[1:]+y[:-1])).sum()

a = trapz(f(grid), grid)
b = np.trapz(f(grid), grid)
# %%

#%%
a  = np.array([[5,3],
          [1, 2]])
# %%
import numpy as np
a = np.array([5, 1])

def func(x):
    return np.append(a, 1)

print(func(a))
print(a)
# %%
from opt_einsum import contract

a1 = np.random.rand(5, 3, 2)
a2 = np.random.rand(5, 3, 2)

b1 = np.random.rand(3, 2)
b2 = np.random.rand(3, 2)

c1 = np.random.rand(2)
c2 = np.random.rand(2)


test_a = contract('dnf, bnf, f -> ndb', a1, a2, c1)


#%%
test_b = contract('...f, ...f, f -> ...', a1, a2, c1)

#%%
test_c = contract('dn...,bn...,... -> n...', a1, a2, c1)

test_d = contract()

# %%
