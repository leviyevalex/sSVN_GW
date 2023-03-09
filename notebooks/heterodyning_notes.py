#%%
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
import gwfast.gwfastUtils as utils
import sys
sys.path.append("..")
from models.GWFAST_heterodyne import gwfast_class
config.update("jax_enable_x64", True)

#%%##################################
# Define class
#####################################
# jax.disable_jit()
model = gwfast_class(chi=1, eps=0.1)
dets = model.detsInNet.keys()

#%%################################################
# Check heterodyned and standard likelihood errors
###################################################
fig, ax = plt.subplots()
nParticles = 1000
X = model._newDrawFromPrior(nParticles)
test1 = model.standard_minusLogLikelihood(X)
test2 = model.heterodyne_minusLogLikelihood(X) 
percent_change = (test1 - test2) / test1 * 100
# Remark: Observe that the heterodyne approximates from below!!! 
fig, ax = plt.subplots()
counts, bins = np.histogram(percent_change, bins=30)
ax.stairs(counts, bins, label='eps=%.2f, chi=%.2f' % (model.eps, model.chi))
ax.set_ylabel('Count')
ax.set_xlabel('Percentage error')
ax.set_title('Distribution of log-likelihood errors over prior support')
ax.legend()

#%%##################################################
# (1) Lets see what the original function looks like
# If we want, we can also see if the PSD and original signal have compatible units
det = 'L1'
fig, ax = plt.subplots()
for det in dets:
    ax.plot(model.fgrid_dense, model.h0_dense[det], label=det)
ax.set_ylabel('Strain')
ax.set_xlabel('Frequency (Hz)')
ax.set_title('Original injected signal')
ax.legend()
fig.show()

#%%#########################################################
# Studying convergence properties of various terms w.r.t grid
ans = []
nParticles = 1
X = model._newDrawFromPrior(nParticles)
full_grid_idx = np.arange(model.nbins_dense)
# for gridsize in [200, 300, 500, 800, 1000, 2000, 3000, 5000, 6000, 8000]:
for gridsize in [200, 300]:
    subgrid_idx = np.round(np.linspace(0, len(full_grid_idx)-1, num=gridsize)).astype(int)
    df = model.fgrid_dense[subgrid_idx][1:] - model.fgrid_dense[subgrid_idx][:-1]
    PSD = {}
    d = {}
    for det in dets:
        PSD[det] = jnp.interp(model.fgrid_dense[subgrid_idx], model.detsInNet[det].strainFreq, model.detsInNet[det].noiseCurve, left=1., right=1.).squeeze()
        d[det] = model.d_dense[det][subgrid_idx]
    # hj0 = model._getJacobianSignal(model.true_params, model.fgrid_dense[subgrid_idx])
    h = model.getSignal(X, model.fgrid_dense[subgrid_idx])
    res = model.overlap(h, d, PSD, df)

    output = 0
    for det in dets:
        output += res[det].real
    ans.append(output)










#%%############################################
# (2) Lets see what the r heterodyne looks like
nParticles = 10
X = model._newDrawFromPrior(nParticles)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,4))
heteros = model.r_heterodyne(X, model.fgrid_dense) 
for i in range(nParticles):
    ax[0].plot(model.fgrid_dense, heteros['L1'][i].real)
    # ax[1].scatter(model.bin_edges, heteros['L1'][i][model.indicies_kept])
    ax[1].plot(model.bin_edges, heteros['L1'][i][model.indicies_kept])
ax[0].set_ylabel('Amplitude')
ax[0].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Amplitude')
ax[1].set_xlabel('Frequency (Hz)')
ax[0].set_title('Heterodyne over dense grid')
ax[1].set_title('Heterodyne over sparse grid')
# ax[0].legend()
# ax[1].legend()
fig.tight_layout()
fig.show()

#%%################################################
# Check heterodyned and standard gradient errors
###################################################
nParticles = 5
X = model._newDrawFromPrior(nParticles)
print('Calculating heterodyne gradient')
test1 = model.getGradientMinusLogPosterior_ensemble(X)
print('Calculating standard gradient')
test2 = model.standard_gradientMinusLogLikelihood(X) 
percent_change = (test1 - test2) / test1 * 100
mpc = np.mean(percent_change, axis=0)
for i in range(model.DoF):
    fig, ax = plt.subplots()
    counts, bins = np.histogram(mpc, bins=30)
    ax.stairs(counts, bins, label='eps=%.2f, chi=%.2f, d=%i' % (model.eps, model.chi, i))
    ax.set_ylabel('Count')
    ax.set_xlabel('Percentage error')
    ax.set_title('Distribution of derivative errors over prior support')
    ax.legend()

print(mpc)

###########################################################################
# Tested
###########################################################################
#%%###########################################
# Confirm that the PSD's are correctly defined
##############################################
for det in dets:
    fig, ax = plt.subplots()
    ax.loglog(model.fgrid_standard, model.PSD_standard[det], label=det)
    ax.legend()

#%%###########################################################
# Lets see what the cross sections of the likelihood look like
##############################################################
from itertools import combinations
pairs = list(combinations(model.gwfast_param_order, 2))
for pair in pairs:
    print(pair)
    model.getCrossSection(pair[0], pair[1], model.standard_minusLogLikelihood, 100)




#%%###########################################################
# Do numerical and JAX derivatives agree with one another?
import numdifftools as nd 
grad1 = nd.Gradient(model.heterodyne_minusLogLikelihood, method='central', step=1e-4)
n = 1
test1 = grad1(X[n])[0]
test2 = model.getGradientMinusLogPosterior_ensemble(X[n][np.newaxis])
(test1-test2)/test1 * 100
#%%

