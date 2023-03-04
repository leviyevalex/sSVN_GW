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

model = gwfast_class(chi=1, eps=0.1)


#%%##################################################
# (1) Lets see what the original function looks like
fig, ax = plt.subplots()
ax.plot(model.fgrid_dense, model.h0_dense['L1'])
fig.show()

#%%############################################
# (2) Lets see what the r function looks like
nParticles = 10
X = model._newDrawFromPrior(nParticles)
def r(X):
    """ 
    Calculate the ratio of signal model with fiducial signal
    """
    r = {}
    signal = model.getSignal(X, model.fgrid_dense)
    for det in model.detsInNet.keys():
        r[det] = signal[det] / model.h0_dense[det]
    return r

heteros = r(X) 
fig, ax = plt.subplots()
fig1, ax1 = plt.subplots()
for i in range(nParticles):
    ax.plot(model.fgrid_dense, heteros['L1'][i].real)
    ax1.plot(model.fgrid_dense, heteros['L1'][i].imag)

fig.show()
fig1.show()

#%%#############################################################
# (3) Lets confirm that binning scheme faithfully represents r 
fig, ax = plt.subplots()
i=1
ax.plot(model.fgrid_dense, heteros['L1'][i])
ax.scatter(model.bin_edges, heteros['L1'][i][model.indicies_kept])
ax.plot(model.bin_edges, heteros['L1'][i][model.indicies_kept])

fig.show()

#%%###########################################################
# Lets see what the cross sections of the likelihood look like
##############################################################
from itertools import combinations
pairs = list(combinations(model.gwfast_param_order, 2))
for pair in pairs:
    print(pair)
    model.getCrossSection(pair[0], pair[1], model.standard_minusLogLikelihood, 100)

#%%################################################
# Check heterodyned and standard likelihood errors
###################################################
nParticles = 500
X = model._newDrawFromPrior(nParticles)
test1 = model.standard_minusLogLikelihood(X)
test2 = model.heterodyne_minusLogLikelihood(X) 
percent_change = (test1 - test2) / test1 * 100
mpc = np.mean(percent_change)
print(mpc)

#%%###########################################################
# Do numerical and JAX derivatives agree with one another?
import numdifftools as nd 
grad1 = nd.Gradient(model.heterodyne_minusLogLikelihood, method='central', step=1e-4)
n = 1
test1 = grad1(X[n])[0]
test2 = model.getGradientMinusLogPosterior_ensemble(X[n][np.newaxis])
(test1-test2)/test1 * 100
#%%


# %%
amplitude = wf_model.Ampl(fgrid_standard, **injParams)
# %%
plt.scatter(fgrid_standard[:200], amplitude[:200] ** 2)
# %%
