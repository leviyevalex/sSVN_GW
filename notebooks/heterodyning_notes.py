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
ans_1 = []
ans_2 = []
nParticles = 1
X = model._newDrawFromPrior(nParticles)
full_grid_idx = np.arange(model.nbins_dense)
# for gridsize in [100, 200, 300, 500, 800, 1000]:#, 2000, 3000, 5000, 6000, 8000]:
# for gridsize in [200, 300]:
for gridsize in [50, 100]:
# for gridsize in (np.floor(np.linspace(500,10000, 200))).astype('int'):
    subgrid_idx = np.round(np.linspace(0, len(full_grid_idx)-1, num=gridsize)).astype(int)
    df = model.fgrid_dense[subgrid_idx][1:] - model.fgrid_dense[subgrid_idx][:-1]
    PSD = {}
    d = {}
    for det in dets:
        PSD[det] = jnp.interp(model.fgrid_dense[subgrid_idx], model.detsInNet[det].strainFreq, model.detsInNet[det].noiseCurve, left=1., right=1.).squeeze()
        d[det] = model.d_dense[det][subgrid_idx]
    hj0 = model._getJacobianSignal(model.true_params[np.newaxis], model.fgrid_dense[subgrid_idx])
    # h = model.getSignal(X, model.fgrid_dense[subgrid_idx])
    res_1 = model.overlap(hj0, d, PSD, df)
    res_2 = model.overlap_trap(hj0, d, PSD, model.fgrid_dense[subgrid_idx])
    # res = model.square_norm(h, PSD, df)
    output_1 = 0
    output_2 = 0
    for det in dets:
        output_1 += res_1[det].real
        output_2 += res_2[det].real
    ans_1.append(output_1)
    ans_2.append(output_2)

#%%
for d in range(11):
    timeseries = []
    for i in range(len(ans)):
        timeseries.append(ans[i][0, d])
    fig, ax = plt.subplots()
    ax.plot([100, 200, 300, 500, 800, 1000], timeseries, label=d)
    ax.set_ylabel('Overlap')
    ax.set_xlabel('nbins')
    ax.set_title('<h_,j, d> overlap')
    ax.legend()
    fig.show()



#%%
grid = (np.floor(np.linspace(500,10000, 200))).astype('int')
fig, ax = plt.subplots()
ax.plot(grid, ans)
ax.set_ylabel('Overlap')
ax.set_xlabel('nbins')
ax.set_title('Re<h,h> convergence w.r.t nbins')
ax.legend()
fig.show()







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
nParticles = 10
for i in range(100):
    X = model._newDrawFromPrior(nParticles)
    # print('Calculating heterodyne gradient')
    test1 = model.getGradientMinusLogPosterior_ensemble(X)
    # print('Calculating standard gradient')
    test2 = model.standard_gradientMinusLogLikelihood(X) 
    if i == 0:
      percent_change = (test1 - test2)
    else:
      percent_change_new = (test1 - test2)
      percent_change = np.concatenate((percent_change,percent_change_new), axis=0)

for i in range(model.DoF):
    var_name = model.gwfast_param_order[i]
    fig, ax = plt.subplots()
    counts, bins = np.histogram(percent_change[:,i], bins=800)
    ax.stairs(counts, bins, label='eps=%.2f, chi=%.2f, d=%s' % (model.eps, model.chi, var_name))
    ax.set_ylabel('Count')
    ax.set_xlabel('Difference')
    ax.set_title('Distribution of derivative errors over prior support')
    ax.legend()
    fig.savefig('%s_plot.png' % model.gwfast_param_order[i])
    print('Variance of %s component of derivative: %f' % (var_name, np.var(percent_change[:,i])))


#%%################################################
# Check heterodyned and standard gradient errors
###################################################
nParticles = 5
X = model._newDrawFromPrior(nParticles)
print('Calculating heterodyne gradient')
test1 = model.getGradientMinusLogPosterior_ensemble(X)
print('Calculating standard gradient')
test2 = model.standard_gradientMinusLogLikelihood(X, 1) 
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

##################
#%%#########################################################
# Studying convergence properties of various terms w.r.t grid
ans_1 = []
ans_2 = []
dets = ['L1', 'H1', 'Virgo']
nParticles = 1
X = model._newDrawFromPrior(nParticles)
full_grid_idx = np.arange(model.nbins_dense)
# evalgrid = [100, 200, 300, 500, 800, 1000, 2000, 3000, 5000, 6000, 8000]
evalgrid = (np.floor(np.linspace(500,1500, 100))).astype('int')
# evalgrad = [2000, 3000, 5000, 6000, 8000]
for gridsize in evalgrid:
    subgrid_idx = np.round(np.linspace(0, len(full_grid_idx)-1, num=gridsize)).astype(int)
    df = model.fgrid_dense[subgrid_idx][1:] - model.fgrid_dense[subgrid_idx][:-1]
    PSD = {}
    res = {}
    hj0 = model._getJacobianSignal(model.true_params[np.newaxis], model.fgrid_dense[subgrid_idx])
    h = model.getSignal(X, model.fgrid_dense[subgrid_idx])
    for det in dets:
      PSD[det] = np.interp(model.fgrid_dense[subgrid_idx], model.detsInNet[det].strainFreq, model.detsInNet[det].noiseCurve, left=1., right=1.).squeeze()
      d[det] = model.d_dense[det][subgrid_idx]
      res[det] = h[det] - d[det]
    res_1 = model.overlap(hj0, res, PSD, df)
    res_2 = model.overlap_trap(hj0, res, PSD, model.fgrid_dense[subgrid_idx]) # res = model.square_norm(h, PSD, df)
    output_1 = 0
    output_2 = 0
    for det in dets:
        output_1 += res_1[det].real
        output_2 += res_2[det].real
    ans_1.append(output_1)
    ans_2.append(output_2)

fig, ax = plt.subplots(nrows=11,ncols=2, figsize=(8,24))
for d in range(11):
    param = model.gwfast_param_order[d]
    timeseries1 = []
    timeseries2 = []
    for i in range(len(ans_1)):
        timeseries1.append(ans_1[i][0, d])
        timeseries2.append(ans_2[i][0, d])
    ax[d, 0].plot(evalgrid[0:len(timeseries1)], timeseries1, label=param)
    ax[d, 0].set_ylabel('Overlap')
    ax[d, 0].set_xlabel('nbins')
    ax[d, 0].set_title('<h_,j, d> overlap riemann')
    ax[d, 0].legend()
    ax[d, 1].plot(evalgrid[0:len(timeseries1)], timeseries2, label=model.gwfast_param_order[d])
    ax[d, 1].set_ylabel('Overlap')
    ax[d, 1].set_xlabel('nbins')
    ax[d, 1].set_title('<h_,j, d> overlap trap')
    ax[d, 1].legend()

plt.tight_layout()
fig.show()
fig.savefig('hj_d_overlap_convergence_imrphenomd.png')o


#%%

#%%
print(np.argmin(counts))
# %%
import numpy_indexed as npi
npi.group_by(bin_index).mean(model.fgrid_dense)
# %%
det = 'L1'
data = 4 * model.h0_dense[det].conjugate() * model.d_dense[det] / model.PSD_dense[det] * model.df_dense
# res = npi.group_by(bin_index).sum(data)[1]
res = npi.group_by(bin_index).multiply(data)[1]
# %%
work = npi.group_by(bin_index)
# %%
work * work
# %%
work.split_array_as_array(data)
# %%
import awkward as ak

array = ak.Array([
    [1, 2, 3], 
    [6, 7, 8, 9]])
# %%
bin_index = (np.digitize(model.fgrid_dense, model.bin_edges)) - 1 # To index bins from 0 to nbins - 1
bin_index[-1] = model.nbins - 1 # Make sure the right endpoint is inclusive!
counts = np.bincount(bin_index)
#%%
model.bin_edges[1:]
# %%
model.fgrid_dense - np.repeat(model.bin_edges[1:], counts)

#%% Test
t = np.cumsum(data)
cumsum = np.zeros(model.nbins_dense + 3).astype('complex128')
cumsum[1:-1] = np.cumsum(data)
testb = cumsum[model.indicies_kept[1:]] - cumsum[model.indicies_kept[:-1]]
np.allclose(testb[:-1], model.A0[det][:-1])



# %%
t = np.cumsum(data)
cumsum = np.zeros(model.nbins_dense + 2).astype('complex128')
cumsum[1:] = np.cumsum(data)
cumsum[-2] = cumsum[-1]
testb = cumsum[model.indicies_kept[1:]] - cumsum[model.indicies_kept[:-1]]
np.allclose(testb, model.A0[det])

# %%



def getSummary_data_new(self):
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

A0, A1, B0, B1 = getSummary_data_new(model)
# %%
for det in dets:
    print(np.allclose(A0[det], model.A0[det]))
    print(np.allclose(A1[det], model.A1[det]))
    print(np.allclose(B0[det], model.B0[det]))
    print(np.allclose(B1[det], model.B1[det]))
# %%
            # tmp[1:] = np.cumsum(integrand)
            # tmp[-2] = tmp[-1]
            # data[det] = tmp[self.indicies_kept[1:]] - tmp[self.indicies_kept[:-1]]


bin_id = (np.digitize(model.fgrid_dense, model.bin_edges)) - 1 # (i)
bin_id[-1] = model.nbins - 1 # (ii)
counts = np.bincount(bin_id)
# %%
bin_id_new = (np.digitize(model.fgrid_dense, model.bin_edges)) # (i)
bin_id_new[-1] = model.nbins # (ii)
counts_new = np.bincount(bin_id_new)
# %%
