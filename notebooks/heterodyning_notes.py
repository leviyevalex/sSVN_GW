""" 
These notes are for debugging the gravitational wave likelihood
"""

#%%
import jax
import jax.numpy as jnp
import numpy as np
from jax.config import config
import matplotlib.pyplot as plt
import sys, os
from pprint import pprint
sys.path.append("..")
from models.GWFAST_heterodyne import gwfast_class
config.update("jax_enable_x64", True)
model = gwfast_class(chi=1, eps=0.5, mode='TaylorF2') # IMRPhenomD | TaylorF2
dets = model.detsInNet.keys()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%############
# The signals must be O(1e-20)
#####################################################################
fig, ax = plt.subplots()
for det in ['L1', 'H1', 'Virgo']:
    ax.plot(model.fgrid_standard, model.h0_standard[det], label=det)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Strain')
ax.set_title('Fiducial signals')
ax.legend()
fig.show()

#%%##################################################################
# The PSD must be O(1e-40)
#####################################################################
fig, ax = plt.subplots()
for det in ['L1', 'H1', 'Virgo']:
    ax.loglog(model.fgrid_standard, model.PSD_standard[det], label=det)
ax.set_xlabel('Frequency (Hz)')
ax.set_title('PSD')
ax.legend()

#%%#######################################################
# Heterodyning significantly reduces oscillatory behavior
##########################################################
det = 'L1'
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,2))
x = model._newDrawFromPrior(1)
h = model.getSignal(x, model.fgrid_standard, det)
ax[0].plot(model.fgrid_standard, h[0].real, label='h')
ax[1].plot(model.fgrid_standard, (h[0] / model.h0_standard[det]).real, label='r')
ax[0].set_xlabel('Frequency (Hz)')
ax[1].set_xlabel('Frequency (Hz)')
ax[0].legend()
ax[1].legend()
fig.show()

#%%###################################################
# Relative error b/w heterodyne, standard likelihoods
######################################################
fig, ax = plt.subplots()
nParticles = 1000
X = model._newDrawFromPrior(nParticles)
test1 = model.standard_minusLogLikelihood(X)
test2 = model.heterodyne_minusLogLikelihood(X) 
percent_change = (test1 - test2) / test1 * 100
counts, bins = np.histogram(percent_change, bins=30)
ax.stairs(counts, bins, label='eps=%.2f, chi=%.2f' % (model.eps, model.chi))
ax.set_ylabel('Count')
ax.set_xlabel('Percentage error')
ax.set_title('Distribution of log-likelihood errors over prior support')
ax.legend()

#%%###########################################################
# Plot cross sections
##############################################################
from itertools import combinations
pairs = list(combinations(model.gwfast_param_order, 2))
for pair in pairs:
    print(pair)
    # model.getCrossSection(pair[0], pair[1], model.standard_minusLogLikelihood, 100)
    # model.getCrossSection(pair[0], pair[1], model.heterodyne_minusLogLikelihood, 300)
    model.getCrossSection(pair[0], pair[1], None, 300)
    # model.getCrossSection(pair[0], pair[1], model.standard_minusLogLikelihood, 300)


#%%########################### 
# Derivative unit tests follow
###############################


#%%###############################################
# Confirm standard derivative is correctly coded
##################################################
x = model._newDrawFromPrior(1)
func = jax.jacrev(model.standard_minusLogLikelihood)
test1 = func(x)
#%%
test2 = model.standard_gradientMinusLogLikelihood(x)
np.allclose(test1, test2)

#%%###############################################
# Confirm heterodyne derivative is properly coded
##################################################
x = model._newDrawFromPrior(1)
func = jax.jacrev(model.heterodyne_minusLogLikelihood)
test1 = func(x)
test2 = model.getGradientMinusLogPosterior_ensemble(x)
print(np.allclose(test1, test2))
test1/test2


#%%############################################################
# Relative error b/w heterodyne, standard likelihood gradients
###############################################################
nParticles = 10
for i in range(20):
    print('Calculating %ith chunk' % i)
    X = model._newDrawFromPrior(nParticles)
    test1 = np.array(model.getGradientMinusLogPosterior_ensemble(X))
    test2 = np.array(model.standard_gradientMinusLogLikelihood(X))
    if i == 0:
      percent_change = np.abs((test1 - test2) / test2) * 100
      store_samples = X
    else:
      percent_change = np.concatenate((percent_change, np.abs((test1 - test2) / test2) * 100), axis=0)
      store_samples = np.concatenate((store_samples, X), axis=0)

fig, ax = plt.subplots(nrows=11, ncols=1, figsize=(4,20))
dict_variances = {}
for i in range(model.DoF):
    var_name = model.gwfast_param_order[i]
    counts, bins = np.histogram(percent_change[:,i], bins=100)
    variance = np.var(percent_change[:,i])
    ax[i].stairs(counts, bins, label='eps=%.2f, chi=%.2f, d=%s' % (model.eps, model.chi, var_name))
    ax[i].set_ylabel('Count')
    ax[i].set_xlabel('Absolute percentage error')
    ax[i].set_title('Distribution of derivative errors. var=%f' % variance)
    ax[i].legend()
    dict_variances[var_name] = variance
fig.tight_layout()
pprint(dict_variances)

#%%
def standard_gradientMinusLogLikelihood(self=model, X): # Checks: XX
    # Remarks:
    # (i) Jacobian is (d, N, f) shaped. sum over final axis gives (d, N), then transpose to give (N, d)
    nParticles = X.shape[0]
    grad_log_like = jnp.zeros((nParticles, self.DoF))
    for det in self.detsInNet.keys():
        template  = self.getSignal(X, self.fgrid_standard, det)
        jacSignal = self._getJacobianSignal(X, self.fgrid_standard, det)
        residual  = template - self.d_standard[det][np.newaxis, ...]
        grad_log_like += self.overlap(jacSignal, residual, self.PSD_standard[det], self.df_standard).real
    return grad_log_like

#%%
y = model.standard_minusLogLikelihood(x)




#%%
#%%




















#%% Extract good and bad samples for investigation
# Remark: We prefer sticking with the id's because of memory constraints
bad_sample_id = {}
good_sample_id = {}
for d in range(11):
    var_name = model.gwfast_param_order[d]
    bad_sample_id[d] =  np.where(percent_change[:,d]>50)[0] 
    good_sample_id[d] = np.where(percent_change[:,d]<5)[0]
    print('Particles leading to large error in derivative component %s' % var_name, bad_sample_id[d])

#%% Define methods that will calculate second heterodyne
# Remark: Compare between standard grid and sparse grid

def jac_r(self, X):
    jac_r = {}
    jacSignal = self._getJacobianSignal(X, self.fgrid_standard)
    for det in self.detsInNet.keys():
        jac_r[det] = jacSignal[det] / self.h0_standard[det]
    return jac_r

def jac_r_sparse(self, X):
    jac_r = {}
    jacSignal = self._getJacobianSignal(X, self.bin_edges)
    for det in self.detsInNet.keys():
        jac_r[det] = jacSignal[det] / self.h0_dense[det][self.indicies_kept]
    return jac_r

bad_sample_jacs = {}
bad_sample_jacs_sparse = {}
for dim in range(11):
    idxs_bad_sample = bad_sample_id[dim]
    for n in idxs_bad_sample:
        if n not in bad_sample_jacs:
            print(n)
            sample = store_samples[n][np.newaxis]
            bad_sample_jacs[n] = jac_r(model, sample)
            bad_sample_jacs_sparse[n] = jac_r_sparse(model, sample)

#%% Investigate bad samples
# var_num = 0
# heterodyne_funcs = jac_r(model,store_samples[bad_sample_id[var_num]])
#%%
fig, ax = plt.subplots(nrows=11, ncols=2, figsize=(8,25))
det = 'H1'
for i in range(model.DoF):
    for n in bad_sample_id[i]:
        var_name = model.gwfast_param_order[i]
        ax[i,0].plot(model.fgrid_standard, bad_sample_jacs[n][det][i,0,:].imag, label='standard')
        ax[i,1].plot(model.bin_edges, bad_sample_jacs_sparse[n][det][i,0,:].imag, label='sparse')
        ax[i,0].set_ylabel('')
        ax[i,0].set_xlabel('Frequency')
        ax[i,0].set_title('r,j where j=%s' % var_name)
fig.tight_layout()


#%%


#################################################################
#%% Compare integrals that go into calculating the gradient 
#################################################################
nParticles = 200
X = model._newDrawFromPrior(nParticles)
chunks = 10
nChunks = int(nParticles / 10)

integrals_heterodyne = {'hj_h' : {'L1': {}, 'H1':{}, 'Virgo':{}},
                        'hj_d' : {'L1': {}, 'H1':{}, 'Virgo':{}}}
integrals_truth = {'hj_h' : {'L1': {}, 'H1':{}, 'Virgo':{}},
                   'hj_d' : {'L1': {}, 'H1':{}, 'Virgo':{}}}

particles = {}
for i in range(nChunks):
    print('Chunk: %i' % i)
    x = X[i * chunks: (i+1)*chunks]
    particles[i] = x
    for det in model.detsInNet.keys():
        jac = model._getJacobianSignal(x, model.fgrid_standard, det)
        h = model.getSignal(x, model.fgrid_standard, det)
        r0, r1 = model.getFirstSplineData(x, det)
        r0j, r1j = model.getSecondSplineData(x, det)

        # integrals_heterodyne['hj_d'][det][i] = np.sum( \
        #                                       (model.A0[det] * r0j.conjugate()) \
        #                                     + (model.A1[det] * r1j.conjugate()), axis=-1).T

        # integrals_heterodyne['hj_h'][det][i] = np.sum( 
        #                                       (model.B0[det] * r0j.conjugate() * r0) \
        #                                     + (model.B1[det] * r0j.conjugate() * r1) \
        #                                     + (model.B1[det] * r1j.conjugate() * r0), axis=-1).T

        # integrals_truth['hj_h'][det][i] = model.overlap(jac, h, model.PSD_standard[det], model.df_standard)
        # integrals_truth['hj_d'][det][i] = model.overlap(jac, model.d_standard[det], model.PSD_standard[det], model.df_standard)

        # integrals_heterodyne['hj_d'][det][i] = np.sum( \
        #                                       (model.C0[det] * r0j.conjugate()) \
        #                                     + (model.C1[det] * r1j.conjugate()), axis=-1).T

        # integrals_heterodyne['hj_h'][det][i] = np.sum( 
        #                                       (model.B0[det] * r0j.conjugate() * (r0-1)) \
        #                                     + (model.B1[det] * r0j.conjugate() * r1) \
        #                                     + (model.B1[det] * r1j.conjugate() * (r0-1)), axis=-1).T

        integrals_heterodyne['hj_h'][det][i] = \
        np.sum((model.B0[det] * r0j.conjugate() * (r0-1)) + (model.B1[det] * (r0j.conjugate() * r1 + r1j.conjugate() * (r0-1))), axis=-1).T





        integrals_truth['hj_h'][det][i] = model.overlap(jac, h - model.h0_standard[det], model.PSD_standard[det], model.df_standard)
        # integrals_truth['hj_d'][det][i] = model.overlap(jac, model.h0_standard[det] - model.d_standard[det], model.PSD_standard[det], model.df_standard)


#%% Extract errors
derivative_component_errors = {'hj_h' : {'L1': {}, 'H1':{}, 'Virgo':{}},
                               'hj_d' : {'L1': {}, 'H1':{}, 'Virgo':{}}}

for det in model.detsInNet.keys():
    for i in range(nChunks):

        # Version 1 - Errors look good
        # a = ((integrals_truth['hj_h'][det][i] - integrals_heterodyne['hj_h'][det][i]) / integrals_truth['hj_h'][det][i]).real * 100
        # b = ((integrals_truth['hj_d'][det][i] - integrals_heterodyne['hj_d'][det][i]) / integrals_truth['hj_d'][det][i]).real * 100
        
        # Version 2 - Errors explode (?)
        a = (integrals_truth['hj_h'][det][i].real - integrals_heterodyne['hj_h'][det][i].real) / integrals_truth['hj_h'][det][i].real * 100
        # b = (integrals_truth['hj_d'][det][i].real - integrals_heterodyne['hj_d'][det][i].real) / integrals_truth['hj_d'][det][i].real * 100

        c = particles[i]

        if i == 0:
            hj_h_errors = a
            # hj_d_errors = b
            samples = c 
        else:
            hj_h_errors = np.concatenate((hj_h_errors, a), axis=0)
            # hj_d_errors = np.concatenate((hj_d_errors, b), axis=0)
            samples = np.concatenate((samples, c), axis=0)

    derivative_component_errors['hj_h'][det] = hj_h_errors
    # derivative_component_errors['hj_d'][det] = hj_d_errors

#%% Plot error distributions
fig, ax = plt.subplots(nrows=11, ncols=1, figsize=(5,25))
for i in range(model.DoF):
    for det in model.detsInNet.keys():
        var_name = model.gwfast_param_order[i]

        counts1, bins1 = np.histogram(derivative_component_errors['hj_h'][det][:, i], bins=100)
        ax[i].stairs(counts1, bins1, label='%s, d=%s' % (det, var_name))

        # counts2, bins2 = np.histogram(derivative_component_errors['hj_d'][det][:, i], bins=50)
        # ax[i,1].stairs(counts2, bins2, label='%s, d=%s' % (det, var_name))

        ax[i].legend()
        # ax[i,1].legend()

    ax[i].set_ylabel('Count')
    ax[i].set_xlabel('Percent error')
    ax[i].set_title('Distribution of <hj,h> error. var=%f' % np.var(derivative_component_errors['hj_h'][det][:,i]))
    # ax[i].set_ylabel('Count')
    # ax[i].set_xlabel('Percent error')
    # ax[i].set_title('Distribution of <hj,d> error.')

fig.tight_layout()

#%%
for i in range(11):
    print()

#%%
bad_samples_idx = {}
for i in range(11):
    bad_samples_idx[i] = np.where(np.abs(derivative_component_errors['hj_h'][det][:,i]) > 100)[0]
pprint(bad_samples_idx)
#%%
# for d in range(11):
n = 83
dims = [9, 10]
p = model.getSignal(X[n][np.newaxis], model.fgrid_standard, 'Virgo') / model.h0_standard[det]
s = model._getJacobianSignal(X[n][np.newaxis], model.fgrid_standard, 'Virgo') / model.h0_standard[det]
#%%
# sother = model._getJacobianSignal(X[3][np.newaxis], model.fgrid_standard, 'Virgo') / model.h0_standard[det]
fig, ax = plt.subplots(nrows=11, figsize=(5,20))
for d in dims:
    # ax[d].plot(model.fgrid_standard, s[d,0].real)
    ax[d].plot(model.fgrid_standard[model.indicies_kept], p[0].real[model.indicies_kept])
    ax[d].plot(model.fgrid_standard[model.indicies_kept], s[d,0].real[model.indicies_kept])

#%%
u, c = np.unique(model.bin_edges, return_counts=True)
dup = u[c > 1]





#%%
s = model.getSignal(point[np.newaxis], model.fgrid_standard, det)
het = (s / model.h0_standard[det])[0]
fig, ax = plt.subplots()
ax.plot(model.fgrid_standard, np.abs(het))
# ax.plot(model.fgrid_standard, het)
# ax.plot(model.bin_edges, het[model.indicies_kept])


#%% ###########################################################
# Confirm that the coefficients are being calculated correctly! 
###############################################################
A0 = jnp.zeros(model.nbins).astype('complex128')
A1 = jnp.zeros(model.nbins).astype('complex128')
for i in range(model.nbins):
    f_minus = model.bin_edges[i]
    f_plus = model.bin_edges[i + 1]
    if i == model.nbins - 1:
        mask = np.logical_and(f_minus <= model.fgrid_dense, model.fgrid_dense <= f_plus)
    else:
        mask = np.logical_and(f_minus <= model.fgrid_dense, model.fgrid_dense < f_plus)
    A0 = A0.at[i].set(4 * jnp.sum(model.h0_dense[det][mask].conjugate() * model.d_dense[det][mask] / model.PSD_dense[det][mask]) * model.df_dense)
    deltaf = model.fgrid_dense[mask] - f_minus
    A1 = A1.at[i].set(4 * jnp.sum(model.h0_dense[det][mask].conjugate() * model.d_dense[det][mask] * deltaf / model.PSD_dense[det][mask]) * model.df_dense)

#%%
fig, ax = plt.subplots()
ax.plot(model.B1[det], label='B1')
ax.plot(model.B0[det], label='B0')
ax.legend()

#%%
fig, ax = plt.subplots(nrows=11, figsize=(5,25))
for n in range(10):
    for i in range(model.DoF):
        ax[i].plot((r0j.conjugate() * (r0-1))[i,n,:])
        # ax[i].plot((r0j.conjugate() * r1)[i,n,:])
        # ax[i].plot((r1j.conjugate() * (r0-1))[i,n,:])
#%%



        # integrals_heterodyne['hj_h'][det][i] = np.sum( 
        #                                       (model.B0[det] * r0j.conjugate() * (r0-1)) \
        #                                     + (model.B1[det] * r0j.conjugate() * r1) \
        #                                     + (model.B1[det] * r1j.conjugate() * (r0-1)), axis=-1).T



#%%
r = np.arange(5)

#%%






















































#%%
# Define functions to plot heterodyne 
def jac_r(self, X, det):
        jacSignal = self._getJacobianSignal(X, self.fgrid_standard, det)
        return jacSignal / self.h0_standard[det]

def jac_r_sparse(self, X, det):
        jacSignal = self._getJacobianSignal(X, self.bin_edges, det)
        return jacSignal / self.h0_dense[det][self.indicies_kept]

def r_standard_f(self, X, det):
        h = self.getSignal(X, self.fgrid_standard, det)
        return h / self.h0_standard[det]

def r_sparse_f(self, X, det):
        h = self.getSignal(X, self.bin_edges, det)
        return h / self.h0_dense[det][self.indicies_kept]

#%% Extract samples responsible 

#%%
fig, ax = plt.subplots(nrows=11, ncols=2, figsize=(10, 25))
det = 'Virgo'
for i in range(11):
    bad_samples = np.where(derivative_component_errors['hj_h'][det][:,i] > 1e7)[0]
    if len(bad_samples) > 3:
        idxs = np.random.choice(bad_samples, size=3)
    else:
        idxs = bad_samples
    for n in idxs:
        print(n)

        jr_standard = jac_r(model, X[n][np.newaxis], det)
        jr_sparse   = jac_r_sparse(model, X[n][np.newaxis], det)
        ax[i,0].plot(model.bin_edges, jr_sparse[i, 0, :].real, label='Sparse grid')
        ax[i,0].plot(model.fgrid_standard, jr_standard[i, 0, :].real, label='Standard grid')
        ax[i,0].set_ylabel('Amplitude')
        ax[i,0].set_xlabel('Frequency')
        ax[i,0].set_title('j-heterodyne on sparse/standard grid')

        r_standard  = r_standard_f(model, X[n][np.newaxis], det)
        r_sparse    = r_sparse_f(model, X[n][np.newaxis], det)
        ax[i,1].plot(model.bin_edges, r_sparse[0, :].real, label='Sparse grid')
        ax[i,1].plot(model.fgrid_standard, r_standard[0, :].real, label='Standard grid')
        ax[i,1].set_ylabel('Amplitude')
        ax[i,1].set_xlabel('Frequency')
        ax[i,1].set_title('heterodyne on sparse/standard grid')

        # ax.legend()

plt.tight_layout()
fig.show()

#%%











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
fig.savefig('hj_d_overlap_convergence_imrphenomd.png')
















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



# def getSummary_data_new(self):
#     """ 
#     Calculate the summary data for heterodyne likelihood
#     Remarks:
#     (i)   Label which frequencies belong to which bin (from 0 to nbins - 1)
#         (ia)   np.digitize labels the first bin with a 1
#         (ib)  We subtract 1 to change the convention in (ia) to begin with 0
#         (ic) The last bin includes the right endpoint in our binning convention
#     (ii)  np.bincount begins tallying from 0. This is why convention (iia) is convenient   
#     (iii) To avoid out of bound error for first bin
#     (iv)  Included to keep (iiia) valid
#     (v)   Indicies have been shifted to the right by previous step
#     """
#     def sumBins(array, bin_indicies):
#         """
#         Given an array over a dense grid, and the indicies of the dense grid which define
#         the subgrid (and by extension the bins), return the sum over elements in each bin.
#         """
#         tmp = np.zeros(len(array) + 1).astype(array.dtype)
#         tmp[1:] = np.cumsum(array) # (iii)
#         tmp[-2] = tmp[-1] # (iv) 
#         return tmp[bin_indicies[1:]] - tmp[bin_indicies[:-1]] # (v) 

#     A0, A1, B0, B1 = {}, {}, {}, {}

#     bin_id = (np.digitize(self.fgrid_dense, self.bin_edges)) - 1 # (ia), (ib)
#     bin_id[-1] = self.nbins - 1 # (ic)
#     elements_per_bin = np.bincount(bin_id) # (ii)

#     deltaf_in_bin = self.fgrid_dense - np.repeat(self.bin_edges[:-1], elements_per_bin)

#     for det in self.detsInNet.keys():
#         A0_integrand = 4 * self.h0_dense[det].conjugate() * self.d_dense[det] / self.PSD_dense[det] * self.df_dense
#         A1_integrand = A0_integrand * deltaf_in_bin
#         B0_integrand = 4 * (self.h0_dense[det].real ** 2 + self.h0_dense[det].imag ** 2) / self.PSD_dense[det] * self.df_dense
#         B1_integrand = B0_integrand * deltaf_in_bin
#         for data, integrand in zip([A0, A1, B0, B1], [A0_integrand, A1_integrand, B0_integrand, B1_integrand]):
#             data[det] = sumBins(integrand, self.indicies_kept)

#     return A0, A1, B0, B1

# A0, A1, B0, B1 = getSummary_data_new(model)
# # %%
# for det in dets:
#     print(np.allclose(A0[det], model.A0[det]))
#     print(np.allclose(A1[det], model.A1[det]))
#     print(np.allclose(B0[det], model.B0[det]))
#     print(np.allclose(B1[det], model.B1[det]))
# # %%
#             # tmp[1:] = np.cumsum(integrand)
#             # tmp[-2] = tmp[-1]
#             # data[det] = tmp[self.indicies_kept[1:]] - tmp[self.indicies_kept[:-1]]


# bin_id = (np.digitize(model.fgrid_dense, model.bin_edges)) - 1 # (i)
# bin_id[-1] = model.nbins - 1 # (ii)
# counts = np.bincount(bin_id)
# # %%
# bin_id_new = (np.digitize(model.fgrid_dense, model.bin_edges)) # (i)
# bin_id_new[-1] = model.nbins # (ii)
# counts_new = np.bincount(bin_id_new)
# # %%




#%%
X = model._newDrawFromPrior(3)
def getGrad_heterodyne(self, X):
    func = jax.jacrev(self.heterodyne_minusLogLikelihood)
    return jax.vmap(func)(X)

testa = getGrad_heterodyne(model, X)
testb = model.getGradientMinusLogPosterior_ensemble(X)

#%%



# kernelKwargs = {'h':h, 'p':1}