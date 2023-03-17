#%%
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt
from jax.config import config
import sys, os
from pprint import pprint
sys.path.append("..")
from models.GWFAST_heterodyne import gwfast_class

#%%##################################
# Define class
#####################################
# jax.disable_jit()
model = gwfast_class(chi=1, eps=0.1)
dets = model.detsInNet.keys()

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
        # Calculate truth
        jac = model._getJacobianSignal(x, model.fgrid_standard, det)
        h = model.getSignal(x, model.fgrid_standard, det)
        integrals_truth['hj_h'][det][i] = model.overlap(jac, h - model.h0_standard[det], model.PSD_standard[det], model.df_standard)

        # Calculate heterodyne
        r0, r1 = model.getFirstSplineData(x, det)
        r0j, r1j = model.getSecondSplineData(x, det)
        integrals_heterodyne['hj_h'][det][i] = \
        jnp.sum((model.B0[det] * r0j.conjugate() * (r0-1)) + (model.B1[det] * (r0j.conjugate() * r1 + r1j.conjugate() * (r0-1)))
                +(model.B2[det] * r1j.conjugate() * r1), axis=-1).T # Remove third term if this doesnt work!


#%% Extract errors
derivative_component_errors = {'hj_h' : {'L1': {}, 'H1':{}, 'Virgo':{}}}
combine = jnp.zeros((200, 11))
for det in model.detsInNet.keys():
    for i in range(nChunks):
        a = (integrals_truth['hj_h'][det][i].real - integrals_heterodyne['hj_h'][det][i].real) / integrals_truth['hj_h'][det][i].real * 100
        combine = combine.at[i * chunks: (i+1)*chunks].set(a)
derivative_component_errors['hj_h'][det] = combine

#%% Plot error distributions
fig, ax = plt.subplots(nrows=11, ncols=1, figsize=(5,25))
for i in range(model.DoF):
    for det in model.detsInNet.keys():
        var_name = model.gwfast_param_order[i]
        counts1, bins1 = np.histogram(derivative_component_errors['hj_h'][det][:, i], bins=100)
        ax[i].stairs(counts1, bins1, label='%s, d=%s, %i' % (det, var_name, i))
        ax[i].legend()

    ax[i].set_ylabel('Count')
    ax[i].set_xlabel('Percent error')
    ax[i].set_title('Distribution of <hj,h> error. var=%f' % np.var(derivative_component_errors['hj_h'][det][:,i]))

fig.tight_layout()

#%%
bad_samples_idx = {}
for i in range(11):
    bad_samples_idx[model.gwfast_param_order[i]] = np.where(np.abs(derivative_component_errors['hj_h'][det][:,i]) > 50)[0]
pprint(bad_samples_idx)
# %%
for d, param in enumerate(model.gwfast_param_order):
    bad_samples = X[bad_samples_idx[param]]
    num = bad_samples.shape[0]
    print(num)
    if num == 0:
        continue
    fig, ax = plt.subplots(ncols=2, nrows=num)
    
    rj_bad = model._getJacobianSignal(bad_samples, model.fgrid_standard, det) / model.h0_standard[det]
    r_bad = model.getSignal(bad_samples, model.fgrid_standard, det) / model.h0_standard[det]
    for n in range(num):
        # if num == 1:
        #     ax[0].plot(model.fgrid_standard, r_bad[n])
        #     ax[1].plot(model.fgrid_standard, rj_bad[d,n])
        # elif num > 1:
        #     ax[n,0].plot(model.fgrid_standard, r_bad[n])
        #     ax[n,1].plot(model.fgrid_standard, rj_bad[d,n])

        if num == 1:
            ax[0].plot(model.fgrid_standard[model.indicies_kept], r_bad[n][model.indicies_kept].real)
            ax[0].plot(model.fgrid_standard[model.indicies_kept], r_bad[n][model.indicies_kept].imag)
            ax[1].plot(model.fgrid_standard[model.indicies_kept], rj_bad[d,n][model.indicies_kept].real)
            ax[1].plot(model.fgrid_standard[model.indicies_kept], rj_bad[d,n][model.indicies_kept].imag)

            ax[0].set_xlabel('Frequency')
            ax[0].set_title('r heterodyne')
            ax[1].set_xlabel('Frequency')
            ax[1].set_title('rj heterodyne')
    
        elif num > 1:
            ax[n,0].plot(model.fgrid_standard[model.indicies_kept], r_bad[n][model.indicies_kept].real)
            ax[n,0].plot(model.fgrid_standard[model.indicies_kept], r_bad[n][model.indicies_kept].imag)
            ax[n,1].plot(model.fgrid_standard[model.indicies_kept], rj_bad[d,n][model.indicies_kept].real)
            ax[n,1].plot(model.fgrid_standard[model.indicies_kept], rj_bad[d,n][model.indicies_kept].imag)

            ax[n,0].set_xlabel('Frequency')
            ax[n,0].set_title('r heterodyne')
            ax[n,1].set_xlabel('Frequency')
            ax[n,1].set_title('rj heterodyne')


        plt.tight_layout()


#%%
for det in model.detsInNet.keys():
    jacp = model._getJacobianSignal(bad_samples, model.fgrid_standard, det)
    hp = model.getSignal(bad_samples, model.fgrid_standard, det)
    r0p, r1p = model.getFirstSplineData(bad_samples, det)
    r0jp, r1jp = model.getSecondSplineData(bad_samples, det)

    # integrals_heterodyne['hj_h'][det]['bad'] = \
    # jnp.sum((model.B0[det] * r0jp.conjugate() * (r0p-1)) + (model.B1[det] * (r0jp.conjugate() * r1p + r1jp.conjugate() * (r0p-1))), axis=-1).T

    # integrals_heterodyne['hj_h'][det]['bad'] = \
    # jnp.sum((model.B0[det] * r0jp.conjugate() * (r0p-1)) + (model.B1[det] * (r0jp.conjugate() * r1p + r1jp.conjugate() * (r0p-1))) + 
    #         (model.B2[det] * r1jp.conjugate * r1p) , axis=-1).T

    integrals_truth['hj_h'][det]['bad'] = model.overlap(jacp, hp - model.h0_standard[det], model.PSD_standard[det], model.df_standard)
#%%


ees = (integrals_heterodyne['hj_h'][det]['bad'].real - integrals_truth['hj_h'][det]['bad'].real) 


#%%

fig, ax = plt.subplots()
for n in range(bad_samples.shape[0]):
# for n in [3]:
    ax.plot(model.fgrid_standard[model.indicies_kept], jacr[d,n,model.indicies_kept])
#%%
fig, ax = plt.subplots()
ax.plot(model.A0[det])
# ax.plot(model.A0[det])
# ax.plot(model.B0[det])
ax.plot(model.B0[det])













# %%
d = 6
fig, ax = plt.subplots()
# for n in range(bad_samples.shape[0]):
for n in [1, 2, 3]:
    print(n)
    ax.plot(model.fgrid_standard, jacr[d,n,:])
#%%



r = (model.getSignal(bad_samples, model.fgrid_standard, det) / model.h0_standard[det])
figj, axj = plt.subplots(ncols=1, nrows=11, figsize=(5,20))
for d in range(11):
    for n in range(bad_samples.shape[0]):
        axj[d].plot(model.fgrid_standard, np.abs(jacr[d,n,:]))


# %%
# fig, ax = plt.subplots()
# for n in range(bad_samples.shape[0]):
#     ax.plot(model.fgrid_standard, r[n])

# tmp = lambda array: jnp.interp(model.fgrid_standard, model.bin_edges, array) # Requires array to be same size as bin_edges
# vinterp = jax.vmap(tmp, in_axes=[0,1]) 
# particles = {}
# fig, ax = plt.subplots(ncols=11, figsize=(5,20))
# nChunks = 1

quality = jnp.zeros((11, 200))
qual = {'real': jnp.zeros((11, 200)), 'imag': jnp.zeros((11, 200))}
for n in range(200):
    print(n)
    rj = model._getJacobianSignal(X[n][np.newaxis], model.fgrid_standard, 'Virgo') / model.h0_standard[det]
    for d in range(11):
        rj_interp = jnp.interp(model.fgrid_standard, model.bin_edges, rj[d,n,:][model.indicies_kept])

        # n1 = np.sqrt(model.square_norm(rj_interp, model.PSD_standard['Virgo'], model.df_standard).real)
        # n2 = np.sqrt(model.square_norm(rj[d,0], model.PSD_standard['Virgo'], model.df_standard).real)
        # q = model.overlap(rj_interp, rj[d,0], model.PSD_standard['Virgo'], model.df_standard) / (n1 * n2)
        
        a = np.float(jnp.median((rj_interp.real - rj[d,0].real) / rj[d,0].real * 100))
        b = np.float(jnp.median((rj_interp.imag - rj[d,0].imag) / rj[d,0].imag * 100))

        qual['real'] = qual['real'].at[d,n].set(a) 
        qual['imag'] = qual['imag'].at[d,n].set(b)
        # q = max(a,b)
        # quality = quality.at[d,n].set(q) 


#%% Plot quality of splines
fig, ax = plt.subplots(nrows=11, ncols=1, figsize=(5,25))
for i in range(model.DoF):
    var_name = model.gwfast_param_order[i]
    counts1, bins1 = np.histogram(qual['real'][i] , bins=100)
    counts2, bins2 = np.histogram(qual['imag'][i] , bins=100)
    ax[i].stairs(counts1, bins1, label='%s, d=%s' % ('real', var_name))
    ax[i].stairs(counts2, bins2, label='%s, d=%s' % ('imag', var_name))
    ax[i].legend()
    ax[i].set_ylabel('Count')
    ax[i].set_xlabel('Maximum error')
    ax[i].set_title('Distribution of maximum spline errors')
plt.tight_layout()

# %% Quality check on splines that yield large error
# d = 8
for d in range(11):
# for d in [9]:
    idxs = np.where(qual['real'][d] < 20)[0]
    idxs2 = np.where(qual['imag'][d] < 20)[0]
    if len(idxs) > 0 or len(idxs2)>0:
    # idxs = np.where(np.abs(derivative_component_errors['hj_h'][det][:, d]) > 50)
        print(derivative_component_errors['hj_h'][det][idxs,d])
        print(derivative_component_errors['hj_h'][det][idxs2,d])

    # print(np.mean(quality[d, idxs]))




# %%
