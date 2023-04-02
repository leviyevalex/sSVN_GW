# import os
# import time
# import gwfast.signal as signal
# from gwfast.network import DetNet
# from opt_einsum import contract
# from functools import partial
# import gwfast.gwfastUtils as utils


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
# nParticles = 10
# X = model._newDrawFromPrior(nParticles)
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





