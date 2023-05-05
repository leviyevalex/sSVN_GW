#%%
import jax.numpy as jnp
import jax
import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
from jax.config import config
from pprint import pprint
sys.path.append("..")
from models.GWFAST_heterodyne import gwfast_class
config.update("jax_enable_x64", True)
from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)
from src.samplers import samplers
from scripts.plot_helper_functions import collect_samples
import corner
from jax.config import config
config.update("jax_debug_nans", True)

#%%################ 
# Create model
###################
# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([0, 1, 3, 4, 5, 6, 7, 8]))
model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))

# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([]))

# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))
# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([0, 1, 3, 4, 5, 6, 7, 8]))
# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([2, 3, 4]))
# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))

#%%##########################
# Setup sampler
#############################
nParticles = 100
h = model.DoF / 10
nIterations = 200

bd_kwargs = {'use': True, 
             'h': 0.05,
             'use_metric': False, 
             'start_iter': -1,
             'end_iter': nIterations+5,
             'eps_bd': 0.01,
             'kernel_type': 'Lp',
             'p':1}

sampler1 = samplers(model=model, nIterations=nIterations, nParticles=nParticles, profile=False, kernel_type='Lp', bd_kwargs=bd_kwargs)
# kernelKwargs = {'h':h, 'p':1}
kernelKwargs = {'h':h, 'p':1} # CHANGED!!!!!!!!!!!!!!!!!!!


sampler1.apply(method='reparam_sSVN', eps=0.1, kernelKwargs=kernelKwargs)
# sampler1.apply(method='langevin', eps=0.01, kernelKwargs=kernelKwargs, schedule=flat_schedule, bd_kernel_kwargs=bd_kernel_kwargs)


# %%
X1 = collect_samples(sampler1.history_path)
a = corner.corner(X1, smooth=0.5, labels=list(np.array(model.gwfast_param_order)[model.active_indicies]), truths=model.true_params[model.active_indicies])

#%%
from scripts.plot_helper_functions import extract_velocity_norms
a = extract_velocity_norms(sampler1.history_path)


# %%
fig, ax = plt.subplots()
# ax.plot(a['vsvn'])
ax.plot(np.log(a['vsvgd']))


#%%
func = jax.jacrev(model.heterodyne_minusLogLikelihood)
for i in range(1):
    x = model._newDrawFromPrior(1)
    test1 = func(x)
    test2 = model.getGradientMinusLogPosterior_ensemble(x)
    print(np.allclose(test1, test2))
    # test1/test2


#%%
import h5py
def extract_gmlpt_norm(file, mode='gmlpt_Y'):
    with h5py.File(file, 'r') as hf:
        iters_performed = hf['metadata']['L'][()]
        for l in range(iters_performed):
            gmlpt = hf['%i' % l][mode][()]
            if l == 0:
                norm_history = np.zeros(iters_performed)
            norm_history[l] = np.linalg.norm(gmlpt)
        return norm_history
#%%
fig, ax = plt.subplots()
a = extract_gmlpt_norm(sampler1.history_path, mode='gmlpt_X')
b = extract_gmlpt_norm(sampler1.history_path, mode='gmlpt_Y')
ax.set_xlabel('Iteration')
ax.set_ylabel('<log|g|>')
ax.set_title('Log Average of gradient norms')
ax.plot(np.log(a), label='Primal')
ax.plot(np.log(b), label='Dual')
ax.legend()

#%%
func = lambda X: np.exp(-model.heterodyne_minusLogLikelihood(X))
# func = lambda X: -1 * model.heterodyne_minusLogLikelihood(X)
# func = lambda X: -1 * model.standard_minusLogLikelihood(X)
# func = lambda X: np.maximum(0, model.get)
# func = lambda X: np.linalg.norm(model.getGradientMinusLogPosterior_ensemble(X), axis=1)
model.getCrossSection('dL', 'chi2z', func, 100)
# model.getCrossSection('theta', 'phi', func, 100)
# %%
