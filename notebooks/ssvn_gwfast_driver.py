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
model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([]))

#%%##########################
# Setup sampler
#############################
nParticles = 100
h = model.DoF / 10
nIterations = 200
flat_schedule = lambda t: 1
cyclic_schedule = lambda t: sampler1._cyclic_schedule(t, nIterations, p=5, C=5)
sampler1 = samplers(model=model, nIterations=nIterations, nParticles=nParticles, profile=False, kernel_type='Lp')
kernelKwargs = {'h':h, 'p':1}
sampler1.apply(method='reparam_sSVN', eps=1, kernelKwargs=kernelKwargs, schedule=cyclic_schedule)

# %%
X1 = collect_samples(sampler1.history_path)
a = corner.corner(X1, smooth=0.5, labels=list(np.array(model.gwfast_param_order)[model.active_indicies]))



#%%
from scripts.plot_helper_functions import extract_velocity_norms
a = extract_velocity_norms(sampler1.history_path)


# %%
fig, ax = plt.subplots()
# ax.plot(a['vsvn'])
ax.plot(a['vsvgd'])










# %%
# import h5py
# def extract_gmlpt_norm(file, mode='gmlpt_X'):
#     with h5py.File(file, 'r') as hf:
#         iters_performed = hf['metadata']['L'][()]
#         for l in range(iters_performed):
#             gmlpt = hf['%i' % l][mode][()]
#             if l == 0:
#                 norm_history = np.zeros(gmlpt.shape)
#             norm_history[l] = np.linalg.norm(gmlpt)
#         return norm_history
# #%%
# a = extract_gmlpt_norm(sampler1.history_path, 'dphi')
# plt.plot(a)
# %%
