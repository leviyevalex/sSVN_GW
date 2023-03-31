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

#%%
model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([9,10]))
# NOTE: 3, 4 are problematic
# NOTE: Maybe problematic: 5, 6, 7 (particles bunch at corners)
# NOTE: BIMODAL: 8
# model = gwfast_class(eps=0.5, chi=1, mode='IMRPhenomD')

#%%
from jax.config import config
config.update("jax_debug_nans", True)

nParticles = 100
h = model.DoF / 10
kernelKwargs = {'h':h, 'p':1}
sampler1 = samplers(model=model, nIterations=200, nParticles=nParticles, profile=False, kernel_type='Lp')
sampler1.apply(method='reparam_sSVN', eps=1, kernelKwargs=kernelKwargs)
# %%
X1 = collect_samples(sampler1.history_path)
a = corner.corner(X1, smooth=0.5, labels=list(np.array(model.gwfast_param_order)[model.active_indicies]))
# a = corner.corner(X1, smooth=0.5, labels=model.gwfast_param_order[model.active_indicies])

# %%
a = ['a', 'b', 'c', 'd']
