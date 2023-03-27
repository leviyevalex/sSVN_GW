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

#%%
model = gwfast_class(eps=0.5, chi=1)

#%%
from jax.config import config
config.update("jax_debug_nans", True)

nParticles = 100
h = model.DoF / 10
kernelKwargs = {'h':h, 'p':1}
sampler1 = samplers(model=model, nIterations=100, nParticles=nParticles, profile=False, kernel_type='Lp')
sampler1.apply(method='mirrorSVN', eps=1, kernelKwargs=kernelKwargs)
# %%
