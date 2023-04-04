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
""" 
Remarks:
(i)   3, 4 are problematic
(ii)  Maybe problematic: 5, 6, 7 (particles bunch at corners)
(iii) BIMODAL: 8
"""
# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([]))
model = gwfast_class(eps=0.1, chi=1, mode='TaylorF2', freeze_indicies=np.array([]))
# Working: 0, 5, 

#%%
from jax.config import config
config.update("jax_debug_nans", True)


def _hyperbolic_schedule(t, T, c=1.3, p=5):
    """
    Hyperbolic annealing schedule
    Args:
        t (int): Current iteration
        T (int): Total number of iterations
        c (float): Controls where transition begins
        p (float): Exponent determining speed of transition between phases

    Returns: (float)

    """
    return np.tanh((c * t / T) ** p)



#%%
nParticles = 100
# h = model.DoF / 10
h = model.DoF / 10
nIterations = 5

def _cyclic_schedule(t, T, p=2, C=int(np.ceil(nIterations / 100))):
# def _cyclic_schedule(t, T, p=5, C=10): # Igot good results with these settings
    """
    Cyclic annealing schedule
    Args:
        t (int): Current iteration
        T (int): Total number of iterations
        p (float): Exponent determining speed of transition between phases
        C (int): Number of cycles

    Returns:

    """
    tmp = T / C
    return (np.mod(t, tmp) / tmp) ** p

sampler1 = samplers(model=model, nIterations=nIterations, nParticles=nParticles, profile=False, kernel_type='Lp')


flat_schedule = lambda a, b: 1
kernelKwargs = {'h':h, 'p':1}
sampler1.apply(method='reparam_sSVN', eps=1, kernelKwargs=kernelKwargs, schedule=flat_schedule, lamb1=0.1, lamb2=0.1)
# sampler1.apply(method='reparam_sSVN', eps=1, kernelKwargs=kernelKwargs, schedule=_cyclic_schedule)



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
import h5py
def extract_gmlpt_norm(file, mode='gmlpt_X'):
    with h5py.File(file, 'r') as hf:
        iters_performed = hf['metadata']['L'][()]
        for l in range(iters_performed):
            gmlpt = hf['%i' % l][mode][()]
            if l == 0:
                norm_history = np.zeros(gmlpt.shape)
            norm_history[l] = np.linalg.norm(gmlpt)
        return norm_history
#%%
a = extract_gmlpt_norm(sampler1.history_path, 'dphi')
plt.plot(a)
# %%
