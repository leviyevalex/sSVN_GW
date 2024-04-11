#%%
import sys
sys.path.append("..")

import jax, os, corner
import jax.numpy as jnp
from jax import grad, config
import matplotlib.pyplot as plt
import numpy as np
config.update("jax_enable_x64", True)
config.update("jax_debug_nans", True)
from functools import partial
from models.priors import Mc_eta_uniform_masses_draw

from models.gw150914_v2 import gwfast_LVGW150914
from models.gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD

from src.pampel import ula_sampler_full_jax_jit
from src.helper import rejection_sampling

from src.helper import plot_cross_section, animate_particles
import matplotlib.pyplot as plt
print(jax.devices())

import os 
#%%
# Initialize model
model = gwfast_LVGW150914(wf_model=IMRPhenomD, nbins=100)

neg_potential = lambda x: -1 * jax.jit(model.potential)(x)

#%%

# fig, ax = plt.subplots()

cwd = os.getcwd()

path = os.path.join(cwd, 'birth_death_flow')

# save_path = os.path.join(path, '%i_%i.png' % (1, 2))

# fig.savefig(save_path)





#%%

n_particles = 200

for i in range(11):
    for j in range(i+1,11):

# for i in [0]:
#     for j in [1]:

        subset_params = jnp.array([i, j]) # Specific example.

        X_injection = jnp.tile(model.true_params, n_particles).reshape(n_particles, model.DoF)

        def potential_subset(X_red):
            X_ = X_injection.at[:, subset_params].set(X_red)
            return model.potential(X_)

        def gradient_subset(X_red):
            X_ = X_injection.at[:, subset_params].set(X_red)
            return model.gradient_potential(X_)[:, subset_params]

        # Subset definitions needed for run 
        n_iter = 10000

        eps = 2e-6

        # Birth death
        stride = 100 
        rate = 2e-4 
        sigmas = jnp.ones(2)
        key = jax.random.PRNGKey(42)
        X0 = model._newDrawFromPrior(n_particles)

        bounded_coordinates_ = []
        periodic_coordinates_ = []

        if subset_params[0] in model.bounded_coordinates:
            bounded_coordinates_.append(0)
        else:
            periodic_coordinates_.append(0)

        if subset_params[1] in model.bounded_coordinates:
            bounded_coordinates_.append(1)
        else:
            periodic_coordinates_.append(1)

        bounded_coordinates_ = jnp.array(bounded_coordinates_)
        periodic_coordinates_ = jnp.array(periodic_coordinates_)

        sam = ula_sampler_full_jax_jit(key, 
                                    potential_subset, 
                                    gradient_subset, 
                                    n_iter, 
                                    eps, 
                                    X0[:, subset_params], 
                                    model.lower_bound[subset_params], 
                                    model.upper_bound[subset_params], 
                                    stride, 
                                    rate, 
                                    sigmas, 
                                    bounded_coordinates_, 
                                    periodic_coordinates_)

        fig, ax = plot_cross_section(subset_params[0], subset_params[1], neg_potential, 200, model.true_params, model.lower_bound, model.upper_bound, model.gwfast_param_order)

        save_path = os.path.join(path, '%i_%i.gif' % (i, j))

        animate_particles(sam, 100, fig, ax, save_path=save_path)