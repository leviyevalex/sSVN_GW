""" 
A notebook to test sSVN+BD on mixture distributions
"""

#%%
import sys
sys.path.append("..")

from models.JAXHRD import hybrid_rosenbrock
from models.mixture_class import Mixture
from models.multivariate_gaussian import multivariate_gaussian
from src.samplers import samplers
from scripts.plot_helper_functions import collect_samples
import numpy as np
%matplotlib inline
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import corner
from jax import random
import jax.numpy as jnp

#%% Mixture of HRD 
n2  = 3
n1  = 4
DoF = n2 * (n1 - 1) + 1

# Component 1 settings
B1 = np.zeros(DoF)
B1[0] = 30
B1[1:] = 20
mu1 = 1
HRD1 = hybrid_rosenbrock(n2, n1, mu1, B1)

# Component 2 settings
B2 = np.zeros(DoF)
B2[0] = 500
B2[1:] = 400
mu2 = 1.5
HRD2 = hybrid_rosenbrock(n2, n1, mu2, B2)

# Mixture settings
mixture_weights = np.array([0.5, 0.5])
lower_bound = np.ones(DoF) * (0.9)
upper_bound = np.ones(DoF) * (1.2)
model = Mixture([HRD1, HRD2], mixture_weights, lower_bound, upper_bound)

#%% Mixture of Gaussians
# Component 1 settings
mu1 = np.ones(DoF) * 2
Sigma1 = np.ones(DoF)
G1 = multivariate_gaussian(mu=mu1, sigma=Sigma1)

# Component 2 settings
mu2 = np.ones(DoF) * -2
Sigma2 = np.ones(DoF)
G2 = multivariate_gaussian(mu=mu2, sigma=Sigma2)

# Mixture settings
mixture_weights = np.array([0.5, 0.5])
lower_bound = np.ones(DoF) * (0.9)
upper_bound = np.ones(DoF) * (1.2)
model = Mixture([G1, G2], mixture_weights, lower_bound, upper_bound)

#%%
N = 300000
samples = model.newDrawFromPosterior(N)
corner.corner(samples)






#%%
#%% Get i.i.d samples from mixture
np.random.seed(2)
N = 5000000
ground_truth_samples = model.newDrawFromPosterior(N)
truth_table = ((ground_truth_samples > model.lower_bound) & (ground_truth_samples < model.upper_bound))
idx = np.where(np.all(truth_table, axis=1))[0]
print('%i samples obtained from rejection sampling' % idx.shape[0])
bounded_iid_samples = ground_truth_samples[idx]

#%% Geometry investigation

corner.corner(bounded_iid_samples)






#%%##########################
# Birth death version
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
             'p':0.5}

sampler1 = samplers(model=model, nIterations=nIterations, nParticles=nParticles, profile=False, kernel_type='Lp', bd_kwargs=bd_kwargs)
kernelKwargs = {'h':h, 'p':1} 

sampler1.apply(method='reparam_sSVN', eps=0.5, kernelKwargs=kernelKwargs)










# %%
