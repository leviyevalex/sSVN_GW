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
import jax



#%% Mixture of Gaussians (Increasing diagonal)
DoF = 10
# Component 1 settings
mu1 = np.ones(DoF) * -2
Sigma1 = np.ones(DoF)
G1 = multivariate_gaussian(mu=mu1, sigma=Sigma1)

# Component 2 settings
mu2 = np.ones(DoF) * 0
Sigma2 = np.ones(DoF)
G2 = multivariate_gaussian(mu=mu2, sigma=Sigma2)

# Component 3 settings
mu3 = np.ones(DoF) * 2
Sigma3 = np.ones(DoF)
G3 = multivariate_gaussian(mu=mu3, sigma=Sigma3)

# Mixture settings
mixture_weights = np.array([0.2, 0.3, 0.5])
lower_bound = np.ones(DoF) * (-4)
upper_bound = np.ones(DoF) * (4)
model = Mixture([G1, G2, G3], mixture_weights, lower_bound, upper_bound, DoF=DoF)

#%% Mixture of Gaussians (Separated modes)
DoF = 10
# Component 1 settings
mu1 = np.ones(DoF) * -6
Sigma1 = np.ones(DoF)
G1 = multivariate_gaussian(mu=mu1, sigma=Sigma1)

# Component 2 settings
mu2 = np.ones(DoF) * 6
Sigma2 = np.ones(DoF)
G2 = multivariate_gaussian(mu=mu2, sigma=Sigma2)

# Mixture settings
mixture_weights = np.array([0.2, 0.8])
lower_bound = np.ones(DoF) * (-10)
upper_bound = np.ones(DoF) * (10)
model = Mixture([G1, G2], mixture_weights, lower_bound, upper_bound, DoF=DoF)






#%% Samples from unconstrained mixture 
N = 300000
samples = model.newDrawFromPosterior(N)
# corner.corner(samples)

#%% Samples from constrained mixture
np.random.seed(2)
N = 5000000
ground_truth_samples = model.newDrawFromPosterior(N)
truth_table = ((ground_truth_samples > model.lower_bound) & (ground_truth_samples < model.upper_bound))
idx = np.where(np.all(truth_table, axis=1))[0]
print('%i samples obtained from rejection sampling' % idx.shape[0])
bounded_iid_samples = ground_truth_samples[idx]
# corner.corner(bounded_iid_samples[0:30000])

#%%##########################
# Birth death version
#############################
nParticles = 100
h = model.DoF / 10
nIterations = 1000
stride = 101
# Remarks:
# h=1 works well for separated modes
# stride = nIterations / 3, where 3 = number of birth-step steps

bd_kwargs = {'use': False, 
             'kernel_type': 'Lp',
             'p':2,
             'h': 0.01,
             'start_iter': -1,
             'tau': 0.01,
             'space': 'primal',
             'stride': stride}

sampler1 = samplers(model=model, nIterations=nIterations, nParticles=nParticles, profile=False, kernel_type='Lp', bd_kwargs=bd_kwargs)
kernelKwargs = {'h':h, 'p':1} 

sampler1.apply(method='reparam_sSVGD', eps=0.1, kernelKwargs=kernelKwargs)



# %%
X1 = collect_samples(sampler1.history_path)
fig1 = corner.corner(bounded_iid_samples[0:30000], hist_kwargs={'density':True})
# fig1 = corner.corner(samples[0:30000], hist_kwargs={'density':True})
corner.corner(X1, color='r', fig=fig1, hist_kwargs={'density':True})
# %%

#%% Test derivative code
x = model.newDrawFromPosterior(1)
test1 = jax.jacrev(model.getMinusLogPosterior_ensemble)(x)
test2, _ = model.getDerivativesMinusLogPosterior_ensemble(x)
np.allclose(test1, test2)

# %%
component_model = model.components[1]
x = component_model.newDrawFromPosterior(1)
test1 = jax.jacrev(component_model.getMinusLogPosterior_ensemble)(x)
test2, _ = component_model.getDerivativesMinusLogPosterior_ensemble(x)
np.allclose(test1, test2)

# %%

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
model = Mixture([HRD1, HRD2], mixture_weights, lower_bound, upper_bound, DoF=DoF)


#%%
import numpy as np
d = 2
N = 3
U = np.random.rand(d, d)
X = np.random.rand(N, d ,d)
# %%
res = U @ X
# %%
