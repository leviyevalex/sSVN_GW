#%%
import sys
sys.path.append("..")

from models.JAXHRD import hybrid_rosenbrock
from models.multivariate_gaussian import multivariate_gaussian
from src.samplers import samplers
from scripts.plot_helper_functions import collect_samples
import numpy as np
%matplotlib inline
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import corner
#%%##############################
# Hybrid Rosenbrock
#################################
n2 = 3
n1 = 4
DoF = n2 * (n1 - 1) + 1
B = np.zeros(DoF)
B[0] = 30
B[1:] = 20
mu=1
model = hybrid_rosenbrock(n2, n1, mu, B)
bounded_iid_samples = np.load('rosenbrock_iid_bounded_samples.npy')
###############################
# Code to draw new iid samples
###############################
# ground_truth_samples = model.newDrawFromPosterior(50000000)
# truth_table = ((ground_truth_samples > model.lower_bound) & (ground_truth_samples < model.upper_bound))
# idx = np.where(np.all(truth_table, axis=1))[0]
# np.random.seed(2)
# print('%i samples obtained from rejection sampling' % idx.shape[0])
# bounded_iid_samples = ground_truth_samples[idx]


#%%
nParticles = 50
nIterations = 300
kernelKwargs = {'h':model.DoF / 10, 'p':1.} # Lp
sampler1 = samplers(model=model, nIterations=nIterations, nParticles=nParticles, profile=False, kernel_type='Lp')
sampler1.apply(method='mirrorSVGD', eps=1, kernelKwargs=kernelKwargs)

#%%
# %%capture
X1 = collect_samples(sampler1.history_path)
# fig1 = corner.corner(bounded_iid_samples[0:X1.shape[0]])
fig1 = corner.corner(bounded_iid_samples, hist_kwargs={'density':True})
# fig1 = corner.corner(ground_truth, hist_kwargs={'density':True})


# ground_truth=model.newDrawFromLikelihood(1000000)
# ground_truth_samples = model.newDrawFromLikelihood(1000000)
# fig1 = corner.corner(ground_truth_samples[0:500000], hist_kwargs={'density':True})

corner.corner(X1, color='r', fig=fig1, hist_kwargs={'density':True})
# fig1.savefig('bounded_gaussian_test.png')
# %%
