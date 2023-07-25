#%%
import os, sys, time
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

import jax.numpy as jnp
import jax
import numpy as np
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
# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))

model = gwfast_class(eps=0.1, chi=1, mode='IMRPhenomD', freeze_indicies=np.array([]))

# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))
# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([0, 1, 3, 4, 5, 6, 7, 8]))
# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([2, 3, 4]))
# model = gwfast_class(eps=0.5, chi=1, mode='TaylorF2', freeze_indicies=np.array([2, 3, 4, 5, 6, 7, 8, 9, 10]))

#%%##########################
# Birth death version
#############################
nParticles = 100
h = model.DoF / 10
nIterations = 10
stride = 200
# Remarks:
# h=1 works well for separated modes
# stride = nIterations / 3, where 3 = number of birth-step steps

bd_kwargs = {'use': True, 
             'kernel_type': 'Lp',
             'p':2,
             'h': 1,
             'start_iter': -1,
             'tau': 0.01,
             'space': 'primal',
             'stride': stride}

sampler1 = samplers(model=model, nIterations=nIterations, nParticles=nParticles, profile=False, kernel_type='Lp', bd_kwargs=bd_kwargs)
kernelKwargs = {'h':h, 'p':2, 'M': jnp.eye(2)} 

sampler1.apply(method='MALA', eps=0.00001, kernelKwargs=kernelKwargs)

# %%
X1 = collect_samples(sampler1.history_path)
a = corner.corner(X1, smooth=0.5, labels=list(np.array(model.gwfast_param_order)[model.active_indicies]), truths=model.true_params[model.active_indicies])

#%% Thin MCMC output ranked by likelihood
importance = model.getMinusLogPosterior_ensemble(X1)

#%%
indicies_sorted = jnp.argsort(importance)
X2 = X1[indicies_sorted[:1000]]
a = corner.corner(X2, smooth=1, labels=list(np.array(model.gwfast_param_order)[model.active_indicies]), truths=model.true_params[model.active_indicies])


#%%
fig, ax = plt.subplots()

counts, bins = np.histogram(importance)
plt.stairs(counts, bins)

#%%

corner.corner(X1[np.argwhere(importance < 1000).squeeze()])










#%%
from scripts.plot_helper_functions import extract_velocity_norms
a = extract_velocity_norms(sampler1.history_path)


# %%
fig, ax = plt.subplots()
# ax.plot(a['vsvn'])
# ax.plot(np.log(a['vsvgd']))
ax.plot((a['vsvgd']))


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
# func = lambda X: np.exp(-model.getMinusLogPosterior_ensemble(X))
func = lambda X: (-model.getMinusLogPosterior_ensemble(X))

# func = lambda X: -1 * model.heterodyne_minusLogLikelihood(X)
# func = lambda X: np.exp(-1 * model.standard_minusLogLikelihood(X))
# func = lambda X: np.maximum(0, model.get)
# func = lambda X: np.linalg.norm(model.getGradientMinusLogPosterior_ensemble(X), axis=1)
model.getCrossSection('Mc', 'eta', func, 200)
# model.getCrossSection('theta', 'phi', func, 100)
# %%
import numpy as np

# Input dimensions
N = 3
I = 4
f = 5

# Random input matrices
J = np.random.randint(1, 10, size=(I, N, f))
P = np.random.randint(1, 10, size=f)

# Perform the einsum operation
result_einsum = np.einsum('iNf, jNf, f->Nij', J, J, P)

result_broadcasting = np.sum(J[:,:,np.newaxis,:]*J.transpose(1,0,2) * P, axis=-1).transpose(1,0,2)
# result_broadcasting = np.swapaxes(result_broadcasting, 0, 1)

# Compare results
assert np.allclose(result_einsum, result_broadcasting)

print("Results are consistent.")

# %%
