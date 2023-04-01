#%%
import sys
sys.path.append("..")

from models.hybrid_rosenbrock import hybrid_rosenbrock as HRD
from models.JAX_hybrid_rosenbrock import hybrid_rosenbrock as JAX_HRD
from src.samplers import samplers
from scripts.plot_helper_functions import collect_samples
import numpy as np
%matplotlib inline
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import corner

n2 = 3
n1 = 4
model_np = HRD(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20, id='')

model_JAX = JAX_HRD(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20, id='')


#%%
particles = model_np.newDrawFromLikelihood(N=2000)
# %%

gmlpt_jax, hmlpt_jax = model_JAX.getDerivativesMinusLogPosterior_ensemble(particles)

#%%
gmlpt_np, hmlpt_np = model_np.getDerivativesMinusLogPosterior_ensemble(particles)

#%%
print(np.allclose(gmlpt_jax, gmlpt_np))
print(np.allclose(hmlpt_jax, hmlpt_np))

# %%

import jax.numpy as jnp

a = jnp.array([[1, 2], [3, 4]])
b = jnp.array([1, 2])
a @ b
# %%
