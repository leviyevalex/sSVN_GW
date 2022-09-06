#%%
import numpy as np
from numdifftools import Hessian, Jacobian, Gradient
from scripts.plot_helper_functions import collect_samples
import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
from chainconsumer import ChainConsumer
import matplotlib.pyplot as plt
from scripts.plot_helper_functions import set_axes_equal

from models.vM_sphere_v2 import vM_sphere
from src.sphere_methods import getEmbedding
from src.samplers import samplers

#%% Setup von Mises
kappa = 10
thetaTrue = np.array([0, np.pi/2])
model = vM_sphere(kappa=kappa, thetaTrue=thetaTrue)

#%% Test forward model
theta = thetaTrue + np.array([34621, 0.45246])
test_1 = model._getJacobianForwardModel(thetaTrue)
test_2 = Jacobian(model._getForwardModel)(thetaTrue)
assert np.allclose(test_1, test_2)

#%% Test residuals
test_3 = Jacobian(model._getResidual)(thetaTrue)
test_4 = model._getJacobianResidual(thetaTrue)
assert np.allclose(test_3, test_4)

#%% Test derivatives
test_5 = Jacobian(model.getMinusLogLikelihood)(theta)
test_6 = model.getGradientMinusLogLikelihood(theta)
assert np.allclose(test_5, test_6)

#%% Setup visualization
ngrid = 500
x = np.linspace(0, 2 * np.pi, ngrid)
y = np.linspace(0, np.pi, ngrid)
X, Y = np.meshgrid(x, y)
Z = np.exp(-1 * model.getMinusLogPosterior_ensemble(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T).reshape(ngrid,ngrid))

#%% Plot
fig1, ax1 = plt.subplots(figsize = (5, 5))
c = ax1.contourf(X, Y, Z, 7)#, colors='black', alpha=0.2)
fig1.colorbar(c)
plt.xlabel(r'$\theta$')
plt.ylabel(r'$\phi$')
plt.title('von Mises-Fisher 2D')
fig1.show()

#%% Wireframe sphere
fig, ax = plt.subplots(1, 1, subplot_kw={'projection':'3d'})

phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))
ax.plot_wireframe(x, y, z, color='k', alpha=0.1, rstride=1, cstride=1)

# von Mises-Fisher samples
ground_truth = model.sample_vMF(mu=model.mu, kappa=kappa, num_samples=1000)
ax.scatter(ground_truth[:,0], ground_truth[:,1], ground_truth[:,2], s=10, c='k', zorder=10)

# Visualize
ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
set_axes_equal(ax) # IMPORTANT - this is also required
fig.show()

#%% Visualize ground truth
params = [r"$\theta$", r'$\phi$']
c = ChainConsumer().add_chain(ground_truth, parameters=params)
summary = c.analysis.get_summary()
fig2 = c.plotter.plot_distributions(truth=thetaTrue)
plt.tight_layout()
fig2.show()
fig2.savefig('summary.png',bbox_inches = "tight")



#%% Run dynamics
sampler1 = samplers(model=model, nIterations=100, nParticles=100, profile=False)
sampler1.apply(method='SVGD', eps=0.1)

#%%% Plot summary
X1 = collect_samples(sampler1.history_path)
params = [r"$\theta$", r'$\phi$']
c = ChainConsumer().add_chain(X1, parameters=params)
summary = c.analysis.get_summary()
fig2 = c.plotter.plot_distributions(truth=thetaTrue)
plt.tight_layout()
fig2.show()
fig2.savefig('summary.png',bbox_inches = "tight")

#%% Visualize samples
fig2, ax2 = plt.subplots(1, 1, subplot_kw={'projection':'3d'})

phi = np.linspace(0, np.pi, 20)
theta = np.linspace(0, 2 * np.pi, 40)
x = np.outer(np.sin(theta), np.cos(phi))
y = np.outer(np.sin(theta), np.sin(phi))
z = np.outer(np.cos(theta), np.ones_like(phi))
ax2.plot_wireframe(x, y, z, color='k', alpha=0.1, rstride=1, cstride=1)

embedding_ensemble = lambda thetas: np.apply_along_axis(getEmbedding, 1, thetas)
embedded_samples = embedding_ensemble(X1)
ax2.scatter(embedded_samples[:,0], embedded_samples[:,1], embedded_samples[:,2])
ax2.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
set_axes_equal(ax2) # IMPORTANT - this is also required
fig2.show()