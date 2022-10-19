""" 
Implementation of https://arxiv.org/abs/1903.09556 with JAX
"""

#%%
import sys
sys.path.append("..")
from opt_einsum import contract
import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from jax.config import config
config.update("jax_enable_x64", True)
import numpy as np

class hybrid_rosenbrock:
    def __init__(self, n2, n1, mu, b):
        """Hybrid Rosenbrock class

        Args:
            mu (float): mean
            b  (array): d-dimensional array. b[0] = a from the paper
            n2 (int):   Number of blocks
            n1 (int):   Block size
        """
        self.n2 = n2
        self.n1 = n1
        self.mu = mu
        self.b = b
        self.DoF = self.n2 * (self.n1 - 1) + 1

        self.B = self._getDependencyStructure(self.b)

        self.Z = self.getPartitionFunction() # Inverse of normalization constant
        self.priorDict = None
        self.id = 'jax_hybrid_rosenbrock'

        # Record evaluations
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0
    
    def _getDependencyStructure(self, x):
        """Get the matrix representation of the dependency structure denoted in Figure 7 - https://arxiv.org/abs/1903.09556
        Remark: This method will be used to express $x_{j,i}$, and $b_{j,i}$ in tensor form. This simplifies the implementation!

        Args:
            x (array): (DoF,) shaped array

        Returns:
            array: (n2, n1) shaped array
        """

        structure = jnp.zeros((self.n2, self.n1))
        structure = structure.at[:, 0].set(x[0])
        structure = structure.at[:, 1:].set(x[1:].reshape(self.n2, self.n1-1))
        return structure

    def _getResiduals(self, x):
        """Get residuals so that Hybrid Rosenbrock may be expressed in "least squares" form. That is,
        $$-\ln \pi(x) = \frac{1}{2} \sum_{d=1}^{DoF} r(x;d)^2$$,

        Args:
            x (array): (DoF,) shaped array representing the point at which we wish to evaluate the residual

        Returns:
            array: (DoF,) shaped array representing the residuals evaluated at 'x'.
        """
        X = self._getDependencyStructure(x)
        res = jnp.zeros(self.DoF)
        res = res.at[0].set(jnp.sqrt(self.b[0]) * (x[0] - self.mu))
        res = res.at[1:].set((jnp.sqrt(self.B[:,1:]) * (X[:, 1:] - X[:,:-1] ** 2)).flatten())
        return res * jnp.sqrt(2)

    def _getJacobianResiduals(self, x):
        """Get the Jacobian of the residuals.

        Args:
            x (array): (DoF,) shaped array representing the point at which we wish to evaluate the residual

        Returns:
            array: (DoF, DoF) shaped array representing the Jacobian of the residuals. axis=1 index the derivative components.
        """
        return jax.jacobian(self._getResiduals)(x)

    def getMinusLogPosterior(self, x):
        """Get the minus log likelihood, also known as the "potential"

        Args:
            x (array): (DoF,) shaped array representing the point at which we wish to evaluate the potential

        Returns:
            float: Potential evaluated at 'x'.
        """
        res = self._getResiduals(x)
        return jnp.sum(res ** 2) / 2 + jnp.log(self.Z)

    def getGradientMinusLogPosterior(self, x):
        """Get the gradient of the potential

        Args:
            x (array): (DoF,) shaped array representing the point at which we wish to evaluate the gradient of the potential

        Returns:
            array: (DoF,) shaped gradient of potential evaluated at 'x'.
        """
        res = self._getResiduals(x)
        jacRes = self._getJacobianResiduals(x)
        return contract('fi, f -> i', jacRes, res)
    
    def getGNHessianMinusLogPosterior(self, x):
        """Get the Gauss-Newton approximation of the potential
        Remark: Yields a positive definite approximation to the Hessian. See introduction to Chapter 10 Nocedal and Wright  

        Args:
            x (array): (DoF,) shaped array representing the point at which we wish to evaluate the Gauss-Newton approximation of the potential

        Returns:
            array: (DoF, DoF) shaped array representing the Gauss-Newton approximation of the potential evaluated at 'x'.
        """
        jacRes = self._getJacobianResiduals(x)
        return contract('fi, fj -> ij', jacRes, jacRes)

    def _getMinusLogPosterior_direct(self, x):
        X = self._getDependencyStructure(x)
        return self.b[0] * (x[0] - self.mu) ** 2 + jnp.sum(self.B[:,1:] * (X[:,1:] - X[:,:-1] ** 2) ** 2)

    def getPartitionFunction(self):
        """Get the partition function for the hybrid Rosenbrock
        Remark: The partition function is the inverse of the so called "normalization constant"

        Returns:
            float: The partition function.
        """
        return (jnp.pi ** (self.DoF / 2)) / (jnp.prod(jnp.sqrt(self.b)))
    
    def newDrawFromPosterior(self, nSamples):
        """Get i.i.d samples from hybrid Rosenbrock

        Args:
            nSamples (int): The number of samples to draw

        Returns:
            array: (nSamples, DoF) shaped array, where each row corresponds to a sample.
        """
        samples = np.zeros((nSamples, self.DoF))
        index_structure = self._getDependencyStructure(np.arange(self.DoF))
        for d in range(self.DoF):
            standard_deviation = 1 / np.sqrt(2 * self.b[d])
            if d == 0:
                samples[:, d] = np.random.normal(self.mu, standard_deviation, nSamples)
            elif d in index_structure[:, 1]:
                samples[:, d] = samples[:, 0] ** 2 + np.random.normal(0, 1, nSamples) * standard_deviation
            else:
                samples[:, d] = samples[:, d - 1] ** 2 + np.random.normal(0, 1, nSamples) * standard_deviation
        return samples

    @partial(jax.jit, static_argnums=(0,))
    def getMinusLogPosterior_ensemble(self, thetas):
        """Batch evaluation of the potential

        Args:
            thetas (array): (N, DoF) shaped array, each row represents a point at which to evaluate the potential

        Returns:
            array: (N,) shaped array of potential avaluated at each row in 'thetas'
        """
        return jax.vmap(self.getMinusLogPosterior)(thetas)

    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogPosterior_ensemble(self, thetas):
        """Batch evaluation of the gradient of the potential

        Args:
            thetas (array): (N, DoF) shaped array, each row represents a point at which to evaluate the potential

        Returns:
            array: (N, DoF) shaped array, each row represents the corresponding gradient evaluation
        """
        return jax.vmap(self.getGradientMinusLogPosterior)(thetas)

    @partial(jax.jit, static_argnums=(0,))
    def getGNHessianMinusLogPosterior_ensemble(self, thetas):
        """Batch evaluation of the Gauss-Newton approximation to the Hessian of the potential

        Args:
            thetas (array): (N, DoF) shaped array, each row represents a point at which to evaluate the potential

        Returns:
            array: (N, d, d) shaped array, each index represents the corresponding GN-Hessian
        """
        return jax.vmap(self.getGNHessianMinusLogPosterior)(thetas)

    def _newDrawFromPrior(self, nSamples):
        """
        Return samples from a uniform prior. Included for convenience.
        Args:
            nParticles (int): Number of samples to draw.

        Returns: 
            array: (nSamples, DoF) shaped array of samples from uniform prior

        """
        return np.random.uniform(low=-6, high=6, size=(nSamples, self.DoF))

    @partial(jax.jit, static_argnums=(0,))
    def getDerivativesMinusLogPosterior_ensemble(self, thetas):
        gmlpt = self.getGradientMinusLogPosterior_ensemble(thetas)
        Hmlpt = self.getGNHessianMinusLogPosterior_ensemble(thetas)
        return (gmlpt, Hmlpt)


#%%
def f(x, **kwargs):
    return x ** 2

f(2, h=5, t=3)

#%%
# #%%
# Define problem
# import numpy as np
# n2 = 2
# n1 = 3
# mu = 1
# a = 1 / 20
# DoF = n2 * (n1 - 1) + 1
# b = jnp.zeros(DoF)
# b = b.at[0].set(a)
# b = b.at[1:].set(100 / 20)
# model = hybrid_rosenbrock(n2, n1, mu, b)
# model_old = old_hybrid_rosenbrock(n2=n2, n1=n1, mu=mu, a=a, b=np.ones((n2, n1-1)) * 100/20)

# #%%
# # Unit tests: Evaluations of the likelihood, gradient, and GN-Hessian all agree!
# x = np.random.rand(DoF)
# x_ = x[np.newaxis,:]
# likelihood1 = model.getMinusLogLikelihood(x) 
# likelihood2 = model_old.getMinusLogLikelihood(x) + np.log(model.Z) # Old code was not normalized
# print(np.allclose(likelihood1, likelihood2))
# grad1 = model.getGradientMinusLogLikelihood(x)
# grad2 = model_old.getGradientMinusLogLikelihood(x)
# print(grad1)
# print(grad2)
# print(np.allclose(grad1, grad2))
# hess1 = model.getGNHessianMinusLogLikelihood(x)
# hess2 = model_old.getGNHessianMinusLogLikelihood(x)
# print(hess1)
# print(hess2)
# print(np.allclose(hess1, hess2))

# #%%
# import corner
# nSamples = 10000
# samples1 = model_old.newDrawFromLikelihood(nSamples)
# samples2 = model.newDrawFromLikelihood(nSamples)

# # fig1 = corner.corner(samples1)
# fig2 = corner.corner(samples2)


# #%%
# from chainconsumer import ChainConsumer
# c = ChainConsumer()
# c.add_chain(samples1, name='old')
# c.add_chain(samples2, name='new')
# c.plotter.plot()





# #%%
# # %%
# # Incorrect for now!
# grad1 = model.getGradientMinusLogLikelihood(x)

# # Correct
# grad2 = jax.grad(model.getMinusLogLikelihood)(x)

# # %%
# def f(x):
#     return x[0] * x + x * x[1]

# x = jnp.array([1., 2.])
# result = jax.jacobian(f)(x)
# # %%

# %%
