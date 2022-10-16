""" 
Hybrid Rosenbrock code
Remark: Hybrid Rosenbrock is referred to as "likelihood" in the code.
"""

#%%
import sys
sys.path.append("..")


#%%
from opt_einsum import contract
import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from jax.config import config
config.update("jax_enable_x64", True)
from models.JAX_hybrid_rosenbrock import hybrid_rosenbrock as old_hybrid_rosenbrock

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
    
    def _getDependencyStructure(self, x):
        """Get the matrix representation of the dependency structure denoted in Figure 7 - https://arxiv.org/abs/1903.09556

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
        """Get residuals so that Hybrid Rosenbrock may be expressed in "least squares" form.

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

    def getMinusLogLikelihood(self, x):
        """Get the minus log likelihood, also known as the "potential"

        Args:
            x (array): (DoF,) shaped array representing the point at which we wish to evaluate the potential

        Returns:
            float: Potential evaluated at 'x'.
        """
        res = self._getResiduals(x)
        return jnp.sum(res ** 2) / 2 + jnp.log(self.Z)

    def getGradientMinusLogLikelihood(self, x):
        """Get the gradient of the potential

        Args:
            x (array): (DoF,) shaped array representing the point at which we wish to evaluate the gradient of the potential

        Returns:
            array: (DoF,) shaped gradient of potential evaluated at 'x'.
        """
        res = self._getResiduals(x)
        jacRes = self._getJacobianResiduals(x)
        return contract('fi, f -> i', jacRes, res)
    
    def getGNHessianMinusLogLikelihood(self, x):
        """Get the Gauss-Newton approximation of the potential
        Remark: Yields a positive definite approximation to the Hessian. See introduction to Chapter 10 Nocedal and Wright  

        Args:
            x (array): (DoF,) shaped array representing the point at which we wish to evaluate the Gauss-Newton approximation of the potential

        Returns:
            array: (DoF, DoF) shaped array representing the Gauss-Newton approximation of the potential evaluated at 'x'.
        """
        jacRes = self._getJacobianResiduals(x)
        return contract('fi, fj -> ij', jacRes, jacRes)

    def _getMinusLogLikelihood_direct(self, x):
        X = self._getDependencyStructure(x)
        return self.b[0] * (x[0] - self.mu) ** 2 + jnp.sum(self.B[:,1:] * (X[:,1:] - X[:,:-1] ** 2) ** 2)

    def getPartitionFunction(self):
        """Get the partition function for the hybrid Rosenbrock
        Remark: The partition function is the inverse of the so called "normalization constant"

        Returns:
            float: The partition function.
        """
        return (jnp.pi ** (self.DoF / 2)) / (jnp.prod(jnp.sqrt(self.b)))

#%%
import numpy as np
n2 = 2
n1 = 3
mu = 1
a = 1 / 20
DoF = n2 * (n1 - 1) + 1
b = jnp.zeros(DoF)
b = b.at[0].set(a)
b = b.at[1:].set(100 / 20)
model = hybrid_rosenbrock(n2, n1, mu, b)
model_old = old_hybrid_rosenbrock(n2=n2, n1=n1, mu=mu, a=a, b=np.ones((n2, n1-1)) * 100/20)
x = np.random.rand(DoF)
x_ = x[np.newaxis,:]
#%%
likelihood1 = model.getMinusLogLikelihood(x)
likelihood2 = model_old.getMinusLogLikelihood(x)
print(likelihood1)
print(likelihood2)
#%%

grad1 = model.getGradientMinusLogLikelihood(x)
grad2 = model_old.getGradientMinusLogLikelihood(x)
print(grad1)
print(grad2)
#%%
hess1 = model.getGNHessianMinusLogLikelihood(x)
hess2 = model_old.getGNHessianMinusLogLikelihood(x)
print(hess1)
print(hess2)
np.allclose(hess1, hess2)







#%%
# %%
# Incorrect for now!
grad1 = model.getGradientMinusLogLikelihood(x)

# Correct
grad2 = jax.grad(model.getMinusLogLikelihood)(x)

# %%
def f(x):
    return x[0] * x + x * x[1]

x = jnp.array([1., 2.])
result = jax.jacobian(f)(x)
# %%