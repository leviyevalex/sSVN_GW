# import numpy as np
import itertools
import logging.config
from scripts.plot_helper_functions import set_size
import os
from opt_einsum import contract
import jax.numpy as jnp
import numpy as np
from functools import partial
import jax
from jax.config import config
config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)
root = os.path.dirname(os.path.abspath(__file__))
class hybrid_rosenbrock:
    def __init__(self, n2, n1, mu, a, b, id=''):
        """
        Hybrid-Rosenbrock class
        Args:
            n2 (int): Height of graph
            n1 (int): Width of graph
            mu (float): Mean
            a (float): Controls the extend of the ridge of the marginals formed
            b (array): n2 x (n1 -1) array controlling shallowness of ridges.
            Entries of b follow block structure and correspond to the x_2, ... x_D (excluding first coordinate)
            id (str): Give the provided settings a name.
        """
        # String identifying a new set of parameters
        self.id = id

        # HRD parameters and setup
        self.n1 = n1
        self.n2 = n2
        self.mu = mu
        self.a = a
        self.b = b
        self.DoF = self.n2 * (self.n1 - 1) + 1

        # Record evaluations
        self.nLikelihoodEvaluations = 0
        self.nGradLikelihoodEvaluations = 0
        self.nHessLikelihoodEvaluations = 0

        # Precalculate theta independent objects
        self.jacGraph = self._getJacobianGraph()
        # self.hessRes = self._getHessianResidual()

        self.priorDict = None
#######################################################################################################################

    def _getGraph(self, theta):
        """
        Produces an array representation of graph denoted in Figure 2 - https://doi.org/10.1111/sjos.12532
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns: (array) n2 x n1 sized array

        """
        graph = jnp.zeros((self.n2, self.n1))
        graph = graph.at[:, 0].set(theta[0])
        graph = graph.at[:, 1:].set(theta[1:].reshape(self.n2, self.n1-1))
        return graph

    def _getResidual(self, theta):
        """
        Residual vector of minus log density. See introduction to Chapter 10 Nocedal and Wright
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns: (array) DoF sized array

        """
        graph = self._getGraph(theta)
        residual = jnp.zeros(self.DoF)
        residual = residual.at[0].set(jnp.sqrt(self.a) * (theta[0] - self.mu))
        residual = residual.at[1:].set((jnp.sqrt(self.b) * (graph[:, 1:] - graph[:, :-1] ** 2)).flatten())
        return jnp.sqrt(2) * residual

    def _index(self, j, i):
        """
        Given position in graph, return corresponding component of input vector
        Args:
            j (int): row
            i (int): col

        Returns: (int)  Corresponding coordinate of theta in graph location (j, i)

        """
        if i == 0:
            return int(0)
        elif i > 0:
            return int(j * (self.n1 - 1) + i)

    def _getJacobianGraph(self):
        """
        Get the derivative of the graph with respect to input.
        Returns: (array) n2 x n1 x DoF
        Note: For given a, b, n2, n1, this remains fixed. Calculate once and forget.

        """
        jacGraph = np.zeros((self.n2, self.n1, self.DoF))
        for j, i in itertools.product(range(self.n2), range(self.n1)):
            jacGraph[j, i, self._index(j, i)] = 1
        return jnp.array(jacGraph)

    def _getJacobianResidual(self, theta):
        """
        Calculate the Jacobian of the residual vector
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns (array): DoF x DoF shaped Jacobian evaluated at theta

        """
        graph = self._getGraph(theta)
        jacResidual = jnp.zeros((self.DoF, self.DoF))
        jacResidual = jacResidual.at[0, 0].set(jnp.sqrt(self.a))
        jacGraphSquared = 2 * contract('db, dbe -> dbe', graph[:, :-1], self.jacGraph[:,:-1])
        jacResidual = jacResidual.at[1:, :].set(contract('ab, abe -> abe', jnp.sqrt(self.b), self.jacGraph[:,1:] - jacGraphSquared).reshape(self.DoF - 1, self.DoF))
        return np.sqrt(2) * jacResidual

    # def _getHessianResidual(self):
    #     """
    #     Calculate the Hessian of the residual vector
    #     Args:
    #         theta (array): DoF sized array, point to evaluate at.

    #     Returns (array): DoF x DoF x DoF shaped Hessian evaluated at theta

    #     """
    #     # hessRes_num = Jacobian(Jacobian(self._getResidual))(np.zeros(self.DoF))
    #     # np.allclose(hessRes_num, hessRes)
    #     hessRes = jnp.zeros((self.DoF, self.DoF, self.DoF))
    #     hessRes = hessRes.at[1:].set(contract('ji, jif, jie -> jief', -jnp.sqrt(8 * self.b), self.jacGraph[:, :-1], self.jacGraph[:, :-1]).reshape(self.DoF - 1, self.DoF, self.DoF))
    #     return hessRes

    def getMinusLogLikelihood(self, theta):
        """
        Returns minus log of Hybrid Rosenbrock
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns: (float) Density evaluation

        """
        r = self._getResidual(theta)
        self.nLikelihoodEvaluations += 1
        return jnp.dot(r, r) / 2
    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogLikelihood(self, theta):
        """
        Evaluates gradient of minus log of Hybrid Rosenbrock
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns: (array) DoF shaped array

        """
        r = self._getResidual(theta)
        jr = self._getJacobianResidual(theta)
        self.nGradLikelihoodEvaluations += 1
        return r.T @ jr
    @partial(jax.jit, static_argnums=(0,))
    def getGNHessianMinusLogLikelihood(self, theta):
        """
        Calculate Gauss-Newton approximation of Hybrid Rosenbrock
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns: (array) DoF x DoF shaped array of Gauss-Newton approximation at theta.

        """
        jr = self._getJacobianResidual(theta)
        return contract('af, ae -> ef', jr, jr)

    def getHessianMinusLogLikelihood(self, theta):
        """
        Calculates Hessian of minus log Hybrid Rosenbrock
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns: (array): DoF x DoF shaped array of Hessian of Hybrid Rosenbrock evaluated at theta

        """
        # hessNum = Jacobian(Jacobian(self.getMinusLogLikelihood))(theta)
        r = self._getResidual(theta)
        jr = self._getJacobianResidual(theta)
        self.nHessLikelihoodEvaluations += 1
        return contract('af, ae -> ef', jr, jr) + contract('a, aef -> ef', r, self.hessRes)

    def getJerkMinusLogLikelihood(self, theta):
        """
        Evaluate third order derivatives
        Args:
            theta (array): DoF sized array, point to evaluate at.

        Returns:

        """
        # TODO
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def getDerivativesMinusLogPosterior_ensemble(self, thetas):
        gmlpt = self.getGradientMinusLogPosterior_ensemble(thetas)
        Hmlpt = self.getGNHessianMinusLogPosterior_ensemble(thetas)
        return (gmlpt, Hmlpt)

    def getNormalizationConstant(self):
        """
        Evaluate normalization constant of Hybrid Rosenbrock given settings.
        Returns:

        """
        return np.sqrt(self.a / (np.pi ** self.DoF)) * np.prod(np.sqrt(self.b))

    def old_newDrawFromLikelihood_old(self, N):
        """
        Draw samples analytically from Hybrid Rosenbrock
        Args:
            N (int): Number of samples

        Returns: (array) N x DoF samples

        """
        samples = np.zeros((N, self.DoF))
        samples[:, 0] = np.random.normal(self.mu, 1 / (np.sqrt(2 * self.a)), N)
        b = self.b.flatten()
        d_not_prev = 1 + np.arange(0,self.n2) * (self.n1 - 1)
        for d in np.arange(1, self.DoF):
            if d in d_not_prev:
                samples[:,d] = np.random.normal(0, 1, N) / np.sqrt(2 * b[d-1]) + samples[:,0] ** 2
            else:
                samples[:,d] = np.random.normal(0, 1, N) / np.sqrt(2 * b[d-1]) + samples[:,d-1] ** 2
        return samples

    def newDrawFromLikelihood(self, N):
        """
        Draw samples analytically from Hybrid Rosenbrock
        Args:
            N (int): Number of samples

        Returns: (array) N x DoF samples

        """
        samples = np.zeros((N, self.DoF))
        samples[:, 0] = self.mu + np.random.normal(0, 1, N) / np.sqrt(2 * self.a)
        for j in range(self.n2):
            for i in np.arange(1, self.n1):
                samples[:, self._index(j, i)] = samples[:, self._index(j, i-1)] ** 2 + np.random.normal(0, 1, N) / np.sqrt(2 * self.b[j, i - 1])
        return samples

    ###################################################################################################################
    # This section is a wrapper so that one may use the Hybrid-Rosenbrock within a Bayesian framework.
    # Specifically, if one wants to use this code downstream in some mainstream sampling algorithm.
    #
    ###################################################################################################################

    def _newDrawFromPrior(self, nParticles):
        """
        Return samples from a uniform prior.
        Included for convenience.
        Args:
            nParticles (int): Number of samples to draw.

        Returns: (array) nSamples x DoF array of representative samples

        """
        return np.random.uniform(low=-6, high=6, size=(nParticles, self.DoF))

    def _getMinusLogPrior(self, theta):
        return 0

    def _getGradientMinusLogPrior(self, theta):
        return np.zeros(self.DoF)

    def _getHessianMinusLogPrior(self, theta):
        return np.zeros((self.DoF, self.DoF))

    def getMinusLogPosterior(self, theta):
        return self.getMinusLogLikelihood(theta) + self._getMinusLogPrior(theta)

    # @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogPosterior(self, theta):
        return self.getGradientMinusLogLikelihood(theta) + self._getGradientMinusLogPrior(theta)
    # @partial(jax.jit, static_argnums=(0,))
    def getGNHessianMinusLogPosterior(self, theta):
        return self.getGNHessianMinusLogLikelihood(theta) + self._getHessianMinusLogPrior(theta)

    def getHessianMinusLogPosterior(self, theta):
        return self.getHessianMinusLogLikelihood(theta) + self._getHessianMinusLogPrior(theta)

    # Vectorized versions of previous methods
    # Args:
    #   thetas (array): N x DoF array, where N is number of samples
    # Returns: (array) N x 1, N x DoF, N x (DoF x DoF), N x (DoF x DoF) respectively

    def getMinusLogPosterior_ensemble(self, thetas):
        return np.apply_along_axis(self.getMinusLogPosterior, 1, thetas)

    @partial(jax.jit, static_argnums=(0,))
    def getGradientMinusLogPosterior_ensemble(self, thetas):
        return jax.vmap(self.getGradientMinusLogPosterior)(thetas)
    @partial(jax.jit, static_argnums=(0,))
    def getGNHessianMinusLogPosterior_ensemble(self, thetas):
        return jax.vmap(self.getGNHessianMinusLogPosterior)(thetas)

    # def getHessianMinusLogPosterior_ensemble(self, thetas):
    #     return np.apply_along_axis(self.getHessianMinusLogPosterior, 1, thetas)

    def getCrossSection(self, index1, index2, func, ngrid):
        import matplotlib.pyplot as plt
        # a, b are the parameters for which we want the marginals:
        x = np.linspace(self.priorDict[a][0], self.priorDict[a][1], ngrid)
        y = np.linspace(self.priorDict[b][0], self.priorDict[b][1], ngrid)
        X, Y = np.meshgrid(x, y)
        particle_grid = np.zeros((ngrid ** 2, self.DoF))
        # index1 = self.gwfast_param_order.index(a)
        # index2 = self.gwfast_param_order.index(b)
        parameter_mesh = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T
        particle_grid[:, index1] = parameter_mesh[:, 0]
        particle_grid[:, index2] = parameter_mesh[:, 1]
        for i in range(self.DoF): # Fix all other parameters
            if i != index1 and i != index2:
                particle_grid[:, i] = np.ones(ngrid ** 2) * self.mu
        Z = np.exp(-1 * func(particle_grid).reshape(ngrid,ngrid))
        fig, ax = plt.subplots(figsize = (5, 5))
        cp = ax.contourf(X, Y, Z)
        # cbar = fig.colorbar(cp)
        plt.colorbar(cp)
        ax.set_xlabel(a)
        ax.set_ylabel(b)
        ax.set_title('Likelihood cross section')
        filename = a + b + '.png'
        path = os.path.join('marginals', filename)
        fig.savefig(path)



def main():

    pass
    #########################################
    # Testing 2D Case
    #########################################
    # def pi(x):
    #     mu_rosen=1
    #     a=1
    #     b=5
    #     HRD_2D = lambda x: np.exp(-a * (x[0] - mu_rosen) ** 2 - b * (x[1] - x[0] ** 2) ** 2)
    #     return HRD_2D(x)
    # def sample_2D_pi(N):
    #     mu_rosen=1
    #     a=1
    #     b=5
    #     samples = np.zeros((N, 2))
    #     samples[:, 0] = np.random.normal(mu_rosen, 1 / (np.sqrt(2 * a)), N)
    #     for n in range(N):
    #         samples[n,1] = np.random.normal(samples[n,0] ** 2, 1 / np.sqrt(2 * b))
    #     return samples
    # samples = sample_2D_pi(2000000)
    # ngrid = 500
    # begin = -10
    # end = 10
    # x = np.linspace(begin, end, ngrid)
    # y = np.linspace(begin, end, ngrid)
    # X, Y = np.meshgrid(x, y)
    # L = lambda x: np.apply_along_axis(pi, 1, x)
    # Z = L(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T).reshape(ngrid,ngrid)
    # width = 469.75502
    # fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
    # plt.xlim(begin, end)
    # plt.ylim(begin, end)
    # plt.hist2d(samples[:,0], samples[:,1], bins=100)
    # cp = ax.contour(X, Y, Z, 9, colors='r', alpha=0.5)
    # fig.show()
    # pass
    #########################################
    # Settings for 2D-HRD (TESTING)
    #########################################
    n2 = 2
    n1 = 3
    # Larger a -> more normal marginals
    # Larger b -> more rosenbrock-like features
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=20, b=np.ones((n2, n1-1)) * 20)
    HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu=1, a=30, b=np.ones((n2, n1-1)) * 20) # i like. how much
    theta = np.array([1.1, 1.2, 1.3, 1.4, 1.5])
    # HRD._getHessianResidual()
    hessHRD = HRD.getHessianMinusLogLikelihood(theta)
    # HRD._getResidual(theta)
    # HRD._getJacobianResidual(theta)

    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=30, b=np.ones((n2, n1-1)) * 1) # almost gaussian
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=1, b=np.ones((n2, n1-1)) * 5)
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=0.5, b=np.ones((n2, n1-1)) * 0.5) # nice
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=0.05, b=np.ones((n2, n1-1)) * 0.5 ) # 10x stretched vertically
    # HRD = hybrid_rosenbrock(n2=n2, n1=n1, mu_rosen=1, a=0.5, b=np.ones((n2, n1-1)) * 5 ) # less thick
    samples = HRD.newDrawFromLikelihood(10)
    # fig1 = corner(samples)
    #########################################
    # Contour calculations
    #########################################
    # ngrid = 500
    # begin = -10
    # end = 10
    # x = np.linspace(begin, end, ngrid)
    # y = np.linspace(begin, end, ngrid)
    # X, Y = np.meshgrid(x, y)
    # L = lambda x: np.apply_along_axis(HRD.getMinusLogLikelihood_individual, 1, x)
    # Z = np.exp(-1 * L(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T).reshape(ngrid,ngrid))
    ####################################################
    # Plot settings
    ####################################################
    # plt.style.use(os.path.join(root, 'latex.mplstyle'))
    width = 469.75502
    # fig, ax = plt.subplots(1, 1, figsize=set_size(width, fraction=1, subplots=(1, 1)))
    plt.rcParams['figure.figsize']=set_size(width, fraction=1, subplots=(1, 1))
    p = palettable.colorbrewer.qualitative.Set1_8.mpl_colors
    mpl.rcParams['axes.prop_cycle'] = cycler(color=p)
    # plt.xlim(begin, end)
    # plt.ylim(begin, end)
    # plt.axis('off')
    # ax.set_title('Collected samples')
    import pandas as pd
    # Solution adapted from https://stackoverflow.com/questions/43924280/pair-plot-with-heat-maps-possibly-logarithmic
    g = sns.PairGrid(pd.DataFrame(samples), corner=True)
    g.map_diag(plt.hist, bins=50)

    def pairgrid_heatmap(x, y, **kws):
        cmap = sns.light_palette(kws.pop("color"), as_cmap=True)
        plt.hist2d(x, y, cmap=cmap, cmin=1, **kws)

    g.map_offdiag(pairgrid_heatmap, bins=100)
    # plt.hist2d(samples[:,0], samples[:,1], bins=100)
    # cp = ax.contour(X, Y, Z, 9, colors='r', alpha=0.5)
    # ax.scatter(samples[:,0], samples[:,1], marker=".", s=1, label='Truth')
    plt.show()
    pass
if __name__ is '__main__':
    main()