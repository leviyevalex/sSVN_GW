import logging
from tqdm import trange  # Use to add a progress bar to loop
import copy
import h5py
import os, sys
import logging.config
from pyinstrument import Profiler
import numpy as np
from opt_einsum import contract
from time import time, sleep
import scipy
from scipy import sparse, linalg, spatial
import scipy.sparse.linalg
import jax
import jax.numpy as jnp
# from src.kernels import get_randomRBF_metric as _getKernelWithDerivatives
from functools import partial
from src.JAX_kernels import kernels
# from sklearn import metrics
from jax.config import config
import random # Used for birth death set selection
import traceback
config.update("jax_enable_x64", True)
from src.reparameterization import reparameterization, logistic_CDF, sigma

log = logging.getLogger(__name__)
# log.addHandler(logging.StreamHandler(stream=sys.stdout))
np.seterr(over='raise')
np.seterr(invalid='raise')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class samplers:
    def __init__(self, model, nIterations, nParticles, kernel_type, bd_kwargs, profile=None, bounded=None, t=None):
        # Quick settings checks
        assert(nParticles > 0)
        assert(nIterations > 0)
        # Setup file paths, logger, and profiler.
        now = str(int(time()))
        sleep(1) # Output path depends on 'now'. Make sure new instances are spaced out enough to be unique.
        log.debug('DEBUG: ROOT path %s' % ROOT_DIR)
        self.OUTPUT_DIR = os.path.join(ROOT_DIR, 'outdir')
        self.RUN_OUTPUT_DIR = os.path.join(self.OUTPUT_DIR, now)
        self.history_path = os.path.join(self.RUN_OUTPUT_DIR, 'output_data.h5')
        # Create folders that organize results
        if os.path.isdir(self.OUTPUT_DIR) == False:
            os.mkdir(self.OUTPUT_DIR)
        if os.path.isdir(self.RUN_OUTPUT_DIR) == False:
            os.mkdir(self.RUN_OUTPUT_DIR)
        # Logger setup
        LOG_OUTPUT_DIR = os.path.join(self.RUN_OUTPUT_DIR, 'info.log')
        fh = logging.FileHandler(filename=LOG_OUTPUT_DIR, mode='w')
        fh.setLevel(logging.INFO)
        log.addHandler(fh)

        if profile == True or profile == 'True' or profile == "'True'":
            self.profile = True
        else:
            self.profile = False
        self.model = model
        self.nParticles = nParticles
        self.nIterations = nIterations
        self.DoF = model.DoF
        self.dim = self.DoF * self.nParticles

        self.bounded = bounded
        self.t = t

        # scipy_cholesky = lambda mat: jax.scipy.linalg.cholesky(mat)
        # self.jit_scipy_cholesky = jax.jit(scipy_cholesky)

        # self.kernelKwargs = kernelKwargs
        kernel_class = kernels(nParticles=self.nParticles, DoF=self.DoF, kernel_type=kernel_type)
        self._getKernelWithDerivatives = kernel_class.getKernelWithDerivatives

        self.bd_kwargs = bd_kwargs
        kernel_class_birth_death= kernels(nParticles=nParticles, DoF=self.DoF, kernel_type=self.bd_kwargs['kernel_type'])
        self.bd_kernel = kernel_class_birth_death.getKernelWithDerivatives

        self.bd_kernel_kwargs = copy.deepcopy(self.bd_kwargs)
        self.bd_kernel_kwargs.pop('use')
        self.bd_kernel_kwargs.pop('kernel_type')
        self.bd_kernel_kwargs.pop('space')
        self.bd_kernel_kwargs.pop('stride')

        self.bd_kernel_kwargs['M'] = None
        self.bd_tau = self.bd_kernel_kwargs['tau']
        self.bd_stride = self.bd_kwargs['stride']
        self.bd_use = self.bd_kwargs['use']
        self.bd_start_iter = self.bd_kwargs['start_iter']


    def apply(self, kernelKwargs, method='SVGD', eps=0.1, schedule=None, lamb1=1, lamb2=2):
        """
        Evolves a set of particles according to (method) with step-size (eps).
        Args:
            method (str): Sampler method chosen.
            Options available: 'SVGD', 'sSVGD', 'BDSVN', 'SVN', 'sSVN'
        Returns (dict):
        'path_to_results' : Path to output h5 file with particle history, etc...
        'outdir_path': Path to output directory

        """
        if self.profile == True:
            profiler = Profiler()
            profiler.start()
        # np.random.seed(int(time())) # Enable for randomness
        np.random.seed(1) # Enable for reproducibility
 
        try:
            # X = self.model._newDrawFromPrior_frozen(self.nParticles) # Initial set of particles
            X = self.model._newDrawFromPrior(self.nParticles) # Initial set of particles
            eta = self._mapHypercubeToReals(X, self.model.lower_bound, self.model.upper_bound)
            key = jax.random.PRNGKey(0)
            with trange(self.nIterations) as ITER:
                for iter_ in ITER:
                    if method == 'reparam_sSVGD':
                        # Reparameterization to R^d
                        mlpt_X = self.model.getMinusLogPosterior_ensemble(X)
                        gmlpt_X, Hmlpt_X = self.model.getDerivativesMinusLogPosterior_ensemble(X)
                        mlpt_Y, gmlpt_Y, Hmlpt_Y = self.getUnboundedPotential(eta, mlpt_X, gmlpt_X, Hmlpt_X)

                        # Calculate SVGD direction
                        kx, gkx1 = self.metric_wrapper(self.k_lp, eta, kernelKwargs)
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt_Y)

                        # Birth-death step
                        self.bd_kernel_kwargs['M'] = None
                        tau = self.bd_kernel_kwargs['tau']
                        stride = self.bd_kwargs['stride']
                        use = self.bd_kwargs['use']
                        start_iter = self.bd_kwargs['start_iter']

                        key, subkey = jax.random.split(key)
                        n_events = 0
                        if use == True and iter_ !=0 and (iter_ % stride) == 0 and (iter_ > start_iter): # Update with teleportation
                            if self.bd_kwargs['space'] == 'primal':
                                # kern_bd = self.gaussian_kernel(X, self.bd_kernel_kwargs['h'])
                                kern_bd, _ = self.metric_wrapper(self.k_lp, X, self.bd_kernel_kwargs)
                                jump_idxs, n_events = self.birthDeathJumpIndicies(kern_bd, mlpt_X, tau=tau*stride)
                            elif self.bd_kwargs['space'] == 'dual':
                                kern_bd, _ = self.metric_wrapper(self.k_lp, eta, self.bd_kernel_kwargs)
                                jump_idxs, n_events = self.birthDeathJumpIndicies(kern_bd, mlpt_Y, tau=tau*stride)
                            noise = self.proposalNoise_SVGD(jump_idxs, kx, subkey)
                            eta = eta[jump_idxs] + eps * v_svgd[jump_idxs] + np.sqrt(eps) * noise
                        else: # Standard update
                            v_stc = self._getSVGD_v_stc(kx, subkey)
                            eta += eps * v_svgd + np.sqrt(eps) * v_stc 

                        X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)


                    elif method == 'reparam_sSVN':
                        # Reparameterization to R^d
                        mlpt_X = self.model.getMinusLogPosterior_ensemble(X)
                        gmlpt_X, Hmlpt_X = self.model.getDerivativesMinusLogPosterior_ensemble(X)
                        mlpt_Y, gmlpt_Y, Hmlpt_Y = self.getUnboundedPotential(eta, mlpt_X, gmlpt_X, Hmlpt_X)

                        # Stein variational Newton step
                        kernelKwargs['M'] = jnp.mean(Hmlpt_Y, axis=0) # jnp.eye(self.DoF)
                        kx, gkx1 = self.metric_wrapper(self.k_lp, eta, kernelKwargs)
                        H = self.get_H_lambda(Hmlpt_Y, kx, gkx1, damping=0.05)
                        UH = self.getCho(H)
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt_Y) 
                        v_svn, alphas = self._getSVN_direction(kx, v_svgd, UH)

                        # Birth-death step
                        self.bd_kernel_kwargs['M'] = None
                        tau = self.bd_kernel_kwargs['tau']
                        stride = self.bd_kwargs['stride']
                        use = self.bd_kwargs['use']
                        start_iter = self.bd_kwargs['start_iter']

                        key, subkey = jax.random.split(key)
                        n_events = 0
                        if use == True and iter_ !=0 and (iter_ % stride) == 0 and (iter_ > start_iter): # Update with teleportation
                            if self.bd_kwargs['space'] == 'primal':
                                kern_bd, _ = self.metric_wrapper(self.k_lp, X, self.bd_kernel_kwargs)
                                jump_idxs, n_events = self.birthDeathJumpIndicies(kern_bd, mlpt_X, tau=tau*stride)
                            elif self.bd_kwargs['space'] == 'dual':
                                kern_bd, _ = self.metric_wrapper(self.k_lp, eta, self.bd_kernel_kwargs)
                                jump_idxs, n_events = self.birthDeathJumpIndicies(kern_bd, mlpt_Y, tau=tau*stride)
                            noise = self.proposalNoise(jump_idxs, kx, UH, subkey)
                            eta = eta[jump_idxs] + eps * v_svn[jump_idxs] + np.sqrt(eps) * noise
                        else: # Standard update
                            v_stc = self._getSVN_v_stc(kx, UH, subkey)
                            eta += eps * v_svn + np.sqrt(eps) * v_stc 


                        X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)

                    elif method=='MALA':
                        n_events = 0
                        # Generate a random Gaussian noise for each point in the ensemble
                        key, subkey = jax.random.split(key)
                        noise = jax.random.normal(subkey, shape=X.shape)
                    
                        # Compute the proposal X_prime for each point
                        grad_V = self.model.getGradientMinusLogPosterior_ensemble(X)
                        X_prime = X - eps * grad_V + jnp.sqrt(2 * eps) * noise
                        # X = X - eps * grad_V + jnp.sqrt(2 * eps) * noise
                    
                        # Calculate log of posterior ratio
                        V = self.model.getMinusLogPosterior_ensemble(X)
                        V_prime = self.model.getMinusLogPosterior_ensemble(X_prime)
                        log_posterior_ratio = V - V_prime

                        # Calculate log of proposal ratio
                        grad_V_prime = self.model.getGradientMinusLogPosterior_ensemble(X_prime)
                        log_proposal_ratio = \
                        jnp.sum((X_prime - X + eps * grad_V) ** 2, axis=-1) / (4 * eps) - \
                        jnp.sum((X - X_prime + eps * grad_V_prime) ** 2, axis=-1) / (4 * eps)

                        # Calculate acceptance probability for each point
                        log_alpha = log_posterior_ratio + log_proposal_ratio
                        alpha = jnp.minimum(1, jnp.exp(log_alpha))

                        # Accept or reject the proposals based on the acceptance probability
                        key, subkey = jax.random.split(key)
                        idxs_accepted = jnp.argwhere(jax.random.uniform(subkey, shape=(self.nParticles,)) < alpha)[:,0]
                        X[idxs_accepted,:] = X_prime[idxs_accepted,:]
                        n_events = len(idxs_accepted)


                    elif method=='langevin':
                        v_svgd = 0 # for output issues

                        V_X = self.model.getMinusLogPosterior_ensemble(X)
                        
                        gmlpt_X, Hmlpt_X = self.model.getDerivativesMinusLogPosterior_ensemble(X)
                        eta, V_Y, gmlpt_Y, Hmlpt_Y = reparameterization(X, V_X, gmlpt_X, Hmlpt_X, self.model.lower_bound, self.model.upper_bound)
                        
                        ### Birth-death step ###

                        # Calculate KDE in bounded space
                        # kern_bd, _ = self.metric_wrapper(self.k_lp, eta, self.bd_kernel_kwargs)

                        # Get jump indicies
                        # jump_idxs, n_events = self.birthDeathJumpIndicies(kern_bd, V_Y, tau=tau*stride)

                        # Perform jumps and diffusion 
                        # eta = eta[jump_idxs] - eps * gmlpt_Y[jump_idxs] + np.sqrt(2 * eps) * np.random.normal(0, 1, size=(self.nParticles, self.DoF))
                        # -----------------------------------------------

                        # Perform update
                        eta += -gmlpt_Y * eps + np.sqrt(2 * eps) * np.random.normal(0, 1, size=(self.nParticles, self.DoF))

                        # Convert samples back to hypercube 
                        X = sigma(logistic_CDF(eta), self.model.lower_bound, self.model.upper_bound)

                        n_events = 0 # For output issues



                    # Update progress bar
                    # ITER.set_description('Stepsize %f | Median bandwidth: %f | SVN norm: %f | Noise norm: %f | SVGD norm %f | Dampening %f' % (eps, self._bandwidth_MED(X), np.linalg.norm(v_svn), np.linalg.norm(v_stc), np.linalg.norm(v_svgd),  lamb))
                    # ITER.set_description('Stepsize %f | Median bandwidth: %f' % (eps1, self._bandwidth_MED(X)))
                    ITER.set_description('Stepsize %f | Median bandwidth: %f | n_events: %i' % (eps[0,0], self._bandwidth_MED(X), n_events))


                    # Store relevant per iteration information
                    with h5py.File(self.history_path, 'a') as f:
                        g = f.create_group('%i' % iter_)
                        g.create_dataset('X', data=copy.deepcopy(X))
                        # if self.model.priorDict == None:
                        #     g.create_dataset('X', data=copy.deepcopy(X))
                        # else:
                        #     g.create_dataset('X', data=copy.deepcopy(self._mapRealsToHypercube(X, self.model.lower_bound, self.model.upper_bound)))
                        # g.create_dataset('h', data=copy.deepcopy(h))
                        g.create_dataset('eps', data=copy.deepcopy(eps))
                        if method != 'MALA':
                            g.create_dataset('v_svgd', data=copy.deepcopy(v_svgd))
                        if method == 'reparam_sSVN':
                            # g.create_dataset('gmlpt_X', data=copy.deepcopy(gmlpt_X))
                            g.create_dataset('gmlpt_Y', data=copy.deepcopy(gmlpt_Y))
                            g.create_dataset('v_svn', data=copy.deepcopy(v_svn))
                            g.create_dataset('DJ', data=np.sum(v_svgd * alphas))
                        g.create_dataset('id', data=copy.deepcopy(self.model.id))

                    # Update particles (I'll probably have to individualize this, since mirrored is a bit more involved than this!)
                    # X += update
                    # Pad particles near boundaries
                    # X = self.model._inBounds(X, self.model.lower_bound, self.model.upper_bound)

                # Dynamics completed: Storing data
                with h5py.File(self.history_path, 'a') as f:
                    g = f.create_group('metadata')
                    g.create_dataset('X', data=copy.deepcopy(X))
                    # if self.model.priorDict == None:
                    #     g.create_dataset('X', data=copy.deepcopy(X))
                    # else:
                    #     g.create_dataset('X', data=copy.deepcopy(self._mapRealsToHypercube(X, self.model.lower_bound, self.model.upper_bound)))

                        # g.create_dataset('X', data=copy.deepcopy(self._F_inv(X, self.model.lower_bound, self.model.upper_bound)))
                    g.create_dataset('nLikelihoodEvals', data=copy.deepcopy(self.model.nLikelihoodEvaluations))
                    g.create_dataset('nGradLikelihoodEvals', data=copy.deepcopy(self.model.nGradLikelihoodEvaluations))
                    g.create_dataset('nParticles', data=copy.deepcopy(self.nParticles))
                    g.create_dataset('DoF', data=copy.deepcopy(self.DoF))
                    g.create_dataset('L', data=copy.deepcopy(iter_ + 1))
                    g.create_dataset('method', data=method)
                    g1 = f.create_group('final_updated_particles')
                    g1.create_dataset('X', data=X)
                    # if self.model.priorDict == None:
                    #     g1.create_dataset('X', data=X)
                    # else:
                    #     g1.create_dataset('X', data=copy.deepcopy(self._mapRealsToHypercube(X, self.model.lower_bound, self.model.upper_bound)))

                        # g1.create_dataset('X', data=copy.deepcopy(self._F_inv(X, self.model.lower_bound, self.model.upper_bound)))

            # Save profiling results
            if self.profile == True:
                log.info('OUTPUT: Saving profile as html')
                profiler.stop()
                try:
                    with open(os.path.join(self.RUN_OUTPUT_DIR, 'output.html'), "w") as f:
                        profile_html = profiler.output_html()
                        log.info(type(profile_html))
                        f.write(profile_html)
                        log.info('OUTPUT: Successfully saved profile to html.')
                        # log.info(profiler.output_text(unicode=True, color=True))
                except:
                    log.error('OUTPUT: Failed to save profile to html. Trying utf-8', exc_info=True)
                    try:
                        with open(os.path.join(self.RUN_OUTPUT_DIR, 'output.html'), "wb") as f:
                            f.write(profile_html.encode('utf-8'))
                            log.info('OUTPUT: Successfully saved profile to html.')
                    except:
                        log.error('OUTPUT: Failed to save utf-8 profile to html.', exc_info=True)

            log.info('OUTPUT: Run completed successfully! Data stored in:\n %s' % self.history_path)

        except Exception as e:
            print(e)
            traceback.print_exc()
            log.error("Error occurred in apply()", exc_info=True)

########################################################################################################################

    # Direction methods (either to make the main code looks clean or so we may reuse it elsewhere)
    @partial(jax.jit, static_argnums=(0,))
    def _getSVGD_direction(self, kx, gkx, gmlpt, reg=None):
        """
        Get SVGD velocity field
        Args:
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            gkx (array): N x N x D array, gradient of kernel gram matrix $\nabla_1 k(z_m, z_n)$
            gmlpt (array): N x D array, gradient of minus log target

        Returns:

        """
        if reg == None:
            # v_svgd = -1 * contract('mn, mo -> no', kx, gmlpt, backend='jax') / self.nParticles + jnp.mean(gkx, axis=0)
            v_svgd = -1 * contract('mn, mo -> no', kx, gmlpt, backend='jax') / self.nParticles + jnp.mean(gkx, axis=0)
        else:
            v_svgd = -1 * contract('mn, mo -> no', kx, gmlpt, backend='jax') / self.nParticles + jnp.mean(gkx * reg[:, np.newaxis, np.newaxis], axis=0)
        return v_svgd

    # @partial(jax.jit, static_argnums=(0,))
    def _getSVGD_v_stc(self, kx, key):
        """
        Get noise injection velocity field for SVGD
        Args:
            L_kx (array): N x N array, lower triangular Cholesky factor of kernel gram matrix $k$
            Bdn (array): D x N array, standard normal noise

        Returns:

        """
        # Bdn = jax.random.normal(key, (self.DoF, self.nParticles)) # standard normal sample
        # alpha, L_kx = self._getMinimumPerturbationCholesky(kx)
        # return jnp.sqrt(2 / self.nParticles) * contract('mn, in -> im', L_kx, Bdn, backend='jax').flatten(order='F').reshape(self.nParticles, self.DoF)
        alpha, L_kx = self._getMinimumPerturbationCholesky(kx)
        # L_kx = jnp.linalg.cholesky(kx)
        b = jax.random.normal(key, (self.nParticles, self.DoF)) # standard normal sample
        return jnp.sqrt(2 / self.nParticles) * (L_kx @ b).reshape(self.DoF,self.nParticles).flatten(order='F').reshape(self.nParticles, self.DoF) 


    @partial(jax.jit, static_argnums=(0,))
    def _getSVN_direction(self, kx, v_svgd, UH):
        """
        Get SVN velocity field
        Args:
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            v_svgd (array): N x D array, SVGD velocity field
            UH (array): ND x ND array, upper triangular Cholesky factor of SVN Hessian $H$

        Returns: (array) N x D array, SVN direction

        """
        # alphas = scipy.linalg.cho_solve((UH, False), v_svgd.flatten()).reshape(self.nParticles, self.DoF)
        alphas = jax.scipy.linalg.cho_solve((UH, False), v_svgd.flatten()).reshape(self.nParticles, self.DoF)
        v_svn = contract('mn, ni -> mi', kx, alphas, backend='jax')
        return v_svn, alphas

    def getUnboundedPotential(self, eta, mlpt_X, gmlpt_X, Hmlpt_X):
        delta = self.model.upper_bound - self.model.lower_bound
        phi =  delta / (4 * jnp.cosh(eta / 2) ** 2)
        mlpt_Y = mlpt_X - jnp.sum(jnp.log(phi), axis=1)
        gmlpt_Y = gmlpt_X * phi + jnp.tanh(eta / 2)
        Hmlpt_Y = Hmlpt_X * contract('Ni,Nj -> Nij', phi, phi)
        Hmlpt_Y = Hmlpt_Y.at[:, jnp.arange(self.DoF), jnp.arange(self.DoF)].add(2 * phi / delta)
        return mlpt_Y, gmlpt_Y, Hmlpt_Y

        # Reparameterization to R^d
        # delta = self.model.upper_bound - self.model.lower_bound
        # phi =  delta / (4 * jnp.cosh(eta / 2) ** 2)
        # mlpt_X = self.model.getMinusLogPosterior_ensemble(X)
        # mlpt_Y = mlpt_X - jnp.sum(jnp.log(phi), axis=1)
        # gmlpt_X, Hmlpt_X = self.model.getDerivativesMinusLogPosterior_ensemble(X)
        # gmlpt_Y = gmlpt_X * phi + jnp.tanh(eta / 2)
        # Hmlpt_Y = Hmlpt_X * contract('Ni,Nj -> Nij', phi, phi)
        # Hmlpt_Y = Hmlpt_Y.at[:, jnp.arange(self.DoF), jnp.arange(self.DoF)].add(2 * phi / delta)


    @partial(jax.jit, static_argnums=(0,))
    def _getSVN_v_stc(self, kx, UH, key):
        """
        Get noise injection velocity field for SVN
        Args:
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            UH (array): ND x ND array, upper triangular Cholesky factor of SVN Hessian $H$

        Returns: (array) N x D array, noise injection for SVN

        """
        # B = np.random.normal(0, 1, self.dim)
        B = jax.random.normal(key, (self.dim,))
        tmp1 = jax.scipy.linalg.solve_triangular(UH, B, lower=False).reshape(self.nParticles, self.DoF)
        return jnp.sqrt(2 / self.nParticles) * contract('mn, ni -> mi', kx, tmp1, backend='jax')




    def _getSteinHessianBlockDiagonal(self, Hmlpt, kx, gkx):
        """
        Calculate block diagonal SVN Hessian $H_{BD}$ by taking block diagonal of both $H_1, H_2$
        Args:
            Hmlpt (array): N x D x D array, Hessian target evaluated at particle n $- \nabla^2 \ln \pi(z_n)$
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            gkx (array): N x N x D array, gradient of kernel gram matrix $\nabla_1 k(z_m, z_n)$

        Returns: (array) N x D x D block diagonal SVN Hessian $H_{BD}$

        """
        return (contract('mn, nij -> mij' , kx ** 2, Hmlpt) + contract('mni, mnj -> mij', gkx, gkx)) / self.nParticles

    def get_H_lambda(self, Hmlpt, kx, gkx, damping=0.05):
        T1 = contract('mk, nk, kij -> mnij', kx, kx, Hmlpt, backend='jax') / self.nParticles 
        T2 = contract('mki, mkj -> mij', gkx, gkx, backend='jax') / self.nParticles
        T1 = T1.at[jnp.arange(self.nParticles), jnp.arange(self.nParticles)].add(T2)
        T1 = T1.at[..., jnp.arange(self.DoF), jnp.arange(self.DoF)].add(damping * kx[..., None])
        return self._reshapeNNDDtoNDND(T1)

    # @partial(jax.jit, static_argnums=(0,))
    def _getSteinHessianPosdef(self, Hmlpt, kx, gkx, reg=None):
        """
        Calculate SVN Hessian $H = H_1 + H_2$.
        Note: If H_1 is made positive-definite (with Gauss-Newton approximation for example),
        then adding H_2 block diagonal ensures H is positive definite.
        Args:
            Hmlpt (array): N x D x D array, Hessian target evaluated at particle n $- \nabla^2 \ln \pi(z_n)$
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            gkx (array): N x N x D array, gradient of kernel gram matrix $\nabla_1 k(z_m, z_n)$

        Returns: (array) ND x ND SVN Hessian $H$

        """
        H1 = contract("xy, xz, xbd -> yzbd", kx, kx, Hmlpt, backend='jax')
        if reg == None:
            # H2 = contract('xzi, xzj -> zij', gkx, gkx, backend='jax')
            # TODO DOUBLE CHECKING THIS ONE (THEY AGREE)
            H2 = contract('xzi, xzj -> xij', gkx, gkx, backend='jax')
        else:
            H2 = contract('xzi, xzj -> zij', gkx, gkx, backend='jax') * reg[:, np.newaxis, np.newaxis]

        # H2 = contract('pni, pmj -> mnij', gkx, gkx) # calculate whole thing

        H1 = H1.at[jnp.array(range(self.nParticles)), jnp.array(range(self.nParticles))].add(H2)
        # H1[range(self.nParticles), range(self.nParticles)] += H2
        # return self._reshapeNNDDtoNDND((H1 + H2) / self.nParticles)
        return self._reshapeNNDDtoNDND(H1 / self.nParticles)


    def _getMinimumPerturbationCholesky(self, x, jitter=1e-9):
        """
        Solution adapted from TensorFlow GitHub page
        Args:
            x (array): "nearly" positive definite matrix
            jitter (float): How much to add to x $x + jitter * I$ where $I$ is the identity matrix.

        Returns: (float, array) Required jitter to produce decomposition and corresponding lower triangular Cholesky factor
        Remarks:
        (i) np.linalg.cholesky returns lower triangular matrix!
        """
        try:
            cholesky = np.linalg.cholesky(x)
            return 0, cholesky
        except Exception:
            while jitter < 1.0:
                try:
                    cholesky = np.linalg.cholesky(x + jitter * np.eye(x.shape[0]))
                    log.warning('CHOLESKY: Matrix not positive-definite. Adding alpha = %.2E' % jitter)
                    return jitter, cholesky
                except Exception:
                    jitter = jitter * 10
            raise Exception('CHOLESKY: Factorization failed.')

    @partial(jax.jit, static_argnums=(0,))
    def _hyperbolic_schedule(self, t, T, c=1.3, p=5):
        """
        Hyperbolic annealing schedule
        Args:
            t (int): Current iteration
            T (int): Total number of iterations
            c (float): Controls where transition begins
            p (float): Exponent determining speed of transition between phases

        Returns: (float)

        """
        return jnp.tanh((c * t / T) ** p) + 1e-11

    @partial(jax.jit, static_argnums=(0,))
    def _cyclic_schedule(self, t, T, p=5, C=5):
        """
        Cyclic annealing schedule
        Args:
            t (int): Current iteration
            T (int): Total number of iterations
            p (float): Exponent determining speed of transition between phases
            C (int): Number of cycles

        Returns:

        """
        tmp = T / C
        return (jnp.mod(t, tmp) / tmp) ** p + 1e-11

    # Stochastic SVN : Reshaping methods
    def _reshapeNNDDtoNDND(self, H):
        """

        Args:
            H (array): Takes N x N x d x d tensor to reshape

        Returns (array): Rearranged H into an ND x ND tensor with N blocks of size d x d along the diagonal

        """
        return H.swapaxes(1, 2).reshape(self.dim, self.dim)

    def _reshapeNDNDtoNNDD(self, H):
        """
        # Adapted from TensorFlow library
        Undoes action of self._reshapeNNDDtoNDND
        Args:
            H (array): Takes Nd x Nd tensor with N blocks of size d x d along the diagonal

        Returns (array): Returns N x N x d x d tensor of blocks

        """
        block_shape = np.array((self.DoF, self.DoF))
        new_shape = tuple(H.shape // block_shape) + tuple(block_shape)
        new_strides = tuple(H.strides * block_shape) + H.strides
        return np.lib.stride_tricks.as_strided(H, shape=new_shape, strides=new_strides)

    # Bandwidth selection
    def _bandwidth_MED(self, X):
        """
        Median bandwidth selection method
        Args:
            X (array): N x D array of particle positions

        Returns: (float) kernel bandwidth $h$

        """
        pairwise_distance = scipy.spatial.distance_matrix(X, X)
        median = np.median(np.trim_zeros(pairwise_distance.flatten()))
        return median ** 2 / np.log(self.nParticles + 1)

    # def _getKernelWithDerivatives(self, X, h, M=None):
    #     """
    #     Computes radial basis function (Gaussian) kernel with optional "metric" - See (Detommasso 2018)
    #     Args:
    #         X (array): N x d array of particles
    #         h (float): Kernel bandwidth
    #         M (array): d x d positive semi-definite metric.

    #     Returns (tuple): N x N kernel gram matrix, N x N x d gradient of kernel (with respect to first slot of kernel)

    #     """

    #     displacement_tensor = self._getPairwiseDisplacement(X)
    #     if M is not None:
    #         U = scipy.linalg.cholesky(M)
    #         X = contract('ij, nj -> ni', U, X)
    #         displacement_tensor = contract('ej, mnj -> mne', M, displacement_tensor)
    #     kx = np.exp(-scipy.spatial.distance_matrix(X, X) ** 2 / h)
    #     gkx1 = -2 * contract('mn, mne -> mne', kx, displacement_tensor) / h
    #     ## test_gkx = -2 / h * contract('mn, ie, mni -> mne', kx, U, displacement_tensor)
        # return kx, gkx1

    #######################################################
    # Methods needed to apply sSVN on bounded domains
    #######################################################

    # def __newDrawFromPrior_(self, nParticles):
    #     X = self.model._newDrawFromPrior(nParticles)
    #     if self.bounded == 'reparam':
    #         Y = self._mapHypercubeToReals(X, self.model.lower_bound, self.model.upper_bound)
    #         return Y
    #     elif self.bounded == 'log_boundary':
    #         return X

    # def _F(self, X, a, b):
    #     return jnp.log((X - a) / (b - X))

    # def _F_inv(self, Y, a, b):
    #     return (a + b * jnp.exp(Y)) / (1 + jnp.exp(Y))

    # def _dF_inv(self, Y, a, b):
    #     return (b - a) * jnp.exp(Y) / (1 + jnp.exp(Y)) ** 2

    # def _diagHessF_inv(self, Y, a, b): # Change name to d2F_inv
    #     return (b - a) * jnp.exp(Y) * (1 - jnp.exp(Y)) / (1 + jnp.exp(Y)) ** 3

    # # @partial(jax.jit, static_argnums=(0,))
    # def _getDerivativesMinusLogPosterior_(self, Y): # Checked (x)
    #     if self.model.priorDict is None:
    #         return self.model.getDerivativesMinusLogPosterior_ensemble(Y)
    #     else:
    #         X = self._F_inv(Y, self.model.lower_bound, self.model.upper_bound)
    #         gmlpt_X, Hmlpt_X = self.model.getDerivativesMinusLogPosterior_ensemble(X)

    #         dF_inv = self._dF_inv(Y, self.model.lower_bound, self.model.upper_bound)
    #         diagHessF_inv = self._diagHessF_inv(Y, self.model.lower_bound, self.model.upper_bound)

    #         gmlpt_Y = dF_inv * gmlpt_X - diagHessF_inv / dF_inv

    #         Hmlpt_Y = contract('Nd, Nb, Ndb -> Ndb', dF_inv, dF_inv, Hmlpt_X, backend='jax')  
    #         Hmlpt_Y = Hmlpt_Y.at[:, jnp.array(range(self.DoF)), jnp.array(range(self.DoF))].add(2 * np.exp(Y) / (1 + np.exp(Y)) ** 2)
    #         # Hmlpt_Y = Hmlpt_Y.at[:, jnp.array(range(self.DoF)), jnp.array(range(self.DoF))].add(diagHessF_inv * gmlpt_X)

    #         return (gmlpt_Y, Hmlpt_Y)




    ########################################################################################################
    # V2
    ########################################################################################################

    @partial(jax.jit, static_argnums=(0,))
    def _mapRealsToHypercube(self, Y, a, b):
        """Map a set of particles with support on R^d (d-dimensional reals) to H^d (d-dimensional hypercube)

        Parameters
        ----------
        Y : array
            (N,d) shaped array of particles with support in R^d
        a : array
            (d,) shaped array of lower bounds constituting H^d
        b : array
            (d,) shaped array of upper bounds constituting H^d

        Returns
        -------
        array
            (N,d) shaped array of transformed particles with support in H^d

        Example
        -------
        Consider the two dimensional hypercube (square) H^2 = [1, 2] X [3, 4].
        Then one would set a = array[1, 3], and b = array[2, 4].

        References
        ----------
        https://mc-stan.org/docs/2_27/reference-manual/logit-transform-jacobian-section.html

        """
        return (a + b * jnp.exp(Y)) / (1 + jnp.exp(Y))
    
    @partial(jax.jit, static_argnums=(0,))
    def _mapHypercubeToReals(self, X, a, b):
        """Map a set of particles with support in H^d (d-dimensional hypercube) to R^d (d-dimensional reals)

        Parameters
        ----------
        X : array
            (N,d) shaped array of particles with support in H^d
        a : array
            (d,) shaped array of lower bounds constituting H^d
        b : array
            (d,) shaped array of upper bounds constituting H^d

        Returns
        -------
        array
            (N,d) shaped array of transformed particles with support in R^d

        References
        ----------
        https://mc-stan.org/docs/2_27/reference-manual/logit-transform-jacobian-section.html

        """
        return jnp.log((X - a) / (b - X))
    
    @partial(jax.jit, static_argnums=(0,))
    def _jacMapRealsToHypercube(self, Y, a, b):
        """Calculate the Jacobian of mapRealsToHypercube (dx/dy)

        Parameters
        ----------
        Y : array
            (N,d) shaped array of particles with support in R^d
        a : array
            (d,) shaped array of lower bounds constituting H^d
        b : array
            (d,) shaped array of upper bounds constituting H^d

        Returns
        -------
        array
            (N,d) shaped array representing the Jacobian

        Remark
        ------
        Normally this map would be (N,d,d) shaped. However, it is diagonal as a consequence of its definition,
        and therefore we need only return (N,d) entries. 
        """
        return (b - a) / (4 * jnp.cosh(Y / 2) ** 2)
    
    @partial(jax.jit, static_argnums=(0,))
    def _getBoundaryGradientCorrection(self, Y):
        """Calculates $\nabla_y \ln \det(dx/dy)$, the "correction" in the transformed gradient expression.

        Parameters
        ----------
        Y : array
            (N,d) shaped array of particles with support in R^d

        Returns
        -------
        array
            (N,d) shaped array correction term
        """
        return jnp.tanh(Y / 2)

    @partial(jax.jit, static_argnums=(0,))
    def _getBoundaryHessianCorrection(self, Y):
        """Calculates $\del_{y_i} \del_{y_j} \ln \det(dx/dy)$, the "correction" in the transformed Hessian expression.

        Parameters
        ----------
        Y : array
            (N,d) shaped array of particles with support in R^d

        Returns
        -------
        array
            (N,d) shaped array correction term
        """
        return 1 / (2 * jnp.cosh(Y / 2) ** 2)

    # def _getDerivativesMinusLogPosterior_new(self, Y, t): 
    #     """Wrapper method to correct for H^d to R^d coordinate transformation

    #     Parameters
    #     ----------
    #     Y : array
    #         (N,d) shaped array of particles with support in R^d

    #     Returns
    #     -------
    #     tuple
    #         (N,d) array representing the gradient and (N,d,d) array representing the Hessian in the unbounded space.
    #     """
    #     if self.bounded == None:
    #         return self.model.getDerivativesMinusLogPosterior_ensemble(Y)
    #     elif self.bounded == 'log_boundary':

    #         gmlpt = jnp.zeros((self.nParticles, self.DoF))
    #         hmlpt = jnp.zeros((self.nParticles, self.DoF, self.DoF))

    #         idx = self.getIndiciesParticlesInBound(Y, self.model.lower_bound, self.model.upper_bound)
    #         gB = self._getGradBarrier(Y[idx], self.model.lower_bound, self.model.upper_bound, t)
    #         hessB = self._getHessBarrier(Y[idx], self.model.lower_bound, self.model.upper_bound, t)

    #         gmlpt_idx, hmlpt_idx = self.model.getDerivativesMinusLogPosterior_ensemble(Y[idx])

    #         gmlpt = gmlpt.at[idx].set(gmlpt_idx + gB)

    #         hmlpt_idx = hmlpt_idx.at[:, jnp.array(range(self.DoF)), jnp.array(range(self.DoF))].add(hessB)

    #         hmlpt = hmlpt.at[idx].set(hmlpt_idx)

    #         return (gmlpt, hmlpt)

    #     elif self.bounded == 'mirrored':
    #         X = self._mapRealsToHypercube(Y, self.model.lower_bound, self.model.upper_bound)
    #         gmlpt_X, Hmlpt_X = self.model.getDerivativesMinusLogPosterior_ensemble(X)

    #         dxdy = self._jacMapRealsToHypercube(Y, self.model.lower_bound, self.model.upper_bound)
    #         boundary_correction_grad = self._getBoundaryGradientCorrection(Y)
    #         boundary_correction_hess = self._getBoundaryHessianCorrection(Y)

    #         gmlpt_Y = dxdy * gmlpt_X + boundary_correction_grad
    #         Hmlpt_Y = contract('Ni, Nj, Nij -> Nij', dxdy, dxdy, Hmlpt_X, backend='jax') 
    #         Hmlpt_Y = Hmlpt_Y.at[:, jnp.array(range(self.DoF)), jnp.array(range(self.DoF))].add(boundary_correction_hess)

    #         return (gmlpt_Y, Hmlpt_Y)

    def getIndiciesParticlesInBound(self, X, a, b):
        truth_table = ((X > a) & (X < b))
        return np.where(np.all(truth_table, axis=1))[0]

    def _getGradBarrier(self, X, a, b, t):
        return (1 / (b - X) - 1 / (X - a)) / t
    
    def _getHessBarrier(self, X, a, b, t):
        return (1 / (b - X) ** 2 + 1 / (X - a) ** 2) / t


#################################################
#################################################
#################################################

    # Standard kernel
    def __getKernelWithDerivatives_(self, Y, params):
        kx, gkx1 = self._getKernelWithDerivatives(Y, params)
        return (kx, gkx1)



    # Useful distance functions

    def _getPairwiseDisplacement(self, X):
        return X[:,np.newaxis,:] - X[np.newaxis,:,:]

    # def get_vSVGD_stc(self, kx): # Checks: X
    #     alpha, L_kx = self._getMinimumPerturbationCholesky(kx)
    #     v_stc = self._getSVGD_v_stc(L_kx)
    #     return v_stc

    #####################################################
    # Matrix and mirrored methods
    # Note: gradient must be taken w.r.t second slot
    #####################################################

    @partial(jax.jit, static_argnums=(0,))
    def getMatrixMirrorKernel(self, X, matrix_kern, grad_matrix_kern):
        # Remarks:
        # (i)   diagonal of $\nabla^2 \psi(x)^{-1}$
        # (ii)  diagonal of Jacobian of tmp1
        # (iii) grad_matrix_kern (Derivative should be taken on the second slot)

        a = self.model.lower_bound
        b = self.model.upper_bound

        tmp = (X - a) * (b - X) / (b - a) # (i)
        tmp_prime = (a + b - 2 * X) / (b - a) # (ii) 

        k_psi = contract('xyij, yj -> xyij', matrix_kern, tmp, backend='jax')

        grad_k_psi = contract('xyijk, yj -> xyijk', grad_matrix_kern, tmp, backend='jax') \
                   + contract('xyij, yj, jk -> xyijk', matrix_kern, tmp_prime, np.eye(self.DoF), backend='jax')

        return k_psi, grad_k_psi

    @partial(jax.jit, static_argnums=(0,))
    def getMatrixSVGD_drift(self, matrix_kern, grad_matrix_kern, gmlpt): # Checks: XX
        uphill = contract('xyij, yj -> xi', matrix_kern, -gmlpt, backend='jax') / self.nParticles
        repulsion = contract('xyijj -> xi', grad_matrix_kern, backend='jax') / self.nParticles 
        return uphill + repulsion

    @partial(jax.jit, static_argnums=(0,))
    def getMatrixSVGD_noise(self, K): # Checks: X
        # LK = self.jit_scipy_cholesky(K).T # Note: scipy cholesky returns upper triangular matrix.
        B = np.random.normal(0, 1, self.dim)
        LK = jax.scipy.linalg.cholesky(K, lower=True) 
        return (np.sqrt(2) * LK @ B).reshape(self.nParticles, self.DoF)

    @partial(jax.jit, static_argnums=(0,))
    def getMatrixSVN_drift(self, UH, v_svgd, K): # Checks: X
        # Remarks:
        # (i) cho_solve expects lower triangular matrix. Pass False if upper triangular
        alphas = jax.scipy.linalg.cho_solve((UH, False), v_svgd.flatten())
        return self.nParticles * (K @ alphas).reshape(self.nParticles, self.DoF)

    @partial(jax.jit, static_argnums=(0,))
    def getMatrixSVN_Hessian(self, matrix_kern, grad_matrix_kern, hmlpt): # Checks: XX
        h1 = contract('myia, yab, nyjb -> mnij', matrix_kern, hmlpt, matrix_kern) / self.nParticles 
        h2 = contract('myaij, mybji -> mab', grad_matrix_kern, grad_matrix_kern) / self.nParticles
        h1 = h1.at[jnp.array(range(self.nParticles)), jnp.array(range(self.nParticles))].add(h2)
        # h1[range(self.nParticles), range(self.nParticles)] += h2
        return self._reshapeNNDDtoNDND(h1)

    @partial(jax.jit, static_argnums=(0,))
    def getMatrixSVN_noise(self, K, UH): # Checks: X
        B = np.random.normal(0, 1, self.dim)
        tmp1 = jax.scipy.linalg.solve_triangular(UH, B, lower=False)#
        return (np.sqrt(2 * self.nParticles) * K @ tmp1).reshape(self.nParticles, self.DoF)


######################################################################
    @partial(jax.jit, static_argnums=(0,))
    def getDerivatives_sharp(self, eta):
        X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
        
        gmlpt_X, Hmlpt_X = self.model.getDerivativesMinusLogPosterior_ensemble(X)
        
        dxdy = self._jacMapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
        boundary_correction_grad = self._getBoundaryGradientCorrection(eta)
        boundary_correction_hess = self._getBoundaryHessianCorrection(eta)

        gmlpt_Y = dxdy * gmlpt_X + boundary_correction_grad
        Hmlpt_Y = contract('Ni, Nj, Nij -> Nij', dxdy, dxdy, Hmlpt_X, backend='jax') 
        Hmlpt_Y = Hmlpt_Y.at[:, jnp.array(range(self.DoF)), jnp.array(range(self.DoF))].add(boundary_correction_hess)

        return gmlpt_X, Hmlpt_X, gmlpt_Y, Hmlpt_Y

    @partial(jax.jit, static_argnums=(0,))
    def getGradient_sharp(self, eta):
        X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
        gmlpt_X = self.model.getGradientMinusLogPosterior_ensemble(X)
        dxdy = self._jacMapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
        boundary_correction_grad = self._getBoundaryGradientCorrection(eta)
        gmlpt_Y = dxdy * gmlpt_X + boundary_correction_grad
        return gmlpt_Y

    @partial(jax.jit, static_argnums=(0,))
    def getCho(self, H):
        return jax.scipy.linalg.cholesky(H, lower=False)
#######################################################
# Linesearch methods
#######################################################
    @partial(jax.jit, static_argnums=(0,))
    def mlpt_sharp(self, eta): # Checks: X
        """ 
        # Remarks:
        # (i) jac returns a N x d matrix, since it is diagonal, not a N x d x d matrix.

        """
        dxdy = self._jacMapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
        det_dxdy = jnp.prod(dxdy, axis=-1) # Determinant is product of diagonal entries 
        X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
        # a = self.model.heterodyne_minusLogLikelihood(X) - jnp.log(jnp.abs(det_dxdy))
        a = self.model.getMinusLogPosterior_ensemble(X) - jnp.log(jnp.abs(det_dxdy))
        return a

    # @partial(jax.jit, static_argnums=(0,))
    # def mlpt_sharp_single(self, eta):
    #     dxdy = self._jacMapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
    #     det_dxdy = jnp.prod(dxdy, axis=-1) # Determinant is product of diagonal entries 
    #     X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
    #     a = self.model.heterodyne_minusLogLikelihood(X) - jnp.log(jnp.abs(det_dxdy))
    #     # a = self.model.getMinusLogPosterior_ensemble(X) - jnp.log(jnp.abs(det_dxdy))
    #     return a


    # def jv(self, gkx1, alphas):
    #     return contract('mnj, ni -> mij', gkx1, alphas)

        # Calculations

        # Dual
        # V = self.mlpt_sharp(eta)
        # kern_bd, _ = self.bd_kernel(eta, self.bd_kernel_kwargs)

        # Primal
        # X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
        # V = self.model.getMinusLogPosterior_ensemble(X)
        # kern_bd, _ = self.bd_kernel(X, self.bd_kernel_kwargs)


    def birthDeathJumpIndicies(self, kern_bd, V, tau=0.01):
        """ 
        Remarks
        -------

        (1) Uses discrepancy from original paper
        (2) uses discrepancy from sequal paper

        """
        # Data structure for calculation
        alive = ListDict()
        for i in range(self.nParticles):
            alive.add_item(i)

        # Get particles with significant mass discrepancy
        # beta = np.log(np.mean(kern_bd, axis=1)) + V
        # Lambda = beta - np.mean(beta) 

        # Get particles with significant mass discrepancy
        tmp1 = np.mean(kern_bd, axis=1)
        Lambda = np.log(tmp1) + V
        Lambda = Lambda - np.mean(Lambda) - 1 + np.mean(kern_bd / tmp1, axis=1)

        # Calculate number of events
        r = np.random.uniform(low=0, high=1, size=self.nParticles)
        xi = np.argwhere(r < 1 - np.exp(-np.abs(Lambda) * tau))[:, 0]
        np.random.shuffle(xi)

        # n_events = len(xi)
        n_events = 0

        # Particle jumps
        output = np.arange(self.nParticles)
        for i in xi:
            if i in alive:
                n_events += 1
                j = alive.choose_random_item()
                if Lambda[i] > 0:
                    output[i] = j
                    alive.remove_item(i)
                elif Lambda[i] < 0:
                    output[j] = i 
                    alive.remove_item(j)

        return output, n_events

    @partial(jax.jit, static_argnums=(0,))
    def proposalNoise(self, idxs, kx, UH, key):
        subkeys = jax.random.split(key, self.nParticles)
        f = lambda n, key: self._getSVN_v_stc(kx, UH, key)[n]
        return jax.vmap(f, [0, 0], 0)(idxs, subkeys)

    # @partial(jax.jit, static_argnums=(0,))
    def proposalNoise_SVGD(self, idxs, kx, key):
        subkeys = jax.random.split(key, self.nParticles)
        f = lambda n, key: self._getSVGD_v_stc(kx, key)[n]
        return jax.vmap(f, [0, 0], 0)(idxs, subkeys)

    def armijoLinesearch(self, X, v_svgd, alphas, w, mlpt, jw, step=1):
        """ 
        Linesearch procedure for SVN
        d = jnp.sum(gmlpt * v) / self.nParticles - jnp.mean(jnp.trace(jv, axis1=1, axis2=2))
        """
        beta = 1e-4

        # Step1: Ensure that the transformation is invertible
        while not np.all(jnp.linalg.det(jnp.eye(self.DoF)[jnp.newaxis] + step * jw) > 0):
            step /= 2
            print('Halfing step size: Non-invertible pushforward')

        # Step2: Ensure sufficient decrease in J
        deltaJ = lambda eps: jnp.mean(mlpt(X) - mlpt(X + eps * w) + jnp.log(jnp.linalg.det(jnp.eye(self.DoF)[jnp.newaxis] + eps * jw)))
        while not deltaJ(step) > beta * step * np.sum(v_svgd * alphas):
            step /= 2
            print('Halfing step size: Insufficient descent')

        return step 

###########################################################################

    def reg(self, mlpt, Hmlpt, lamb1, lamb2):
        a = jnp.trace(Hmlpt,axis1=1, axis2=2)
        # print(jnp.any(a>0))
        return 1 - lamb1 * (mlpt) - lamb2 * jnp.maximum(0, a)

    @partial(jax.jit, static_argnums=(0,))
    def k_lp(self, X, kernel_kwargs):
        """ 
        Lp kernel implementation
        """
        # Definitions

        p = kernel_kwargs['p'] # Order
        h = kernel_kwargs['h'] # Bandwidth
        
        # Get separation vectors
        separation_vectors = X[:, jnp.newaxis, :] - X[jnp.newaxis, :, :]
        
        # Calculate kernel
        k = jnp.exp(-jnp.sum(jnp.abs(separation_vectors) ** p, axis=-1) / (p * h))
        
        # Calculate gradient of kernel
        gk = -k[..., jnp.newaxis] * jnp.abs(separation_vectors) ** (p - 1) * jnp.sign(separation_vectors) / h
        return k, gk

    # @partial(jax.jit, static_argnums=(0,))
    def metric_wrapper(self, getKern, X, kernel_kwargs):
        """ 
        Wrapper to implement kernel metric reparameterization
        """
        metric = kernel_kwargs['M']
        if metric is not None:
            U = jnp.linalg.cholesky(metric).T
            X = contract('ij, mj -> mi', U, X)
        k, gk = getKern(X, kernel_kwargs)
        if metric is not None:
            gk = contract('ij, mnj -> mni', U.T, gk)
        return k, gk

########################
# Stackexchange class 
########################
class ListDict(object):
    """  
    Solution adapted from 
    https://stackoverflow.com/questions/15993447/python-data-structure-for-efficient-add-remove-and-random-choice
    """
    def __init__(self):
        self.item_to_position = {}
        self.items = []

    def add_item(self, item):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items)-1

    def remove_item(self, item):
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def choose_random_item(self):
        return random.choice(self.items)

    def __contains__(self, item):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

