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
config.update("jax_enable_x64", True)

log = logging.getLogger(__name__)
# log.addHandler(logging.StreamHandler(stream=sys.stdout))
np.seterr(over='raise')
np.seterr(invalid='raise')
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class samplers:
    def __init__(self, model, nIterations, nParticles, kernel_type, profile=None, bounded=None, t=None):
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

    def apply(self, kernelKwargs, method='SVGD', eps=0.1):
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
            # X = self.__newDrawFromPrior_(self.nParticles) # Initial set of particles
            X = self.model._newDrawFromPrior(self.nParticles) # Initial set of particles
            eta = self._mapHypercubeToReals(X, self.model.lower_bound, self.model.upper_bound)
            key = jax.random.PRNGKey(0)
            with trange(self.nIterations) as ITER:
                for iter_ in ITER:
                    if method == 'SVGD':
                        gmlpt, GN_Hmlpt = self.model.getDerivativesMinusLogPosterior_ensemble(X)
                        M = np.mean(GN_Hmlpt, axis=0) # None # np.eye(self.DoF)
                        kernelKwargs['M'] = M
                        kx, gkx1 = self._getKernelWithDerivatives(X, kernelKwargs)
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        X += v_svgd * eps

                    elif method == 'sSVGD':
                        gmlpt, GN_Hmlpt = self.model.getDerivativesMinusLogPosterior_ensemble(X)
                        M = np.mean(GN_Hmlpt, axis=0) # None # np.eye(self.DoF)
                        kernelKwargs['M'] = M
                        kx, gkx1 = self._getKernelWithDerivatives(X, kernelKwargs)
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        alpha, L_kx = self._getMinimumPerturbationCholesky(kx)
                        if alpha != 0:
                            kx += alpha * np.eye(self.nParticles)
                        v_stc = self._getSVGD_v_stc(L_kx)
                        X += v_svgd * eps + v_stc * np.sqrt(eps)

                    elif method == 'BDSVN':
                        gmlpt, GN_Hmlpt = self._getDerivativesMinusLogPosterior_(X)
                        M = np.mean(GN_Hmlpt, axis=0) # None
                        kx, gkx1 = self.__getKernelWithDerivatives_(X, h, M)
                        import tensorflow as tf
                        solve_method = 'CG' # 'Cholesky'
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        HBD = self._getSteinHessianBlockDiagonal(GN_Hmlpt, kx, gkx1)
                        if solve_method == 'Cholesky':
                            LHBD = tf.linalg.cholesky(HBD)
                            v_svn = np.squeeze(tf.linalg.cholesky_solve(LHBD, v_svgd[..., np.newaxis]))
                        elif solve_method == 'CG':
                            cg_maxiter = 10
                            HBDop = tf.linalg.LinearOperatorFullMatrix(HBD, is_self_adjoint=True, is_positive_definite=True)
                            v_svn = tf.linalg.experimental.conjugate_gradient(HBDop, tf.constant(v_svgd), max_iter=cg_maxiter).x.numpy()
                        X += v_svn * eps

                    elif method == 'SVN':
                        gmlpt, GN_Hmlpt = self.model.getDerivativesMinusLogPosterior_ensemble(X)
                        M = np.mean(GN_Hmlpt, axis=0) # None # np.eye(self.DoF)
                        kernelKwargs['M'] = M
                        kx, gkx1 = self._getKernelWithDerivatives(X, kernelKwargs)
                        solve_method = 'Cholesky' # 'CG'
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        H = self._getSteinHessianPosdef(GN_Hmlpt, kx, gkx1)
                        if solve_method == 'CG':
                            cg_maxiter = 50
                            alphas = scipy.sparse.linalg.cg(H, v_svgd.flatten(), maxiter=cg_maxiter)[0].reshape(self.nParticles, self.DoF)
                            v_svn = contract('xd, xn -> nd', alphas, kx)
                        elif solve_method == 'Cholesky':
                            lamb = 0.01 # 0.
                            NK = self._reshapeNNDDtoNDND(contract('mn, ij -> mnij', kx, np.eye(self.DoF)))
                            H = H + NK * lamb
                            UH = scipy.linalg.cholesky(H)
                            v_svn = self._getSVN_direction(kx, v_svgd, UH)
                        X += v_svn * eps

                    elif method == 'sSVN':
                        gmlpt, GN_Hmlpt = self.model.getDerivativesMinusLogPosterior_ensemble(X)
                        M = jnp.mean(GN_Hmlpt, axis=0) # jnp.eye(self.DoF)
                        kernelKwargs['M'] = M
                        kx, gkx1 = self.__getKernelWithDerivatives_(X, kernelKwargs)
                        NK = self._reshapeNNDDtoNDND(contract('mn, ij -> mnij', kx, jnp.eye(self.DoF)))
                        H1 = self._getSteinHessianPosdef(GN_Hmlpt, kx, gkx1)
                        lamb = 0.01 # 0.1
                        H = H1 + NK * lamb
                        UH = self.jit_scipy_cholesky(H)
                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt)
                        v_svn = self._getSVN_direction(kx, v_svgd, UH)
                        v_stc = self._getSVN_v_stc(kx, UH)
                        X += (v_svn) * eps + v_stc * np.sqrt(eps)

                        # THIS CODE IS FOR mirror SVGD MATRIX KERNELS. CLEAN UP LATER
                        # Lift to matrix kernel

                        # minv = np.linalg.inv(M)
                        # matrix_kern = contract('mn, ij -> mnij', kx, minv)
                        # grad_matrix_kern = contract('mnk, ij -> mnijk', -1 * gkx1, minv)

                    elif method == 'mirrorSVGD':
                        # Calculate derivatives
                        # gmlpt, Hmlpt = self.model.getDerivativesMinusLogPosterior_ensemble(X)
                        gmlpt = self.model.getGradientMinusLogPosterior_ensemble(X)

                        # Calculate matrix kernel
                        M = jnp.eye(self.DoF) # jnp.mean(Hmlpt, axis=0)  
                        kernelKwargs['M'] = M
                        kx_scalar, gkx1_scalar = self._getKernelWithDerivatives(X, kernelKwargs)

                        kx_matrix = contract('mn, ij -> mnij', kx_scalar, np.eye(self.DoF), backend='jax')
                        gkx2_matrix = contract('mnk, ij -> mnijk', -1 * gkx1_scalar, np.eye(self.DoF), backend='jax')

                        # Get Jacobian adjusted kernel
                        k_psi, grad_k_psi = self.getMatrixMirrorKernel(X, kx_matrix, gkx2_matrix)

                        # Calculate SVGD drift
                        v_svgd = self.getMatrixSVGD_drift(k_psi, grad_k_psi, gmlpt)

                        # Calculate SVGD noise
                        # Remark: This first method appears to be numerically unstable!
                        # K = self._reshapeNNDDtoNDND(kx_matrix) / self.nParticles
                        # v_stc = self.getMatrixSVGD_noise(K)
                        key, subkey = jax.random.split(key)
                        Bdn = jax.random.normal(subkey, (self.DoF, self.nParticles)) # standard normal sample

                        v_stc = self.getSVGD_v_stc(kx_scalar, Bdn)
                        # v_stc = self.get_vSVGD_stc(kx_scalar, Bdn)

                        # Perform dual update
                        eta += eps * v_svgd + np.sqrt(eps) * v_stc

                        # Transform back
                        X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)

                    elif method == 'mirrorSVN':                        
                        # Calculate derivatives
                        gmlpt, Hmlpt = self.model.getDerivativesMinusLogPosterior_ensemble(X)

                        # Calculate matrix kernel
                        M = jnp.mean(Hmlpt, axis=0)
                        kernelKwargs['M'] = M
                        kx_scalar, gkx1_scalar = self._getKernelWithDerivatives(X, kernelKwargs)

                        kx_matrix = contract('mn, ij -> mnij', kx_scalar, np.eye(self.DoF), backend='jax')
                        gkx2_matrix = contract('mnk, ij -> mnijk', -1 * gkx1_scalar, np.eye(self.DoF), backend='jax')

                        # Get Jacobian adjusted kernel
                        k_psi, grad_k_psi = self.getMatrixMirrorKernel(X, kx_matrix, gkx2_matrix)

                        # Calculate SVGD drift
                        v_svgd = self.getMatrixSVGD_drift(k_psi, grad_k_psi, gmlpt)

                        K = self._reshapeNNDDtoNDND(kx_matrix) / self.nParticles
                        h_psi = self.getMatrixSVN_Hessian(k_psi, grad_k_psi, Hmlpt) + 0.01 * self.nParticles * K
                        UH_psi = jax.scipy.linalg.cholesky(h_psi, lower=False)

                        # Calculate SVN noise
                        v_stc = self.getMatrixSVN_noise(K, UH_psi)

                        # Calculate drift
                        v_svn = self.getMatrixSVN_drift(UH_psi, v_svgd, K)

                        # Perform dual update
                        eta += eps * v_svn + np.sqrt(eps) * v_stc

                        # Transform Back
                        X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)

                    elif method == 'mSVGD':
                        # Calculate derivatives
                        gmlpt, Hmlpt = self.model.getDerivativesMinusLogPosterior_ensemble(X)

                        # Calculate scalar primal kernel
                        M = jnp.mean(Hmlpt, axis=0) # np.eye(self.DoF)
                        kernelKwargs['M'] = M
                        kx, gkx1 = self._getKernelWithDerivatives(X, kernelKwargs)

                        # Lift to matrix kernel
                        # matrix_kern = contract('mn, ij -> mnij', kx, np.eye(self.DoF))
                        minv = np.linalg.inv(M)
                        matrix_kern = contract('mn, ij -> mnij', kx, minv)
                        grad_matrix_kern = contract('mnk, ij -> mnijk', -1 * gkx1, minv)
                        
                        # SVGD diffusion matrix
                        K = self._reshapeNNDDtoNDND(matrix_kern) / self.nParticles

                        # SVGD Drift
                        v_svgd = self.getMatrixSVGD_v_drift(matrix_kern, grad_matrix_kern, gmlpt)

                        # SVGD thermal noise
                        v_stc = self.getMatrixSVGD_v_stc(K)

                        X += eps * v_svgd + np.sqrt(eps) * v_stc

                    elif method == 'mSVN':
                        gmlpt, Hmlpt = self.model.getDerivativesMinusLogPosterior_ensemble(X)

                        # Calculate scalar primal kernel
                        M = jnp.mean(Hmlpt, axis=0) # np.eye(self.DoF)
                        kernelKwargs['M'] = M
                        kx, gkx1 = self._getKernelWithDerivatives(X, kernelKwargs)

                        # Lift to matrix kernel (Matrix Second order)

                        minv = np.linalg.inv(M)

                        # matrix_kern = contract('mn, ij -> mnij', kx, minv)
                        # grad_matrix_kern = contract('mnk, ij -> mnijk', -1 * gkx1, minv)

                        matrix_kern = contract('mn, ij -> mnij', kx, np.eye(self.DoF))
                        grad_matrix_kern = contract('mnk, ij -> mnijk', -1 * gkx1, np.eye(self.DoF))

                        # Calculate dual update
                        v_svgd = self.getMatrixSVGD_v_drift(matrix_kern, grad_matrix_kern, gmlpt)

                        K = self._reshapeNNDDtoNDND(matrix_kern) / self.nParticles

                        H = self.getMatrixSVN_Hessian(matrix_kern, grad_matrix_kern, Hmlpt) + 0.01 * self.nParticles * K

                        UH = self.jit_scipy_cholesky(H) # Used in both v_det and v_stc 

                        v_svn = self.getMatrixSVN_v_drift(UH, v_svgd, K)

                        v_stc = self.getMatrixSVN_v_stc(K, UH)

                        X += eps * v_svn + np.sqrt(eps) * v_stc

                    elif method == 'reparam_sSVGD':
                        gmlpt_X, Hmlpt_X = self.model.getDerivativesMinusLogPosterior_ensemble(X)

                        dxdy = self._jacMapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
                        boundary_correction_grad = self._getBoundaryGradientCorrection(eta)
                        boundary_correction_hess = self._getBoundaryHessianCorrection(eta)

                        gmlpt_Y = dxdy * gmlpt_X + boundary_correction_grad
                        Hmlpt_Y = contract('Ni, Nj, Nij -> Nij', dxdy, dxdy, Hmlpt_X, backend='jax') 
                        Hmlpt_Y = Hmlpt_Y.at[:, jnp.array(range(self.DoF)), jnp.array(range(self.DoF))].add(boundary_correction_hess)

                        M = jnp.eye(self.DoF) # jnp.mean(Hmlpt_Y, axis=0) # 
                        kernelKwargs['M'] = M
                        kx, gkx1 = self.__getKernelWithDerivatives_(eta, kernelKwargs)

                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt_Y)


                        v_stc = self.get_vSVGD_stc(kx)
                        eta += (v_svgd) * eps + v_stc * np.sqrt(eps)

                        X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)


                    elif method == 'reparam_sSVN':
                        gmlpt_X, Hmlpt_X = self.model.getDerivativesMinusLogPosterior_ensemble(X)

                        dxdy = self._jacMapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
                        boundary_correction_grad = self._getBoundaryGradientCorrection(eta)
                        boundary_correction_hess = self._getBoundaryHessianCorrection(eta)

                        gmlpt_Y = dxdy * gmlpt_X + boundary_correction_grad
                        Hmlpt_Y = contract('Ni, Nj, Nij -> Nij', dxdy, dxdy, Hmlpt_X, backend='jax') 
                        Hmlpt_Y = Hmlpt_Y.at[:, jnp.array(range(self.DoF)), jnp.array(range(self.DoF))].add(boundary_correction_hess)

                        M = jnp.mean(Hmlpt_Y, axis=0) # jnp.eye(self.DoF)
                        kernelKwargs['M'] = M
                        kx, gkx1 = self.__getKernelWithDerivatives_(eta, kernelKwargs)
                        NK = self._reshapeNNDDtoNDND(contract('mn, ij -> mnij', kx, jnp.eye(self.DoF), backend='jax'))
                        H1 = self._getSteinHessianPosdef(Hmlpt_Y, kx, gkx1)
                        lamb = 0.01 # 0.1
                        H = H1 + NK * lamb
                        UH = jax.scipy.linalg.cholesky(H, lower=False)

                        v_svgd = self._getSVGD_direction(kx, gkx1, gmlpt_Y)
                        v_svn = self._getSVN_direction(kx, v_svgd, UH)

                        key, subkey = jax.random.split(key)
                        B = jax.random.normal(subkey, (self.dim,)) # standard normal sample
                        v_stc = self._getSVN_v_stc(kx, UH, B)
                        eta += (v_svn) * eps + v_stc * np.sqrt(eps)

                        X = self._mapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)




                    # Update progress bar
                    # ITER.set_description('Stepsize %f | Median bandwidth: %f | SVN norm: %f | Noise norm: %f | SVGD norm %f | Dampening %f' % (eps, self._bandwidth_MED(X), np.linalg.norm(v_svn), np.linalg.norm(v_stc), np.linalg.norm(v_svgd),  lamb))
                    ITER.set_description('Stepsize %f | Median bandwidth: %f' % (eps, self._bandwidth_MED(X)))

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
                        # g.create_dataset('gmlpt', data=copy.deepcopy(gmlpt))
                        g.create_dataset('v_svgd', data=copy.deepcopy(v_svgd))
                        # g.create_dataset('v_svn', data=copy.deepcopy(v_svn))
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

        except Exception:
            log.error("Error occurred in apply()", exc_info=True)

########################################################################################################################

    # Direction methods (either to make the main code looks clean or so we may reuse it elsewhere)
    @partial(jax.jit, static_argnums=(0,))
    def _getSVGD_direction(self, kx, gkx, gmlpt):
        """
        Get SVGD velocity field
        Args:
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            gkx (array): N x N x D array, gradient of kernel gram matrix $\nabla_1 k(z_m, z_n)$
            gmlpt (array): N x D array, gradient of minus log target

        Returns:

        """
        v_svgd = -1 * contract('mn, mo -> no', kx, gmlpt, backend='jax') / self.nParticles + jnp.mean(gkx, axis=0)
        return v_svgd

    # @partial(jax.jit, static_argnums=(0,))
    def getSVGD_v_stc(self, kx, Bdn):
        """
        Get noise injection velocity field for SVGD
        Args:
            L_kx (array): N x N array, lower triangular Cholesky factor of kernel gram matrix $k$
            Bdn (array): D x N array, standard normal noise

        Returns:

        """
        # if Bdn is None:
            # Bdn = np.random.normal(0, 1, (self.DoF, self.nParticles))
        alpha, L_kx = self._getMinimumPerturbationCholesky(kx)
        return jnp.sqrt(2 / self.nParticles) * contract('mn, in -> im', L_kx, Bdn, backend='jax').flatten(order='F').reshape(self.nParticles, self.DoF)

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
        return v_svn

    @partial(jax.jit, static_argnums=(0,))
    def _getSVN_v_stc(self, kx, UH, B):
        """
        Get noise injection velocity field for SVN
        Args:
            kx (array): N x N array, kernel gram matrix $k(z_m, z_n)$
            UH (array): ND x ND array, upper triangular Cholesky factor of SVN Hessian $H$

        Returns: (array) N x D array, noise injection for SVN

        """
        # B = np.random.normal(0, 1, self.dim)
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

    @partial(jax.jit, static_argnums=(0,))
    def _getSteinHessianPosdef(self, Hmlpt, kx, gkx):
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
        H2 = contract('xzi, xzj -> zij', gkx, gkx, backend='jax') # Only calculate block diagonal
        # H2 = contract('pni, pmj -> mnij', gkx, gkx) # calculate whole thing

        H1 = H1.at[jnp.array(range(self.nParticles)), jnp.array(range(self.nParticles))].add(H2)
        # H1[range(self.nParticles), range(self.nParticles)] += H2
        return self._reshapeNNDDtoNDND(H1 / self.nParticles)
        # return self._reshapeNNDDtoNDND((H1 + H2) / self.nParticles)


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
        return np.tanh((c * t / T) ** p)

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
        return (np.mod(t, tmp) / tmp) ** p

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


    # MODIFIED KERNEL (v1)
    # def __getKernelWithDerivatives_(self, Y, params):
    #     if self.model.priorDict is None:
    #         kx, gkx1 = self._getKernelWithDerivatives(Y, params)
    #         return (kx, gkx1)
    #     else:
    #         X = self._F_inv(Y, self.model.lower_bound, self.model.upper_bound)
    #         dF_inv = self._dF_inv(Y, self.model.lower_bound, self.model.upper_bound)
    #         k, gk1_ = self._getKernelWithDerivatives(X, params)
    #         gk1 = contract('md, mnd -> mnd', dF_inv, gk1_)
    #         return (k, gk1)

    # MODIFIED KERNEL (v2)
    # def __getKernelWithDerivatives_(self, Y, params):
    #     if self.model.priorDict is None:
    #         kx, gkx1 = self._getKernelWithDerivatives(Y, params)
    #         return (kx, gkx1)
    #     else:
    #         # Precondition wrapper
    #         U = jax.scipy.linalg.cholesky(params['M'])
    #         Y = contract('ij, Nj -> Ni', jax.scipy.linalg.cholesky(params['M']), Y)
    #         params['M'] = jnp.eye(self.DoF)

    #         # Original kernel
    #         X = self._F_inv(Y, self.model.lower_bound, self.model.upper_bound)
    #         k_X, gk1_X = self._getKernelWithDerivatives(X, params)

    #         # Quantities needed for the transformation
    #         dxdy = self._dF_inv(Y, self.model.lower_bound, self.model.upper_bound)
            
    #         det_dxdy = jnp.prod(dxdy, axis=1)
            
    #         d_dxdy = self._diagHessF_inv(Y, self.model.lower_bound, self.model.upper_bound)
            
    #         d_det_dxdy = contract('m, mj -> mj', det_dxdy, d_dxdy / dxdy)

    #         # Transformed kernel and its derivative
    #         k_Y = contract('mn, m, n -> mn', k_X, det_dxdy, det_dxdy)

    #         gk1_Y = contract('md, mnd -> mnd', dxdy, gk1_X)
    #         gk1_Y = contract('mnd, m, n -> mnd', gk1_Y, det_dxdy, det_dxdy) + contract('mn, md, n -> mnd', k_X, d_det_dxdy, det_dxdy)
    #         gk1_Y = contract('ij, mni -> mnj', U, gk1_Y) # Final multiplication

            # return (k_Y, gk1_Y)

#################################################
#################################################
#################################################

    # Standard kernel
    def __getKernelWithDerivatives_(self, Y, params):
        kx, gkx1 = self._getKernelWithDerivatives(Y, params)
        return (kx, gkx1)

    # Distance aware kernel
    # def __getKernelWithDerivatives_(self, Y, params):
    #     X = self._mapRealsToHypercube(Y, self.model.lower_bound, self.model.upper_bound)
    #     kx, gkx1 = self._getKernelWithDerivatives(X, params)
    #     dxdy = self._jacMapRealsToHypercube(Y, self.model.lower_bound, self.model.upper_bound)
    #     gkx1 = contract('mi, mni -> mni', dxdy, gkx1)
    #     return (kx, gkx1)


#################################################
#################################################
#################################################


    # def _getKernelWithDerivatives(self, X, h, M=None):
    #     # Geodesic Gaussian kernel on S1
    #     # Returns kernel and gradient of the kernel
    #     def d_circ(thetas):
    #         # Get n x n matrix of distances given vector of thetas
    #         tmp = spatial.distance_matrix(thetas, thetas)
    #         return np.minimum(tmp, 2 * np.pi - tmp)
    #
    #     d = d_circ(X)
    #     kx = np.exp(-d ** 2 / h)
    #     gkx1 = (2 * kx * d / h)[..., np.newaxis]
    #     return kx, gkx1

    # def _getGeodesicDist_sphere(self, x, y):
        # theta_1 = x[0]
        # phi_1 = x[1]
        # theta_2 = y[0]
        # phi_2 = y[1]

        # tmp1 = np.sin(theta_1) * np.sin(theta_2) * np.cos(phi_1 - phi_2) + np.cos(theta_1) * np.cos(theta_2)
        # tmp = np.sin(x[0]) * np.sin(y[0]) * np.cos(x[1] - y[1]) + np.cos(x[0]) * np.cos(y[0])
        #
        # tol = 1e-9
        # if tmp > 1 and tmp < 1 + tol:
        #     tmp = 1
        # elif tmp < 0 and tmp > -tol:
        #     tmp = 0
        #
        # return np.arccos(tmp)

    # def _getKernelWithDerivatives(self, X, h, M=None):
    #     # d = metrics.pairwise_distances(X, metric=self._getGeodesicDist_sphere)
    #
    #     gd = np.zeros((self.nParticles, self.nParticles, 2))
    #
    #     tmp0 = lambda x, y: np.sin(x[1]) * np.sin(y[1]) * np.cos(x[0] - y[0]) + np.cos(x[1]) * np.cos(y[1]) # X
    #     tmp1 = lambda x, y: np.sin(x[1]) * np.sin(y[1]) * np.sin(x[0] - y[0])
    #     tmp2 = lambda x, y: np.sin(x[1]) * np.cos(y[1]) - np.cos(x[1]) * np.sin(y[1]) * np.cos(x[0] - y[0]) #
    #
    #     eta = metrics.pairwise_distances(X, metric=tmp0)
    #     eta[range(self.nParticles), range(self.nParticles)] = np.zeros(self.nParticles) # Set to analytic value to avoid numerical out of support
    #
    #     d = np.arccos(eta)
    #     kx = np.exp(-d ** 2 / h)
    #
    #     # temp = 1 - eta ** 2
    #
    #     denominator = np.sqrt(1 - eta ** 2)
    #
    #     a = metrics.pairwise_distances(X, metric=tmp1) / denominator
    #     b = metrics.pairwise_distances(X, metric=tmp2) / denominator
    #
    #     gd[..., 0] = a
    #     gd[..., 1] = b
    #
    #     gkx1 = -2 * contract('mn, mni -> mni', kx * d, gd) / h
    #
    #     return kx, gkx1




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



#######################################################




    # def get_mirror_kernel_new(self, X, kx, gkx2):
    #     a = self.model.lower_bound
    #     b = self.model.upper_bound
        
    #     tmp1 = (b - X) * (X - a) / (b - a)
    #     tmp2 = (a + b - 2 * X) / (b - a)

    #     hess_psi_inv = contract('mi, ij -> mij', tmp1, jnp.eye(self.DoF))        
    #     k_psi = contract('xy, yij -> xyij', kx, hess_psi_inv) 

    #     grad_k_psi = np.zeros((self.nParticles, self.nParticles, self.DoF, self.DoF, self.DoF))
    #     grad_k_psi[:, :, range(self.DoF), range(self.DoF), ...] += contract('mnk, nj -> mnjk', gkx2, tmp1)
    #     grad_k_psi[..., range(self.DoF), range(self.DoF), range(self.DoF)] += contract('mn, nj -> mnj', kx, tmp2)

    #     return k_psi, grad_k_psi





    #####################################################
    # SVN CODE
    #####################################################







    ###################################################
    # 1/24/23 updates
    ###################################################

    # def hess_psi_inv(self, X): # Checks: X
    #     a = self.model.lower_bound
    #     b = self.model.upper_bound
    #     tmp = (b - X) * (X - a) / (b - a)
    #     return contract('mi, ij -> mij', tmp, jnp.eye(self.DoF))
    #     # return tmp


    # def grad_hess_psi_inv(self, X, hess_psi_inv): # Checks: X
    #     a = self.model.lower_bound
    #     b = self.model.upper_bound
    #     tmp = (a - b) * (a + b - 2 * X) / ((a - X) ** 2 * (b - X) ** 2)
    #     # tmp = (self.model.lower_bound - self.model.upper_bound) * (self.model.lower_bound + self.model.upper_bound - 2 * X) / ((self.model.lower_bound - X) ** 2 * (self.model.upper_bound - X) ** 2)
    #     return -1 * contract('yaj, yj, yji -> yaij', hess_psi_inv, tmp, hess_psi_inv)
    #     # return -1 * hess_psi_inv * tmp * hess_psi_inv
    #     # return -1 * contract('mi, mi, mi -> yaij', hess_psi_inv, tmp, hess_psi_inv)


    # def getMirrorKernelWithDerivative(self, X, kx, gkx2): # Checks: X
    #     hess_psi_inv = self.hess_psi_inv(X)
    #     grad_hess_psi_inv = self.grad_hess_psi_inv(X, hess_psi_inv)
    #     k_psi = contract('xy, yij -> xyij', kx, hess_psi_inv) # 
    #     grad_k_psi = contract('xyj, yai -> xyaij', gkx2, hess_psi_inv) \
    #                + contract('xy, yaij -> xyaij', kx, grad_hess_psi_inv)
    #     return k_psi, grad_k_psi



###################################
# Alternative metric test
###################################
# dxdy = self._jacMapRealsToHypercube(eta, self.model.lower_bound, self.model.upper_bound)
# M = contract('Ni, Nj, Nij -> ij', dxdy, dxdy, Hmlpt, backend='jax') / self.nParticles 
