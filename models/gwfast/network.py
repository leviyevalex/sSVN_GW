#
#    Copyright (c) 2022 Francesco Iacovelli <francesco.iacovelli@unige.ch>, Michele Mancarella <michele.mancarella@unige.ch>
#
#    All rights reserved. Use of this source code is governed by the
#    license that can be found in the LICENSE file.

import os
import jax
#Enable 64bit on JAX, fundamental
from jax.config import config
config.update("jax_enable_x64", True)
#config.update("TF_CPP_MIN_LOG_LEVEL", 0)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_ALLOCATOR']='platform'

import numpy as onp
import jax.numpy as np
from jax.interpreters import xla
from jax import pmap, vmap, jacrev, jit, jacfwd

import copy

# from gwfast_light.gwfast import gwfastUtils as utils
from models.gwfast import gwfastUtils as utils

from functools import partial


class DetNet(object):
    """
    Class to build a network of multiple detectors.
    
    The functions defined within this class allow to get e.g. the *network* SNR and Fisher matrix.
    
    :param dict(GWSignal, ...) signals: Dictionary containing one or multiple individual detector objects.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    
    """
    def __init__(self, signals, verbose=True, fixed_fgrid=None, wf_model=None):
        """
        Constructor method
        """
        # signals is a dictionary of the form
        # {'detector_name': GWSignal object }
        
        self.signals = signals
        self.verbose=verbose
        if fixed_fgrid is not None:
            self.fixed_fgrid=fixed_fgrid
        else:
            self.fixed_fgrid=np.linspace(10., 2048., num=2000)
        if wf_model is not None:
            self.wf_model = wf_model
        else:
            self.wf_model = self.signals[list(self.signals.keys())[0]].wf_model
        self.signals_list = list(self.signals.keys())
    
    def SNR(self, evParams, res=1000):
        """
        Compute the *network signal-to-noise-ratio*, SNR, as a function of the parameters of the event(s).
        
        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param bool, optional return_all: Boolean specifying if the SNRs of the individual detectors have to be returned separately, together with the network SNR(s). In this case the return type is *dict(array, array, ...)*.
        
        :return: Network SNR(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm events})`.
        :rtype: 1-D array
        
        """
        totSNR = np.zeros_like(evParams['Mc'])

        fcut = self.wf_model.fcut(**evParams)
        fcut = np.where(fcut > self.signals[list(self.signals.keys())[0]].fmax, self.signals[list(self.signals.keys())[0]].fmax, fcut)
        fminarr = np.full(fcut.shape, self.signals[list(self.signals.keys())[0]].fmin)
        
        fgrids = np.geomspace(fminarr,fcut,num=int(res))

        wfAmpl = self.wf_model.Ampl(fgrids, **evParams)
        Ap = wfAmpl*0.5*(1.+(np.cos(evParams['iota']))**2)
        Ac = wfAmpl*np.cos(evParams['iota'])
        
        def tmpSNR(i, totSNR):
            strainGrids = np.interp(fgrids, list(self.signals.values())[i].strainFreq, list(self.signals.values())[i].noiseCurve, left=1., right=1.)
            Fp, Fc = list(self.signals.values())[i]._PatternFunction(evParams['theta'], evParams['phi'], evParams['tcoal'], evParams['psi'], rot=0.)
                
            Atot = Ap*Ap*Fp*Fp + Ac*Ac*Fc*Fc
            return totSNR + 4.*np.trapz(Atot/strainGrids, fgrids, axis=0)
        '''
        def tmpSNR(i, totSNR):
            strainGrids = np.interp(fgrids, list(self.signals.values())[i].strainFreq, list(self.signals.values())[i].noiseCurve, left=1., right=1.)
            Fp, Fc = list(self.signals.values())[i]._PatternFunction(evParams['theta'], evParams['phi'], evParams['tcoal'], evParams['psi'], rot=0.)
                
            Atot = Ap*Ap*Fp*Fp + Ac*Ac*Fc*Fc
            return totSNR + 4.*np.trapz(Atot/strainGrids, fgrids, axis=0), totSNR + 4.*np.trapz(Atot/strainGrids, fgrids, axis=0)
        '''
        for i in range(len(self.signals.keys())):
            totSNR = tmpSNR(i, totSNR)
        #totSNR = jax.lax.fori_loop(0, len(self.signals.keys()), tmpSNR, totSNR)
        #totSNR = jax.lax.scan(tmpSNR, totSNR, np.arange(len(self.signals.keys())).astype(int))

        net_snr = np.sqrt(totSNR)

        return net_snr 
        
    def FisherMatr(self, evParams):
        """
        Compute the *Fisher information matrix*, FIM, as a function of the parameters of the event(s) for a network of detectors. In this case the derivative with respect to the intrinsic parameters is computed only once, resulting in a faster evaluation.
        
        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :return: FIM(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
        :rtype: 3-D array
        
        """
        
        McOr, dL, theta, phi = evParams['Mc'].astype('complex128'), evParams['dL'].astype('complex128'), evParams['theta'].astype('complex128'), evParams['phi'].astype('complex128')
        iota, psi, tcoal, etaOr, Phicoal = evParams['iota'].astype('complex128'), evParams['psi'].astype('complex128'), evParams['tcoal'].astype('complex128'), evParams['eta'].astype('complex128'), evParams['Phicoal'].astype('complex128')
        chi1z, chi2z = evParams['chi1z'].astype('complex128'), evParams['chi2z'].astype('complex128')
            
        nParams = self.wf_model.nParams
        
        tcelem = self.wf_model.ParNums['tcoal']
        fgrids = np.repeat(self.fixed_fgrid, McOr.shape[0]).reshape((self.fixed_fgrid.shape[0], McOr.shape[0]))

        FisherDerivs = self._SignalDerivatives(McOr, etaOr, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z)
        Fisher = np.zeros((nParams,nParams,len(McOr)))
        def tmpFisher(d):
            strainGrids = np.interp(fgrids, self.signals[d].strainFreq, self.signals[d].noiseCurve, left=1., right=1.)
            FisherIntegrands = (np.conjugate(FisherDerivs[d][:,:,np.newaxis,:])*FisherDerivs[d].transpose(1,0,2))
            Fisher = np.zeros((nParams,nParams,len(McOr)))

            Fisher = jax.lax.fori_loop(0, nParams, lambda alpha, Fisher: jax.lax.fori_loop(alpha, nParams, lambda beta, Fisher: Fisher.at[alpha,beta,:].add(np.trapz(FisherIntegrands[alpha,:,beta,:].T.real/strainGrids.real, fgrids, axis=0)*4.), Fisher), Fisher)
            Fisher = jax.lax.fori_loop(0, nParams, lambda alpha, Fisher: jax.lax.fori_loop(alpha, nParams, lambda beta, Fisher: Fisher.at[beta,alpha,:].set(Fisher[alpha,beta,:]), Fisher), Fisher)
            
            return Fisher
        
        for d in self.signals.keys():
            FisherDerivs.at[tcelem].divide(3600.*24.)
            Fisher = Fisher + tmpFisher(d)
        #Fisher = jax.lax.fori_loop(0, len(self.signals.keys()), lambda i, Fisher: Fisher + tmpFisher(list(self.signals.keys())[i]), Fisher)
            
        return Fisher

    partial(jit, static_argnums=(0,))
    def _SignalDerivatives(self, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z):
        """
        Compute the derivatives of the GW strain with respect to the parameters of the event(s) at given frequencies (in :math:`\\rm Hz`) for a network of detectors at fixed frequency grid. In this case the derivative with respect to the intrinsic parameters is computed only once, resulting in a faster evaluation, and less memory is allocated thanks to the fixed grid.
        
        :param array or float Mc: The chirp mass(es), :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`. 
        :param array or float eta:  The symmetric mass ratio(s), :math:`\eta`.
        :param array or float dL: The luminosity distance(s), :math:`d_L`, in :math:`\\rm Gpc`.
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float iota: The inclination angle(s), with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`. 
        :param array or float psi: The polarisation angle(s), :math:`\psi`, in :math:`\\rm rad`.
        :param array or float tcoal: The time(s) of coalescence, :math:`t_{\\rm coal}`, as a GMST.
        :param array or float Phicoal: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
        :param array or float chi1z: The spin component(s) of the primary object(s) along the axis :math:`z`, :math:`\chi_{1,z}`.
        :param array or float chi2z: The spin component(s) of the secondary object(s) along the axis :math:`z`, :math:`\chi_{2,z}`.
        :return: Complete signal strain derivatives (complex), evaluated at the given parameters and frequency(ies).
        :rtype: array
        
        """
        
        derivargs = (0,1,4,5)
            
        def wfcall(Mc, eta, dL, iota, chi1z, chi2z):
            f = self.fixed_fgrid
            evParams = {'Mc':Mc, 'eta':eta, 'chi1z':chi1z, 'chi2z':chi2z, 'dL':dL}
            
            AmpWF = self.wf_model.Ampl(f, **evParams)
            #print(AmpWF.shape)
            phaseWF = self.wf_model.Phi(f, **evParams)
            #print((AmpWF*np.exp(1j*(-phaseWF))*0.5*(1.+(np.cos(iota))**2)).shape)
            return AmpWF*np.exp(1j*(-phaseWF))*0.5*(1.+(np.cos(iota))**2), 1j*AmpWF*np.exp(1j*(-phaseWF))*np.cos(iota)
            
        def wf_derivative_holo(Mc, eta, dL, iota, chi1z, chi2z):
            GWstrainUse = lambda Mc, eta, dL, iota, chi1z, chi2z: wfcall(Mc, eta, dL, iota, chi1z, chi2z)
            FisherDerivs_tot =  np.asarray(vmap(jacfwd(GWstrainUse, argnums=derivargs, holomorphic=True))(Mc, eta, dL, iota, chi1z, chi2z))
            return FisherDerivs_tot[0], FisherDerivs_tot[1]
        
        def wf_derivative_nonholo(Mc, eta, dL, iota, chi1z, chi2z):
            Mc, eta, dL, iota, chi1z, chi2z = np.real(Mc), np.real(eta), np.real(dL), np.real(iota), np.real(chi1z), np.real(chi2z)
            
            GWstrainUse_real = lambda Mc, eta, dL, iota, chi1z, chi2z: np.real(np.asarray(wfcall(Mc, eta, dL, iota, chi1z, chi2z)))
            GWstrainUse_imag = lambda Mc, eta, dL, iota, chi1z, chi2z: np.imag(np.asarray(wfcall(Mc, eta, dL, iota, chi1z, chi2z)))
            
            realDerivs = np.asarray(vmap(jacfwd(GWstrainUse_real, argnums=derivargs))(Mc, eta, dL, iota, chi1z, chi2z))
            imagDerivs = np.asarray(vmap(jacfwd(GWstrainUse_imag, argnums=derivargs))(Mc, eta, dL, iota, chi1z, chi2z))

            realDerivs_p, imagDerivs_p = realDerivs[:,:,0], imagDerivs[:,:,0]
            realDerivs_c, imagDerivs_c = realDerivs[:,:,1], imagDerivs[:,:,1]

            return realDerivs_p + 1j*imagDerivs_p, realDerivs_c + 1j*imagDerivs_c
        
        FisherDerivs_p, FisherDerivs_c = jax.lax.cond(self.wf_model.is_holomorphic, lambda x: wf_derivative_holo(*x), lambda x: wf_derivative_nonholo(*x), (Mc, eta, dL, iota, chi1z, chi2z))
        
        fgrids = np.repeat(self.fixed_fgrid, Mc.shape[0]).reshape((self.fixed_fgrid.shape[0], Mc.shape[0]))

        allDerivs_Fish = {}

        def singleFish(d):
            t = tcoal*np.ones_like(fgrids)
            tmpDeltLoc = self.signals[d]._DeltLoc(theta, phi, t)
            tmpDeltLoc = self.signals[d]._DeltLoc(theta, phi, t) # in seconds
            t = t + tmpDeltLoc/(3600.*24.)

            phiL = (2.*np.pi*fgrids)*tmpDeltLoc
            Fp, Fc = self.signals[d]._PatternFunction(theta, phi, t, psi, rot=0.)

            FisherDerivs_p_tmp = FisherDerivs_p.transpose(0,2,1)*Fp*np.exp(1j*(phiL + 2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal))
            FisherDerivs_c_tmp = FisherDerivs_c.transpose(0,2,1)*Fc*np.exp(1j*(phiL + 2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal))

            FisherDerivs = (FisherDerivs_p_tmp + FisherDerivs_c_tmp).transpose(0,2,1)
    
            dL_deriv, theta_deriv, phi_deriv, iota_deriv, psi_deriv, tc_deriv, Phicoal_deriv = self.signals[d]._AnalyticalDerivatives(Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z)
            FisherDerivs = np.vstack((FisherDerivs, np.asarray(dL_deriv).T[np.newaxis,:], np.asarray(theta_deriv).T[np.newaxis,:], np.asarray(phi_deriv).T[np.newaxis,:], np.asarray(iota_deriv).T[np.newaxis,:], np.asarray(psi_deriv).T[np.newaxis,:], np.asarray(tc_deriv).T[np.newaxis,:], np.asarray(Phicoal_deriv).T[np.newaxis,:]))
            
            #allDerivs_Fish[d] = FisherDerivs

            return FisherDerivs
        
        #allDerivs_Fish = jax.lax.fori_loop(0, len(self.signals.keys()), lambda i, allDerivs_Fish: allDerivs_Fish + singleFish(list(self.signals.keys())[i]), allDerivs_Fish)
        for d in self.signals.keys():
            allDerivs_Fish[d] = singleFish(d)

        return allDerivs_Fish
            
