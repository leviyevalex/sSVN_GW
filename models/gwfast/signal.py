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


# We use both the original numpy, denoted as onp, and the JAX implementation of numpy, denoted as np
import numpy as onp
import jax.numpy as np
from jax import vmap, jacrev

# from gwfast_light.gwfast import gwfastGlobals as glob
from models.gwfast import gwfastGlobals as glob

class GWSignal(object):
    """
    Class to compute the GW signal emitted by a coalescing binary system as seen by a detector on Earth.
    
    The functions defined within this class allow to get e.g. the amplitude of the signal, its phase, SNR and Fisher matrix elements.
    
    :param WaveFormModel wf_model: Object containing the waveform model.
    :param str psd_path: Full path to the file containing the detector's *Power Spectral Density*, PSD, or *Amplitude Spectral Density*, ASD, including the file extension. The file is assumed to have two columns, the first containing the frequencies (in :math:`\\rm Hz`) and the second containing the detector's PSD/ASD at each frequency.
    :param str detector_shape: The shape of the detector, to be chosen among ``'L'`` for an L-shaped detector (90°-arms) and ``'T'`` for a triangular detector (3 nested detectors with 60°-arms).
    :param float det_lat: Latitude of the detector, in degrees.
    :param float det_long: Longitude of the detector, in degrees.
    :param float det_xax: Angle between the bisector of the detector's arms (the first detector in the case of a triangle) and local East, in degrees.
    :param bool, optional verbose: Boolean specifying if the code has to print additional details during execution.
    :param bool, optional is_ASD: Boolean specifying if the provided file is a PSD or an ASD.
    :param float fmin: Minimum frequency to use for the grid in the analysis, in :math:`\\rm Hz`.
    :param float fmax: Maximum frequency to use for the grid in the analysis, in :math:`\\rm Hz`. The cut frequency of the waveform (which depends on the events parameters) will be used as maximum frequency if ``fmax=None`` or if it is smaller than ``fmax``.
    :param array or float fixed_fgrid: Frequency grid to use for the analysis, in :math:`\\rm Hz`. 
    """
    '''
    Inputs are an object containing the waveform model, the coordinates of the detector (latitude and longitude in deg),
    its shape (L or T), the angle with respect to East of the bisector of the arms (deg)
    and its ASD or PSD (given in a .txt file containing two columns: one with the frequencies and one with the ASD or PSD values,
    remember ASD=sqrt(PSD))
    
    '''
    def __init__(self, wf_model,
                psd_path=None,
                detector_shape = 'L',
                det_lat=40.44,
                det_long=9.45,
                det_xax=0.,
                verbose=True,
                is_ASD=True,
                fmin=10., fmax=2048.,
                fixed_fgrid=None
                ):
        """
        Constructor method
        """
        if (detector_shape!='L') and (detector_shape!='T'):
            raise ValueError('Enter valid detector configuration')
        
        if psd_path is None:
            raise ValueError('Enter a valid PSD or ASD path')
        
        if verbose:
            if not is_ASD:
                print('Using PSD from file %s ' %psd_path)
            else:
                print('Using ASD from file %s ' %psd_path)
        
        self.wf_model = wf_model
        
        self.psd_base_path = ('/').join(psd_path.split('/')[:-1])
        self.psd_file_name = psd_path.split('/')[-1]
 
        self.verbose = verbose
        self.detector_shape = detector_shape
        
        self.det_lat_rad  = det_lat*np.pi/180.
        self.det_long_rad = det_long*np.pi/180.
        
        self.det_xax_rad  = det_xax*np.pi/180.
        
        noise = onp.loadtxt(psd_path, usecols=(0,1))
        f = noise[:,0]
        if is_ASD:
            S = (noise[:,1])**2
        else:
            S = noise[:,1]
        
        self.strainFreq = f
        self.noiseCurve = S
        
        self.fmin = fmin #Hz
        self.fmax = fmax #Hz or None
        
        if detector_shape == 'L':
            self.angbtwArms = 0.5*np.pi
        elif detector_shape == 'T':
            self.angbtwArms = np.pi/3.
        
        if fixed_fgrid is not None:
            self.fixed_fgrid=fixed_fgrid
        else:
            self.fixed_fgrid=np.linspace(10., 2048., num=2000)
        
    def _ra_dec_from_th_phi(self, theta, phi):
        ra = phi 
        dec = 0.5*np.pi - theta
        return ra, dec
        
    def _PatternFunction(self, theta, phi, t, psi, rot=0.):
        """
        Compute the value of the so-called pattern functions of the detector for a set of sky coordinates, GW polarisation(s) and time(s).
        
        For the definition of the pattern functions see `arXiv:gr-qc/9804014 <https://arxiv.org/abs/gr-qc/9804014>`_ eq. (10)--(13).
        
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float t: The time(s) given as GMST.
        :param array or float psi: The GW polarisation angle(s) :math:`\psi`, in :math:`\\rm rad`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry. In this case, the three arms will have orientations 1 --> :py:data:`self.xax`, 2 --> :py:data:`self.xax` + 60°, 3 --> :py:data:`self.xax` + 120°.
        :return: Plus and cross pattern functions of the detector evaluated at the given parameters.
        :rtype: tuple(array, array) or tuple(float, float)
        
        """
        # See P. Jaranowski, A. Krolak, B. F. Schutz, PRD 58, 063001, eq. (10)--(13)
        
    
        def afun(ra, dec, t, rot):
            phir = self.det_long_rad
            a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(3.-np.cos(2.*dec))*np.cos(2.*(ra - phir - 2.*np.pi*t))
            a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*np.sin(2.*(ra - phir - 2.*np.pi*t))
            a3 = 0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)
            a4 = 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)
            a5 = 3.*0.25*np.sin(2*(self.det_xax_rad+rot))*(np.cos(self.det_lat_rad)*np.cos(dec))**2.
            return a1 - a2 + a3 - a4 + a5
        
        def bfun(ra, dec, t, rot):
            phir = self.det_long_rad
            b1 = np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*np.cos(2.*(ra - phir - 2.*np.pi*t))
            b2 = 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*np.sin(2.*(ra - phir - 2.*np.pi*t))
            b3 = np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)
            b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)
            
            return b1 + b2 + b3 + b4
        
        rot_rad = rot*np.pi/180.
        
        ras, decs = self._ra_dec_from_th_phi(theta, phi)
        afac = afun(ras, decs, t, rot_rad)
        bfac = bfun(ras, decs, t, rot_rad)
        
        Fp = np.sin(self.angbtwArms)*(afac*np.cos(2.*psi) + bfac*np.sin(2*psi))
        Fc = np.sin(self.angbtwArms)*(bfac*np.cos(2.*psi) - afac*np.sin(2*psi))
        
        return Fp, Fc
    
    def _DeltLoc(self, theta, phi, t):
        """
        Compute the time needed to go from Earth center to detector location for a set of sky coordinates and time(s). The result is given in seconds.
        
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float t: The time(s) given as GMST.
        
        :return: Time shift(s) to go from Earth center to detector location.
        :rtype: array or float
        
        """
        # Time needed to go from Earth center to detector location
        
        ras, decs = self._ra_dec_from_th_phi(theta, phi)
        
        comp1 = np.cos(decs)*np.cos(ras)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)
        comp2 = np.cos(decs)*np.sin(ras)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)
        comp3 = np.sin(decs)*np.sin(self.det_lat_rad)
        # The minus sign arises from the definition of the unit vector pointing to the source
        Delt = - glob.REarth*(comp1+comp2+comp3)/glob.clight
        
        return Delt # in seconds
    
    def GWAmplitudes(self, evParams, f, rot=0.):
        """
        Compute the amplitude of the signal(s) as seen by the detector, as a function of the parameters, at given frequencies.
        
        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param float rot: Further rotation of the interferometer with respect to the :py:data:`self.xax` orientation, in degrees, needed for the triangular geometry.
        :return: Plus and cross amplitudes at the detector, evaluated at the given parameters and frequency(ies).
        :rtype: tuple(array, array) or tuple(float, float)
        
        """
        # evParams are all the parameters characterizing the event(s) under exam. It has to be a dictionary containing the entries:
        # Mc -> chirp mass (Msun), dL -> luminosity distance (Gpc), theta & phi -> sky position (rad), iota -> inclination angle of orbital angular momentum to l.o.s toward the detector,
        # psi -> polarisation angle, tcoal -> time of coalescence as GMST (fraction of days), eta -> symmetric mass ratio, Phicoal -> GW frequency at coalescence.
        # chi1z, chi2z -> dimensionless spin components aligned to orbital angular momentum [-1;1], Lambda1,2 -> tidal parameters of the objects,
        # f is the frequency (Hz)
        
        theta, phi, iota, psi, tcoal = evParams['theta'], evParams['phi'], evParams['iota'], evParams['psi'], evParams['tcoal']
        
        t = tcoal 
        t = t + self._DeltLoc(theta, phi, t)/(3600.*24.)
        # wfAmpl = self.wf_model.Ampl(f, **evParams)
        Fp, Fc = self._PatternFunction(theta, phi, t, psi, rot=rot)
        
        wfAmpl = self.wf_model.Ampl(f, **evParams)
        Ap = wfAmpl*Fp*0.5*(1.+(np.cos(iota))**2)
        Ac = wfAmpl*Fc*np.cos(iota)
        
        return Ap, Ac
    
    def GWPhase(self, evParams, f):
        """
        Compute the complete phase of the signal(s), as a function of the parameters, at given frequencies.
        
        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        
        :return: Complete signal phase, evaluated at the given parameters and frequency(ies).
        :rtype: array or float
        
        """
        # Phase of the GW signal
        tcoal, Phicoal =  evParams['tcoal'], evParams['Phicoal']
        PhiGw = self.wf_model.Phi(f, **evParams)

        return 2.*np.pi*f*(tcoal*3600.*24.) - Phicoal - PhiGw

    def GWstrain(self, f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, rot=0.):
        """
        Compute the full GW strain (complex) as a function of the parameters, at given frequencies.
        
        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
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
        
        :return: Complete signal strain (complex), evaluated at the given parameters and frequency(ies).
        :rtype: array or float
        
        """
        # Full GW strain expression (complex)
        # Here we have the decompressed parameters and we put them back in a dictionary just to have an easier
        # implementation of the JAX module for derivatives
        
        McUse  = Mc
        etaUse = eta
            
        evParams = {'Mc':McUse, 'eta':etaUse, 'dL':dL, 'theta':theta, 'phi':phi, 'iota':iota, 'psi':psi, 'tcoal':tcoal, 'Phicoal':Phicoal, 'chi1z':chi1z, 'chi2z':chi2z}
            
        tmpDeltLoc = self._DeltLoc(theta, phi, t) # in seconds
        t = t + tmpDeltLoc/(3600.*24.)
        
        phiL = (2.*np.pi*f)*tmpDeltLoc
        Ap, Ac = self.GWAmplitudes(evParams, f, rot=rot)
        Psi = self.GWPhase(evParams, f)
        Psi = Psi + phiL 
            
        return (Ap + 1j*Ac)*np.exp(Psi*1j)
    
    def SNRInteg(self, evParams, res=1000):
        """
        Compute the *signal-to-noise-ratio*, SNR, as a function of the parameters of the event(s).
        
        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :param int res: The resolution of the frequency grid to use.
        :param bool, optional return_all: Boolean specifying if, in the case of a triangular detector, the SNRs of the individual instruments have to be returned separately. In this case the return type is *list(array, array, array)*.
        
        :return: SNR(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm events})`.
        :rtype: 1-D array
        
        """
                            
        fcut = self.wf_model.fcut(**evParams)
        fcut = np.where(fcut > self.fmax, self.fmax, fcut)
        fminarr = np.full(fcut.shape, self.fmin)
        
        fgrids = np.geomspace(fminarr,fcut,num=int(res))
        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
        strainGrids = np.interp(fgrids, self.strainFreq, self.noiseCurve, left=1., right=1.)
        
        Aps, Acs = self.GWAmplitudes(evParams, fgrids)
        Atot = Aps*Aps + Acs*Acs
        SNRsq = np.trapz(Atot/strainGrids, fgrids, axis=0)
        
        return 2*np.sqrt(SNRsq)
            
    def FisherMatr_fixedgrid(self, evParams):
        """
        Compute the *Fisher information matrix*, FIM, as a function of the parameters of the event(s).
        
        :param dict(array, array, ...) evParams: Dictionary containing the parameters of the event(s), as in :py:data:`events`.
        :return: FIM(s) as a function of the parameters of the event(s). The shape is :math:`(N_{\\rm parameters}`, :math:`N_{\\rm parameters}`, :math:`N_{\\rm events})`.
        :rtype: 3-D array
        
        """
        # If use_m1m2=True the Fisher is computed w.r.t. m1 and m2, not Mc and eta
        # If use_chi1chi2=True the Fisher is computed w.r.t. chi1z and chi2z, not chiS and chiA
        
        McOr, dL, theta, phi = evParams['Mc'].astype('complex128'), evParams['dL'].astype('complex128'), evParams['theta'].astype('complex128'), evParams['phi'].astype('complex128')
        iota, psi, tcoal, etaOr, Phicoal = evParams['iota'].astype('complex128'), evParams['psi'].astype('complex128'), evParams['tcoal'].astype('complex128'), evParams['eta'].astype('complex128'), evParams['Phicoal'].astype('complex128')
        chi1z, chi2z = evParams['chi1z'].astype('complex128'), evParams['chi2z'].astype('complex128')
            
        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
        strainGrids = np.interp(self.fixed_fgrid, self.strainFreq, self.noiseCurve, left=1., right=1.)

        nParams = self.wf_model.nParams
        
        tcelem = self.wf_model.ParNums['tcoal']
        
        FisherDerivs = self._SignalDerivatives_use(McOr, etaOr, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z)
        FisherDerivs.at[tcelem].divide(3600.*24.)

        FisherIntegrands = (np.conjugate(FisherDerivs[:,:,np.newaxis,:])*FisherDerivs.transpose(1,0,2))
    
        Fisher = np.zeros((nParams,nParams,len(McOr)))
        for alpha in range(nParams):
            for beta in range(alpha,nParams):
                tmpElem = FisherIntegrands[alpha,:,beta,:].T
                Fisher[alpha,beta, :] = np.trapz(tmpElem.real/strainGrids.real, self.fixed_fgrid.real, axis=0)*4.

                Fisher[beta,alpha, :] = Fisher[alpha,beta, :]
            
        return Fisher
    
    def _SignalDerivatives(self, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, rot=0.):
        """
        Compute the derivatives of the GW strain with respect to the parameters of the event(s) at given frequencies (in :math:`\\rm Hz`).
        
        :param array or float fgrids: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param array or float Mc: The chirp mass(es), :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`. If ``use_m1m2=True`` this is interpreted as the primary mass, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
        :param array or float eta:  The symmetric mass ratio(s), :math:`\eta`. If ``use_m1m2=True`` this is interpreted as the secondary mass, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
        :param array or float dL: The luminosity distance(s), :math:`d_L`, in :math:`\\rm Gpc`.
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float iota: The inclination angle(s), with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`. If ``is_prec_ang=True`` this is interpreted as the inclination angle(s) with respect to total angular momentum, :math:`\\theta_{JN}`, in :math:`\\rm rad`.
        :param array or float psi: The polarisation angle(s), :math:`\psi`, in :math:`\\rm rad`.
        :param array or float tcoal: The time(s) of coalescence, :math:`t_{\\rm coal}`, as a GMST.
        :param array or float Phicoal: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
        :param array or float chiS: The symmetric spin component(s), :math:`\chi_s`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the primary object(s) along the axis :math:`z`, :math:`\chi_{1,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the primary object(s), :math:`\chi_1`.
        :param array or float chiA: The antisymmetric spin component(s) :math:`\chi_a`. If :py:class:`self.wf_model` is precessing or ``use_chi1chi2=True`` this is interpreted as the spin component(s) of the secondary object(s) along the axis :math:`z`, :math:`\chi_{2,z}`. If ``use_prec_ang=True`` this is interpreted as the spin magnitude(s) of the secondary object(s), :math:`\chi_2`.
        
        :return: Complete signal strain derivatives (complex), evaluated at the given parameters and frequency(ies).
        :rtype: array
        
        """
        
        derivargs = (1,2,10,11)
        
        def wf_derivative_holo(Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z):
            GWstrainUse = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z: self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, rot=rot)
            return np.asarray(vmap(jacrev(GWstrainUse, argnums=derivargs, holomorphic=True))(self.fixed_fgrid, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, rot=rot))
        
        def wf_derivative_nonholo(Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z):
            Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z = np.real(Mc), np.real(eta), np.real(dL), np.real(theta), np.real(phi), np.real(iota), np.real(psi), np.real(tcoal), np.real(Phicoal), np.real(chi1z), np.real(chi2z)
            
            GWstrainUse_real = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z: np.real(self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, rot=rot))
            GWstrainUse_imag = lambda f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z: np.imag(self.GWstrain(f, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, rot=rot))
            
            realDerivs = np.asarray(vmap(jacrev(GWstrainUse_real, argnums=derivargs))(self.fixed_fgrid, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z))
            imagDerivs = np.asarray(vmap(jacrev(GWstrainUse_imag, argnums=derivargs))(self.fixed_fgrid, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z))

            return realDerivs + 1j*imagDerivs
        
        FisherDerivs = jax.lax.cond(self.wf_model.is_holomorphic, lambda x: wf_derivative_holo(*x), lambda x: wf_derivative_nonholo(*x), (Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z))
                
        dL_deriv, theta_deriv, phi_deriv, iota_deriv, psi_deriv, tc_deriv, Phicoal_deriv = self._AnalyticalDerivatives(Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, rot=rot)
        FisherDerivs = np.vstack((FisherDerivs, np.asarray(dL_deriv).T[np.newaxis,:], np.asarray(theta_deriv).T[np.newaxis,:], np.asarray(phi_deriv).T[np.newaxis,:], np.asarray(iota_deriv).T[np.newaxis,:], np.asarray(psi_deriv).T[np.newaxis,:], np.asarray(tc_deriv).T[np.newaxis,:], np.asarray(Phicoal_deriv).T[np.newaxis,:]))    
        
        return FisherDerivs
        
    def _AnalyticalDerivatives(self, Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, rot=0.):
        """
        Compute analytical derivatives with respect to ``dL``, ``theta``, ``phi``, ``psi``, ``tcoal``, ``Phicoal`` and ``iota`` (the latter only for the fundamental mode in the non-precessing case).
        
        :param array or float f: The frequency(ies) at which to perform the calculation, in :math:`\\rm Hz`.
        :param array or float Mc: The chirp mass(es), :math:`{\cal M}_c`, in units of :math:`\\rm M_{\odot}`. If ``use_m1m2=True`` this is interpreted as the primary mass, :math:`m_1`, in units of :math:`\\rm M_{\odot}`.
        :param array or float eta:  The symmetric mass ratio(s), :math:`\eta`. If ``use_m1m2=True`` this is interpreted as the secondary mass, :math:`m_2`, in units of :math:`\\rm M_{\odot}`.
        :param array or float dL: The luminosity distance(s), :math:`d_L`, in :math:`\\rm Gpc`.
        :param array or float theta: The :math:`\\theta` sky position angle(s), in :math:`\\rm rad`.
        :param array or float phi: The :math:`\phi` sky position angle(s), in :math:`\\rm rad`.
        :param array or float iota: The inclination angle(s), with respect to orbital angular momentum, :math:`\iota`, in :math:`\\rm rad`. If ``is_prec_ang=True`` this is interpreted as the inclination angle(s) with respect to total angular momentum, :math:`\\theta_{JN}`, in :math:`\\rm rad`.
        :param array or float psi: The polarisation angle(s), :math:`\psi`, in :math:`\\rm rad`.
        :param array or float tcoal: The time(s) of coalescence, :math:`t_{\\rm coal}`, as a GMST.
        :param array or float Phicoal: The phase(s) at coalescence, :math:`\Phi_{\\rm coal}`, in :math:`\\rm rad`.
        :param array or float chi1z: The spin component(s) of the primary object(s) along the axis :math:`z`, :math:`\chi_{1,z}`.
        :param array or float chi2z: The spin component(s) of the secondary object(s) along the axis :math:`z`, :math:`\chi_{2,z}`.
        :return: Analytical derivatives with respect to ``dL``, ``theta``, ``phi``, ``iota``, ``psi``, ``tcoal`` and ``Phicoal``. If the :py:class:`self.wf_model` is precessing or includes higher order modes the derivative with respect to ``iota`` will be ``None``
        :rtype: tuple(array, array, array, array, array, array, array)
        
        """
        # Module to compute analytically the derivatives w.r.t. dL, theta, phi, psi, tcoal, Phicoal and also iota in absence of HM or precessing spins. Each derivative is inserted into its own function with representative name, for ease of check.
        evParams = {'Mc':Mc, 'dL':dL, 'theta':theta, 'phi':phi, 'iota':iota, 'psi':psi, 'tcoal':tcoal, 'eta':eta, 'Phicoal':Phicoal, 'chi1z':chi1z, 'chi2z':chi2z}
        fgrids = np.repeat(self.fixed_fgrid, Mc.shape[0]).reshape((self.fixed_fgrid.shape[0], Mc.shape[0]))
        
        wfPhiGw = self.wf_model.Phi(fgrids, **evParams)
        wfAmpl  = self.wf_model.Ampl(fgrids, **evParams)
        wfhp, wfhc = wfAmpl*np.exp(-1j*wfPhiGw)*0.5*(1.+(np.cos(iota))**2), 1j*wfAmpl*np.exp(-1j*wfPhiGw)*np.cos(iota)
        tmpDeltLoc = self._DeltLoc(theta, phi, tcoal) # in seconds
        t = tcoal + tmpDeltLoc/(3600.*24.)
        
        phiL = (2.*np.pi*fgrids)*tmpDeltLoc
        
        rot_rad = rot*np.pi/180.

        def afun(ra, dec, t, rot):
            phir = self.det_long_rad
            a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(3.-np.cos(2.*dec))*np.cos(2.*(ra - phir - 2.*np.pi*t))
            a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*np.sin(2.*(ra - phir - 2.*np.pi*t))
            a3 = 0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)
            a4 = 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)
            a5 = 3.*0.25*np.sin(2*(self.det_xax_rad+rot))*(np.cos(self.det_lat_rad)*np.cos(dec))**2.
            return a1 - a2 + a3 - a4 + a5
        
        def bfun(ra, dec, t, rot):
            phir = self.det_long_rad
            b1 = np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*np.cos(2.*(ra - phir - 2.*np.pi*t))
            b2 = 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*np.sin(2.*(ra - phir - 2.*np.pi*t))
            b3 = np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)
            b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)
            
            return b1 + b2 + b3 + b4
        
        ras, decs = self._ra_dec_from_th_phi(theta, phi)

        afac = afun(ras, decs, t, rot_rad)
        bfac = bfun(ras, decs, t, rot_rad)
        
        Fp = np.sin(self.angbtwArms)*(afac*np.cos(2.*psi) + bfac*np.sin(2*psi))
        Fc = np.sin(self.angbtwArms)*(bfac*np.cos(2.*psi) - afac*np.sin(2*psi))

        hp, hc = wfhp*Fp*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL)), wfhc*Fc*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL))
        def psi_par_deriv():
            
            Fp_psider = 2*np.sin(self.angbtwArms)*(-afac*np.sin(2.*psi) + bfac*np.cos(2*psi))
            Fc_psider = 2*np.sin(self.angbtwArms)*(-bfac*np.sin(2.*psi) - afac*np.cos(2*psi))
            
            return wfhp*Fp_psider*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL)) + wfhc*Fc_psider*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL))
        
        def phi_par_deriv():
            
            def Delt_loc_phider(ra, dec, t):
                
                comp1 = -np.cos(dec)*np.sin(ra)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)
                comp2 = np.cos(dec)*np.cos(ra)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)
                
                Delt_phider = - glob.REarth*(comp1+comp2)/glob.clight
                
                return Delt_phider/(3600.*24.) # in days
    
            def afun_phider(ra, dec, t, rot):
                phir = self.det_long_rad
                a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(3.-np.cos(2.*dec))*(-2.)*np.sin(2.*(ra - phir - 2.*np.pi*t))
                a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*2.*np.cos(2.*(ra - phir - 2.*np.pi*t))
                a3 = -0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)
                a4 = 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)

                return a1 - a2 + a3 - a4
            
            def bfun_phider(ra, dec, t, rot):
                phir = self.det_long_rad
    
                b1 = np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*(-2.)*np.sin(2.*(ra - phir - 2.*np.pi*t))
                b2 = 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*2.*np.cos(2.*(ra - phir - 2.*np.pi*t))
                b3 = -np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)
                b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)
                
                return b1 + b2 + b3 + b4
            locDt_phider = Delt_loc_phider(ras, decs, tcoal)
            afac_phider = afun_phider(ras, decs, t, rot_rad)*(1.-2.*np.pi*locDt_phider)
            bfac_phider = bfun_phider(ras, decs, t, rot_rad)*(1.-2.*np.pi*locDt_phider)
            
            Fp_phider = np.sin(self.angbtwArms)*(afac_phider*np.cos(2.*psi) + bfac_phider*np.sin(2*psi))
            Fc_phider = np.sin(self.angbtwArms)*(bfac_phider*np.cos(2.*psi) - afac_phider*np.sin(2*psi))
            
            ampP_phider = wfhp*Fp_phider*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL))
            ampC_phider = wfhc*Fc_phider*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL))
            phiD_phideriv = 0.
            phiL_phideriv = 2.*np.pi*fgrids*locDt_phider*(3600.*24.)
            
            return ampP_phider + 1j*(phiD_phideriv + phiL_phideriv)*hp + ampC_phider + 1j*(phiD_phideriv + phiL_phideriv)*hc
        
        def theta_par_deriv():
            def Delt_loc_thder(ra, dec, t):
                
                comp1 = np.sin(dec)*np.cos(ra)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)
                comp2 = np.sin(dec)*np.sin(ra)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)
                comp3 = -np.cos(dec)*np.sin(self.det_lat_rad)
                
                Delt_thder = - glob.REarth*(comp1+comp2+comp3)/glob.clight
                
                return Delt_thder/(3600.*24.) # in days
            
            def afun_thder(ra, dec, t, rot, loc_thder):
                phir = self.det_long_rad
                a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(-2.*np.sin(2.*dec))*np.cos(2.*(ra - phir - 2.*np.pi*t)) + 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad)) *(3.-np.cos(2.*dec))*np.sin(2.*(ra - phir - 2.*np.pi*t))*(4.*np.pi*loc_thder)
                a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(-2.*np.sin(2.*dec))*np.sin(2.*(ra - phir - 2.*np.pi*t)) - 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*np.cos(2.*(ra - phir - 2.*np.pi*t))*(4.*np.pi*loc_thder)
                a3 = -0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*2.*np.cos(2.*dec)*np.cos(ra - phir - 2.*np.pi*t) + 0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)*(2.*np.pi*loc_thder)
                a4 = -0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*2.*np.cos(2.*dec)*np.sin(ra - phir - 2.*np.pi*t) - 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)*(2.*np.pi*loc_thder)
                a5 = 2.*3.*0.25*np.sin(2*(self.det_xax_rad+rot))*((np.cos(self.det_lat_rad))**2)*np.cos(dec)*np.sin(dec)
                return a1 - a2 + a3 - a4 + a5
            
            def bfun_thder(ra, dec, t, rot, loc_thder):
                phir = self.det_long_rad
                b1 = -np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.cos(dec)*np.cos(2.*(ra - phir - 2.*np.pi*t)) + np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*np.sin(2.*(ra - phir - 2.*np.pi*t))*(4.*np.pi*loc_thder)
                b2 = -0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.cos(dec)*np.sin(2.*(ra - phir - 2.*np.pi*t)) - 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*np.cos(2.*(ra - phir - 2.*np.pi*t))*(4.*np.pi*loc_thder)
                b3 = np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(dec)*np.cos(ra - phir - 2.*np.pi*t) + np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)*(2.*np.pi*loc_thder)
                b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(dec)*np.sin(ra - phir - 2.*np.pi*t) - 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)*(2.*np.pi*loc_thder)
                
                return b1 + b2 + b3 + b4
            
            locDt_thder = Delt_loc_thder(ras, decs, tcoal)
            afac_thder = afun_thder(ras, decs, t, rot_rad, locDt_thder)
            bfac_thder = bfun_thder(ras, decs, t, rot_rad, locDt_thder)
            
            Fp_thder = np.sin(self.angbtwArms)*(afac_thder*np.cos(2.*psi) + bfac_thder*np.sin(2*psi))
            Fc_thder = np.sin(self.angbtwArms)*(bfac_thder*np.cos(2.*psi) - afac_thder*np.sin(2*psi))
            
            ampP_thder = wfhp*Fp_thder*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL))
            ampC_thder = wfhc*Fc_thder*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL))
            phiD_thderiv = 0.
            phiL_thderiv = 2.*np.pi*fgrids*locDt_thder*(3600.*24.)
            
            return ampP_thder + 1j*(phiD_thderiv + phiL_thderiv)*hp + ampC_thder + 1j*(phiD_thderiv + phiL_thderiv)*hc
        
        def tcoal_par_deriv():
            
            def Delt_loc_tcder(ra, dec, t):
    
                comp1 = -np.cos(dec)*np.cos(ra)*np.cos(self.det_lat_rad)*np.sin(self.det_long_rad + 2.*np.pi*t)
                comp2 = np.cos(dec)*np.sin(ra)*np.cos(self.det_lat_rad)*np.cos(self.det_long_rad + 2.*np.pi*t)
                
                Delt_tcder = - 2.*np.pi*glob.REarth*(comp1+comp2)/glob.clight
                
                return Delt_tcder/(3600.*24.) # in days
    
            def afun_tcder(ra, dec, t, rot):
                phir = self.det_long_rad
                a1 = 0.0625*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*(3.-np.cos(2.*dec))*(-2.)*np.sin(2.*(ra - phir - 2.*np.pi*t))
                a2 = 0.25*np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*(3.-np.cos(2.*dec))*2.*np.cos(2.*(ra - phir - 2.*np.pi*t))
                a3 = -0.25*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.sin(2.*dec)*np.sin(ra - phir - 2.*np.pi*t)
                a4 = 0.5*np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.sin(2.*dec)*np.cos(ra - phir - 2.*np.pi*t)

                return a1 - a2 + a3 - a4
            
            def bfun_tcder(ra, dec, t, rot):
                phir = self.det_long_rad
                b1 = np.cos(2*(self.det_xax_rad+rot))*np.sin(self.det_lat_rad)*np.sin(dec)*(-2.)*np.sin(2.*(ra - phir - 2.*np.pi*t))
                b2 = 0.25*np.sin(2*(self.det_xax_rad+rot))*(3.-np.cos(2.*self.det_lat_rad))*np.sin(dec)*2.*np.cos(2.*(ra - phir - 2.*np.pi*t))
                b3 = -np.cos(2*(self.det_xax_rad+rot))*np.cos(self.det_lat_rad)*np.cos(dec)*np.sin(ra - phir - 2.*np.pi*t)
                b4 = 0.5*np.sin(2*(self.det_xax_rad+rot))*np.sin(2.*self.det_lat_rad)*np.cos(dec)*np.cos(ra - phir - 2.*np.pi*t)
                
                return b1 + b2 + b3 + b4
            locDt_tcder = Delt_loc_tcder(ras, decs, tcoal)
            afac_tcder = -2.*np.pi*afun_tcder(ras, decs, t, rot_rad)*(1.+locDt_tcder)
            bfac_tcder = -2.*np.pi*bfun_tcder(ras, decs, t, rot_rad)*(1.+locDt_tcder)
            
            Fp_tcder = np.sin(self.angbtwArms)*(afac_tcder*np.cos(2.*psi) + bfac_tcder*np.sin(2*psi))
            Fc_tcder = np.sin(self.angbtwArms)*(bfac_tcder*np.cos(2.*psi) - afac_tcder*np.sin(2*psi))
            
            ampP_tcder = wfhp*Fp_tcder*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL))
            ampC_tcder = wfhc*Fc_tcder*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL))
            phiD_tcderiv = 0.
            phiL_tcderiv = 2.*np.pi*fgrids*locDt_tcder*(3600.*24.)

            return ampP_tcder + 1j*(phiD_tcderiv + phiL_tcderiv + 2.*np.pi*fgrids*3600.*24.)*hp + ampC_tcder + 1j*(phiD_tcderiv + phiL_tcderiv + 2.*np.pi*fgrids*3600.*24.)*hc
        
        def iota_par_deriv():
            
            wfhp_iotader, wfhc_iotader = -wfAmpl*np.exp(-1j*wfPhiGw)*(np.cos(iota)*np.sin(iota)), -1j*wfAmpl*np.exp(-1j*wfPhiGw)*np.sin(iota)
            
            return wfhp_iotader*Fp*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL)) + wfhc_iotader*Fc*np.exp(1j*(2.*np.pi*fgrids*(tcoal*3600.*24.) - Phicoal + phiL))
        
        dL_deriv = -(hp+hc)/dL
        Phicoal_deriv = -1j*(hp+hc)
        psi_deriv = psi_par_deriv()
        phi_deriv = phi_par_deriv()
        theta_deriv = theta_par_deriv()
        tc_deriv = tcoal_par_deriv()
        iota_deriv = iota_par_deriv()
        
        return dL_deriv, theta_deriv, phi_deriv, iota_deriv, psi_deriv, tc_deriv, Phicoal_deriv
            