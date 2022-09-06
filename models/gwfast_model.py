import jax.numpy as jnp
from gwfast import gwfastUtils as utils
import gwfast.signal as signal

class gwfast_model:

    def __init__(self, NetDict, WaveForm, fmax=None, fmin=10, EarthMotion=False):
        """
        Args:
            NetDict (dict): dictionary containing the specifications of the detectors in the network, e.g.
            {"H1": {"lat": 46.455, "long": -119.408, "xax": 170.99924234706103, "shape": "L", "psd_path":"path/to/psd"}}
            WaveForm (WaveFormModel): waveform model to use
        """
        self.fmin = fmin
        self.fmax = fmax
        self.wf_model = WaveForm
        self.detsInNet = {}
        for d in NetDict.keys():
            self.detsInNet[d] = signal.GWSignal(self.wf_model,
                                                psd_path=NetDict[d]['psd_path'],
                                                detector_shape = NetDict[d]['shape'],
                                                det_lat= NetDict[d]['lat'],
                                                det_long=NetDict[d]['long'],
                                                det_xax=NetDict[d]['xax'],
                                                verbose=True,
                                                useEarthMotion = EarthMotion,
                                                fmin=fmin, fmax=fmax,
                                                is_ASD=True)





    def _getJacobianResidual_Vec(self, theta, res=1000, df=None, spacing='geom'):
        """
        Args:
         theta (nd.array): (d, Nev) shaped array
         Remark: Represents point in d-dimensional parameter space $\chi$
                 The order is Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, La
         res (int): resolution of the frequency grid
         df (float): spacing of the frequency grid, alternative to res
         spacing: 'geom' for logarithmic spacing and 'lin' for linear spacing
        Returns (nd.array): (d, F, Nev) shaped array
            Remark: Represents Jacobian of residual evaluated at theta.
        """
        # Read parameters, assuming the order in theta is:
        #    Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, LambdaTilde, deltaLambda
        Mc, eta, dL, theta, phi   = theta[0,:].astype('complex128'), theta[1,:].astype('complex128'), theta[2,:].astype('complex128'), theta[3,:].astype('complex128'), theta[4,:].astype('complex128')
        iota, psi, tcoal, Phicoal = theta[5,:].astype('complex128'), theta[6,:].astype('complex128'), theta[7,:].astype('complex128'), theta[8,:].astype('complex128'),
        chi1z, chi2z = theta[9,:].astype('complex128'), theta[10,:].astype('complex128')
        # For the moment no precessing spins
        chi1x, chi2x, chi1y, chi2y = Mc*0, Mc*0, Mc*0, Mc*0
        evParams = {'Mc':Mc, 'eta':eta,  'dL':dL, 'theta':theta, 'phi':phi, 'iota':iota, 'psi':psi, 'tcoal':tcoal,
                    'Phicoal':Phicoal, 'chi1z':chi1z, 'chi2z':chi2z}
        if self.wf_model.is_tidal:
            LambdaTilde, deltaLambda = theta[11,:].astype('complex128'), theta[12,:].astype('complex128')
            Lambda1, Lambda2 = utils.Lam12_from_Lamt_delLam(LambdaTilde, deltaLambda, eta)
            evParams['Lambda1'] = Lambda1
            evParams['Lambda2'] = Lambda2
        else:
            LambdaTilde, deltaLambda = Mc*0, Mc*0
        nParams = self.wf_model.nParams

        # Frequency grid
        fcut = self.wf_model.fcut(**evParams)
        if self.fmax is not None:
            fcut = jnp.where(fcut > self.fmax, self.fmax, fcut)
        fminarr = jnp.full(fcut.shape, self.fmin)
        if res is None and df is not None:
            res = jnp.floor( jnp.real((1+(fcut-fminarr)/df)))
            res = jnp.amax(res)
        elif res is None and df is None:
            raise ValueError('Provide either resolution in frequency or step size.')
        if spacing=='lin':
            fgrids = jnp.linspace(fminarr, fcut, num=int(res))
        elif spacing=='geom':
            fgrids = jnp.geomspace(fminarr, fcut, num=int(res))


        derivsSum = jnp.zeros((nParams,)+fgrids.shape)
        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
        tcelem = self.wf_model.ParNums['tcoal']
        for det in self.detsInNet.keys():
            strainGrids = jnp.interp(fgrids, self.detsInNet[det].strainFreq, self.detsInNet[det].noiseCurve, left=1., right=1.)
            # Compute derivatives
            FisherDerivs = self.detsInNet[det]._SignalDerivatives_use(fgrids, Mc, eta, dL, theta, phi,
                                                                 iota, psi, tcoal, Phicoal, chi1z, chi2z,
                                                                 chi1x, chi2x, chi1y, chi2y,
                                                                 LambdaTilde, deltaLambda, ecc,
                                                                 rot=0., use_m1m2=False, use_chi1chi2=True,
                                                                 use_prec_ang=False, computeAnalyticalDeriv=True)
            # Change the units of the tcoal derivative from days to seconds (this improves conditioning)
            FisherDerivs = jnp.array(FisherDerivs)/strainGrids.real
            FisherDerivs[tcelem,:,:] /= (3600.*24.)
            derivsSum = derivsSum + FisherDerivs
        return derivsSum


    def _getResidual_Vec(self, theta, res=1000, df=None, spacing='lin'):
        """
        Args:
         theta (nd.array): (d, Nev) shaped array
         Remark: Represents point in d-dimensional parameter space $\chi$
                 The order is Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1x, chi1y, chi1z, chi2x, chi2y, chi2z, La
         res (int): resolution of the frequency grid
         df (float): spacing of the frequency grid, alternative to res
         spacing: 'geom' for logarithmic spacing and 'lin' for linear spacing
        Returns (nd.array): (F, Nev) shaped array
            Remark: Represents residuals for each frequency bin, up to bin $F$.
        """
        # Read parameters, assuming the order in theta is:
        #    Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z, LambdaTilde, deltaLambda
        Mc, eta, dL, theta, phi   = theta[0,:].astype('complex128'), theta[1,:].astype('complex128'), theta[2,:].astype('complex128'), theta[3,:].astype('complex128'), theta[4,:].astype('complex128')
        iota, psi, tcoal, Phicoal = theta[5,:].astype('complex128'), theta[6,:].astype('complex128'), theta[7,:].astype('complex128'), theta[8,:].astype('complex128'),
        chi1z, chi2z = theta[9,:].astype('complex128'), theta[10,:].astype('complex128')
        # For the moment no precessing spins
        chi1x, chi2x, chi1y, chi2y = Mc*0, Mc*0, Mc*0, Mc*0
        evParams = {'Mc':Mc, 'eta':eta,  'dL':dL, 'theta':theta, 'phi':phi, 'iota':iota, 'psi':psi, 'tcoal':tcoal,
                    'Phicoal':Phicoal, 'chi1z':chi1z, 'chi2z':chi2z}
        if self.wf_model.is_tidal:
            LambdaTilde, deltaLambda = theta[11,:].astype('complex128'), theta[12,:].astype('complex128')
            Lambda1, Lambda2 = utils.Lam12_from_Lamt_delLam(LambdaTilde, deltaLambda, eta)
            evParams['Lambda1'] = Lambda1
            evParams['Lambda2'] = Lambda2
        else:
            LambdaTilde, deltaLambda = Mc*0, Mc*0
        nParams = self.wf_model.nParams

        # Frequency grid
        fcut = self.wf_model.fcut(**evParams)
        if self.fmax is not None:
            fcut = jnp.where(fcut > self.fmax, self.fmax, fcut)
        fminarr = jnp.full(fcut.shape, self.fmin)
        if res is None and df is not None:
            res = jnp.floor( jnp.real((1+(fcut-fminarr)/df)))
            res = jnp.amax(res)
        elif res is None and df is None:
            raise ValueError('Provide either resolution in frequency or step size.')
        if spacing=='lin':
            fgrids = jnp.linspace(fminarr, fcut, num=int(res))
        elif spacing=='geom':
            fgrids = jnp.geomspace(fminarr, fcut, num=int(res))


        signalsSum = jnp.zeros(fgrids.shape)
        for det in self.detsInNet.keys():
            signal = self.detsInNet[det].GWstrain(fgrids, Mc, eta, dL, theta, phi,
                                             iota, psi, tcoal, Phicoal, chi1z, chi2z,
                                             chi1x, chi2x, chi1y, chi2y,
                                             LambdaTilde, deltaLambda, ecc,
                                             rot=0., is_m1m2=False, is_chi1chi2=True,
                                             is_prec_ang=False)
            signalsSum = signalsSum + signal
        return signalsSum