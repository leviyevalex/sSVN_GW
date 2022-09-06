#%%
import jax.numpy as jnp
from gwfast import gwfastUtils as utils
import gwfast.signal as signal
import numpy as onp
from astropy.cosmology import Planck18
#  from gwfastUtils import GPSt_to_LMST

#%% Note: Here we use GW170817 parameters
z = onp.array([0.00980])
tGPS = onp.array([1187008882.4])

Mc = onp.array([1.1859])*(1.+z)
dL = Planck18.luminosity_distance(z).value/1000
dec = onp.array([onp.pi/2. + 0.4080839999999999])
ra = onp.array([3.4461599999999994])
iota = onp.array([2.545065595974997])
psi = onp.array([0.])
tcoal = utils.GPSt_to_LMST(tGPS, lat=0., long=0.) # GMST is LMST computed at long = 0°
eta = onp.array([0.24786618323504223])
Phicoal = onp.array([0.])
chi1z = onp.array([0.005136138323169717])
chi2z = onp.array([0.003235146993487445])
# Lambda1 = onp.array([368.17802383555687])
# Lambda2 = onp.array([586.5487031450857])

theta = jnp.array([Mc, eta, dL, dec, ra, iota, psi, tcoal, Phicoal, chi1z, chi2z])
#%% Remark: Provide real array, convert to complex128, then unpack to interface with gwfast methods
Mc, eta, dL, dec_, ra, iota, psi, tcoal, Phicoal, chi1z, chi2z = theta.astype('complex128')
# dec_ is pi/2 - dec
# For the moment no precessing spins
chi1x, chi2x, chi1y, chi2y = Mc * 0, Mc * 0, Mc * 0, Mc * 0
evParams = {'Mc': Mc, 'eta': eta, 'dL': dL, 'theta': dec_, 'phi': ra, 'iota': iota, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal, 'chi1z': chi1z, 'chi2z': chi2z}

#%%





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

        # Setup detector network
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

        # Frequency grid stuff
        self.res = 1000 # Question: What is the relationship between this and fmin, fmax? Can't we get this using df?
        self.df = None
        self.spacing = 'lin'


    def forward(self, theta):
        # Unpack parameters
        Mc, eta, dL, dec, ra, iota, psi, tcoal, Phicoal, chi1z, chi2z = theta.astype('complex128')

        # For the moment no precessing spins
        chi1x, chi2x, chi1y, chi2y = Mc*0, Mc*0, Mc*0, Mc*0

        # These parameters are used to determine the cut frequency?
        evParams = {'Mc': Mc, 'eta': eta,  'dL': dL, 'theta': dec, 'phi': ra, 'iota': iota, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal, 'chi1z': chi1z, 'chi2z': chi2z}

        # Theta is 13 dimensional?
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
        if self.res is None and self.df is not None:
            self.res = jnp.floor( jnp.real((1+(fcut-fminarr)/self.df)))
            self.res = jnp.amax(self.res)
        elif self.res is None and self.df is None:
            raise ValueError('Provide either resolution in frequency or step size.')
        if self.spacing=='lin':
            self.fgrids = jnp.linspace(fminarr, fcut, num=int(self.res))
        elif self.spacing=='geom':
            self.fgrids = jnp.geomspace(fminarr, fcut, num=int(self.res))



    def _getJacobianResidual_Vec(self, theta, ):
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



        derivsSum = jnp.zeros((nParams,)+fgrids.shape)
        # Out of the provided PSD range, we use a constant value of 1, which results in completely negligible conntributions
        tcelem = self.wf_model.ParNums['tcoal']
        for det in self.detsInNet.keys():
            strainGrids = jnp.interp(fgrids, self.detsInNet[det].strainFreq, self.detsInNet[det].noiseCurve, left=1., right=1.)
            # Compute derivatives
            FisherDerivs = self.detsInNet[det]._SignalDerivatives_use(fgrids, Mc, eta, dL, theta, phi,
                                                                 iota, psi, tcoal, Phicoal, chi1z, chi2z,
                                                                 chi1x, chi2x, chi1y, chi2y,
                                                                 LambdaTilde, deltaLambda, ecc=0,
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