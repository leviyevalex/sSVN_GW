""" 
Compare sSVN runs with outputs from other samplers
"""

#%% 
# Import libraries
import sys
sys.path.append("..")
import numpy as np
import bilby
import copy
import gwfast.gwfastGlobals as glob
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD
import os
# from models.gwfastWrapClass import gwfast_class
from models.GWFAST_REWRITE import gwfast_class
# from collections import OrderedDict

label = 'gwfast_run'

#%%
# Define for convenience
tGPS = np.array([1.12625946e+09])
# (GW150914) like parameters
injParams = dict()
injParams['Mc']      = np.array([34.3089283])        # Units: M_sun
injParams['eta']     = np.array([0.2485773])         # Units: Unitless
injParams['dL']      = np.array([2.634])             # Units: Gigaparsecs 
injParams['theta']   = np.array([2.78560281])        # Units: Rad
injParams['phi']     = np.array([1.67687425])        # Units: Rad
injParams['iota']    = np.array([2.67548653])        # Units: Rad
injParams['psi']     = np.array([0.78539816])        # Units: Rad
injParams['tcoal']   = np.array([0.])                # Units: Fraction of day 
# injParams['tcoal'] = np.array(utils.GPSt_to_LMST(tGPS, lat=0., long=0.) * (3600 * 24)) # Coalescence time, in units of fraction of day (GMST is LMST computed at long = 0°) 
injParams['Phicoal'] = np.array([0.])                # Units: Rad
injParams['chi1z']   = np.array([0.27210419])        # Units: Unitless
injParams['chi2z']   = np.array([0.33355909])        # Units: Unitless

#%%
#  Setup gravitational wave network problem

all_detectors = copy.deepcopy(glob.detectors) # Geometry of every available detector

LV_detectors = {det:all_detectors[det] for det in ['L1', 'H1', 'Virgo']} # Extract only LIGO/Virgo detectors
# LV_detectors = {det:all_detectors[det] for det in ['L1', 'H1']}

print('Using detectors ' + str(list(LV_detectors.keys())))

detector_ASD = dict() # Remark: Providing ASD path to psd_path with flag "is_ASD = True" in
detector_ASD['L1']    = 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
detector_ASD['H1']    = 'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'
detector_ASD['Virgo'] = 'O3-V1_sensitivity_strain_asd.txt'

LV_detectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['L1']) # Add paths to detector sensitivities
LV_detectors['H1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['H1'])
LV_detectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['Virgo'])

waveform = TaylorF2_RestrictedPN() # Choice of waveform
# waveform = IMRPhenomD()

priorDict = {}

priorDict['Mc']      = [29., 39.]          # (1)   # (0)
priorDict['eta']     = [0.22, 0.25]        # (2)   # (1)
priorDict['dL']      = [0.1, 3.]           # (3)   # (2)
priorDict['theta']   = [0., np.pi]         # (4)   # (3)
priorDict['phi']     = [0., 2*np.pi]       # (5)   # (4)
priorDict['iota']    = [0., np.pi]         # (6)   # (5)
priorDict['psi']     = [0., np.pi]         # (7)   # (6)
priorDict['tcoal']   = [injParams['tcoal'][0] - 0.001, injParams['tcoal'][0] + 0.001]           # (8)   # (7)
priorDict['Phicoal'] = [0., 0.1]           # (9)   # (8)
priorDict['chi1z']   = [-1., 1.]           # (10)  # (9)
priorDict['chi2z']   = [-1., 1.]           # (11)  # (10)

#%%
nParticles = 1
model = gwfast_class(LV_detectors, waveform, injParams, priorDict, nParticles=nParticles)









#%%
priors = {}
for param in model.gwfast_param_order:
    priors[param] = bilby.prior.Uniform(minimum=priorDict[param][0], maximum=priorDict[param][1], name=param)



# Wrap likelihood
#%%
class bilby_gwfast_wrapper(bilby.Likelihood):
    def __init__(self):
        """

        Parameters
        ----------
        """
        params = {'Mc': None, 'eta': None, 'dL': None, 'theta': None, 'phi': None, 'iota': None, 'psi': None, 'tcoal': None, 'Phicoal': None, 'chi1z': None, 'chi2z': None}

        super().__init__(parameters=params)

    def log_likelihood(self):
        Mc = self.parameters['Mc'] 
        eta = self.parameters['eta'] 
        dL = self.parameters['dL'] 
        theta = self.parameters['theta'] 
        phi = self.parameters['phi'] 
        iota = self.parameters['iota'] 
        psi = self.parameters['psi'] 
        tcoal = self.parameters['tcoal'] 
        Phicoal = self.parameters['Phicoal'] 
        chi1z = self.parameters['chi1z'] 
        chi2z = self.parameters['chi2z']

        theta = np.array([Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z])
        return -1 * model.getMinusLogPosterior_ensemble(theta[np.newaxis, ...]).squeeze()

bilby_likelihood = bilby_gwfast_wrapper()
# %%
# nlive = 1000          # live points
# stop = 0.1            # stopping criterion
# method = "unif"       # method of sampling
# # sampler = "dynesty"   # sampler to use
# sampler = "nessai"   # sampler to use

result = bilby.run_sampler(
    label=label,
    resume=False,
    plot=True,
    likelihood=bilby_likelihood,
    priors=priors,
    sampler="dynesty",
    injection_parameters=injParams,
    analytic_priors=True,
    seed=1234,
    nlive=1000
)



# result = bilby.core.sampler.nessai.Nessai(bilby_likelihood, priors)

# result = bilby.run_sampler(
#     bilby_likelihood, priors, sampler=sampler, label=label,
#     sample=method, nlive=nlive, dlogz=stop) 


# %%
import pandas as pd
import os
a = os.getcwd()
pickle_path = os.path.join(a, 'outdir', 'gwfast_run_dynesty.pickle')
obj = pd.read_pickle(pickle_path)
# %%
import corner
#%%
samples = obj['samples']
#%%
corner.corner(samples, smooth=1)
# %%
import matplotlib.pyplot as plt
#%%
plt.hist2d(samples[:,0], samples[:,1], bins=1000)
# %%
plt.scatter(samples[:,0], samples[:,1])
# %%
a = [1, 2, 3]
a.size()
# %%
