#%%################################
# Export PATH and import libraries
###################################
import sys
sys.path.append("..")
# 
# from models.gwfastWrapClass import gwfast_class
from models.GWFAST_heterodyne import gwfast_class
from src.samplers import samplers
from scripts.plot_helper_functions import collect_samples
import numpy as np
import logging
import sys
from collections import OrderedDict
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import os
import time
import copy
import time
import gwfast.gwfastUtils as utils
import gwfast.gwfastGlobals as glob
import gwfast.signal as signal
from gwfast.network import DetNet
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD
from astropy.cosmology import Planck18
import matplotlib.pyplot as plt
from numdifftools import Jacobian, Gradient, Hessdiag
from jax import jacobian, grad
from corner import corner
import jax
import jax.numpy as jnp
# import matplotlib.pyplot as plt

#%%##########################
# (GW150914) like parameters
#############################
# Remarks:
# (i) `tcoal` is accepted in units of fraction of a day
# (ii) GPSt_to_LMST returns GMST in units of fraction of day (GMST is LMST computed at long = 0°)
seconds_per_day = 86400. 
tGPS = np.array([1.12625946e+09])
# tcoal = float(utils.GPSt_to_LMST(tGPS, lat=0., long=0.) * seconds_per_day) # Units of seconds
tcoal = float(utils.GPSt_to_LMST(tGPS, lat=0., long=0.)) # [0, 1] Units of fraction of the day
injParams = dict()
injParams['Mc']      = np.array([34.3089283])                                                # [M_solar]
injParams['eta']     = np.array([0.2485773])                                                 # [Unitless]
injParams['dL']      = np.array([2.634])                                                     # [Gigaparsecs] 
injParams['theta']   = np.array([2.78560281])                                                # [Rad]
injParams['phi']     = np.array([1.67687425])                                                # [Rad]
injParams['iota']    = np.array([2.67548653])                                                # [Rad]
injParams['psi']     = np.array([0.78539816])                                                # [Rad]
injParams['tcoal']   = np.array([tcoal])                                                     # [Sec]
injParams['Phicoal'] = np.array([0.])                                                        # [Rad]
injParams['chi1z']   = np.array([0.27210419])                                                # [Unitless]
injParams['chi2z']   = np.array([0.33355909])                                                # [Unitless]

priorDict = {}
priorDict['Mc']      = [29., 39.]                                             # (1)   # (0)  # [M_solar]
priorDict['eta']     = [0.22, 0.25]                                           # (2)   # (1)  # [Unitless]
priorDict['dL']      = [0.1, 4.]                                              # (3)   # (2)  # [GPC]
priorDict['theta']   = [0., np.pi]                                            # (4)   # (3)  # [Rad]
priorDict['phi']     = [0., 2 * np.pi]                                        # (5)   # (4)  # [Rad]
priorDict['iota']    = [0., np.pi]                                            # (6)   # (5)  # [Rad]
priorDict['psi']     = [0., np.pi]                                            # (7)   # (6)  # [Rad]
priorDict['tcoal']   = [tcoal - 1e-7, tcoal + 1e-7]                           # (8)   # (7)  # [Sec]
priorDict['Phicoal'] = [0., 2 * np.pi]                                        # (9)   # (8)  # [Rad]
priorDict['chi1z']   = [-1., 1.]                                              # (10)  # (9)  # [Unitless]
priorDict['chi2z']   = [-1., 1.]                                              # (11)  # (10) # [Unitless]

#%%########################################
# Setup gravitational wave network problem
###########################################
# Notes:
# (i)   Geometry of every available detector
# (ii)  Extract only LIGO/Virgo detectors
# (iii) Providing ASD path to psd_path with flag "is_ASD = True"
# (iv)  Add paths to detector sensitivities
all_detectors = copy.deepcopy(glob.detectors) # (i)

LV_detectors = {det:all_detectors[det] for det in ['L1', 'H1', 'Virgo']} # (ii)
# LV_detectors = {det:all_detectors[det] for det in ['L1']}

print('Using detectors ' + str(list(LV_detectors.keys())))

detector_ASD = dict() # (iii)
detector_ASD['L1']    = 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt'
detector_ASD['H1']    = 'O3-H1-C01_CLEAN_SUB60HZ-1251752040.0_sensitivity_strain_asd.txt'
detector_ASD['Virgo'] = 'O3-V1_sensitivity_strain_asd.txt'

LV_detectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['L1']) # (iv) 
LV_detectors['H1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['H1'])
LV_detectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', detector_ASD['Virgo'])

# waveform = TaylorF2_RestrictedPN() # Choice of waveform
waveform = IMRPhenomD()


#%%###############################################
# Class setup
##################################################
model = gwfast_class(LV_detectors, waveform, injParams, priorDict)


#%%####################################
# Likelihood accuracy test
#######################################
nParticles = 500
X = model._newDrawFromPrior(nParticles)
test1 = model.standard_minusLogLikelihood(X)
test2 = model.heterodyne_minusLogLikelihood(X) 
percent_change = (test1 - test2) / test1 * 100
mpc = np.mean(percent_change)
print(mpc)

#%%
import bilby
class bilby_gwfast(bilby.Likelihood):
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
        return float(-1 * model.heterodyne_minusLogLikelihood(theta[np.newaxis, ...]).squeeze())

priors = {}
for param in model.gwfast_param_order:
    priors[param] = bilby.prior.Uniform(minimum=priorDict[param][0], maximum=priorDict[param][1], name=param)

#%%
bilby_likelihood = bilby_gwfast()
# nlive = 1000          # live points
# stop = 0.1            # stopping criterion
# method = "unif"       # method of sampling
# # sampler = "dynesty"   # sampler to use
# sampler = "nessai"   # sampler to use


result = bilby.run_sampler(
    label='try004',
    resume=False,
    plot=True,
    likelihood=bilby_likelihood,
    priors=priors,
    sampler="nessai",
    injection_parameters=injParams,
    analytic_priors=True,
    seed=1234,
    nlive=200,
    # nlive=1000,
    # nact=15
)

#%%#################################
# Loading nessai results
####################################
import json 
home = os.getcwd()
filepath = os.path.join(home, 'outdir', 'try002_result.json')
f = open(filepath, "rb" )


jsonObject = json.load(f)

# data = json.loads(filepath)
#%%
f.close()

#%%
import pandas as pd
home = os.getcwd()
filepath = os.path.join(home, 'outdir', 'try001_dynesty.pickle')
obj = pd.read_pickle(filepath)
#%%
fig = corner(obj['samples'])
fig.savefig('corner_bilby.png')



#%%






































#########################################
#


#%%###########################################
# jaxns
##############################################
from jax.config import config

config.update("jax_enable_x64", True)

import pylab as plt
import tensorflow_probability.substrates.jax as tfp
from jax import random, numpy as jnp
from jax import vmap

import jaxns
from jaxns import ExactNestedSampler
from jaxns import Model
from jaxns import PriorModelGen, Prior
from jaxns import TerminationCondition
from jaxns import analytic_log_evidence

tfpd = tfp.distributions

#%%
def log_likelihood(x):
    X = x[np.newaxis, ...]
    return -1 * model.heterodyne_minusLogLikelihood(X)


def prior_model() -> PriorModelGen:
    x = yield Prior(tfpd.Uniform(low=model.lower_bound, high=model.upper_bound), name='x')
    return x


model = Model(prior_model=prior_model,
              log_likelihood=log_likelihood)

ns = exact_ns = ExactNestedSampler(model=model, num_live_points=200, num_parallel_samplers=1,
                                   max_samples=1e4)

termination_reason, state = exact_ns(random.PRNGKey(42),
                                     term_cond=TerminationCondition(live_evidence_frac=1e-4))
results = exact_ns.to_results(state, termination_reason)
# # %%

# %%
