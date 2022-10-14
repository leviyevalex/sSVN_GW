"""
Minimum working example for derivative methods
"""

#%%
# Import methods
import sys
sys.path.append("..")
import numpy as np
import copy, os
import gwfast.gwfastUtils as utils
import gwfast.gwfastGlobals as glob
from gwfast.network import DetNet
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD
from models.gwfastWrapClass import gwfast_class
import jax.numpy as jnp
from jax import jacobian

#%%
# Define 11 parameters for signal injection
injParams = dict()
injParams['Mc']      = np.array([34.3089283])        # (0)  Units: M_sun
injParams['eta']     = np.array([0.2485773])         # (1)  Units: Unitless
injParams['dL']      = np.array([2.634])             # (2)  Units: Gigaparsecs 
injParams['theta']   = np.array([2.78560281])        # (3)  Units: Rad
injParams['phi']     = np.array([1.67687425])        # (4)  Units: Rad
injParams['iota']    = np.array([2.67548653])        # (5)  Units: Rad
injParams['psi']     = np.array([0.78539816])        # (6)  Units: Rad
injParams['tcoal']   = np.array([0.])                # (7)  Units: Fraction of day in seconds
injParams['Phicoal'] = np.array([0.])                # (8)  Units: Rad
injParams['chi1z']   = np.array([0.27210419])        # (9)  Units: Unitless
injParams['chi2z']   = np.array([0.33355909])        # (10) Units: Unitless

#%%
# Settings prior ranges (The class expects this as an input, but is only needed by
# the sampler, and not the derivative method.)
priorDict = {}
priorDict['Mc']      = [29., 39.]      # (0)
priorDict['eta']     = [0.22, 0.25]    # (1)
priorDict['dL']      = [0.1, 3.]       # (2)
priorDict['theta']   = [0., np.pi]     # (3)
priorDict['phi']     = [0., 2*np.pi]   # (4)
priorDict['iota']    = [0., np.pi]     # (5)
priorDict['psi']     = [0., np.pi]     # (6)
priorDict['tcoal']   = [0, 0.001]      # (7)
priorDict['Phicoal'] = [0., 0.1]       # (8)
priorDict['chi1z']   = [-1., 1.]       # (9)
priorDict['chi2z']   = [-1., 1.]       # (10)

#%%
# Setup detector (This is indeed needed to define the likelihood)
all_detectors = copy.deepcopy(glob.detectors)
LV_detectors = {det:all_detectors[det] for det in ['L1']}
LV_detectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', 'O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt')

#%%
# Define class with relevant methods. 
model = gwfast_class(LV_detectors, TaylorF2_RestrictedPN(), injParams, priorDict)
# model = gwfast_class(LV_detectors, IMRPhenomD(), injParams, priorDict)


#%%
# Derivative tests
# Note: Derivatives agree for TaylorF2_RestrictedPN(), but disagree for IMRPhenomD()
particles = jnp.array(model.true_params[np.newaxis] + 0.001)
grad1, Fisher1 = model.getDerivativesMinusLogPosterior_ensemble(particles)
grad2 = jacobian(model.getMinusLogPosterior___)(particles)[0,0]
np.allclose(grad1[0], grad2)

#%%
# Fisher tests
injParams_copy  = copy.deepcopy(injParams)
injParams_copy['tcoal'] /= model.time_scale # Put back into units of days for gwfast methods.
net = DetNet(model.detsInNet)
true_injection  = model.true_params[np.newaxis]
grad1, Fisher1  = model.getDerivativesMinusLogPosterior_ensemble(true_injection)

fisher_manual   = np.array(Fisher1[0])
fisher_gwfast   = net.FisherMatr(copy.deepcopy(injParams_copy), res=float(model.grid_resolution), df=None, spacing='geom', use_prec_ang=False)

np.allclose(fisher_gwfast, fisher_manual)

# %%
