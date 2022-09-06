import gwfast

#%%

#%%
import copy
import gwfast.gwfastGlobals as glob
import os
#%%

alldetectors = copy.deepcopy(glob.detectors)
print('All available detectors are: '+str(list(alldetectors.keys())))

# select only LIGO and Virgo
LVdetectors = {det:alldetectors[det] for det in ['L1', 'H1', 'Virgo']}
print('Using detectors '+str(list(LVdetectors.keys())))
#%%
# We use the O2 psds
LVdetectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt')
LVdetectors['H1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt')
LVdetectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', 'Hrec_hoft_V1O2Repro2A_16384Hz.txt')
#%%
# from gwfast.waveforms import IMRPhenomD_NRTidalv2
from gwfast.waveforms import TaylorF2_RestrictedPN
from gwfast.waveforms import IMRPhenomD
from gwfast.signal import GWSignal
from gwfast.network import DetNet
from fisherTools import CovMatr, compute_localization_region, check_covariance, fixParams
#%%
myLVSignals = {}
#%%
for d in LVdetectors.keys():
    # myLVSignals[d] = GWSignal(IMRPhenomD_NRTidalv2(),
    myLVSignals[d] = GWSignal(TaylorF2_RestrictedPN(),
                              psd_path=LVdetectors[d]['psd_path'],
                              detector_shape=LVdetectors[d]['shape'],
                              det_lat=LVdetectors[d]['lat'],
                              det_long=LVdetectors[d]['long'],
                              det_xax=LVdetectors[d]['xax'],
                              verbose=True,
                              useEarthMotion=False,
                              fmin=10.,
                              IntTablePath=None)
#%%
myLVNet = DetNet(myLVSignals)
#%%






