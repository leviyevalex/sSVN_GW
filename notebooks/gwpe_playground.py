#%%
from models.ripple.GWPE_JAX.Notebooks.likelihood import single_event_likelihood
from models.ripple.waveforms import IMRPhenomD
# single_detector_likelihood(waveform_model, params, data, data_f, PSD, detector):

waveform = IMRPhenomD()
single_event_likelihood()