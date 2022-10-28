""" 
Perform BILBY runs exclusively using BILBY.
"""

#%% 
# Import libraries
import sys
sys.path.append("..")
import numpy as np
import bilby
import copy
import gwfast.gwfastGlobals as glob
import os
import numpy as np
import matplotlib.pyplot as plt

#%% 

# Parameters chosen to simulate GW150914 with gwfast

tGPS    = 1.12625946e+09 
Mc      = 34.3089283        # Units: M_sun
eta     = 0.2485773         # Units: Unitless           (Symmetric mass ratio)
dL      = 2.634             # Units: Gigaparsecs 
theta   = 2.78560281        # Units: Rad
phi     = 1.67687425        # Units: Rad
iota    = 2.67548653        # Units: Rad
psi     = 0.78539816        # Units: Rad
tcoal   = 0.                # Units: Fraction of day 
Phicoal = 0.                # Units: Rad
chi1z   = 0.27210419        # Units: Unitless
chi2z   = 0.33355909        # Units: Unitless

# params = ['chirp_mass','symmetric_mass_ratio', 'chi_1', 'chi_2', 'ra', 'dec', 'luminosity_distance', 'theta_jn', 'psi', 'phase', 'geocent_time']

#%%

# Model details

approximant = "IMRPhenomD"
injection_parameters = dict(chirp_mass            = Mc,              # Question: Does BILBY expect chirp mass in units of M_sun?
                            symmetric_mass_ratio  = eta,             # Alternative: mass_ratio
                            luminosity_distance   = dL,
                            dec                   = np.pi/2 - theta,
                            ra                    = theta,
                            theta_jn              = iota,
                            psi                   = psi,
                            geocent_time          = tGPS,
                            phase                 = Phicoal,
                            chi_1                 = chi1z,
                            chi_2                 = chi2z)

#%%

# Grid details

minimum_frequency = 10
reference_frequency = 100
duration = 256
sampling_frequency = 4096

#%%

prior_ranges = {}
prior_ranges['chirp_mass']            = [29., 39.]
prior_ranges['symmetric_mass_ratio']  = [0.22, 0.25]
prior_ranges['luminosity_distance']   = [0.1, 4.]
prior_ranges['dec']                   = [-np.pi / 2, np.pi / 2]
prior_ranges['ra']                    = [0., 2 * np.pi]
prior_ranges['theta_jn']              = [0, np.pi]
prior_ranges['psi']                   = [0, np.pi]
prior_ranges['geocent_time']          = [tGPS - 2, tGPS + 2]
prior_ranges['phase']                 = [Phicoal - 1e-7, Phicoal + 1e-7]
prior_ranges['chi_1']                 = [-1., 1.]
prior_ranges['chi_2']                 = [-1., 1.]

priors = {}
for param in prior_ranges.keys():
    priors[param] = bilby.core.prior.Uniform(prior_ranges[param][0], prior_ranges[param][1])

# Setup Bilby uniform prior over all parameters.

# priors = {}
# priors['chirp_mass']            = bilby.core.prior.Uniform(29., 39.)
# priors['symmetric_mass_ratio']  = bilby.core.prior.Uniform(0.22, 0.25)
# priors['luminosity_distance']   = bilby.core.prior.Uniform(0.1, 4.)
# priors['dec']                   = bilby.core.prior.Uniform(-np.pi / 2, np.pi / 2)
# priors['ra']                    = bilby.core.prior.Uniform(0., 2 * np.pi)
# priors['theta_jn']              = bilby.core.prior.Uniform(0, np.pi)
# priors['psi']                   = bilby.core.prior.Uniform(0, np.pi)
# priors['geocent_time']          = bilby.core.prior.Uniform(tGPS - 2, tGPS + 2)
# priors['phase']                 = bilby.core.prior.Uniform(Phicoal - 1e-7, Phicoal + 1e-7)
# priors['chi_1']                 = bilby.core.prior.Uniform(-1., 1.)
# priors['chi_2']                 = bilby.core.prior.Uniform(-1., 1.)

#%%

# Waveform generators provide a unified method to call disparate source models. Inject signal

waveform_generator = bilby.gw.WaveformGenerator(duration                      = duration,
                                                sampling_frequency            = sampling_frequency,
                                                frequency_domain_source_model = bilby.gw.source.lal_binary_black_hole,
                                                waveform_arguments            = dict(waveform_approximant=approximant, reference_frequency=reference_frequency),
                                                parameter_conversion          = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)
                                                

ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

#%%

# What exactly does this do? Zero noise injection ? 

# for det in ifos:
#     det.set_strain_data_from_zero_noise(sampling_frequency = sampling_frequency, 
#                                         duration           = duration, 
#                                         start_time         = injection_parameters["geocent_time"] - duration + 2)

ifos.set_strain_data_from_power_spectral_densities(sampling_frequency = sampling_frequency,
                                                   duration           = duration,
                                                   start_time         = injection_parameters["geocent_time"] - duration + 2)

ifos.inject_signal(waveform_generator=waveform_generator, parameters=injection_parameters)

#%%

# Sets the minimum frequency for each detector

for ifo in ifos:
    ifo.minimum_frequency = minimum_frequency

# make waveform generator for likelihood evaluations

search_waveform_generator =  \
bilby.gw.waveform_generator.WaveformGenerator(duration                      = duration,
                                              sampling_frequency            = sampling_frequency,
                                              frequency_domain_source_model = bilby.gw.source.binary_black_hole_frequency_sequence,
                                              waveform_arguments            = dict(waveform_approximant=approximant, reference_frequency=reference_frequency),
                                              parameter_conversion          = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters)

likelihood = bilby.gw.likelihood.BasicGravitationalWaveTransient(interferometers = ifos, waveform_generator = waveform_generator)

#%%
def L(theta):
    param_dict = {}
    for i in range(11):
        param_dict[list(prior_ranges.keys())[i]] = theta[i]
    likelihood.parameters = param_dict
    return likelihood.log_likelihood

getMinusLogPosterior_ensemble = lambda X: np.apply_along_axis(L, 1, X)
nParticles = 200 ** 2
DoF = 11
true_params = [list(injection_parameters.values())]

#%%
def getCrossSection(a, b):
    # a, b are the parameters for which we want the marginals
    ngrid = int(np.sqrt(nParticles))
    x = np.linspace(prior_ranges[a][0], prior_ranges[a][1], ngrid)
    y = np.linspace(prior_ranges[b][0], prior_ranges[b][1], ngrid)
    X, Y = np.meshgrid(x, y)

    particle_grid = np.tile(true_params, ngrid ** 2).reshape(ngrid ** 2, DoF)

    index1 = list(prior_ranges.keys()).index(a)
    index2 = list(prior_ranges.keys()).index(b)
    
    parameter_mesh = np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y)))
    
    particle_grid[:, index1] = parameter_mesh[0]
    particle_grid[:, index2] = parameter_mesh[1]
    
    Z = np.exp(-1 * getMinusLogPosterior_ensemble(particle_grid).reshape(ngrid,ngrid))

    # Plot
    fig, ax = plt.subplots(figsize = (5, 5))
    cp = ax.contourf(X, Y, Z)
    plt.colorbar(cp)
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.set_title('Analytically calculated marginal')
    filename = a + b + '.png'
    path = os.path.join('marginals', filename)
    fig.savefig(path)

getCrossSection('chirp_mass', 'symmetric_mass_ratio')
# %%
