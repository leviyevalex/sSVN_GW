#%%
import sys
sys.path.append("..")
import numpy as np
import bilby
import copy
import gwfast.gwfastGlobals as glob
from gwfast.waveforms import TaylorF2_RestrictedPN 
import os
from models.gwfastWrapClass import gwfast_class

label = 'gwfast_run'
# outdir = "bilby_outdir"
# bilby.utils.check_directory_exists_and_if_not_mkdir(outdir)



# priors['dL'] = bilby.prior.Uniform(minimum=0.2, maximum=0.5, name='$d_L$')
# priors['theta'] = bilby.prior.Uniform(minimum=0., maximum=np.pi, name='$\theta$')
# priors['phi'] = bilby.prior.Uniform(minimum=0., maximum=2*np.pi, name='$\phi$')
# priors['cos_iota'] = bilby.prior.Uniform(minimum=-1., maximum=1., name='$\cos(\iota)$')
# priors['psi'] = bilby.prior.Uniform(minimum=0., maximum=np.pi, name='$\psi$')
# priors['tcoal'] = bilby.prior.Uniform(minimum=0, maximum=0.001, name='$t_c$')
# # priors['tcoal_prior'] = bilby.prior.Uniform(minimum=0, maximum=0.001, name='$t_c$')
# priors['Phicoal'] = bilby.prior.Uniform(minimum=0., maximum=0.001, name='$\phi_c$')
# priors['chi1z'] = bilby.prior.Uniform(minimum=-1., maximum=1., name='$\chi_{1z}$')
# priors['chi2z'] = bilby.prior.Uniform(minimum=-1., maximum=1., name='$\chi_{2z}$')

#%%
# Define likelihood
Mc = np.array([34.3089283]) # (1)
eta = np.array([0.2485773]) # (2)
dL = np.array([0.43929891]) # (3) # TODO: In gigaparsecs? true: proportional to dl^2 (for now use unif)
theta = np.array([2.78560281]) # (4) unif in cos
phi = np.array([1.67687425]) # (5) unif
iota = np.array([2.67548653]) # (6) unif in cos
psi = np.array([0.78539816]) # (7) 
tcoal = np.array([0.]) # (8) Unif super narrow
Phicoal = np.array([0.]) # (9) Unif
chi1z = np.array([0.27210419]) # (10) Unif
chi2z = np.array([0.33355909]) # (11) Unif
injParams = {'Mc': Mc, 'eta': eta, 'dL': dL, 'theta': theta, 'phi': phi, 'iota': iota, 'psi': psi, 'tcoal': tcoal, 'Phicoal': Phicoal, 'chi1z': chi1z, 'chi2z': chi2z}
theta_true = np.array([Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z])
alldetectors = copy.deepcopy(glob.detectors)
LVdetectors = {det:alldetectors[det] for det in ['L1', 'H1', 'Virgo']}
print('Using detectors '+str(list(LVdetectors.keys())))
LVdetectors['L1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-08-06_DCH_C02_L1_O2_Sensitivity_strain_asd.txt')
LVdetectors['H1']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', '2017-06-10_DCH_C02_H1_O2_Sensitivity_strain_asd.txt')
LVdetectors['Virgo']['psd_path'] = os.path.join(glob.detPath, 'LVC_O1O2O3', 'Hrec_hoft_V1O2Repro2A_16384Hz.txt')
waveform = TaylorF2_RestrictedPN()
# waveform = IMRPhenomD()
fmin = 20.
fmax = 325.
df = 1./5
model = gwfast_class(LVdetectors, waveform, injParams, df, fmin=fmin, fmax=fmax)
model.get_signal(method='sim', add_noise=False)

#%%
# Define priors
priors = {}

# priors['Mc'] = bilby.prior.Uniform(minimum=34, maximum=34.6, name='$\mathcal{M_c}$')
# priors['eta'] = bilby.prior.Uniform(minimum=0.247, maximum=0.25, name='$\ets$')
# priors['dL'] = dL[0]
# priors['theta'] = theta[0]
# priors['phi'] = phi[0]
# priors['iota'] = iota[0]
# priors['psi'] = psi[0]
# priors['tcoal'] = tcoal[0]
# priors['Phicoal'] = Phicoal[0]
# priors['chi1z'] = chi1z[0]
# priors['chi2z'] = chi2z[0]

bounds = {'Mc': [33.75, 34.75], 
          'eta': [0.245, 0.25], 
          'dL': [0.4, 0.47], 
          'theta': [2.7, 2.9], 
          'phi': [1.5, 1.8], 
          'iota': [2.5, 4.1], 
          'psi': [0.7, 0.9], 
          'tcoal': [0, 0.00000001], 
          'Phicoal': [0., 0.2], 
          'chi1z': [0.26, 0.285], 
          'chi2z': [0.31, 0.35]}

latex_names = {'Mc': '$\mathcal{M}_c$', 
               'eta': '$\eta$', 
               'dL': '$d_L$', 
               'theta': '$\theta$', 
               'phi': '$\phi$', 
               'iota': '$\iota$', 
               'psi': '$\psi$', 
               'tcoal': '$t_c$', 
               'Phicoal': '$\phi_c$', 
               'chi1z': '$\chi_{1z}$', 
               'chi2z': '$\chi_{2z}$'}

for param in model.names:
    priors[param] = bilby.prior.Uniform(minimum=bounds[param][0], maximum=bounds[param][1], name=latex_names[param])



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
        return -1 * model.getMinusLogLikelihood_ensemble(theta[np.newaxis, ...]).squeeze()

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