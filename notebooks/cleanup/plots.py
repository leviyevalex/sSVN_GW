#%%
import sys
sys.path.append("..")
import deepdish as dd
from chainconsumer import ChainConsumer
from scripts.plot_helper_functions import collect_samples
import numpy as np


Mc = np.array([34.3089283]) # (1)
eta = np.array([0.2485773]) # (2)
dL = np.array([0.43929891]) # (3) # TODO: In gigaparsecs? true: proportional to dl^2 (for now use unif)
theta = np.array([2.78560281]) # (4) unif in cos
phi = np.array([1.67687425]) # (5) unif
iota = np.array([2.67548653]) # (6) unif in cos
psi = np.array([0.78539816]) # (7) 
# tcoal = utils.GPSt_to_LMST(tGPS, lat=0., long=0.) # Coalescence time (GMST is LMST computed at long = 0°)
tcoal = np.array([0.]) # (8) Unif super narrow
Phicoal = np.array([0.]) # (9) Unif
chi1z = np.array([0.27210419]) # (10) Unif
chi2z = np.array([0.33355909]) # (11) Unif

X1 = dd.io.load('test2.h5')
params = [r"$\mathcal{M}_c$", r"$\eta$", r"$d_L$", r"$\theta$", r"$\phi$", r"$\cos(\iota)$", r"$\psi$", r"$t_c$", r"$\phi_c$", r"$\chi_1$", r"$\chi_2$"]
c = ChainConsumer().add_chain(X1, parameters=params)
true_params = np.array([Mc, eta, dL, theta, phi, iota, psi, tcoal, Phicoal, chi1z, chi2z])
summary = c.analysis.get_summary()
fig = c.plotter.plot_distributions(truth=true_params)
# %%
