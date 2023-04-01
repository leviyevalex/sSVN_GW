""" Perform sanity checks to make sure likelihood is calculated correctly. """
#%%
# Export PATH
import sys
sys.path.append("..")
from gwfast.waveforms import TaylorF2_RestrictedPN, IMRPhenomD
import numpy as np
from astropy.cosmology import Planck18
import pycbc.waveform
import pycbc.types as pcbcdt
import matplotlib.pyplot as plt
ourcodenameString = 'WF4Py'
from matplotlib import gridspec

def m1m2_from_Mceta(Mc, eta):
    # Define a function to compute the component masses of a binary given its chirp mass and symmetric mass ratio
    m1 = 0.5*(Mc/(eta**(3./5.)))*(1.+np.sqrt(1.-4.*eta))
    m2 = 0.5*(Mc/(eta**(3./5.)))*(1.-np.sqrt(1.-4.*eta))
    return m1, m2

#%%
### Define injection parameters and waveform model to compare ###

zs = np.array([.2])
events = {'Mc':np.array([30])*(1.+zs), 'dL':(Planck18.luminosity_distance(zs).value/1000.), 
         'iota':np.array([.0]), 'eta':np.array([0.24]), 'chi1z':np.array([0.8]), 'chi2z':np.array([-0.8]), 
          'Lambda1':np.array([0.]), 'Lambda2':np.array([0.])}

# approximant, gwfast_WF = ('IMRPhenomD', IMRPhenomD())
approximant, gwfast_WF = ('TaylorF2', TaylorF2_RestrictedPN())

# Get frequency grid

fcut      = gwfast_WF.fcut(**events)
fminarr   = np.full(fcut.shape, 5)
fgrids    = np.geomspace(fminarr, fcut, num=int(1000))


### Get GWfast waveform ###
gwfast_ampl    = gwfast_WF.Ampl(fgrids, **events)
gwfast_phase   = gwfast_WF.Phi(fgrids, **events)

### Get pyCBC waveform ###
m1s, m2s = m1m2_from_Mceta(events['Mc'], events['eta'])
hp, hc = pycbc.waveform.get_fd_waveform_sequence(approximant=approximant,                                        
                                                 mass1=m1s,
                                                 mass2=m2s,
                                                 spin1z=events['chi1z'],
                                                 spin2z=events['chi2z'],
                                                 sample_points = pcbcdt.array.Array(fgrids[:,0]),
                                                 distance=events['dL']*1000.)

PyCBCwfAmpl  = np.array(abs(hp))
PyCBCwfPhase = np.array(hp/abs(hp)) # Why divide by abs hp?


#%%
fig = plt.figure(figsize=(10,12))
gs = gridspec.GridSpec(3,1,height_ratios=[2,1,.9])
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1], sharex=ax1)
ax3 = plt.subplot(gs[2], sharex=ax1)
ax1.plot(fgrids, PyCBCwfAmpl, 'C1', label='LAL', alpha=.35, linewidth=6.)
ax1.plot(fgrids, gwfast_ampl, 'C0', label=ourcodenameString)
ax1.set_yscale('log')
ax1.set_ylabel(r'$A_+ \, ({\rm Hz}^{-1})$',fontsize=20)
ax1.grid()
ax2.plot(fgrids, PyCBCwfPhase.real, 'C1', alpha=.35, linewidth=6.)
ax2.plot(fgrids[:,0], np.cos(gwfast_phase[:,0]), 'C0',)
ax2.set_ylabel(r'$\cos(\Psi_{+})$', fontsize=20)
plt.xlabel(r'$f \, (\rm Hz)$', fontsize=20)
yticks = ax2.yaxis.get_major_ticks()
yticks[-1].label1.set_visible(False)
ax3.plot(fgrids[:,0], abs(1.-(gwfast_ampl[:,0])/(PyCBCwfAmpl)), '.', color='C2', ms=3)
ax3.set_ylabel(r'$\rm res$', fontsize=20)
ax1.set_xscale('log')
ax3.set_yscale('log')

plt.subplots_adjust(hspace=0.)

handles, labels = ax1.get_legend_handles_labels()
ax1.legend(handles[::-1], labels[::-1], loc='lower left', fontsize=20)

ax1.tick_params(axis='both', which='major', labelsize=14)
ax1.tick_params(axis='both', which='minor', labelsize=14)
ax2.tick_params(axis='both', which='major', labelsize=14)
ax2.tick_params(axis='both', which='minor', labelsize=14)
ax3.tick_params(axis='both', which='major', labelsize=14)
ax3.tick_params(axis='both', which='minor', labelsize=14)

plt.xlim(min(fgrids), max(fgrids))
ax1.grid(True, which='both', ls='dotted', linewidth='0.8', alpha=.8)
ax2.grid(True, which='both', ls='dotted', linewidth='0.8', alpha=.8)
ax3.grid(True, which='both', ls='dotted', linewidth='0.8', alpha=.8)

# ax1.set_title(r"{$\bf \texttt{IMRPhenomD}$}", fontsize=25)
#%%
plt.show()
# %%
