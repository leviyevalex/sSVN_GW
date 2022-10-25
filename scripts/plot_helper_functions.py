import h5py
import numpy as np
import os
import logging.config
import matplotlib.pyplot as plt
log = logging.getLogger(__name__)

#%% Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def set_size(width, fraction=1, subplots=(1, 1)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predefined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])
    # fig_height_in = fig_width_in

    return (fig_width_in, fig_height_in)

def collect_samples(file, window=None, skipsize=None):
    with h5py.File(file, 'r') as f:
        if window is None:
            iter_window_max = f['metadata']['L'][()]
            iter_window_min = int(np.floor(iter_window_max * .75))
        else:
            iter_window_max = window[1]
            iter_window_min = window[0]
        if skipsize is None:
            skipsize = 1

        n = f['metadata']['nParticles'][()]
        d = f['metadata']['DoF'][()]

        # window = int(iter_window_max - iter_window_min)
        collect_at = np.arange(iter_window_min, iter_window_max, skipsize)
        number_iters = len(collect_at)
        samples = np.zeros((number_iters * n, d))
        for l in range(number_iters):
            samples[l * n : n * (l + 1), 0 : d] = f['%i' % collect_at[l]]['X'][()]
    return samples

def extract_moments(file):
    with h5py.File(file, 'r') as hf:
        iters_performed = hf['metadata']['L'][()]
        DoF = hf['metadata']['DoF'][()]
        mean_history = np.zeros((iters_performed, DoF))
        cov_history = np.zeros((iters_performed, DoF))
        # for l in np.arange(0, iters_performed, 10):
        for l in range(iters_performed):
            X = hf['%i' % l]['X'][()]
            mean = np.mean(X, axis=0)
            cov = np.cov(X.T)
            for d in range(DoF):
                mean_history[l,d] = mean[d]
                cov_history[l,d] = cov[d, d]
    return {'mean_history': mean_history, 'cov_history': cov_history, 'DoF':DoF}

def extract_gmlpt_norm(file):
    with h5py.File(file, 'r') as hf:
        iters_performed = hf['metadata']['L'][()]
        norm_history = np.zeros((iters_performed))
        for l in range(iters_performed):
            gmlpt = hf['%i' % l]['gmlpt'][()]
            norm_history[l] = np.linalg.norm(gmlpt)
        return norm_history

def extract_velocity_norms(file):
    with h5py.File(file, 'r') as hf:
        iters_performed = hf['metadata']['L'][()]
        norm_history_svgd = np.zeros((iters_performed))
        norm_history_svn = np.zeros((iters_performed))
        for l in range(iters_performed):
            v_svgd = hf['%i' % l]['v_svgd'][()]
            v_svn = hf['%i' % l]['v_svn'][()]
            norm_history_svgd[l] = np.linalg.norm(v_svgd)
            norm_history_svn[l] = np.linalg.norm(v_svn)
        return {'vsvn': norm_history_svn, 'vsvgd': norm_history_svgd}

