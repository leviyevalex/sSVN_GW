import numpy as np
import h5py
import os
import logging
# Instantiate logging
log = logging.getLogger(__name__)

def create_contour(sampler, lower_bound, upper_bound):
    # Create a background contour file if necessary for given settings
    contour_file = "%s_%.2f_%.2f_%.2f_%.2f.h5" % (sampler.model.id, lower_bound[0], upper_bound[0], lower_bound[1], upper_bound[1])
    contour_file_path = os.path.join(sampler.OUTPUT_DIR, contour_file)
    if not os.path.exists(contour_file_path):
        log.info('Contour file does not exist for given settings! Creating...')
        ngrid = 500
        # x = np.linspace(-5, 4, ngrid)
        # y = np.linspace(-5, 10, ngrid)
        x = np.linspace(lower_bound[0], upper_bound[0], ngrid)
        y = np.linspace(lower_bound[1], upper_bound[1], ngrid)
        X, Y = np.meshgrid(x, y)
        Z = np.exp(-1 * sampler.model.getMinusLogPosterior_ensemble(np.vstack((np.ndarray.flatten(X), np.ndarray.flatten(Y))).T).reshape(ngrid,ngrid))
        log.info('Successfully created contour file for given settings.')
        with h5py.File(contour_file_path, 'a') as hf:
            hf.create_dataset("X", data=X)
            hf.create_dataset("Y", data=Y)
            hf.create_dataset("Z", data=Z)
    else:
        log.info('Using contour file for given settings found in /outdir ')
    return contour_file_path
