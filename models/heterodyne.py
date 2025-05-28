import numpy as np
import jax.numpy as jnp
import jax 

class heterodyne:
    def __init__(self, model, chi, eps):
        self.strain = model.strain
        self.d_dense = model.data
        self.fgrid_dense = model.frequency 
        self.h0_dense = self.strain(model.injection, self.fgrid_dense)
        self.d_d = model.SNR ** 2
        self.fmin = model.fmin
        self.fmax = model.fmax
        self.df_dense = model.deltaf
        self.nbins_dense = model.n_bins
        self.PSD_dense = model.PSD

        self.getHeterodyneBins(chi=chi, eps=eps)
        self.subgrid = self.fgrid_dense[self.subgrid_idxs]
        self.precomputeSummaryData()

    def precomputeSummaryData(self):
        def sumBins(array, bin_indicies):
            """
            Given an `array`, and `bin_indicies` which defines how to partition `array`, return
            sum in each partition
            """
            tmp = np.zeros(len(array) + 1).astype(array.dtype)
            tmp[1:] = np.cumsum(array) # (iii)
            tmp[-2] = tmp[-1] # (iv) 
            return tmp[bin_indicies[1:]] - tmp[bin_indicies[:-1]] # (v) 

        def getBinIds(grid, bins):
            """ 
            Given bins, returns an array labeling which bin each point in grid belongs to.
            Bins are labeled beginning from 0 to nbins - 1!
            """
            bin_ids = (np.digitize(grid, bins)) - 1 # (ia), (ib)
            bin_ids[-1] = len(bins) - 2 # (ic)
            return bin_ids

        elements_per_bin = np.bincount(getBinIds(self.fgrid_dense, self.bin_edges)) # (ii)

        deltaf_in_bin = self.fgrid_dense - np.repeat(self.bin_edges[:-1], elements_per_bin)

        A0_integrand = 4 * self.h0_dense.conjugate() * self.d_dense / self.PSD_dense * self.df_dense 
        A1_integrand = A0_integrand * deltaf_in_bin
        B0_integrand = 4 * (self.h0_dense.real ** 2 + self.h0_dense.imag ** 2) / self.PSD_dense * self.df_dense
        B1_integrand = B0_integrand * deltaf_in_bin

        self.A0 = jnp.array(sumBins(A0_integrand, self.subgrid_idxs))
        self.A1 = jnp.array(sumBins(A1_integrand, self.subgrid_idxs))
        self.B0 = jnp.array(sumBins(B0_integrand, self.subgrid_idxs))
        self.B1 = jnp.array(sumBins(B1_integrand, self.subgrid_idxs))

    # Remark: The output of this function should be () shaped for autodiff to correctly parallelize
    def getMinusLogLikelihood(self, x):
        h = self.strain(x, self.subgrid)
        r = h / self.h0_dense[self.subgrid_idxs] # (i)
        r0 = r[:-1] # (ii)
        r1 = (r[1:] - r[:-1]) / self.bin_widths

        h_d = jnp.sum(self.A0[jnp.newaxis] * r0.conjugate() + self.A1[jnp.newaxis] * r1.conjugate(), axis=1).squeeze()
        h_h = jnp.sum(self.B0[jnp.newaxis] * jnp.abs(r0) ** 2 + 2 * self.B1[jnp.newaxis] * (r0.conjugate() * r1).real, axis=1).squeeze()
        
        return 0.5 * h_h - h_d.real + 0.5 * self.d_d

    def getHeterodyneBins(self, chi, eps): # Checks X
        # TODO: Replace with simpler binning scheme!
        gamma = np.array([-5/3, -2/3, 1, 5/3, 7/3])
        f_star = self.fmax * np.heaviside(gamma, 0.5) + self.fmin * np.heaviside(-gamma, 0.5) # (i) 
        delta = lambda f_minus, f_plus: 2 * np.pi * chi * np.sum(np.abs((f_plus[:, np.newaxis] / f_star) ** gamma - (f_minus[:, np.newaxis] / f_star) ** gamma), axis=-1)
        delta_single = np.max(delta(self.fgrid_dense[:-1], self.fgrid_dense[1:]))
        assert delta_single < eps # (iii)
        subindex = [0] # (ii)
        index_f_minus = 0
        j = 1
        while j <= self.nbins_dense:
            if j == self.nbins_dense:
                subindex.append(j)
                break
            d = delta(self.fgrid_dense[index_f_minus, np.newaxis], self.fgrid_dense[j, np.newaxis])[0]
            if d >= eps:
                subindex.append(j - 1)
                index_f_minus = j - 1
                continue
            j += 1

        self.subgrid_idxs = np.array(subindex)
        self.bin_edges = self.fgrid_dense[self.subgrid_idxs]
        self.nbins = len(subindex) - 1
        self.bin_widths = self.bin_edges[1:] - self.bin_edges[:-1]
        print('Heterodyne binning scheme: %i' % self.nbins)
