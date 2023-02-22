#%%
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
import scipy

#%%
# Adaptive bin tolerance scheme
f_max = 512
f_min = 20
gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3])
chi = 0.1
eps = 0.5
num_grid_ticks = 100000
f_grid = np.linspace(f_min, f_max, num_grid_ticks) # Begin with dense grid
bound = lambda f_minus, f_plus: 2 * np.pi * chi * np.sum((1 - (f_minus / f_plus) ** np.abs(gamma)))
bin_edges = [f_min]
indicies_kept = [0]
max_iter = num_grid_ticks
i = 0 # grid index
j = 0 # bin edge index
while i < num_grid_ticks:
    while i < num_grid_ticks and bound(bin_edges[j], f_grid[i]) < eps:
        i += 1
    bin_edges.append(f_grid[i - 1])
    indicies_kept.append(i - 1)
    j += 1
print(len(bin_edges))
#%% Calculating summary data


#%% 

#%%
#%%
# Evidence suggesting a finer sampling at lower frequencies is better
f_max = 600
f_min = 20
gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3]) # Array of powers coming in the different PN corrections
m = 100
eps = 0.03
chi = 0.1
tau = (f_max - f_min) / m
b = np.arange(1, m + 1)
tmp = f_min / (f_max - (m - b) * tau) + (b - 1) * tau / (f_max - (m - b) * tau)
tmp_1 = tmp[..., np.newaxis] ** np.abs(gamma)
tmp_2 = 1 - tmp_1
deltas = 2 * np.pi * chi * np.sum(tmp_2, axis=1)
fig, ax = plt.subplots()
ax.plot(b, deltas, c='b')
ax.plot(b, np.ones(b.shape[0]) * eps, c='r')
fig.show()

#%%
f_max = 600
f_min = 20
gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3]) # Array of powers coming in the different PN corrections
# delta_support = np.sum((1 - (f_min / f_max) ** np.abs(gamma)))
# m = int(np.ceil(delta_support / eps))
eps = 0.03
chi = 0.1
m = 

tau = (f_max - f_min) / m
b = np.arange(1, m + 1)

tmp = f_min / (f_max - (m - b) * tau) + (b - 1) * tau / (f_max - (m - b) * tau)
tmp_1 = tmp[..., np.newaxis] ** np.abs(gamma)
tmp_2 = 1 - tmp_1
deltas = 2 * np.pi * chi * np.sum(tmp_2, axis=1)

fig, ax = plt.subplots()
ax.plot(b, deltas, c='b')
ax.plot(b, np.ones(b.shape[0]) * eps, c='r')
#%%
    

# while i < num_grid_ticks:
#     # print(i)
#     # print(j)
#     if bound(bin_edges[j], f_grid[i]) >= eps:
#         i += 1
#     else:
#         while i < num_grid_ticks and bound(bin_edges[j], f_grid[i]) < eps:
#             i += 1
#         bin_edges.append(f_grid[i-1])
#         indicies_kept.append(i-1)
#         j += 1




#%%
# Scheme 1
eps = 0.03
chi = 0.1
gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3]) 
for m in np.arange(20, 1000):
    tau = (f_max - f_min) / m
    print(m)
    bound = 2 * np.pi * chi * np.sum(1 - (f_min / (f_max - (m-1) * tau)) ** np.abs(gamma))
    if bound < eps:
        n_target = m
        break






#%%








#%%
#########################
# Preliminary definitions
#########################

delta = 0.03     # Error tolerance for each bin
f_min = 20       # 
f_max = 600      #
f_grid = np.linspace(f_min, f_max, 10000) # Preliminary dense grid
gamma = np.array([-5./3, -2./3, 1., 5./3, 7./3]) # Array of powers coming in the different PN corrections
chi = 0.1
bound = 2 * np.pi * chi * np.sum((1 - (f_min / f_max) ** np.abs(gamma))) # Get bound over [f_min, f_max]
n_bins = int(np.ceil(bound / delta)) # 

print(n_bins)

#%%
tau = (f_max - f_min) / n_bins
b = np.arange(1, n_bins)
np.sum(1 - ((f_min + (b - 1) * tau) / (f_min + b * tau))^np.abs(gamma))


#%%
bs = np.arange(1, n_bins)
tmp_func = lambda b: np.sum(1 - ((f_min + (b - 1) * tau) / (f_min + b * tau))^np.abs(gamma))
tmp_func(bs)
#%%

import matplotlib.pyplot as plt
plt.plot(func(bs))






#%%
#%%
bin_edges = np.linspace(f_min, f_max, n_bins + 1) # n_bins = ticks - 1

frequencies_in_interval = lambda f_minus, f_plus: np.where(np.logical_and(f_minus <= f_grid, f_grid <= f_plus))[0]

f_bin0 = frequencies_in_interval(bin_edges[0], bin_edges[1])


#%% Calculating summary data

A0 = 0j
A1 = 0j
B1 = 0j
B2 = 0j
for b in range(n_bins):
    index = frequencies_in_interval(bin_edges[b], bin_edges[b+1])
    tmp1 = d[index] * h0[index].conjugate() / 
    A0 += 



dig = np.digitize(f_grid, bin_edges)
dig[-1] = 
# f_bar = (f_heterodyned[1:] - f_heterodyned[:-1]) / 2

# np.random.rand(f_dense.size())

# scipy.stats.binned_statistic(x, values, statistic='mean', bins=10, range=None)

A_0 = np.zeros(n_bins) 
A_1 = np.zeros(n_bins)
B_0 = np.zeros(n_bins) 
B_1 = np.zeros(n_bins) 

for i in range(f_grid.size()):





#%%

#%%
import scipy
# arr = np.sort([20, 2, 7, 1, 34])
arr = np.sort([1 + 5j, 2, 7, 20, 34])


f = lambda x: np.mean(x)

A_0 = lambda f: np.sum()



a = scipy.stats.binned_statistic(arr, np.arange(5), statistic=f, bins=2)

# a = scipy.stats.binned_statistic(arr, strain * h0.conjugate() / PSD / T, statistic='sum', bins=n_bins)

print(a[0])



#%%



#%%





#%%

























#%%
 # make array of frequencies

delta = 0.03

f_min = 20
f_max = 600

f_nt = np.linspace(f_min, f_max, 10000)

# array of powers coming in the different PN corrections
ga = np.array([-5./3, -2./3, 1., 5./3, 7./3])

# compute the coefficient from https://arxiv.org/pdf/1806.08792.pdf
dalpha = 2 * np.pi / np.abs(f_min**ga - f_max**ga)
dphi = np.sum([np.sign(g) * d * f_nt**g for d, g in zip(dalpha, ga)], axis = 0)
dphi -= dphi[0]

# construct the frequency bins
nbin = int(np.ceil(dphi[-1] / delta))
dphi_grid = np.linspace(dphi[0], dphi[-1], nbin+1)
fbin = np.interp(dphi_grid, dphi, f_nt)

#%%


# set the bin edges to values for which we have template values
fbin_idxs = np.unique(np.argmin(np.abs(waveform_generator.frequency_array[:, np.newaxis] - self.fbin), axis = 0))
fbin = waveform_generator.frequency_array[self.fbin_idxs]
Nbin = len(fbin) - 1 # number of bins
fm = (fbin[1:] + self.fbin[:-1])/2.
binwidths = fbin[1:] - self.fbin[:-1] # binwidths

























































#%%
# rv = norm()
def trunc_gauss_pdf(x, a, b):
    y = np.zeros(x.shape[0])
    idx = np.where(np.logical_and(x > a, x < b))
    y[idx] = norm.pdf(x[idx], loc=0, scale=1)
    return y

def smoothed_trunc_gauss(x, a, b, t):
    y = np.zeros(x.shape[0])
    idx = np.where(np.logical_and(x > a, x < b))
    y[idx] = norm.pdf(x[idx], loc=0, scale=1) * ((x[idx]-a) * (b-x[idx])) ** (1 / t)
    return y


a = -.5
b = 1
x = np.linspace(a - 5, a + 5, 1000)
y_trunc = trunc_gauss_pdf(x, a, b)
fig, ax = plt.subplots(1, 1)
y_smooth_a = smoothed_trunc_gauss(x, a, b, t=100)
# y_smooth_b = smoothed_trunc_gauss(x, a, b, t=0.1)
# y_smooth_c = smoothed_trunc_gauss(x, a, b, t=0.01)

# y_b = y * ((x-a) * (b-x)) ** t
max_trunc = np.max(y_trunc)

plt.plot(x, y_trunc)
plt.plot(x, y_smooth_a)
# plt.plot(x, y_smooth_b)
# plt.plot(x, y_smooth_c)

#%%








































#%%
import numpy as np
from numdifftools import Hessian, Jacobian, Gradient
from models.vM_sphere import vM_sphere

kappa = 100
thetaTrue = np.array([0, np.pi/2])
model = vM_sphere(kappa=kappa, thetaTrue=thetaTrue)
theta = np.array([np.pi / 4, np.pi])
#%%
model.getMinusLogLikelihood(theta)
test_a = Gradient(model.getMinusLogLikelihood)(theta)
test_b = model.getGradientMinusLogLikelihood(theta)






#%% Confirm Jacobian is calculated properly
test5 = model.getJacobianEmbedding(theta)
test6 = Jacobian(model.embedding)(theta)
np.allclose(test5, test6)
#%%
testa = model.getHessianEmbedding(theta)
testb = Jacobian(Jacobian(model.embedding))(theta)
np.allclose(testa, testb)
#%%
test1 = model.getGradientMinusLogLikelihood(theta)
test2 = Jacobian(model.getMinusLogLikelihood)(theta)
np.allclose(test1, test2)

#%%
test3 = model.getGNHessianMinusLogLikelihood(theta)
test4 = Hessian(model.getMinusLogLikelihood)(theta)
np.allclose(test3, test4)


#%% Failing!
test_A = model.getGradientJacobian(theta)
test_B = Jacobian(model.getJacobianEmbedding)(theta)
test_C = Jacobian(Jacobian(model.embedding))(theta)
print(np.allclose(test_A[:, :, 0], test_B[:, :, 0]))
print(np.allclose(test_A[:, :, 1], test_B[:, :, 1]))

#%%
test5 = Jacobian(model.getJacobianEmbedding)(theta)
test6 = Jacobian(Jacobian(model.embedding))(theta)
np.allclose(test5, test6)




#%%
a = model.getGradientJacobian(theta)
b = Jacobian(Jacobian(model.embedding))(theta)
np.allclose(b, a)


#%%
def getGeodesicDist_circle(theta_1, theta_2):
    diff = np.abs(theta_2 - theta_1) % (2 * np.pi)
    return min(diff, 2 * np.pi - diff)

    # if tmp > np.pi:
    #     return 2 * np.pi - tmp
    # else:
    #     return tmp
#%%
# Test case 1
theta_1 = np.pi/4
theta_2 = 7 * np.pi / 4
res = getGeodesicDist_circle(theta_1, theta_2)
assert res == np.pi/2
#%%
# Test case 2
theta_1 = 3 * np.pi/4
theta_2 = np.pi
res = getGeodesicDist_circle(theta_1, theta_2)
assert res == np.pi/4
#%%
def getGeodesicDist_sphere(x, y):
    theta_1 = x[0]
    phi_1 = x[1]
    theta_2 = y[0]
    phi_2 = y[1]

    tmp = np.sin(theta_1) * np.sin(theta_2) * np.cos(phi_1 - phi_2) + np.cos(theta_1) * np.cos(theta_2)

    tol = 1e-9
    if tmp > 1 and tmp < 1 + tol:
        tmp = 1
    elif tmp < 0 and tmp > -tol:
        tmp = 0

    return np.arccos(tmp)
#%%
# Test case 1
x = np.array([0, 0])
y = np.array([np.pi/2, np.pi/2])
getGeodesicDist_sphere(x, y)
#%%
# Test case 2
x = np.array([0, 0])
y = np.array([np.pi/2, 0])
getGeodesicDist_sphere(x, y)
#%%
# Test case 3
x = np.array([0, 0])
y = np.array([np.pi, 0])
getGeodesicDist_sphere(x, y)
#%%
# Test case 4
x = np.array([np.pi / 4, np.pi/4])
y = np.array([0, 0])
getGeodesicDist_sphere(x, y)

#%%
import scipy
from scipy import spatial
from opt_einsum import contract


# Test 1
thetas = np.array([np.pi/4, 3 * np.pi/4, 5 * np.pi / 4])
# res = d_circ(thetas[...,np.newaxis])

# def g_d_circ(thetas):
#     n = thetas.shape[0]
#     return -1 * np.ones((n, n))

def getKernelWithDerivatives_S1(X, h):
    # Geodesic Gaussian kernel on S1
    # Returns kernel and gradient of the kernel
    def d_circ(thetas):
        # Get n x n matrix of distances given vector of thetas
        tmp = spatial.distance_matrix(thetas, thetas)
        return np.minimum(tmp, 2 * np.pi - tmp)
    d = d_circ(X)
    kx = np.exp(-d ** 2 / h)
    gkx = (2 * kx * d / h)[..., np.newaxis]
    return kx, gkx
#%%
# thetas = np.array([[np.pi/4, np.pi/2],
#                    [np.pi/3, np.pi/5],
#                    [np.pi/7, np.pi]])


# How do we generalize the single geodesic distance on a sphere to an n sample distance?

# from scipy.spatial import distance
# pdist with custom metric is buggy in scipy
# dm = distance.pdist(thetas, getGeodesicDist_sphere)

#%%
from sklearn import metrics
from itertools import product
thetas = np.array([[0.1, 0.4],
                   [0.2, 0.5],
                   [0.3, 0.6]])
res1 = metrics.pairwise_distances(thetas, thetas, getGeodesicDist_sphere)
for m, n in product(range(3), range(3)):
    x = thetas[m]
    y = thetas[n]
    test1 = getGeodesicDist_sphere(x, y)
    assert np.allclose(test1, res1[m,n])

#%%
res2 = metrics.pairwise_distances(thetas, metric=getGeodesicDist_sphere)
#%%
def getKernelWithDerivatives_S2(X, h):
    n = X.shape[0]
    d = metrics.pairwise_distances(thetas, metric=getGeodesicDist_sphere)
    gd = np.zeros((n, n, 2))
    eta = np.cos(d)
    denominator = np.sqrt(1 - eta ** 2)

    kx = np.exp(-d ** 2 / h)

    tmp1 = lambda x, y: np.sin(x[0]) * np.cos(y[0]) - np.cos(x[0]) * np.sin(y[0]) * np.cos(x[1] - y[1])
    tmp2 = lambda x, y: np.sin(x[0]) * np.sin(y[0]) * np.sin(x[1] - y[1])

    a = metrics.pairwise_distances(X, metric=tmp1) / denominator
    b = metrics.pairwise_distances(X, metric=tmp2) / denominator

    gd[..., 0] = a
    gd[..., 1] = b

    gkx1 = -2 * contract('mn, mni -> mni', kx * d, gd) / h

    return kx, gkx1

#%%
import numpy as np
np.array([[1, 2],
          [3, 4],
          [5, 6]]).T
