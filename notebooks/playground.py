#%%
import numpy as np

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
res = d_circ(thetas[...,np.newaxis])

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

def getKernelWithDerivatives_S2(X, h):
