#%%
import numpy as np
from numdifftools import Hessian, Jacobian, Gradient
from models.vM_sphere import vM_sphere
#%%
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
