import numpy as np

def getEmbedding(theta):
    """ Take coordinates in (theta,phi) space and map to (x,y,z) on sphere.
    Args:
        theta: (theta, phi), $\theta \in (0, 2 \pi), \phi \in (0, \pi)$

    Returns (np.array): (x, y, z) coordinates in $R^3$

    """
    x1 = np.sin(theta[1]) * np.cos(theta[0])
    x2 = np.sin(theta[1]) * np.sin(theta[0])
    x3 = np.cos(theta[1])
    return np.array([x1, x2, x3])

def getJacobianEmbedding(theta):
    """ Get Jacobian of embedding

    Args:
        theta:

    Returns:

    """
    theta_ = theta[0]
    phi = theta[1]
    return np.array([[-np.sin(theta_) * np.sin(phi), np.cos(theta_) * np.cos(phi)],
                     [np.cos(theta_) * np.sin(phi), np.sin(theta_) * np.cos(phi)],
                     [0, -np.sin(phi)]])

def getHessianEmbedding(x):
    """ Get Hessian of embedding

    Args:
        x:

    Returns:

    """
    gJ = np.zeros((3, 2, 2))
    layer0 = np.array([[-np.cos(x[0]) * np.sin(x[1]), -np.sin(x[0]) * np.cos(x[1])],
                       [-np.sin(x[0]) * np.sin(x[1]), np.cos(x[0]) * np.cos(x[1])],
                       [0, 0]])

    layer1 = np.array([[-np.sin(x[0]) * np.cos(x[1]), -np.cos(x[0]) * np.sin(x[1])],
                       [np.cos(x[0]) * np.cos(x[1]), -np.sin(x[0]) * np.sin(x[1])],
                       [0, -np.cos(x[1])]])

    gJ[:, :, 0] = layer0
    gJ[:, :, 1] = layer1

    return gJ

def embedding_ensemble(thetas):
    return np.apply_along_axis(getEmbedding, 1, thetas)