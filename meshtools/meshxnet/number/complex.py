import numpy as np

def conjugate(x):
    """
    x is an (n, 4) array
    """
    
    z = np.copy(x)
    z[:,1:] *= -1
    return z


def inverse(x):
    """
    x is an (n, 4) array
    """
    
    z = conjugate(x)
    z /= sq_norm(x)[:, np.newaxis]
    return z


def sq_norm(x):
    """
    x is an (n, d) array, the output is (n,)
    """    

    return np.sum(x**2,1)


def c_one(d):
    res = np.zeros((d,2))
    res[:,0] = 1
    return res


def matrix_representation(x, conjugation=False, pure_imaginary=False):
    """
    x is an (n,2) array.
    X is an (n,2,2) array.
    """
    
    if pure_imaginary:
        a = np.zeros((len(x),1))
        b = x[:,0]
    else:
        a = x[:,0]
        b = x[:,1]
    mb = -b
    
    X = np.c_[a,mb, \
              b,a]
    
    # The representation of the conjugate is the transpose of the representation.
    result = X.reshape((-1,2,2))
    if conjugation:
        return result.transpose(0,2,1)
    else:
        return result


def logarithm(x):
    # compute the polar decomposition
    sq_norm_Im = x[:,1]**2   
    rho = np.sqrt(x[:,0]**2+sq_norm_Im)
    defined = rho != 0 
    
    # will put log(0) = -infty + 0*i, which is fine (we just don't want NaNs in the imaginary part of the log)    
    cos = np.ones(len(rho))
    cos[defined] = x[defined,0]/rho[defined]
    theta = np.arccos(cos)
    
    #
    norm_Im = np.sqrt(sq_norm_Im)
    mask = np.isclose(norm_Im,0)
    not_mask = np.logical_not(mask)
    
    # Adjust the sign of theta
    theta[not_mask] *= x[not_mask,1]
    theta[not_mask] /= norm_Im[not_mask]
    
    return np.c_[np.log(rho),theta]


def exponential(z):
    rho = np.exp(z[:,0])
    theta = z[:,1]
    
    return rho[:,np.newaxis]*np.c_[np.cos(theta), np.sin(theta)]
    
    
def complex_to_quaternion(z):
    q = np.zeros((len(z),4))

    q[:,:2] = z
    return q


def center(x, weights=None, rescale=True, iterations=1):   
    """
    Try removing the average complex (rotation/scaling), but non linear stuff not allowed...
    And left multiplying not allowed *v'
    """
    
    # Average, it's like doing infinitesimal rotations theta_i u_i / T (T -> infty), cycling over the dataset i=1...n,
    # for t=1...T
    mean_log = np.average(logarithm(x), weights=weights, axis=0)
    i_avg_x = exponential((-mean_log)[np.newaxis,:])    
    if not rescale:
        i_avg_x /= np.sqrt(sq_norm(i_avg_x))
    
    # Remove the average
    res = x.dot(matrix_representation(i_avg_x).squeeze().transpose())
    
    if iterations > 1:
        return center(res, weights, rescale=rescale, iterations=iterations-1)
    else:
        return res
