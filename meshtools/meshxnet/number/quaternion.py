import numpy as np
from .complex import complex_to_quaternion

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

    return np.sum(x**2,-1)


def qu_one(d):
    res = np.zeros((d,4))
    res[:,0] = 1
    return res


def matrix_representation(x, right_action=False, conjugation=False, pure_imaginary=False):
    """
    x is an (n,4) array.
    X is an (n,4,4) array.
    """
    
    if pure_imaginary:
        a = np.zeros((len(x),1))
        b = x[:,0]
        c = x[:,1]
        d = x[:,2]
    else:
        a = x[:,0]
        b = x[:,1]
        c = x[:,2]
        d = x[:,3]
    mb = -b
    mc = -c
    md = -d
    
    if right_action:
        # doesn't have to represent valid quaternions, only the output of q*x is correctly given by X*q for any q
        X = np.c_[a,mb,mc,md, \
                     b,a,d,mc, \
                     c,md,a,b, \
                     d,c,mb,a]
    else:
        # actual matrix representation, so that x*q is given by X*q
        X = np.c_[a,mb,mc,md, \
                     b,a,md,c, \
                     c,d,a,mb, \
                     d,mc,b,a]
    
    # For both left and right actions, the representation of the conjugate is the transpose of the representation.
    result = X.reshape((-1,4,4))
    if conjugation:
        return result.transpose(0,2,1)
    else:
        return result


def logarithm(x):
    shape = x.shape
    x = x.squeeze()
    if x.ndim == 1:
        x = np.reshape(x, (1,-1))
    
    # compute the polar decomposition
    sq_norm_Im = np.sum(x[:,1:]**2,axis=-1)    
    rho = np.sqrt(x[:,0]**2+sq_norm_Im)
    defined = rho != 0 
    undefined = np.logical_not(defined)
    
    # will put log(0) = -infty + 0*{i,j,k} which is fine (we just don't want NaNs in the imaginary part of the log)    
    cos = np.ones(len(rho))
    cos[defined] = x[defined,0]/rho[defined]
    theta = np.arccos(cos)
    
    norm_Im = np.sqrt(sq_norm_Im)
    mask = np.isclose(norm_Im,0)
    not_mask = np.logical_not(mask)
    
    # will put log(-r) = log(r) + pi*average(u, weights=theta), averaged over regular theta, this is also arbitrary but fine
    singular = np.logical_and(mask, np.logical_not(np.isclose(theta,0)))
    regular = np.logical_not(singular)
    regular[undefined] = False
    
    u_theta = np.zeros((len(x),3))
    theta_regular = theta[regular]
    u_theta[regular,:] = x[regular,1:]*np.reshape(theta_regular,(-1,1))
    u_theta[not_mask,:] /= np.reshape(norm_Im[not_mask],(-1,1)) # some regular points are positive reals :)
    
    if np.any(singular):
        u_mean = np.sum(u_theta_regular, axis=-1)
        sum_theta = np.sum(theta_regular)
        if np.isclose(np.sqrt(np.sum(u_mean**2)),0) or np.isclose(sum_theta,0): # set to i instead
            u_mean = np.zeros(3)
            u_mean[0] = 1
        else:
            u_mean /= sum_theta
    
        u_theta[singular,:] = u_mean*np.pi
    
    #
    res = np.c_[np.log(rho),u_theta]
    res = np.reshape(res.squeeze(), shape)
        
    return res


def exponential(z):
    shape = z.shape
    z = z.squeeze()
    if z.ndim == 1:
        z = np.reshape(z, (1,-1))
    
    theta = np.sqrt(sq_norm(z[:,1:]))
    rho = np.exp(z[:,0])
    
    mask = np.isclose(theta,0)
    not_mask = ~mask

    u_sin = z[:,1:].copy()
    u_sin[not_mask,:] *= np.reshape(np.sin(theta[not_mask])/theta[not_mask], (-1,1))
        # otherwise u*theta is a pretty good approximation ;p
    
    res = rho[:,np.newaxis]*np.c_[np.cos(theta), u_sin]
    res = np.reshape(res.squeeze(), shape)
    
    return res


def center(x, weights=None, rescale=True, iterations=1):   
    """
    Try removing the average quaternion (rotation/scaling), but non linear stuff not allowed...
    And left multiplying not allowed *v'
    """
    
    # Average, it's like doing infinitesimal rotations theta_i u_i / T (T -> infty), cycling over the dataset i=1...n,
    # for t=1...T
    mean_log = np.average(logarithm(x), weights=weights, axis=0)
    i_avg_x = exponential((-mean_log)[np.newaxis,:])    
    if not rescale:
        i_avg_x /= np.sqrt(sq_norm(i_avg_x))
    
    # Remove the average
    res = x.dot(matrix_representation(i_avg_x,right_action=True).squeeze().transpose())
    
    if iterations > 1:
        return center(res, weights, rescale=rescale, iterations=iterations-1)
    else:
        return res
    
    
def svd(X, rtol=None, atol=None):
    """
    quaternionic svd
    X: (d, N, 4)
    
    What about negative eigenvalues? -_-
    """
    
    #
    gram = np.einsum('ijq,ikq->jk', X, X) # N*N
    l, V = np.linalg.eigh(gram)
    
    mask = l <= 0
    if atol:
        mask *= np.isclose(l, 0, atol=atol)
    if rtol:
        lmax = np.max(l)
        mask *= l<(lmax*rtol)
    mask = ~mask
    l = l[mask]
    V = V[:,mask]
    
    #
    if V.ndim == 1:
        V = V.reshape((-1,1))
    
    #
    s = np.sqrt(l) # K
    U = np.einsum('ijq,jk->ikq', X, V) # N*K*4
    U = U*(1/s).reshape((1,-1,1))
    Vh = V.transpose() # K*N
    
    return U, s, Vh