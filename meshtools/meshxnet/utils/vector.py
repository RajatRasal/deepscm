import numpy as np


def veclen(vectors):
    """ return L2 norm (vector length) along the last axis, for example to compute the length of an array of vectors """
    square_norms = sq_norm(vectors)
    mask = square_norms>0
    
    res = np.zeros(vectors.shape[:-1])
    res[mask] = np.sqrt(square_norms[mask])
    return res

def sq_norm(vectors):
    """ return squared L2 norm along the last axis """
    return np.sum(vectors**2, axis=-1)

def normalized(vectors):
    """ normalize array of vectors along the last axis """
    return vectors / veclen(vectors)[..., np.newaxis]

def homogenize(v, value=1):
    """ returns v as homogeneous vectors by inserting one more element into the last axis 
    the parameter value defines which value to insert (meaningful values would be 0 and 1) 
    >>> homogenize([1, 2, 3]).tolist()
    [1, 2, 3, 1]
    >>> homogenize([1, 2, 3], 9).tolist()
    [1, 2, 3, 9]
    >>> homogenize([[1, 2], [3, 4]]).tolist()
    [[1, 2, 1], [3, 4, 1]]
    """
    v = np.asanyarray(v)
    return np.insert(v, v.shape[-1], value, axis=-1)

def dehomogenize(a):
    """ makes homogeneous vectors inhomogenious by dividing by the last element in the last axis 
    >>> dehomogenize([1, 2, 4, 2]).tolist()
    [0.5, 1.0, 2.0]
    >>> dehomogenize([[1, 2], [4, 4]]).tolist()
    [[0.5], [1.0]]
    """
    a = np.asfarray(a)
    return a[...,:-1] / a[...,np.newaxis,-1]

def transform(v, M, w=1):
    """ transforms vectors in v with the matrix M
    if matrix M has one more dimension then the vectors 
    this will be done by homogenizing the vectors 
    (with the last dimension filled with w) and 
    then applying the transformation """
    if M.shape[0] == M.shape[1] == v.shape[-1] + 1:
        v1 = homogenize(v, value=w)
        return dehomogenize(np.dot(v1.reshape((-1,v1.shape[-1])), M.T)).reshape(v.shape)
    else:
        return np.dot(v.reshape((-1,v.shape[-1])), M.T).reshape(v.shape)

def filter_reindex(condition, target):
    """
    >>> indices = np.array([1, 4, 1, 4])
    >>> condition = np.array([False, True, False, False, True])
    >>> filter_reindex(condition, indices).tolist()
    [0, 1, 0, 1]
    """
    if condition.dtype != np.bool:
        raise ValueError("condition must be a binary array")
    reindex = np.cumsum(condition) - 1
    return reindex[target]

