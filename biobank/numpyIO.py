import numpy as np
from scipy import sparse

def save_sparse_csr(filename, sparse_csr_matrix):
    np.savez(filename, data=sparse_csr_matrix.data, indices=sparse_csr_matrix.indices,
             indptr=sparse_csr_matrix.indptr, shape=sparse_csr_matrix.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])