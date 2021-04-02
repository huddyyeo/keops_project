import numpy as np

from pykeops.numpy import LazyTensor

from scipy.sparse.linalg import aslinearoperator, eigsh
from scipy.sparse.linalg.interface import IdentityOperator

from nystrom_common import GenericNystrom
from numpy_utils import numpytools



class Nystrom_NK(GenericNystrom):
    '''Nystrom class to work with Numpy arrays'''

    def __init__(self, n_components=100, kernel='rbf', sigma:float = None,
                 eps:float = 0.05, mask_radius:float = None, k_means = 10, 
                 n_iter:int = 10, inv_eps:float = None, dtype = np.float32, 
                 backend = None, verbose = False, random_state=None, tools = None):

        super().__init__(n_components, kernel, sigma, eps, mask_radius,
                         k_means, n_iter, inv_eps, dtype, backend, verbose, random_state)
        
        self.tools = numpytools
        
    def _decomposition_and_norm(self, X:LazyTensor) -> np.array:

        K_linear = aslinearoperator(X)
        # K <- K + eps
        K_linear = K_linear + IdentityOperator(K_linear.shape, dtype=self.dtype) * self.inv_eps
        k = K_linear.shape[0] - 1
        S, U = eigsh(K_linear, k=k, which='LM')
        S = np.maximum(S, 1e-12)

        return np.dot(U / np.sqrt(S), U.T)

    def K_approx(self, x:np.array) -> 'LinearOperator':
        ''' Function to return Nystrom approximation to the kernel.
        
        Args:
            X = data used in fit(.) function.
        Returns
            K = Nystrom approximation to kernel'''
    
        K_nq = self._pairwise_kernels(x, self.components_, dense=True)
        K_q_inv = self.normalization_.T @ self.normalization_
        K_approx = K_nq @ K_q_inv @ K_nq.T
        return aslinearoperator(K_approx) 

    def _astype(self, data, d_type):
        return data.astype(d_type)
