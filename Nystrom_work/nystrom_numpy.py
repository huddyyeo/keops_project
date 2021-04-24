import numpy as np

from scipy.linalg import eigh
from scipy.sparse.linalg import aslinearoperator

from nystrom_common import GenericNystroem
from numpy_utils import numpytools 
from pykeops.numpy import LazyTensor

from typing import Tuple, List

class Nystroem(GenericNystroem):
    '''Nystrom class to work with Numpy arrays'''

    def __init__(self, n_components=100, kernel='rbf', sigma:float = None,
                 inv_eps:float = None, verbose = False, random_state=None, 
                 eigvals:List[int]=None):

        '''
        Args:
            eigvals = eigenvalues index interval [a,b] for constructed K_q,
             where 0 <= a < b < length of K_q
            
        '''
        super().__init__(n_components, kernel, sigma, inv_eps, verbose, 
                         random_state)
        
        self.tools = numpytools
        self.lazy_tensor = LazyTensor
        self.eigvals = eigvals

        if eigvals:
            assert eigvals[0] < eigvals[1], 'eigvals = [a,b] needs a < b'
            assert eigvals[1] < n_components, 'max eigenvalue index needs to be less\
            than size of K_q = n_components'

    def _decomposition_and_norm(self, X:np.array) -> np.array:
        '''Computes K_q^{-1/2}'''

        X = X + np.eye(X.shape[0], dtype=self.dtype)*self.inv_eps   # (Q,Q)  Q - num_components     
        S,U = eigh(X, eigvals=self.eigvals) # (Q,), (Q,Q)
        S = np.maximum(S, 1e-12)
        
        return np.dot(U / np.sqrt(S), U.T) # (Q,Q)

    def _get_kernel(self, x, y, kernel=None):
    
        D_xx = np.sum((x ** 2), axis=-1)[:, None]  # (N,1)
        D_xy = x @ y.T  # (N,D) @ (D,M) = (N,M)
        D_yy = np.sum((y ** 2), axis=-1)[None, :]  # (1,M)
        D_xy = D_xx - 2 * D_xy + D_yy  # (N,M)
        if kernel == 'exp':
            D_xy = np.sqrt(D_xy)
        return np.exp(-D_xy)  # (N,M)
        
    def K_approx(self, x:np.array) -> 'LinearOperator':
        ''' Function to return Nystrom approximation to the kernel.
        
        Args:
            x = data used in fit(.) function.
        Returns
            K = Nystrom approximation to kernel'''
    
        K_nq = self._pairwise_kernels(x, self.components_, dense=False) # (N, Q)
        K_nq.backend="GPU_2D"
        K_nq = aslinearoperator(K_nq)
        K_q_inv = (aslinearoperator(self.normalization).T @ 
                    aslinearoperator(self.normalization) 
                ) # (Q,Q), (Q,Q)
        K_approx = K_nq @ K_q_inv @ K_nq.T # (N,Q), (Q,Q), (Q,N)
        return K_approx # (N, N)
    
