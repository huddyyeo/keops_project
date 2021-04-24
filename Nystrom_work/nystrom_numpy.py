import numpy as np

from scipy.linalg import svd, eigh
from scipy.sparse.linalg import aslinearoperator

from nystrom_common import GenericNystroem 
from numpy_utils import numpytools 
from pykeops.numpy import LazyTensor

from typing import Tuple, List

class Nystroem(GenericNystroem):
    '''Nystrom class to work with Numpy arrays'''

    def __init__(self, n_components=100, kernel='rbf', sigma:float = None,
                 eps:float = 0.05, inv_eps:float = None,
                  verbose = False, random_state=None, eigvals:List[int]=None):

        '''
        Args:
            eigvals(list): eigenvalues index interval [a,b] for constructed K_q,
             where 0 <= a < b < length of K_q
            
        '''
        super().__init__(n_components, kernel, sigma, eps, inv_eps, verbose, 
                        random_state)
        
        self.tools = numpytools
        self.LazyTensor = LazyTensor
        self.eigvals = eigvals

        if eigvals:
            assert eigvals[0] < eigvals[1], 'eigvals = [a,b] needs a < b'
            assert eigvals[1] < n_components, 'max eigenvalue index needs to be less\
            than size of K_q = n_components'

    def _decomposition_and_norm(self, X:np.array) -> np.array:
        '''Computes K_q^{-1/2}'''

        X = X + np.eye(X.shape[0], dtype=self.dtype)*self.inv_eps        
        S,U = eigh(X, eigvals=self.eigvals)
        S = np.maximum(S, 1e-12)
        
        return np.dot(U / np.sqrt(S), U.T)
        

    def K_approx(self, x:np.array) -> 'LinearOperator':
        ''' Function to return Nystrom approximation to the kernel.
        
        Args:
            x(array): data used in fit(.) function.
        Returns
            K(operator): Nystrom approximation to kernel'''
    
        K_nq = self._pairwise_kernels(x, self.components, dense=False)
        K_nq.backend="GPU_2D"
        K_nq = aslinearoperator(K_nq)
        K_q_inv = (aslinearoperator(self.normalization).T @
                    aslinearoperator(self.normalization)
                )
        K_approx = K_nq @ K_q_inv @ K_nq.T
        return K_approx 

    def _astype(self, data, d_type):
        return data.astype(d_type)
