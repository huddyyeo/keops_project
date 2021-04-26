import numpy as np

from scipy.linalg import eigh
from scipy.sparse.linalg import aslinearoperator

from nystrom_common import GenericNystroem
from numpy_utils import numpytools 
from pykeops.numpy import LazyTensor

from typing import Tuple, List

class Nystroem(GenericNystroem):
    """
    Nystroem class to work with Numpy arrays.
    """

    def __init__(self, n_components=100, kernel='rbf', sigma:float = None,
                 inv_eps:float = None, verbose = False, random_state=None, 
                 top_k:int=None):

        """
        Args:
             n_components (int): how many samples to select from data.
            kernel: str: type of kernel to use. Current options = {rbf:Gaussian,
                exp: exponential}.
            sigma (float): exponential constant for the RBF and exponential kernels.
            inv_eps (float): additive invertibility constant for matrix decomposition.
            top_k (int): keep the top-k eigenpairs after the decomposition of K_q.
            verbose  (boolean): set True to print details.
            random_state (int): to set a random seed for the random sampling of the 
                samples. To be used when reproducibility is needed.
            
        """
        super().__init__(n_components, kernel, sigma, inv_eps, verbose, 
                         random_state)
        
        self.tools = numpytools
        self.lazy_tensor = LazyTensor
        self.top_k = top_k

        if top_k:
            assert top_k <= n_components, 'min_eigval needs to satisfy\
                min_eigval < n_components'

    def _decomposition_and_norm(self, X:np.array) -> np.array:
        """
        Computes K_q^{-1/2}.

        Returns:
            K_q^{-1/2}: np.array
        """

        X = X + np.eye(X.shape[0], dtype=self.dtype)*self.inv_eps   # (Q,Q)  Q - num_components     
        
        if self.top_k:
            eigvals = [self.n_components - self.top_k, self.n_components-1]
        else:
            eigvals = None
        
        S,U = eigh(X, eigvals=eigvals) # (Q,), (Q,Q)
        S = np.maximum(S, 1e-12)
        
        return np.dot(U / np.sqrt(S), U.T) # (Q,Q)

    def _get_kernel(self, x:np.array, y:np.array, kernel=None) -> np.array:
    
        D_xx = np.sum((x ** 2), axis=-1)[:, None]  # (N,1)
        D_xy = x @ y.T  # (N,D) @ (D,M) = (N,M)
        D_yy = np.sum((y ** 2), axis=-1)[None, :]  # (1,M)
        D_xy = D_xx - 2 * D_xy + D_yy  # (N,M)
        if kernel == 'exp':
            D_xy = np.sqrt(D_xy)
        return np.exp(-D_xy)  # (N,M)
        
    def K_approx(self, x:np.array) -> 'LinearOperator':
        """
        Method to return Nystrom approximation to the kernel.
        
        Args:
            x: np.array: data used in fit(.) function.
        Returns
            K: LinearOperator: Nystrom approximation to kernel
        """
        K_nq = self._pairwise_kernels(x, self.components_, dense=False) # (N, Q)
        

        K_qn = K_nq.T
        K_nq.backend="GPU_2D"
        K_qn = aslinearoperator(K_qn)
        K_nq = aslinearoperator(K_nq)

        K_q_inv = self.normalization.T @ self.normalization  # (Q,Q)
        K_q_inv = aslinearoperator(K_q_inv)  

        K_approx = K_nq @ K_q_inv @ K_qn # (N,Q), (Q,Q), (Q,N)

        return K_approx # (N, N)
    
