import numpy as np

from scipy.linalg import svd, eigh
from scipy.sparse.linalg import aslinearoperator

from nystrom_common import GenericNystrom 
from numpy_utils import numpytools
from pykeops.numpy import LazyTensor

from typing import Tuple

class Nystrom(GenericNystrom):
    '''Nystrom class to work with Numpy arrays'''

    def __init__(self, n_components=100, kernel='rbf', sigma:float = None,
                 eps:float = 0.05, mask_radius:float = None, k_means = 10, 
                 n_iter:int = 10, inv_eps:float = None,
                  verbose = False, random_state=None, tools = None):

        super().__init__(n_components, kernel, sigma, eps, mask_radius,
                         k_means, n_iter, inv_eps, verbose, random_state)
        
        self.tools = numpytools
        self.LazyTensor = LazyTensor

    def _decomposition_and_norm(self, X:np.array) -> np.array:
        '''Computes K_q^{-1/2}'''

        X = X + np.eye(X.shape[0], dtype=self.dtype)*self.inv_eps
        S,U = eigh(X)
        S = np.maximum(S, 1e-12)
        
        return np.dot(U / np.sqrt(S), U.T)
        

    def K_approx(self, x:np.array) -> 'LinearOperator':
        ''' Function to return Nystrom approximation to the kernel.
        
        Args:
            x = data used in fit(.) function.
        Returns
            K = Nystrom approximation to kernel'''
    
        K_nq = aslinearoperator(
            self._pairwise_kernels(x, self.components_, dense=False)
            )
        
        K_q_inv = (aslinearoperator(self.normalization_).T @ 
                    aslinearoperator(self.normalization_) 
                )
        K_approx = K_nq @ K_q_inv @ K_nq.T
        return K_approx 

    def _astype(self, data, d_type):
        return data.astype(d_type)


    def _KMeans(self,x:np.array) -> Tuple[np.array]:
        ''' KMeans with Pykeops to do binning of original data.
        Args:
            x = data
        Returns:
            labels = class labels for each point in x
            clusters = coordinates for each centroid
        '''
        N, D = x.shape  
        clusters = np.copy(x[:self.k_means, :])  # initialization of clusters
        x_i = LazyTensor(x[:, None, :])  

        for i in range(self.n_iter):

            clusters_j = LazyTensor(clusters[None, :, :])  
            D_ij = ((x_i - clusters_j) ** 2).sum(-1)  # points-clusters kernel
            labels = D_ij.argmin(axis=1).astype(int).reshape(N)  # Points -> Nearest cluster
            Ncl = np.bincount(labels).astype(self.dtype)  # Class weights
            for d in range(D):  # Compute the cluster centroids with np.bincount:
                clusters[:, d] = np.bincount(labels, weights=x[:, d]) / Ncl

        return labels, clusters