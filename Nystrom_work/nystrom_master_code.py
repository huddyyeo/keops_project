# If using colab, make sure you install via:  !pip install pykeops[full] > log.log


import numpy as np
import torch
import pykeops

from pykeops.numpy import LazyTensor as LazyTensor_n
from pykeops.numpy.cluster import grid_cluster
from pykeops.numpy.cluster import from_matrix
from pykeops.numpy.cluster import cluster_ranges_centroids
from pykeops.numpy.cluster import sort_clusters
from pykeops.torch import LazyTensor

from sklearn.utils import check_random_state, as_float_array
from scipy.linalg import svd
from sklearn.kernel_approximation import Nystroem
from scipy.sparse.linalg import aslinearoperator, eigsh

import matplotlib.pyplot as plt
import time




##############################################################################

'''
The two classes below implement the Nystrom algorithm. One can transform
the data into the approximated feature-space and/or obtain the approximated 
kernel.

Example of usage:

Let x be a numpy array of shape =  (length, features), then 

NN = Nystrom_N(n_components=100 ,kernel='rbf', gamma=1.) # creates an instance
NN.fit(X)  # fits to data         
X_new_i = NN.transform(X)  # transform data to approximated features
K_approx = NN.K_approx(X)  # obtain approximated kernel

'''

class Nystrom_N:
    '''
        This class implements Nystrom on numpy arrays.
    
        * The fit method computes K^{-1}_q.

        * The transform method maps the data into the feature space underlying
        the Nystrom-approximated kernel.

        * The method K_approx directly computes the Nystrom approximation.

        Parameters:

        n_components [int] = how many samples to select from data.
        kernel [str] = type of kernel to use. Current options = {linear, rbf}.
        gamma [float] = exponential constant for the RBF kernel. 
        random_state=[None, float] = to set a random seed for the random
                                     sampling of the samples. To be used when 
                                     reproducibility is needed.

    '''
  
    def __init__(self, n_components=100, kernel='linear', gamma:float = 1., 
                 random_state=None): 

        self.n_components = n_components
        self.kernel = kernel
        self.random_state = random_state
        self.gamma = gamma


    def fit(self, X:np.array):
        ''' 
        Args:   X = data array with (n_samples, n_features)

        Returns: Fitted instance of the class
        '''

        # Basic checks
        assert type(X) == np.ndarray, 'Input to fit(.) must be an array.'
        assert X.shape[0] >= self.n_components, f'The application needs X.shape[0] >= n_components.'

        # Number of samples
        n_samples = X.shape[0]
        # Define basis
        rnd = check_random_state(self.random_state)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:self.n_components]
        basis = X[basis_inds]
        # Build smaller kernel
        basis_kernel = self._pairwise_kernels(basis, kernel = self.kernel)  
        # Get SVD
        U, S, V = svd(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), V)
        self.components_ = basis
        self.component_indices_ = inds
        return self


    def _pairwise_kernels(self, x:np.array, y:np.array = None, kernel='linear',
                          gamma = 1.) -> np.array:
        '''Helper function to build kernel
        
        Args:   X = numpy array of dimension 2.
                kernel = type of Kernel to return
        '''
        
        if y is None:
            y = x
        if kernel == 'linear':
            K = x @ y.T 
        elif kernel == 'rbf':
            K =  ( (x[:,None,:] - y[None,:,:])**2 ).sum(-1)
            K = np.exp(- gamma* K)
  
        return K

    def transform(self, x:np.array) -> np.array:
        ''' Applies transform on the data.
        
        Args:
            x [np.array] = data to transform
        Returns
            x_new [np.array] = data after transformation
        '''
        K_nq = self._pairwise_kernels(x, self.components_, self.kernel)
        x_new = K_nq @ self.normalization_.T

        return x_new

    
    def K_approx(self, x:np.array) -> np.array:
        ''' Function to return Nystrom approximation to the kernel.
        
        Args:
            X[np.array] = data used in fit(.) function.
        Returns
            K[np.array] = Nystrom approximation to kernel'''
        
        K_nq = self._pairwise_kernels(x, self.components_, self.kernel)
        K_approx = K_nq @ self.normalization_ @ K_nq.T

        return K_approx




##########################################################################

# Same as LazyNystrom_N but written with Pytorch

class LazyNystrom_T:
    '''
        Class to implement Nystrom on torch LazyTensors.
        This class works as an interface between lazy tensors and 
        the Nystrom algorithm in NumPy.

        * The fit method computes K^{-1}_q.

        * The transform method maps the data into the feature space underlying
        the Nystrom-approximated kernel.

        * The method K_approx directly computes the Nystrom approximation.

        Parameters:

        n_components [int] = how many samples to select from data.
        kernel [str] = type of kernel to use. Current options = {linear, rbf}.
        gamma [float] = exponential constant for the RBF kernel. 
        random_state=[None, float] = to set a random seed for the random
                                     sampling of the samples. To be used when 
                                     reproducibility is needed.

    '''
  
    def __init__(self, n_components=100, kernel='linear',  gamma:float = 1., 
                 random_state=None ):
        
        self.n_components = n_components
        self.kernel = kernel
        self.random_state = random_state
        self.gamma = gamma


    def fit(self, X:LazyTensor):
        ''' 
        Args:   X = torch lazy tensor with features of shape 
                (1, n_samples, n_features)

        Returns: Fitted instance of the class
        '''

        # Basic checks: we have a lazy tensor and n_components isn't too large
        assert type(X) == LazyTensor, 'Input to fit(.) must be a LazyTensor.'
        assert X.shape[1] >= self.n_components, f'The application needs X.shape[1] >= n_components.'

        X = X.sum(dim=0) 
        # Number of samples
        n_samples = X.size(0)
        # Define basis
        rnd = check_random_state(self.random_state)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:self.n_components]
        basis = X[basis_inds]
        # Build smaller kernel
        basis_kernel = self._pairwise_kernels(basis, kernel = self.kernel)  
        # Get SVD
        U, S, V = torch.svd(basis_kernel)
        S = torch.maximum(S, torch.ones(S.size()) * 1e-12)
        self.normalization_ = torch.mm(U / np.sqrt(S), V.t())
        self.components_ = basis
        self.component_indices_ = inds
        
        return self


    def _pairwise_kernels(self, x:torch.tensor, y:torch.tensor = None, kernel='linear',
                          gamma = 1.) -> torch.tensor:
        '''Helper function to build kernel
        
        Args:   X = torch tensor of dimension 2.
                K_type = type of Kernel to return
        '''
        
        if y is None:
            y = x
        if kernel == 'linear':
            K = x @ y.T
        elif kernel == 'rbf':
            K =  ( (x[:,None,:] - y[None,:,:])**2 ).sum(-1)
            K = torch.exp(- gamma * K )

        return K

    def transform(self, X:LazyTensor) -> LazyTensor:
        ''' Applies transform on the data.
        
        Args:
            X [LazyTensor] = data to transform
        Returns
            X [LazyTensor] = data after transformation
        '''
        
        X = X.sum(dim=0)
        K_nq = self._pairwise_kernels(X, self.components_, self.kernel)
        return LazyTensor((K_nq @ self.normalization_.t())[None,:,:])

    
    def K_approx(self, X:LazyTensor) -> LazyTensor:
        ''' Function to return Nystrom approximation to the kernel.
        
        Args:
            X[LazyTensor] = data used in fit(.) function.
        Returns
            K[LazyTensor] = Nystrom approximation to kernel'''
        
        X = X.sum(dim=0)
        K_nq = self._pairwise_kernels(X, self.components_, self.kernel)
        K_approx = K_nq @ self.normalization_ @ K_nq.t()
        return LazyTensor(K_approx[None,:,:])




################################################################################
class Nystrom_NK:
    '''
        Class to implement Nystrom using numpy and PyKeops.

        * The fit method computes K^{-1}_q.

        * The transform method maps the data into the feature space underlying
        the Nystrom-approximated kernel.

        * The method K_approx directly computes the Nystrom approximation.

        Parameters:

        n_components [int] = how many samples to select from data.
        kernel [str] = type of kernel to use. Current options = {rbf}.
        sigma [float] = exponential constant for the RBF kernel. 
        eps[float] = size for square bins
        random_state=[None, float] = to set a random seed for the random
                                     sampling of the samples. To be used when 
                                     reproducibility is needed.

    '''
  
    def __init__(self, n_components=100, kernel='rbf', sigma:float = 1., 
                 eps:float = 0.05, random_state=None): 

        self.n_components = n_components
        self.kernel = kernel
        self.random_state = random_state
        self.sigma = sigma
        self.eps = eps


    def fit(self, x:np.ndarray):
        ''' 
        Args:   x = numpy array of shape (n_samples, n_features)

        Returns: Fitted instance of the class

        '''

        # Basic checks
        assert type(x) == np.ndarray, 'Input to fit(.) must be an array.'
        assert x.shape[0] >= self.n_components, f'The application needs X.shape[0] >= n_components.'

        # Number of samples
        n_samples = x.shape[0]
        # Define basis
        rnd = check_random_state(self.random_state)
        inds = rnd.permutation(n_samples) 
        basis_inds = inds[:self.n_components] 
        basis = x[basis_inds]
        # Build smaller kernel
        basis_kernel = self._pairwise_kernels(basis, kernel = self.kernel)  
        # Spectral decomposition
        S, U = self._spectral(basis_kernel)
        S = np.maximum(S, 1e-12)
        self.normalization_ = np.dot(U / np.sqrt(S), U.T)
        self.components_ = basis
        self.component_indices_ = inds
        return self


    def _pairwise_kernels(self, x:np.array, y:np.array = None, kernel='rbf',
                          sigma = 1.) -> LazyTensor:
        '''Helper function to build kernel
        
        Args:   X = torch tensor of dimension 2,
                K_type = type of Kernel to return.

        Returns:
                K_ij[LazyTensor]
        '''
        if y is None:
            y = x
        if kernel == 'linear': 
            K = x @ y.T 
        elif kernel == 'rbf':
            x /= sigma
            y /= sigma
            x_i, x_j = LazyTensor_n(x[:, None, :]), LazyTensor_n(y[None, :, :])
            K_ij = ((-1*(x_i - x_j)**2).sum(2)).exp()
            # block-sparse reduction preprocess
            K_ij = self._Gauss_block_sparse_pre(x, y, K_ij, self.sigma, self.eps)
            
        return K_ij


    def _spectral(self, X_i:LazyTensor):
        '''
        Helper function to compute eigendecomposition of K_q.

        Args: X_i[numpy LazyTensor]

        Returns S[np.array] eigenvalues,
                U[np.array] eigenvectors
        '''
        K_linear = aslinearoperator(X_i)
        k = K_linear.shape[0] - 1
        S, U = eigsh(K_linear, k=k, which='LM')
        return S, U
        

    def transform(self, x:np.ndarray) -> np.array:
        ''' Applies transform on the data.
        
        Args:
            X [np.array] = data to transform
        Returns
            X [np.array] = data after transformation
        '''
        
        K_nq = self._pairwise_kernels(x, self.components_, self.kernel)
        x_new = K_nq @ self.normalization_.T
        return x_new

    
    def K_approx(self, x:np.array) -> np.array:
        ''' Function to return Nystrom approximation to the kernel.
        
        Args:
            X[np.array] = data used in fit(.) function.
        Returns
            K[np.array] = Nystrom approximation to kernel'''
       
        K_nq = self._pairwise_kernels(x, self.components_, self.kernel)
        # For arrays: K_approx = K_nq @ self.normalization_ @ K_nq.T
        # But to use @ with lazy tensors we have:
        K_approx = K_nq @ (K_nq @ self.normalization_ ).T
        
        return K_approx.T 


    def _Gauss_block_sparse_pre(self, x:np.array, y:np.array, K_ij:LazyTensor, 
                               sigma:float = 1., eps:float = 0.05):
        ''' 
        Helper function to preprocess data for block-sparse reduction
        of the Gaussian kernel
    
        Args: 
            x[np.array], y[np.array] = arrays giving rise to Gaussian kernel K(x,y)
            K_ij[LazyTensor_n] = symbolic representation of K(x,y)
            eps[float] = size for square bins

        Returns:
            K_ij[LazyTensor_n] = symbolic representation of K(x,y) with 
                                set sparse ranges
        '''

        # class labels
        x_labels = grid_cluster(x, eps) 
        y_labels = grid_cluster(y, eps) 
        # compute one range and centroid per class
        x_ranges, x_centroids, _ = cluster_ranges_centroids(x, x_labels)
        y_ranges, y_centroids, _ = cluster_ranges_centroids(y, y_labels)
        # sort points
        x, x_labels = sort_clusters(x, x_labels)
        y, y_labels = sort_clusters(y, y_labels) 
        # Compute a coarse Boolean mask:
        D = np.sum((x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2, 2)
        keep = D < (4 * sigma) ** 2  # self.sigma 
        # mask -> set of integer tensors
        ranges_ij = from_matrix(x_ranges, y_ranges, keep)
        K_ij.ranges = ranges_ij  # block-sparsity pattern

        return K_ij