# If using colab, make sure you install:

# !apt-get install cuda=10.2.89-1
# !pip install si_prefix
# !pip install sphinx
# !pip install pykeops[full] > log.log

# from common.nystrom import GenericNystrom -> needed when integrating into the lib

import numpy as np

import pykeops
from pykeops.numpy.cluster import grid_cluster
from pykeops.numpy.cluster import from_matrix
from pykeops.numpy.cluster import cluster_ranges_centroids, cluster_ranges
from pykeops.numpy.cluster import sort_clusters

from typing import Tuple, TypeVar, Union
# Generic placeholder for numpy and torch variables.
generic_array = TypeVar('generic_array')
GenericLazyTensor = TypeVar('GenericLazyTensor')

""" Abstract class"""

class GenericNystrom:
    '''Super class defining the Nystrom operations. The end user should
    use numpy.nystrom or torch.nystrom subclasses.'''

    def __init__(self, n_components:int=100, kernel:str='rbf', sigma:float = None, 
                 eps:float = 0.05, mask_radius:float = None, k_means:int = 10, 
                 n_iter:int = 10, inv_eps:float = None, dtype = np.float32, 
                 backend:str = None, verbose:bool = False, 
                 random_state:Union[None,int] = None, tools = None): 

        ''' 
        n_components  = how many samples to select from data.
        kernel  = type of kernel to use. Current options = {rbf:Gaussian, 
                                                                 exp: exponential}.
        sigma  = exponential constant for the RBF and exponential kernels.
        eps = size for square bins in block-sparse preprocessing.
        k_means = number of centroids for KMeans algorithm in block-sparse 
                       preprocessing.
        n_iter = number of iterations for KMeans.
        dtype = type of data: np.float32 or np.float64
        inv_eps = additive invertibility constant for matrix decomposition.
        backend = "GPU" or "CPU" mode for LazyTensors.
        verbose = set True to print details.
        random_state = to set a random seed for the random sampling of the samples. 
                        To be used when  reproducibility is needed.
        '''
        self.n_components = n_components
        self.kernel = kernel
        self.sigma = sigma
        self.eps = eps
        self.mask_radius = mask_radius
        self.k_means = k_means
        self.n_iter = n_iter
        self.dtype = dtype
        self.verbose = verbose
        self.random_state = random_state
        self.tools = None

        if not backend:
            self.backend = 'GPU' if pykeops.config.gpu_available else 'CPU'
        else:
            self.backend = backend

        if inv_eps:
            self.inv_eps = inv_eps
        else:
            self.inv_eps = 1e-8

    def fit(self, x:generic_array) -> 'Nystrom_common':
        ''' 
        Args:   x = array or tensor of shape (n_samples, n_features)
        Returns: Fitted instance of the class
        '''

        if self.verbose:
            print(f'Working with backend = {self.backend}')
        
        # Basic checks
        assert self.tools.is_tensor(x) , 'Input to fit(.) must be an array\
        if using numpy and tensor if using torch.'
        assert x.shape[0] >= self.n_components, 'The application needs\
        X.shape[0] >= n_components.'
        if self.kernel == 'exp' and not (self.sigma is None):
            assert self.sigma > 0, 'Should be working with decaying exponential.'
        
        # Set default sigma
        if self.sigma is None and self.kernel == 'rbf':
            self.sigma = np.sqrt(x.shape[1])

        if self.mask_radius is None:
            if self.kernel == 'rbf':
                self.mask_radius = 2* np.sqrt(2) * self.sigma
            elif self.kernel == 'exp':
                self.mask_radius = 8 * self.sigma

        # Update dtype
        self._update_dtype(x)
        # Number of samples
        n_samples = x.shape[0]
        # Define basis
        rnd = self._check_random_state(self.random_state)
        inds = rnd.permutation(n_samples) 
        basis_inds = inds[:self.n_components] 
        basis = x[basis_inds]
        # Build smaller kernel
        basis_kernel = self._pairwise_kernels(basis, dense=False)
        # Decomposition is an abstract method that needs to be defined in each class
        self.normalization_ = self._decomposition_and_norm(basis_kernel)
        self.components_ = basis
        self.component_indices_ = inds

        return self

    def _decomposition_and_norm(self, X:GenericLazyTensor):
        """
        To be defined in the subclass
        """
        print('_decomposition_and_norm needs to be implemented in the subclass')

    def transform(self, x):
        ''' 
        Applies transform on the data.
        
        Args:
            X [np.array or torch.tensor] = data to transform
        Returns
            X [np.array or torch.tensor] = data after transformation
        '''

        K_nq = self._pairwise_kernels(x, self.components_, dense=True)
        x_new = K_nq @ self.tools.transpose(self.normalization_)
        return x_new

    def _pairwise_kernels(self, x, y = None, dense = False):
        '''Helper function to build kernel
        
        Args:   x[np.array or torch.tensor] = data
                y[np.array or torch.tensor] = array/tensor 
                dense[bool] = False to work with lazy tensor reduction,
                              True to work with dense arrays/tensors
        Returns:
                K_ij[LazyTensor] if dense = False
                K_ij[np.array or torch.tensor] if dense = True
        '''

        if y is None:
            y = x

        if self.kernel == 'rbf':
            x = x / self.sigma
            y = y / self.sigma

            if dense:
                x_i, x_j = x[:, None, :], y[None, :, :]
                D_ij = ( (x_i - x_j)**2 ).sum(axis=2)
                K_ij = self.tools.exp(-D_ij )

            else:
                x_i, x_j = self.tools.LazyTensor(x[:, None, :]), self.tools.LazyTensor(y[None, :, :])
                D_ij = ( (x_i - x_j)**2 ).sum(dim=2)
                K_ij = (-D_ij).exp()
                
                # block-sparse reduction preprocess
                K_ij = self._Gauss_block_sparse_pre(x, y, K_ij)

        elif self.kernel == 'exp':
            x = x / self.sigma
            y = y / self.sigma

            if dense:
                x_i, x_j = x[:, None, :], y[None, :, :]
                K_ij =  self.tools.exp(-self.tools.sqrt( ( ((x_i - x_j) ** 2).sum(axis=2) )))

            else:
                x_i, x_j = self.tools.LazyTensor(x[:, None, :]), self.tools.LazyTensor(y[None, :, :])
                K_ij = (-(((x_i - x_j) ** 2).sum(-1)).sqrt()).exp()

                # block-sparse reduction preprocess
                K_ij = self._Gauss_block_sparse_pre(x, y, K_ij)
       
        if not dense:
            K_ij.backend = self.backend
        
        return K_ij

    def _Gauss_block_sparse_pre(self, x:generic_array, y:generic_array, 
                                K_ij:GenericLazyTensor):
        ''' 
        Helper function to preprocess data for block-sparse reduction
        of the Gaussian kernel
    
        Args: 
            x, y =  arrays or tensors giving rise to Gaussian kernel K(x,y)
            K_ij = symbolic representation of K(x,y)
            eps[float] = size for square bins
        Returns:
            K_ij =  symbolic representation of K(x,y) with 
                                set sparse ranges
        '''

        # labels for low dimensions
        if x.shape[1] < 4 or y.shape[1] < 4:
            x_labels = grid_cluster(x, self.eps) 
            y_labels = grid_cluster(y, self.eps)

            # range and centroid per class
            x_ranges, x_centroids, _ = cluster_ranges_centroids(x, x_labels)
            y_ranges, y_centroids, _ = cluster_ranges_centroids(y, y_labels)

        else:
        # labels for higher dimensions
            x_labels, x_centroids = self._KMeans(x)
            y_labels, y_centroids = self._KMeans(y)
            # compute ranges
            x_ranges = cluster_ranges(x_labels)
            y_ranges = cluster_ranges(y_labels)

        # sort points
        x, x_labels = sort_clusters(x, x_labels)
        y, y_labels = sort_clusters(y, y_labels)

        # Compute a coarse Boolean mask:
        if self.kernel == 'rbf':
            D = self.tools.arraysum((x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2, 2)

        elif self.kernel == 'exp':
            D = self.tools.sqrt(self.tools.arraysum((x_centroids[:, None, :] - y_centroids[None, :, :]) ** 2, 2))

        keep = D < (self.mask_radius) ** 2
        # mask -> set of integer tensors
        ranges_ij = from_matrix(x_ranges, y_ranges, keep)
        K_ij.ranges = ranges_ij  # block-sparsity pattern

        return K_ij

    def _KMeans(self,x):
        ''' KMeans with Pykeops to do binning of original data.
        Args:
            x[np.array or torch.tensor] = data
            k_means[int] = number of bins to build
            n_iter[int] = number iterations of KMeans loop
        Returns:
            labels[np.array or torch.tensor] = class labels for each point in x
            clusters[np.array or torch.tensor] = coordinates for each centroid
        '''
        N, D = x.shape  
        clusters = self.tools.copy(x[:self.k_means, :])  # initialization of clusters
        x_i = self.tools.LazyTensor(x[:, None, :])  

        for i in range(self.n_iter):

            clusters_j = self.tools.LazyTensor(clusters[None, :, :])  
            D_ij = ((x_i - clusters_j) ** 2).sum(-1)  # points-clusters kernel
            labels = self._astype(D_ij.argmin(axis=1), int).reshape(N)  # Points -> Nearest cluster
            Ncl = self._astype(self.tools.bincount(labels), self.dtype)  # Class weights
            for d in range(D):  # Compute the cluster centroids with bincount:
                clusters[:, d] = self.tools.bincount(labels, weights=x[:, d]) / Ncl

        return labels, clusters

    def _astype(self, data, type):
        return data

    def _update_dtype(self, x):
        ''' Helper function that sets dtype to that of 
            the given data in the fitting step.
            
        Args:
            x [np.array or torch.tensor] = raw data to remap
        Returns:
            None
        '''
        self.dtype = x.dtype
        self.inv_eps = np.array([self.inv_eps]).astype(self.dtype)[0]

    def _check_random_state(self, seed:Union[None,int]) -> None:
        '''Set/get np.random.RandomState instance for permutation
        Args
            seed[None, int] 
        Returns:
            numpy random state
        '''

        if seed is None:
            return np.random.mtrand._rand

        elif type(seed) == int:
            return np.random.RandomState(seed)
            
        raise ValueError(f'Seed {seed} must be None or an integer.')
