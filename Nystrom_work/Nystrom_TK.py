# !pip install pykeops[full] > install.log
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

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import variable
from torch.utils.data import dataloader
import cv2

# Same as LazyNystrom_T but written with pyKeOps

class LazyNystrom_TK:
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

    def __init__(self, n_components=100, kernel='rbf', sigma:float = 1.,
                 eps:float = 0.05, random_state=None):

        self.n_components = n_components
        self.kernel = kernel
        self.random_state = random_state
        self.sigma = sigma
        self.eps = eps

    def fit(self, X: torch.tensor):
        '''
        Args:   X = torch tensor with features of shape
                (1, n_samples, n_features)

        Returns: Fitted instance of the class
        '''
        print(type(X))
        print(X.shape)
        # Basic checks: we have a lazy tensor and n_components isn't too large
        assert type(X) == torch.Tensor, 'Input to fit(.) must be a Tensor.'
        assert X.size(0) >= self.n_components, f'The application needs X.shape[1] >= n_components.'

        # X = X.sum(dim=0)
        # Number of samples
        n_samples = X.size(0)
        # Define basis
        rnd = check_random_state(self.random_state)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:self.n_components]
        basis = X[basis_inds]
        # Build smaller kernel
        basis_kernel = self._pairwise_kernels(basis, kernel=self.kernel)
        if type(basis_kernel)==LazyTensor:
            basis_kernel = basis_kernel.sum(dim=1)
        # Get SVD
        U, S, V = torch.svd(basis_kernel)
        S = torch.maximum(S, torch.ones(S.size()) * 1e-12)
        self.normalization_ = torch.mm(U / np.sqrt(S), V.t())
        self.components_ = basis
        self.component_indices_ = inds

        return self

    def _pairwise_kernels(self, x: torch.tensor, y: torch.tensor = None, kernel='rbf',
                          sigma:float = 1.) -> LazyTensor:
        '''Helper function to build kernel

        Args:   X = torch tensor of dimension 2.
                K_type = type of Kernel to return

        Returns:
                K_ij[LazyTensor]
        '''

        if y is None:
            y = x
        if kernel == 'linear':
            K_ij = x @ y.T
        elif kernel == 'rbf':
            x /= sigma
            y /= sigma
            x_i, x_j = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])
            K_ij = (-1 * ((x_i - x_j) ** 2).sum(2)).exp()
            # block-sparse reduction preprocess
            K_ij = self._Gauss_block_sparse_pre(x, y, K_ij, self.sigma, self.eps)
        elif kernel == 'exp':
            x_i, x_j = LazyTensor(x[:, None, :]), LazyTensor(y[None, :, :])
            K_ij = ((-1 * sqrt((x_i - x_j) ** 2)).sum(2)).exp()
            # block-sparse reduction preprocess
            K_ij = self._Gauss_block_sparse_pre(x, y, K_ij, self.sigma, self.eps)
        return K_ij

    """def _spectral(self, X_i: LazyTensor):
        '''
        Helper function to compute eigendecomposition of K_q.
        Written using LinearOperators which are lazy
        representations of sparse and/or structured data.
        Args: X_i[numpy LazyTensor]
        Returns S[np.array] eigenvalues,
                U[np.array] eigenvectors
        '''

        K_linear = aslinearoperator(X_i)
        # K <- K + eps
        K_linear = K_linear + IdentityOperator(K_linear.shape, dtype=self.dtype) * self.inv_eps
        k = K_linear.shape[0] - 1
        S, U = eigsh(K_linear, k=k, which='LM')
        return S, U"""

    def transform(self, X: torch.tensor) -> torch.tensor:
        ''' Applies transform on the data.

        Args:
            X [LazyTensor] = data to transform
        Returns
            X [LazyTensor] = data after transformation
        '''


        K_nq = self._pairwise_kernels(X, self.components_, self.kernel)
        return K_nq @ self.normalization_.t()

    def K_approx(self, X: torch.tensor) -> torch.tensor:
        ''' Function to return Nystrom approximation to the kernel.

        Args:
            X[torch.tensor] = data used in fit(.) function.
        Returns
            K[torch.tensor] = Nystrom approximation to kernel'''

        K_nq = self._pairwise_kernels(X, self.components_, self.kernel)
        K_approx = K_nq @ self.normalization_ @ K_nq.t()
        return K_approx

    def _Gauss_block_sparse_pre(self, x: torch.Tensor, y: torch.Tensor, K_ij: torch.Tensor,
        sigma: float = 1., eps:float = 0.05):
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
        x = x.numpy()
        y = y.numpy()
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
###############################
# run on MNIST
###############################
n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)
print(type(example_data))
print(example_data.shape)
#%%timeit
length = 5000
num_sampling = 100

x = example_data.permute(1,0,2,3)
X_i = x.view(1,1000,784)

# Instatiate & fit on lazy tensor version
LN_test = LazyNystrom_TK(num_sampling, kernel='rbf', random_state=0).fit(example_data)
X_new_i = LN_test.transform(X_i)

#NUMPY on MNIST
#%%timeit
# Instatiate & fit Nystroem for comparison
x_ = x.reshape(1, 1000,784)


# Instatiate & fit Nystroem for comparison
sk_N = Nystroem(kernel='rbf', n_components=num_sampling, random_state=0).fit(x[0].numpy())
x_new = sk_N.transform(x[0].numpy())      # (length, num_sampling) array

# Instatiate & fit on lazy tensor version
LN_test = LazyNystrom_TK(num_sampling, kernel='rbf', random_state=0).fit(x_)
X_new_i = LN_test.transform(x_)          # (1,length,num_sampling) lazy tensor

# Print the L2 error
err = np.linalg.norm(x_new - X_new_i.sum(dim=0).numpy()) / x_new.size
print(f'Error when compared to sklearn = {err}')
#>>Error when compared to sklearn = 0.000224494037628173823