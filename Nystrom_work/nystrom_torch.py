# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 15:30:06 2021

@author: ahled
"""
##############################################################################
""" Torch class"""

import numpy as np
import torch
import pykeops

from pykeops.torch import LazyTensor
from pykeops.torch.cluster import grid_cluster
from pykeops.torch.cluster import from_matrix
from pykeops.torch.cluster import cluster_ranges_centroids, cluster_ranges
from pykeops.torch.cluster import sort_clusters

from nystrom_common import Nystrom_common
from torch_utils import torchtools

class Nystrom_TK(Nystrom_common):
    
    def __init__(self, n_components=100, kernel='rbf', sigma:float = None, 
                 eps:float = 0.05, mask_radius:float = None, k_means = 10, 
                 n_iter:int = 10, inv_eps:float = None, dtype = np.float32, 
                 backend = None, verbose = False, random_state=None, tools = None):
        
        super().__init__(n_components, kernel, sigma, eps, mask_radius, k_means, 
                         n_iter, inv_eps, dtype, backend, verbose, random_state)
        
        self.tools = torchtools

        # as of now torch doesn't use these
        self.backend = None
        self.verbose = None

    def _update_dtype(self, x):
        pass
    
    def _decomposition_and_norm(self, basis_kernel):
        
        basis_kernel = basis_kernel @ torch.diag(torch.ones(basis_kernel.shape[1]))
        U, S, V = torch.svd(basis_kernel)
        S = torch.maximum(S, torch.ones(S.size()) * 1e-12)

        return torch.mm(U / np.sqrt(S), V.t())
    
    def K_approx(self, X: torch.tensor) -> torch.tensor:
        ''' Function to return Nystrom approximation to the kernel.
        Args:
            X[torch.tensor] = data used in fit(.) function.
        Returns
            K[torch.tensor] = Nystrom approximation to kernel'''

        K_nq = self._pairwise_kernels(X, self.components_, dense = True)
        K_approx = K_nq @ self.normalization_ @ K_nq.t()
        return K_approx