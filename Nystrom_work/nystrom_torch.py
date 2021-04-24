import torch

from nystrom_common import GenericNystroem 
from torch_utils import torchtools 
from pykeops.torch import LazyTensor


class Nystroem(GenericNystroem):

    def __init__(self, n_components=100, kernel='rbf', sigma: float = None,
                 eps: float = 0.05, n_iter: int = 10, inv_eps: float = None, 
                 verbose=False, random_state=None, tools=None):

        super().__init__(n_components, kernel, sigma, eps, inv_eps, 
                    verbose, random_state)

        self.tools = torchtools
        self.verbose = verbose
        self.LazyTensor = LazyTensor

    def _update_dtype(self, x):
        pass


    def _decomposition_and_norm(self, basis_kernel):
        '''Function to return self.normalization used in fit(.) function
        Args:
            basis_kernel(torch LazyTensor): subset of input data
        Returns:
            self.normalization(torch.tensor):  X_q is the q x D-dimensional sub matrix of matrix X
            '''
        basis_kernel = basis_kernel # dim: num_components x num_components
        U, S, V = torch.linalg.svd(basis_kernel, full_matrices=False) # dim: [100,100] x [100] x [100,100]
        S = torch.maximum(S, torch.ones(S.size()) * 1e-12)
        return torch.mm(U / torch.sqrt(S), V)   # dim: num_components x num_components

    def K_approx(self, X: torch.tensor) -> 'K_approx operator':
        ''' Function to return Nystrom approximation to the kernel.
        Args:
            X(tensor): data used in fit(.) function.
        Returns
            K_approx(operator): Nystrom approximation to kernel which can be applied
                        downstream as K_approx @ v for some 1d tensor v'''

        K_nq = self._pairwise_kernels(X, self.components, dense=False)
        K_approx = K_approx_operator(K_nq, self.normalization)
        return K_approx

class K_approx_operator():
    ''' Helper class to return K_approx as an object 
    compatible with @ symbol'''

    def __init__(self, K_nq, normalization):

        self.K_nq = K_nq # dim: number of samples x num_components
        self.K_nq.backend="GPU_2D"
        self.normalization = normalization

    def __matmul__(self, x:torch.tensor) -> torch.tensor:

        x = self.K_nq.T @ x 
        x = self.normalization @ self.normalization.T @ x
        x = self.K_nq @ x
        return x 

