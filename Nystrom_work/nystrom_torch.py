import torch

from nystrom_common import GenericNystrom 
from torch_utils import torchtools 
from pykeops.torch import LazyTensor


class Nystrom(GenericNystrom):

    def __init__(self, n_components=100, kernel='rbf', sigma: float = None,
                 eps: float = 0.05, mask_radius: float = None, k_means=10,
                 n_iter: int = 10, inv_eps: float = None, verbose=False, random_state=None, tools=None):
        super().__init__(n_components, kernel, sigma, eps, mask_radius, k_means,
                         n_iter, inv_eps, verbose, random_state)

        self.tools = torchtools
        self.verbose = verbose
        self.LazyTensor = LazyTensor

    def _update_dtype(self, x):
        pass

    def _to_device(self, x):
        return x.to(self.backend)

    def _decomposition_and_norm(self, basis_kernel):
        '''Function to return self.nomalization_ used in fit(.) function
        Args:
            basis_kernel[torch LazyTensor] = subset of input data
        Returns:
            self.normalization_[torch.tensor]  X_q is the q x D-dimensional sub matrix of matrix X
            '''
        id = torch.diag(torch.ones(basis_kernel.shape[1], dtype=self.dtype)).to(self.backend)
        basis_kernel = basis_kernel.to(self.backend) @ id
        U, S, V = torch.linalg.svd(basis_kernel, full_matrices=False)
        S = torch.maximum(S, torch.ones(S.size()).to(self.backend) * 1e-12)
        return torch.mm(U / torch.sqrt(S), V)

    def K_approx(self, X: torch.tensor) -> torch.tensor:
        ''' Function to return Nystrom approximation to the kernel.
        Args:
            X[torch.tensor] = data used in fit(.) function.
        Returns
            K[torch.tensor] = Nystrom approximation to kernel'''

        K_nq = self._pairwise_kernels(X, self.components_, dense=True)
        K_approx = K_approx_operator(K_nq, self.normalization_)
        return K_approx

class K_approx_operator():
    ''' Helper class to return K_approx as an object 
    compatible with @ symbol'''

    def __init__(self, K_nq, normalization):

        self.K_nq = K_nq
        self.normalization = normalization

    def __matmul__(self, x:torch.tensor) -> torch.tensor:
        print(self.K_nq.dtype)
        x = self.K_nq.T @ x 
        x = self.normalization @ self.normalization.T @ x
        x = self.K_nq @ x
        return x 
