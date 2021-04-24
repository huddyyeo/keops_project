import torch

from nystrom_common import GenericNystroem 
from torch_utils import torchtools 
from pykeops.torch import LazyTensor


class Nystroem(GenericNystroem):

    def __init__(self, n_components=100, kernel='rbf', sigma: float = None,
                 inv_eps: float = None, verbose=False, random_state=None):

        super().__init__(n_components, kernel, sigma, inv_eps,
                    verbose, random_state)

        self.tools = torchtools
        self.verbose = verbose
        self.lazy_tensor = LazyTensor

    def _update_dtype(self, x):
        pass


    def _decomposition_and_norm(self, basis_kernel):
        '''Function to return self.normalization used in fit(.) function
        Args:
            basis_kernel(torch LazyTensor): subset of input data
        Returns:
            self.normalization(torch.tensor):  X_q is the q x D-dimensional sub matrix of matrix X
            '''
        U, S, V = torch.linalg.svd(basis_kernel, full_matrices=False) # (Q,Q), (Q,), (Q,Q)
        S = torch.maximum(S, torch.ones(S.size()) * 1e-12)
        return U / torch.sqrt(S)@ V   # (Q,Q)

    def kernel(self, x,y, kernel):
        D_xx = (x * x).sum(-1).unsqueeze(1)  # (N,1)
        D_xy = torch.matmul(x, y.permute(1, 0))  # (N,D) @ (D,M) = (N,M)
        D_yy = (y * y).sum(-1).unsqueeze(0)  # (1,M)
        D_xy = D_xx - 2 * D_xy + D_yy  # (N,M)
        if kernel == 'exp':
            D_xy = torch.sqrt(D_xy)
        return (-D_xy).exp()  # (N,M)

    def K_approx(self, X: torch.tensor) -> 'K_approx operator':
        ''' Function to return Nystrom approximation to the kernel.
        Args:
            X(tensor): data used in fit(.) function.
        Returns
            K_approx(operator): Nystrom approximation to kernel which can be applied
                        downstream as K_approx @ v for some 1d tensor v'''

        K_nq = self._pairwise_kernels(X, self.components, dense=False) # (N, Q)
        return K_approx_operator(K_nq, self.normalization) # (N, B), with v[N, B]

class K_approx_operator():
    ''' Helper class to return K_approx as an object 
    compatible with @ symbol'''

    def __init__(self, K_nq, normalization):

        self.K_nq = K_nq # dim: number of samples x num_components
        self.K_nq.backend="GPU_2D"
        self.normalization = normalization

    def __matmul__(self, v:torch.tensor) -> torch.tensor:

        x = self.K_nq.T @ v # (Q,N), (N,B)
        x = self.normalization @ self.normalization.T @ x # (Q,Q), (Q,Q), (Q, B)
        x = self.K_nq @ x # (N,Q), (Q,B)
        return x # (N,B)



