import numpy as np
import pykeops
from typing import TypeVar, Union, Tuple
import warnings

# Generic placeholder for numpy and torch variables.
generic_array = TypeVar('generic_array')
GenericLazyTensor = TypeVar('GenericLazyTensor')


class GenericNystrom:
    '''Super class defining the Nystrom operations. The end user should
    use numpy.nystrom or torch.nystrom subclasses.'''

    def __init__(self, n_components: int = 100, kernel: Union[str, callable] = 'rbf', sigma: float = None,
                 eps: float = 0.05, inv_eps: float = None,
                 verbose: bool = False,
                 random_state: Union[None, int] = None, tools=None):

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
        verbose = set True to print details.
        random_state = to set a random seed for the random sampling of the samples.
                        To be used when  reproducibility is needed.
        '''
        self.n_components = n_components
        self.kernel = kernel
        self.sigma = sigma
        self.eps = eps
        self.dtype = None
        self.verbose = verbose
        self.random_state = random_state
        self.tools = None
        self.LazyTensor = None

        self.device = 'cuda' if pykeops.config.gpu_available else 'cpu'

        if inv_eps:
            self.inv_eps = inv_eps
        else:
            self.inv_eps = 1e-8

    def fit(self, x: generic_array) -> 'GenericNystrom':
        '''
        Args:   x = array or tensor of shape (n_samples, n_features)
        Returns: Fitted instance of the class
        '''
        x = self._to_device(x)
        self.dtype = x.dtype

        # Basic checks
        assert self.tools.is_tensor(x), 'Input to fit(.) must be an array\
        if using numpy and tensor if using torch.'
        assert x.shape[0] >= self.n_components, 'The application needs\
        X.shape[0] >= n_components.'
        if self.kernel == 'exp' and not (self.sigma is None):
            assert self.sigma > 0, 'Should be working with decaying exponential.'

        # Set default sigma
        # if self.sigma is None and self.kernel == 'rbf':
        if self.sigma is None:
            self.sigma = np.sqrt(x.shape[1])

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
        basis_kernel = self._pairwise_kernels(basis, dense=True)

        # Decomposition is an abstract method that needs to be defined in each class
        self.normalization_ = self._decomposition_and_norm(basis_kernel)
        self.components_ = basis
        self.component_indices_ = inds

        return self

    def _decomposition_and_norm(self, X: GenericLazyTensor):
        """
        To be defined in the subclass
        """
        raise NotImplementedError('Subclass must implement the method _decomposition_and_norm.')
        

    def transform(self, x:generic_array, dense=True) -> generic_array:
        '''
        Applies transform on the data mapping it to the feature space
        which supports the approximated kernel.
        Args:
            X = data to transform
        Returns
            X = data after transformation
        '''
        if type(x) == np.ndarray and not dense:
            warnings.warn("For Numpy transform it is best to use dense=True")
            
        x = self._to_device(x)
        K_nq = self._pairwise_kernels(x, self.components_, dense=dense)
        x_new = K_nq @ self.normalization_
        return x_new

    def _pairwise_kernels(self, x:generic_array, y:generic_array=None, dense=False):
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
        x = x / self.sigma
        y = y / self.sigma

        x_i, x_j = self.tools.contiguous(self._to_device(x[:, None, :])), self.tools.contiguous(
            self._to_device(y[None, :, :]))

        if self.kernel == 'rbf':
            if dense:
                D_ij = ((x_i - x_j) ** 2).sum(axis=2)
                K_ij = self.tools.exp(-D_ij)

            else:
                x_i, x_j = self.LazyTensor(x_i), self.LazyTensor(x_j)
                D_ij = ((x_i - x_j) ** 2).sum(dim=2)
                K_ij = (-D_ij).exp()

        elif self.kernel == 'exp':
            if dense:
                K_ij = self.tools.exp(-self.tools.sqrt((((x_i - x_j) ** 2).sum(axis=2))))

            else:
                x_i, x_j = self.LazyTensor(x_i), self.LazyTensor(x_j)
                K_ij = (-(((x_i - x_j) ** 2).sum(-1)).sqrt()).exp()

        # computation with custom kernel
        else:
            print('Please note that computations on custom kernels are dense-only.')
            K_ij = self.kernel(x_i, x_j)

        return K_ij

    def _astype(self, data, type):
        return data

    def _to_device(self, data):
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

    def _check_random_state(self, seed: Union[None, int]) -> None:
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

