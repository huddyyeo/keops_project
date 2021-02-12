# Same as LazyNystrom_T but written with Pytorch

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
                 eps:float = 0.05,
                 random_state=None):

        self.n_components = n_components
        self.kernel = kernel
        self.random_state = random_state
        self.sigma = sigma
        self.eps = eps

    def fit(self, X: LazyTensor):
        '''
        Args:   X = torch lazy tensor with features of shape
                (1, n_samples, n_features)

        Returns: Fitted instance of the class
        '''

        # Basic checks: we have a lazy tensor and n_components isn't too large
        assert type(X) == torch.Tensor, 'Input to fit(.) must be a Tensor.'
        assert X.shape[0] >= self.n_components, f'The application needs X.shape[1] >= n_components.'

        # X = X.sum(dim=0)
        # Number of samples
        n_samples = X.shape(0)
        # Define basis
        rnd = check_random_state(self.random_state)
        inds = rnd.permutation(n_samples)
        basis_inds = inds[:self.n_components]
        basis = X[basis_inds]
        # Build smaller kernel
        basis_kernel = self._pairwise_kernels(basis, kernel=self.kernel)
        # Get SVD
        U, S, V = torch.svd(basis_kernel)
        S = torch.maximum(S, torch.ones(S.size()) * 1e-12)
        self.normalization_ = torch.mm(U / np.sqrt(S), V.t())
        self.components_ = basis
        self.component_indices_ = inds

        return self

    def _pairwise_kernels(self, x: torch.tensor, y: torch.tensor = None, kernel='linear',
                          sigma:float = 1.) -> torch.tensor:
        '''Helper function to build kernel

        Args:   X = torch tensor of dimension 2.
                K_type = type of Kernel to return
        '''

        if y is None:
            y = x
        if kernel == 'linear':
            K_ij = x @ y.T
        elif kernel == 'rbf':
            x /= sigma
            y /= sigma
            x_i, x_j = LazyTensor_n(x[:, None, :]), LazyTensor_n(y[None, :, :])
            K_ij = (-1 * ((x_i - x_j) ** 2).sum(2)).exp()
            # block-sparse reduction preprocess
            K_ij = self._Gauss_block_sparse_pre(x, y, K_ij, self.sigma, self.eps)
        elif kernel == 'exp':
            x_i, x_j = LazyTensor_n(x[:, None, :]), LazyTensor_n(y[None, :, :])
            K_ij = (-1 * sqrt((x_i - x_j) ** 2)).sum(2)).exp()
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

    def transform(self, X: torch.Tensor) -> Tensor:
        ''' Applies transform on the data.

        Args:
            X [LazyTensor] = data to transform
        Returns
            X [LazyTensor] = data after transformation
        '''


        K_nq = self._pairwise_kernels(X, self.components_, self.kernel)
        return K_nq @ self.normalization_.t()

    def K_approx(self, X: LazyTensor) -> LazyTensor:
        ''' Function to return Nystrom approximation to the kernel.

        Args:
            X[LazyTensor] = data used in fit(.) function.
        Returns
            K[LazyTensor] = Nystrom approximation to kernel'''

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