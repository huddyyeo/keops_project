import torch
import time
from pykeops.torch import LazyTensor, Genred, KernelSolve, default_dtype
from pykeops.torch.cluster import swap_axes as torch_swap_axes
from pykeops.torch.cluster import cluster_ranges_centroids, from_matrix

# from pykeops.torch.generic.generic_red import GenredLowlevel


def is_on_device(x):
    return x.is_cuda



class torchtools:
    copy = torch.clone
    exp = torch.exp
    log = torch.log
    norm = torch.norm

    swap_axes = torch_swap_axes

    Genred = Genred
    KernelSolve = KernelSolve

    arraytype = torch.Tensor
    float_types = [float]

    # GenredLowlevel = GenredLowlevel

    @staticmethod
    def eq(x, y):
        return torch.eq(x, y)

    @staticmethod
    def transpose(x):
        return x.t()

    @staticmethod
    def permute(x, *args):
        return x.permute(*args)

    @staticmethod
    def contiguous(x):
        return x.contiguous()

    @staticmethod
    def solve(A, b):
        return torch.solve(b, A)[0].contiguous()

    @staticmethod
    def arraysum(x, axis=None):
        return x.sum() if axis is None else x.sum(dim=axis)

    @staticmethod
    def long(x):
        return x.long()

    @staticmethod
    def size(x):
        return x.numel()

    @staticmethod
    def tile(*args):
        return torch.Tensor.repeat(*args)

    @staticmethod
    def numpy(x):
        return x.detach().cpu().numpy()

    @staticmethod
    def view(x, s):
        return x.view(s)

    @staticmethod
    def is_tensor(x):
        return isinstance(x, torch.Tensor)

    @staticmethod
    def dtype(x):
        if hasattr(x, "dtype"):
            return x.dtype
        else:
            return type(x)

    @staticmethod
    def detect_complex(x):
        if type(x) == list:
            return any(type(v) == complex for v in x)
        elif type(x) == torch.Tensor:
            return torch.is_complex(x)
        else:
            return type(x) == complex

    @staticmethod
    def view_as_complex(x):
        sh = list(x.shape)
        sh[-1] //= 2
        sh += [2]
        x = x.view(sh)
        return torch.view_as_complex(x)

    @staticmethod
    def view_as_real(x):
        sh = list(x.shape)
        sh[-1] *= 2
        return torch.view_as_real(x).view(sh)

    @staticmethod
    def dtypename(dtype):
        if dtype == torch.float32:
            return "float32"
        elif dtype == torch.float64:
            return "float64"
        elif dtype == torch.float16:
            return "float16"
        elif dtype == int:
            return int
        elif dtype == list:
            return "float32"
        else:
            raise ValueError(
                "[KeOps] {} data type incompatible with KeOps.".format(dtype)
            )

    @staticmethod
    def rand(m, n, dtype=default_dtype, device="cpu"):
        return torch.rand(m, n, dtype=dtype, device=device)

    @staticmethod
    def randn(m, n, dtype=default_dtype, device="cpu"):
        return torch.randn(m, n, dtype=dtype, device=device)

    @staticmethod
    def zeros(shape, dtype=default_dtype, device="cpu"):
        return torch.zeros(shape, dtype=dtype, device=device)

    @staticmethod
    def eye(n, dtype=default_dtype, device="cpu"):
        return torch.eye(n, dtype=dtype, device=device)

    @staticmethod
    def array(x, dtype=default_dtype, device="cpu"):
        if dtype == "float32":
            dtype = torch.float32
        elif dtype == "float64":
            dtype = torch.float64
        elif dtype == "float16":
            dtype = torch.float16
        else:
            raise ValueError("[KeOps] data type incompatible with KeOps.")
        return torch.tensor(x, dtype=dtype, device=device)

    @staticmethod
    def device(x):
        if isinstance(x, torch.Tensor):
            return x.device
        else:
            return None

    @staticmethod
    def distance_function(metric):
        def euclidean(x,y):
            return ((x-y) ** 2).sum(-1)
        def manhattan(x,y):
            return ((x-y).abs()).sum(-1)
        def angular(x,y):
            return -(x | y)
        def angular_full(x,y):
            return angular(x,y)/((angular(x,x)*angular(y,y)).sqrt())
        def hyperbolic(x,y):
            return ((x - y) ** 2).sum(-1) / (x[0] * y[0])
        if metric=='euclidean':
            return euclidean
        elif metric=='manhattan':
            return manhattan
        elif metric=='angular':
            return angular
        elif metric=='angular_full':
            return angular_full  
        elif metric=='hyperbolic':
            return hyperbolic      
        else:
            raise ValueError('Unknown metric')  

    @staticmethod
    def sort(x):
        return torch.sort(x)

    @staticmethod
    def unsqueeze(x,n):
        return torch.unsqueeze(x,n)
    @staticmethod
    def arange(n,device="cpu"):
        return torch.arange(n,device=device)
    @staticmethod
    def repeat(x,n):
        return torch.repeat_interleave(x,n)
      
    @staticmethod
    def to(x,device):
        return x.to(device)
      
    @staticmethod
    def index_select(input,dim,index):
        return torch.index_select(input,dim,index)

    @staticmethod
    def norm(x,p=2,dim=-1):
        return torch.norm(x,p=p,dim=dim)
     
    @staticmethod
    def kmeans(x, distance=None, K=10, Niter=10, device="cuda", approx=False, n=10):

        from pykeops.torch import LazyTensor

        if distance is None:
            distance = torchtools.distance_function("euclidean")

        def calc_centroid(x, c, cl, n=10):
            "Helper function to optimise centroid location"
            c = torch.clone(c.detach()).to(device)
            c.requires_grad = True
            x1 = LazyTensor(x.unsqueeze(0))
            op = torch.optim.Adam([c], lr=1 / n)
            scaling = 1 / torch.gather(torch.bincount(cl), 0, cl).view(-1, 1)
            scaling.requires_grad = False
            with torch.autograd.set_detect_anomaly(True):
                for _ in range(n):
                    c.requires_grad = True
                    op.zero_grad()
                    c1 = LazyTensor(torch.index_select(c, 0, cl).unsqueeze(0))
                    d = distance(x1, c1)
                    loss = (
                        d.sum(0) * scaling
                    ).sum()  # calculate distance to centroid for each datapoint, divide by total number of points in that cluster, and sum
                    loss.backward(retain_graph=False)
                    op.step()
            return c.detach()

        N, D = x.shape
        c = x[:K, :].clone()
        x_i = LazyTensor(x.view(N, 1, D).to(device))

        for i in range(Niter):
            c_j = LazyTensor(c.view(1, K, D).to(device))
            D_ij = distance(x_i, c_j)
            cl = D_ij.argmin(dim=1).long().view(-1)

            # updating c: either with approximation or exact
            if approx:
                # approximate with GD optimisation
                c = calc_centroid(x, c, cl, n)

            else:
                # exact from average
                c.zero_()
                c.scatter_add_(0, cl[:, None].repeat(1, D), x)
                Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
                c /= Ncl

            if torch.any(torch.isnan(c)):
                raise ValueError(
                    "NaN detected in centroids during KMeans, please check metric is correct"
                )
        return cl, c



def squared_distances(x, y):
    x_norm = (x ** 2).sum(1).reshape(-1, 1)
    y_norm = (y ** 2).sum(1).reshape(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.matmul(x, torch.transpose(y, 0, 1))
    return dist


def torch_kernel(x, y, s, kernel):
    sq = squared_distances(x, y)
    _kernel = {
        "gaussian": lambda _sq, _s: torch.exp(-_sq / (_s * _s)),
        "laplacian": lambda _sq, _s: torch.exp(-torch.sqrt(_sq) / _s),
        "cauchy": lambda _sq, _s: 1.0 / (1 + _sq / (_s * _s)),
        "inverse_multiquadric": lambda _sq, _s: torch.rsqrt(1 + _sq / (_s * _s)),
    }
    return _kernel[kernel](sq, s)   

class GenericIVF:
    """Abstract class to compute IVF functions
    End-users should use 'pykeops.numpy.ivf' or 'pykeops.torch.ivf'
    """

    def __init__(self, k, metric, normalise, LazyTensor):

        self.__k = k
        self.__normalise = normalise
        self.__update_metric(metric)
        self.__LazyTensor = LazyTensor
        self.__c = None

    def __update_metric(self, metric):
        if isinstance(metric, str):
            self.__distance = self.tools.distance_function(metric)
            self.__metric = metric
        elif callable(metric):
            self.__distance = metric
            self.__metric = "custom"
        else:
            raise ValueError("Unrecognised metric input type")

    @property
    def metric(self):
        """Returns the metric used in the search"""
        return self.__metric

    @property
    def c(self):
        """Returns the clusters obtained through K-Means"""
        if self.__c is not None:
            return self.__c
        else:
            raise ValueError("Run .fit() first!")

    def __get_tools(self):
        pass

    def __k_argmin(self, x, y, k=1):
        x_LT = self.__LazyTensor(
            self.tools.to(self.tools.unsqueeze(x, 1), self.__device)
        )
        y_LT = self.__LazyTensor(
            self.tools.to(self.tools.unsqueeze(y, 0), self.__device)
        )

        d = self.__distance(x_LT, y_LT)
        if not self.tools.is_tensor(x):
            if self.__backend:
                d.backend = self.__backend

        if k == 1:
            return self.tools.view(self.tools.long(d.argmin(dim=1)), -1)
        else:
            return self.tools.long(d.argKmin(K=k, dim=1))

    def __sort_clusters(self, x, lab, store_x=True):
        lab, perm = self.tools.sort(self.tools.view(lab, -1))
        if store_x:
            self.__x_perm = perm
        else:
            self.__y_perm = perm
        return x[perm], lab

    def __unsort(self, nn):
        return self.tools.index_select(self.__x_perm[nn], 0, self.__y_perm.argsort())

    def _fit(
        self,
        x,
        clusters=50,
        a=5,
        Niter=15,
        device=None,
        backend=None,
        approx=False,
        n=50,
    ):
        """
        Fits the main dataset
        """
        if type(clusters) != int:
            raise ValueError("Clusters must be an integer")
        if clusters >= len(x):
            raise ValueError("Number of clusters must be less than length of dataset")
        if type(a) != int:
            raise ValueError("Number of clusters to search over must be an integer")
        if a > clusters:
            raise ValueError(
                "Number of clusters to search over must be less than total number of clusters"
            )
        if len(x.shape) != 2:
            raise ValueError("Input must be a 2D array")
        if self.__normalise:
            x = x / self.tools.repeat(self.tools.norm(x, 2, -1), x.shape[1]).reshape(
                -1, x.shape[1]
            )

        # if we want to use the approximation in Kmeans, and our metric is angular, switch to full angular metric
        if approx and self.__metric == "angular":
            self.__update_metric("angular_full")

        x = self.tools.contiguous(x)
        self.__device = device
        self.__backend = backend

        cl, c = self.tools.kmeans(
            x,
            self.__distance,
            clusters,
            Niter=Niter,
            device=self.__device,
            approx=approx,
            n=n,
        )

        self.__c = c
        cl = self.__assign(x)

        ncl = self.__k_argmin(c, c, k=a)
        self.__x_ranges, _, _ = cluster_ranges_centroids(x, cl)

        x, x_labels = self.__sort_clusters(x, cl, store_x=True)
        self.__x = x
        r = self.tools.repeat(self.tools.arange(clusters, device=self.__device), a)
        self.__keep = self.tools.to(
            self.tools.zeros([clusters, clusters], dtype=bool), self.__device
        )
        self.__keep[r, ncl.flatten()] = True

        return self

    def __assign(self, x, c=None):
        if c is None:
            c = self.__c
        return self.__k_argmin(x, c)

    def _kneighbors(self, y):
        """
        Obtain the k nearest neighbors of the query dataset y
        """
        if self.__x is None:
            raise ValueError("Input dataset not fitted yet! Call .fit() first!")
        if self.__device and self.tools.device(y) != self.__device:
            raise ValueError("Input dataset and query dataset must be on same device")
        if len(y.shape) != 2:
            raise ValueError("Query dataset must be a 2D tensor")
        if self.__x.shape[-1] != y.shape[-1]:
            raise ValueError("Query and dataset must have same dimensions")
        if self.__normalise:
            y = y / self.tools.repeat(self.tools.norm(y, 2, -1), y.shape[1]).reshape(
                -1, y.shape[1]
            )
        y = self.tools.contiguous(y)
        y_labels = self.__assign(y)

        y_ranges, _, _ = cluster_ranges_centroids(y, y_labels)
        self.__y_ranges = y_ranges
        y, y_labels = self.__sort_clusters(y, y_labels, store_x=False)
        x_LT = self.__LazyTensor(self.tools.unsqueeze(self.__x, 0))
        y_LT = self.__LazyTensor(self.tools.unsqueeze(y, 1))
        D_ij = self.__distance(y_LT, x_LT)
        ranges_ij = from_matrix(y_ranges, self.__x_ranges, self.__keep)
        D_ij.ranges = ranges_ij
        nn = D_ij.argKmin(K=self.__k, axis=1)
        return self.__unsort(nn)

    def brute_force(self, x, y, k=5):
        """Performs a brute force search with KeOps
        Args:
          x (array): Input dataset
          y (array): Query dataset
          k (int): Number of nearest neighbors to obtain
        """
        x_LT = self.__LazyTensor(self.tools.unsqueeze(x, 0))
        y_LT = self.__LazyTensor(self.tools.unsqueeze(y, 1))
        D_ij = self.__distance(y_LT, x_LT)
        return D_ij.argKmin(K=k, axis=1)


class IVF(GenericIVF):
    """IVF-Flat is a KNN approximation algorithm that first clusters the data and then performs the query search on a subset of the input dataset."""

    def __init__(self, k=5, metric="euclidean", normalise=False):
        """Initialise the IVF-Flat class.
        IVF-Flat is a KNN approximation algorithm that first clusters the data and then performs the query search on a subset of the input dataset.
        Args:
          k (int): Number of nearest neighbours to obtain
          metric (str,function): Metric to use
            Currently, "euclidean", "manhattan", "angular" and "hyperbolic" are directly supported, apart from custom metrics
            Hyperbolic metric requires the use of approx = True, during the fit() function later
            Custom metrics should be in the form of a function with 2 inputs and returns their distance
            For more information, refer to the tutorial
          normalise (bool): Whether or not to normalise all input data to norm 1
            This is used mainly for angular metric
            In place of this, "angular_full" metric may be used instead
        """
        from pykeops.torch import LazyTensor

        self.__get_tools()
        super().__init__(k=k, metric=metric, normalise=normalise, LazyTensor=LazyTensor)

    def __get_tools(self):
        # from pykeops.torch.utils import torchtools

        self.tools = torchtools

    def fit(self, x, clusters=50, a=5, Niter=15, approx=False, n=50):
        """Fits a dataset to perform the nearest neighbour search over
        K-Means is performed on the dataset to obtain clusters
        Then the closest clusters to each cluster is stored for use during query time
        Args:
          x (torch.Tensor): Torch tensor dataset of shape N, D
            Where N is the number of points and D is the number of dimensions
          clusters (int): Total number of clusters to create in K-Means
          a (int): Number of clusters to search over, must be less than total number of clusters created
          Niter (int): Number of iterations to run in K-Means algorithm
          approx (bool): Whether or not to use an approximation step in K-Means
            In hyperbolic metric and custom metric, this should be set to True
            This is because the optimal cluster centroid may not have a simple closed form expression
          n (int): Number of iterations to optimise the cluster centroid, when approx = True
            A value of around 50 is recommended
            Lower values are faster while higher values give better accuracy in centroid location
        """
        if type(x) != torch.Tensor:
            raise ValueError("Input dataset must be a torch tensor")
        return self._fit(
            x, clusters=clusters, a=a, Niter=Niter, device=x.device, approx=approx, n=n
        )

    def kneighbors(self, y):
        """Obtains the nearest neighbors for an input dataset from the fitted dataset
        Args:
          y (torch.Tensor): Input dataset to search over
        """
        if type(y) != torch.Tensor:
            raise ValueError("Query dataset must be a torch tensor")
        return self._kneighbors(y)