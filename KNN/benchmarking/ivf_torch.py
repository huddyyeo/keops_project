import torch
import time
from pykeops.torch import Genred, KernelSolve, default_dtype
from pykeops.torch.cluster import swap_axes as torch_swap_axes


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
        return (x | y)
      def hyperbolic(x,y):
          return ((x - y) ** 2).sum(-1) / (x[0] * y[0])
      if metric=='euclidean':
        return euclidean
      elif metric=='manhattan':
        return manhattan
      elif metric=='angular':
        return angular
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
      if isinstance(x,torch.Tensor):
        return x.to(device)
      return x
      
    @staticmethod
    def index_select(input,dim,index):
      return torch.index_select(input,dim,index)

    @staticmethod
    def norm(x,p=2,dim=-1):
      return torch.norm(x,p=p,dim=dim)

    @staticmethod
    def kmeans(x,K=10,Niter=15,metric='euclidean',device='cuda'):
      from pykeops.torch import LazyTensor
      distance=torchtools.distance_function(metric)
      N, D = x.shape  
      c = x[:K, :].clone() 
      x_i = LazyTensor(x.view(N, 1, D).to(device))  
      for i in range(Niter):
          c_j = LazyTensor(c.view(1, K, D).to(device))  
          D_ij=distance(x_i,c_j)
          cl = D_ij.argmin(dim=1).long().view(-1)  
          c.zero_() 
          c.scatter_add_(0, cl[:, None].repeat(1, D), x) 
          Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
          c /= Ncl  
          if torch.any(torch.isnan(c)) and metric=='angular':
            raise ValueError("Please normalise inputs")
      return cl, c        

class GenericIVF:
    def __init__(
        self, k, metric, normalise, LazyTensor, cluster_ranges_centroids, from_matrix
    ):
        self.__k = k
        self.__normalise = normalise
        self.__distance = self.tools.distance_function(metric)
        self.__metric = metric
        self.__LazyTensor = LazyTensor
        self.__cluster_ranges_centroids = cluster_ranges_centroids
        self.__from_matrix = from_matrix

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

    def _fit(self, x, clusters=50, a=5, Niter=15, device=None, backend=None):
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
        x = self.tools.contiguous(x)
        self.__device = device
        self.__backend = backend

        cl, c = self.tools.kmeans(
            x, clusters, Niter=Niter, metric=self.__metric, device=self.__device
        )

        self.__c = c

        cl = self.__assign(x)

        ncl = self.__k_argmin(c, c, k=a)
        self.__x_ranges, _, _ = self.__cluster_ranges_centroids(x, cl)

        x, x_labels = self.__sort_clusters(x, cl, store_x=True)
        self.__x = x
        r = self.tools.repeat(self.tools.arange(clusters, device=self.__device), a)
        self.__keep = self.tools.zeros(
            [clusters, clusters], dtype=bool, device=self.__device
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

        y_ranges, _, _ = self.__cluster_ranges_centroids(y, y_labels)
        self.__y_ranges = y_ranges
        y, y_labels = self.__sort_clusters(y, y_labels, store_x=False)
        x_LT = self.__LazyTensor(self.tools.unsqueeze(self.__x, 0))
        y_LT = self.__LazyTensor(self.tools.unsqueeze(y, 1))
        D_ij = self.__distance(y_LT, x_LT)
        ranges_ij = self.__from_matrix(y_ranges, self.__x_ranges, self.__keep)
        D_ij.ranges = ranges_ij
        nn = D_ij.argKmin(K=self.__k, axis=1)
        return self.__unsort(nn)

    def brute_force(self, x, y, k=5):
        x_LT = self.__LazyTensor(self.tools.unsqueeze(x, 0))
        y_LT = self.__LazyTensor(self.tools.unsqueeze(y, 1))
        D_ij = self.__distance(y_LT, x_LT)
        return D_ij.argKmin(K=k, axis=1)

    
class IVF(GenericIVF):
  def __init__(self,k=5,metric='euclidean',normalise=False):
    self.__get_tools()
    super().__init__(k=k,metric=metric,normalise=normalise,LazyTensor=LazyTensor,cluster_ranges_centroids=cluster_ranges_centroids,from_matrix=from_matrix)

  def __get_tools(self):
    self.tools = torchtools
  def fit(self,x,clusters=50,a=5,Niter=15):
    if type(x)!=torch.Tensor:
      raise ValueError("Input dataset must be a torch tensor")    
    return self._fit(x,clusters=clusters,a=a,Niter=Niter,device=x.device)
  def kneighbors(self,y):
    if type(y)!=torch.Tensor:
      raise ValueError("Query dataset must be a torch tensor")
    return self._kneighbors(y)