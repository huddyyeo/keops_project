import pykeops
pykeops.clean_pykeops()
import torch
from pykeops.torch import LazyTensor
import numpy as np
from pykeops.torch.cluster import cluster_ranges_centroids
#from pykeops.torch.cluster import sort_clusters
from pykeops.torch.cluster import from_matrix
use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
if use_cuda:
    torch.cuda.synchronize()

def KMeans(x, K=10, Niter=10):
    """Implements Lloyd's algorithm for the Euclidean metric."""


    N, D = x.shape 
    c = x[:K, :].clone()  
    x_i = LazyTensor(x.view(N, 1, D)) 
    c_j = LazyTensor(c.view(1, K, D)) 
    for i in range(Niter):

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster
        c.zero_() #sets c to 0
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  

    return cl, c

def k_argmin(x,y,k=5):
  x_LT=LazyTensor(x.unsqueeze(1))
  y_LT=LazyTensor(y.unsqueeze(0))
  d=((x_LT-y_LT)**2).sum(-1)
  return d.argmin(dim=1).long().view(-1)   

def k_argmin_torch(x,y,k=5):
  d=((x.unsqueeze(1)-y.unsqueeze(0))**2).sum(-1)
  sort,idx=torch.sort(d,dim=1)
  return idx[:,:k]

class IVF_flat():
  def __init__(self,k=5):
    self.c=None
    self.cl=None
    self.k=k
    self.ncl=None
    self.x=None
    self.keep=None
    self.x_ranges=None
    self.x_perm=None
    self.y_perm=None
    self.y_sorted=None

  def sort_clusters(self,x,lab,store_x=True):
    lab, perm = torch.sort(lab.view(-1))
    if store_x:
      self.x_perm=perm #store permutations
    else:
      self.y_perm=perm
    return x[perm],lab

  def unsort(self,nn):

    un_x=self.x_perm[nn]

    out=torch.index_select(un_x,0,self.y_perm.argsort())

    return out

  def fit(self,x,use_torch=True,clusters=50,a=5):
    cl, c = KMeans(x,clusters)
    self.c=c
    #update cluster assignment
    if use_torch:
      d=((x.unsqueeze(1)-c.unsqueeze(0))**2).sum(-1)
      self.cl=torch.argmin(d,dim=1)
    else:
      self.cl=k_argmin(x,c)
    if use_cuda:
        torch.cuda.synchronize()

    #get KNN graph for the clusters
    if use_torch:

      self.ncl=k_argmin_torch(c,c,k=a)
    else:
        
      c1=LazyTensor(c.unsqueeze(1)) 
      c2=LazyTensor(c.unsqueeze(0))
      d=((c1-c2)** 2).sum(-1)
      self.ncl=d.argKmin(K=a,dim=1) #get a nearest clusters

    #get the ranges and centroids 
    self.x_ranges, _, _ = cluster_ranges_centroids(x, self.cl)
    
    x, x_labels = self.sort_clusters(x,self.cl,store_x=True) #sort dataset to match ranges
    self.x=x#store dataset
      
    r=torch.arange(clusters).repeat(a,1).T.reshape(-1).long()
    self.keep= torch.zeros([clusters,clusters], dtype=torch.bool)    
   
    self.keep[r,self.ncl.flatten()]=True        
    return self

  def sorted(self,x,labels=None):
    if labels is None:
      labels=self.cl
    x,_=sort_clusters(x,labels)
    return x
  def clusters(self):
    return self.c
  def assign(self,x,c=None):
    if c is None:
      c=self.c
    return k_argmin(x,c,self.k)
    
  def kneighbors(self,y,sparse=True):
    if use_cuda:
        torch.cuda.synchronize()
    d=((y.unsqueeze(1)-self.c.unsqueeze(0))**2).sum(-1)
    y_labels=torch.argmin(d,dim=1)

    y_ranges,_,_ = cluster_ranges_centroids(y, self.cl)

    
    y, y_labels = self.sort_clusters(y, y_labels,store_x=False)   
    self.y_sorted=y
    
    x_LT=LazyTensor(self.x.unsqueeze(0))
    y_LT=LazyTensor(y.unsqueeze(1))
    D_ij=((y_LT-x_LT)**2).sum(-1)
    if sparse:
      ranges_ij = from_matrix(self.x_ranges, y_ranges, self.keep)    
      D_ij.ranges=ranges_ij
    nn=D_ij.argKmin(K=self.k,axis=1)

    return self.unsort(nn)

  def reduce(self,x,y):
    x_LT=LazyTensor(x.unsqueeze(0))
    y_LT=LazyTensor(y.unsqueeze(1))
    D_ij=((y_LT-x_LT)**2).sum(-1) 
    return D_ij.argKmin(K=self.k,axis=1)
