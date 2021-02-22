import pykeops
pykeops.clean_pykeops()
import torch
from pykeops.torch import LazyTensor
import numpy as np
from pykeops.torch.cluster import cluster_ranges_centroids
from pykeops.torch.cluster import from_matrix
use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
if use_cuda:
    torch.cuda.synchronize()
    
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
    self.device=None

  def __KMeans(self,x, K=10, Niter=15):
      """Implements Lloyd's algorithm for the Euclidean metric."""
      
      N, D = x.shape  
      c = x[:K, :].clone() 
      x_i = LazyTensor(x.view(N, 1, D))  
      c_j = LazyTensor(c.view(1, K, D))  

      for i in range(Niter):
          D_ij = ((x_i - c_j) ** 2).sum(-1)  
          cl = D_ij.argmin(dim=1).long().view(-1)  
          c.zero_() #sets c to 0
          c.scatter_add_(0, cl[:, None].repeat(1, D), x) 
          Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
          c /= Ncl  

      return cl, c    

  def __k_argmin(self,x,y,k=1):
    #print(x)
    x_LT=LazyTensor(x.unsqueeze(1))
    y_LT=LazyTensor(y.unsqueeze(0))
    d=((x_LT-y_LT)**2).sum(-1)
    if k==1:
      return d.argmin(dim=1).long().view(-1)
    else:
      return d.argKmin(K=k,dim=1).long()  

  def __sort_clusters(self,x,lab,store_x=True):
    lab, perm = torch.sort(lab.view(-1))
    if store_x:
      self.x_perm=perm #store permutations
    else:
      self.y_perm=perm
    return x[perm],lab

  def __unsort(self,nn):

    un_x=self.x_perm[nn]

    out=torch.index_select(un_x,0,self.y_perm.argsort())

    return out

  def fit(self,x,clusters=50,a=5):
    '''
    Fits the main dataset
    '''
    self.device=x.device
    cl, c = self.__KMeans(x,clusters)
    self.c=c

    #print('check:',type(x))
    self.__k_argmin(x,c)
    self.cl=self.__k_argmin(x,c)

    if use_cuda:
        torch.cuda.synchronize()

    self.ncl=self.__k_argmin(c,c,k=a)

    #get the ranges and centroids 
    self.x_ranges, _, _ = cluster_ranges_centroids(x, self.cl)
    
    x, x_labels = self.__sort_clusters(x,self.cl,store_x=True) #sort dataset to match ranges
    self.x=x#store dataset
    r=torch.arange(clusters).repeat(a,1).T.reshape(-1).long()
    self.keep= torch.zeros([clusters,clusters], dtype=torch.bool).to(self.device)  
    self.keep[r,self.ncl.flatten()]=True        
    return self


  def __assign(self,x,c=None):
    if c is None:
      c=self.c
    return self.__k_argmin(x,c)
    
  def kneighbors(self,y,sparse=True):
    '''
    Obtain the k nearest neighbors of the query dataset y
    '''
    assert(y.device==self.device)
    if use_cuda:
        torch.cuda.synchronize()
    y_labels=self.__assign(y,self.c)

    y_ranges,_,_ = cluster_ranges_centroids(y, self.cl)

    y, y_labels = self.__sort_clusters(y, y_labels,store_x=False)   
    
    x_LT=LazyTensor(self.x.unsqueeze(0).to(self.device))
    y_LT=LazyTensor(y.unsqueeze(1).to(self.device))
    D_ij=((y_LT-x_LT)**2).sum(-1)
  
    ranges_ij = from_matrix(self.x_ranges, y_ranges, self.keep)
    D_ij.ranges=ranges_ij
    nn=D_ij.argKmin(K=self.k,axis=1)

    return self.__unsort(nn)

  def nn(self,x,y):
    x_LT=LazyTensor(x.unsqueeze(0))
    y_LT=LazyTensor(y.unsqueeze(1))
    D_ij=((y_LT-x_LT)**2).sum(-1) 
    return D_ij.argKmin(K=self.k,axis=1)
