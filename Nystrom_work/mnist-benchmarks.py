# !pip install pykeops[full] > install.log
import numpy as np
import torch
import pykeops

from pykeops.numpy import LazyTensor as LazyTensor_n
from pykeops.numpy.cluster import grid_cluster
from pykeops.numpy.cluster import from_matrix
from pykeops.numpy.cluster import cluster_ranges_centroids
from pykeops.numpy.cluster import sort_clusters
from pykeops.torch import LazyTensor

from sklearn.utils import check_random_state, as_float_array
from scipy.linalg import svd
from sklearn.kernel_approximation import Nystroem
from scipy.sparse.linalg import aslinearoperator, eigsh

import matplotlib.pyplot as plt
import time

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import variable
from torch.utils.data import dataloader
import cv2

from nystrom_master_code import LazyNystrom_T

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

# Load MNIST dataset
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
examples = enumerate(test_loader)
batch_idx, (example_data, example_targets) = next(examples)

#%%timeit
length = 50000
num_sampling = 100

x = example_data.permute(1,0,2,3)
X_i = LazyTensor(x.view(1,1000,784))

# Instatiate & fit on lazy tensor version
LN_test = LazyNystrom_T(num_sampling, kernel='rbf', random_state=0).fit(X_i)
X_new_i = LN_test.transform(X_i)

#NUMPY on MNIST
#%%timeit
# Instatiate & fit Nystroem for comparison
x_ = x.reshape(1, 1000,784)

X_i = LazyTensor(x_)

# Instatiate & fit Nystroem for comparison
sk_N = Nystroem(kernel='rbf', n_components=num_sampling, random_state=0).fit(x[0].numpy())
x_new = sk_N.transform(x[0].numpy())      # (length, num_sampling) array

# Instatiate & fit on lazy tensor version
LN_test = LazyNystrom_T(num_sampling, kernel='rbf', random_state=0).fit(X_i)
X_new_i = LN_test.transform(X_i)          # (1,length,num_sampling) lazy tensor

# Print the L2 error
err = np.linalg.norm(x_new - X_new_i.sum(dim=0).numpy()) / x_new.size
print(f'Error when compared to sklearn = {err}')
#>>Error when compared to sklearn = 0.00022449403762817382