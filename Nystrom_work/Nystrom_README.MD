# README file for Nystrom code


## Nystrom approximation
The Nystrom method produces a low-rank approximation of an arbitrary kernel. 

## Basic Nystrom
A basic implementation of the Nystrom algorithm using Numpy/Scipy is under the class `Nystrom_N`, and equivalent code written in Pytorch is under the class ` LazyNystrom_T`. (The Pytorch version right now is wrapped by LazyTensor interface which should be removed.) Both versions work with a linear or Gaussian kernel.

## Nystrom optimized 

 `Nystrom_NK`, for Nystrom Numpy Keops, implements the Nystrom approximation in Numpy/Scipy but with upgrades coming from Pykeops. Currently only written only for the Gaussian kernel.
 
 Example of usage:
```
from nystrom_master_code import Nystrom_NK

NK = Nystrom_NK(n_components=100 ,kernel='rbf', gamma=1.).fit(X) # creates an instance and fits the data        
X_new_i = NK.transform(X)  # transform data to approximated features
K_approx = NK.K_approx(X)  # obtain approximated kernel
```
