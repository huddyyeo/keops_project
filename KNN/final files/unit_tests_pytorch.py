import os.path
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + "..") * 2)
print(os.path.dirname(os.path.abspath(__file__)) + (os.path.sep + "..") * 2)

import unittest

'''
keops file structure:
keops
    pykeops
        torch
            ivf_torch
            utils (to update)
        numpy
            ivf_np
            utils (to update)
        common
            ivf_generic
        test
            unit_test (to add the below function)
        benchmarks
            plot_benchmark_IVF
            
'''


class PytorchUnitTestCase(unittest.TestCase):
    ############################################################
    def test_IVF(self):
        ############################################################
#         from pykeops.torch import ivf_torch
        import torch
        from KNN.common.ivf_torch import ivf

        torch.manual_seed(0)
        N, D, K, k = 10**4, 3, 50, 5
        
        # test over a - number of clusters to search over
        for a in range(1,11):
            # Generate random datapoints x, y
            x = 0.7 * torch.randn(N, D) + 0.3
            y = 0.7 * torch.randn(N, D) + 0.3

            # Ground truth K nearest neighbours
            truth = torch.argsort(((y.unsqueeze(1)-x.unsqueeze(0))**2).sum(-1),dim=1)
            truth = truth[:,:k]

            # IVF K nearest neighbours
            IVF = ivf()
            IVF.fit(x,a=a)
            ivf_fit = IVF.kneighbors(y)

            # Calculate accuracies of ivf.fit and ivf.reduce
            accuracy = 0
            for i in range(k):
                accuracy += torch.sum(ivf_fit == truth).float()/N
                truth = torch.roll(truth, 1, -1) # Create a rolling window (index positions may not match)
            # Record accuracies
            accuracy = float(accuracy/k)
            
            print(a,accuracy)
            self.assertTrue(accuracy >= 0.8, f'Failed at {a}, {accuracy}')
                

if __name__ == "__main__":
    """
    run tests
    """
    unittest.main()