import unittest
import numpy as np
import pyro as po
from numpy.testing import assert_allclose

class SSCALTest(unittest.TestCase):
    def test_vs_numpy(self):
        # Pyro
        N = int(1e4)
        array = po.gpuarray((N,))
        array.numpy[:] = np.arange(N)
        array.to_dev()
        array *= 2
        array.to_host()

        # Numpy
        base = np.arange(N)
        base *= 2
        assert_allclose(array.numpy, base)


