import ctypes as c
print("Loading required DLLs")
dll = c.CDLL('librocblas.so')

from .gpu_array import gpuarray
