from ._common import *
import numpy as np

class gpuarray:
    def __init__(self, shape, dtype=np.float32):
        assert dtype == np.float32, f"Only supporting float32."
        self.length = prod(shape)
        self.shape = shape
        d = self.device_ptr = float_ptr(c.c_float(0.0))
        status = dll.hipMalloc(c.pointer(d), self.length*4)
        self._numpy = np.empty(shape, dtype=np.float32)
        self.host_dirty = False

    @property
    def numpy(self):
        return self._numpy

    def to_dev(self):
        host_ptr = self._numpy.ctypes.data_as(float_ptr)
        status = dll.hipMemcpy(self.device_ptr, host_ptr, 4*self.length, hipMemcpyHostToDevice)

    def to_host(self):
        host_ptr = self._numpy.ctypes.data_as(float_ptr)
        status = dll.hipMemcpy(host_ptr, self.device_ptr, 4 * self.length, hipMemcpyDeviceToHost)

    def sync_host(self):
        if self.host_dirty:
            self.to_host()


    def __mul__(self, rhs):
        if isinstance(rhs, float) or isinstance(rhs, int):
            call('rocblas_sscal', handle, self.length, float_ptr(c.c_float(rhs)), self.device_ptr, 1)
            self.host_dirty = True
        else:
            raise NotImplementedError

        return self

    def __repr__(self):
        return 'hi'

    def __str__(self):
        self.sync_host()
        return f'rocm_array shape={self.shape} [{self.numpy}]'
