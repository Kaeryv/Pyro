from math import prod
import ctypes as c
from . import dll

#/opt/rocm/include/hip/driver_types.h
hipMemcpyHostToDevice=1
hipMemcpyDeviceToHost=2
float_ptr = c.POINTER(c.c_float)


def call(function, *args):
    assert hasattr(dll, function), f"Function {function} does not exist."
    ret = getattr(dll, function)(*args)
    assert ret == 0, f"{function} signal error."
    return ret

def rocblas_handle():
    handle = c.create_string_buffer(b"", 4096)
    dll.rocblas_create_handle(handle)
    return handle


handle = rocblas_handle()
