import numpy as np
cimport numpy as cnp
cimport cython
from cython import boundscheck, wraparound

ctypedef cnp.uint8_t DTYPE_t

def create_mask_cy(cnp.ndarray[DTYPE_t, ndim=3] src, DTYPE_t threshold):
    cdef int x
    cdef int y
    cdef cnp.ndarray[DTYPE_t, ndim=1] one_array
    cdef cnp.ndarray[DTYPE_t, ndim=1] zero_array

    one_array = np.empty(3, dtype=np.uint8)
    zero_array = np.empty(3, dtype=np.uint8)
    
    one_array[:] = 1
    zero_array[:] = 0

    with boundscheck(False), wraparound(False):
        for y in range(len(src)):
            for x in range(len(src[y])):
                if src[y, x, 0] + src[y, x, 1] + src[y, x, 2] >= threshold:
                    src[y, x] = one_array
                else:
                    src[y, x] = zero_array
    return src