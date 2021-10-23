import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint8_t DTYPE_t

def get_centroid(np.ndarray[DTYPE_t, ndim=3] frame):
    cdef int rcx = 0
    cdef int rcy = 0
    cdef int gcx = 0
    cdef int gcy = 0
    cdef int bcx = 0
    cdef int bcy = 0
    cdef int nonzero_r = 0
    cdef int nonzero_g = 0
    cdef int nonzero_b = 0
    cdef int x
    cdef int y

    for y in range(len(frame)):
        for x in range(len(frame[y])):
            if frame[y, x, 0] > 125:
                bcx += x
                bcy += y
                nonzero_b += 1
            if frame[y, x, 1] > 125:
                gcx += x
                gcy += y
                nonzero_g += 1
            if frame[y, x, 2] > 125:
                rcx += x
                rcy += y
                nonzero_r += 1
    if nonzero_r != 0:
        rcx /= nonzero_r
        rcy /= nonzero_r
    if nonzero_g != 0:
        gcx /= nonzero_g
        gcy /= nonzero_g
    if nonzero_b != 0:
        bcx /= nonzero_b
        bcy /= nonzero_b
    return rcx, rcy, gcx, gcy, bcx, bcy, nonzero_r, nonzero_g, nonzero_b