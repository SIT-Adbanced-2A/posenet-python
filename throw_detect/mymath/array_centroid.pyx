import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint8_t DTYPE_t

def get_centroid_rgb_cy(np.ndarray[DTYPE_t, ndim=3] frame, DTYPE_t threshold):
    cdef int rcnt[2]
    cdef int gcnt[2]
    cdef int bcnt[2]
    cdef int nonzero[3]
    cdef int x
    cdef int y

    rcnt[:] = [0, 0]
    gcnt[:] = [0, 0]
    bcnt[:] = [0, 0]
    nonzero[:] = [0, 0, 0]
    x = 0
    y = 0
    
    for y in range(len(frame)):
        for x in range(len(frame[y])):
            if frame[y, x, 0] > threshold:
                bcnt[0] += x
                bcnt[1] += y
                nonzero[0] += 1
            if frame[y, x, 1] > threshold:
                gcnt[0] += x
                gcnt[1] += y
                nonzero[1] += 1
            if frame[y, x, 2] > threshold:
                rcnt[0] += x
                rcnt[1] += y
                nonzero[2] += 1

    if nonzero[0] != 0:
        rcnt[0] /= nonzero[0]
        rcnt[1] /= nonzero[0]
    if nonzero[1] != 0:
        gcnt[0] /= nonzero[1]
        gcnt[1] /= nonzero[1]
    if nonzero[2] != 0:
        bcnt[0] /= nonzero[2]
        bcnt[1] /= nonzero[2]
    return rcnt, gcnt, bcnt, nonzero