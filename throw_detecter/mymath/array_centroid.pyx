import numpy as np
cimport numpy as np
cimport cython

ctypedef np.uint8_t DTYPE_t

def get_rgb_array_cnt_cy(np.ndarray[DTYPE_t, ndim=3] frame):
    cdef int cnt[2]
    cdef int x
    cdef int y
    cdef int nonzero

    cnt[:] = [0, 0]
    nonzero = 0
    for y in range(len(frame)):
        for x in range(len(frame[y])):
            if frame[y, x, 0] + frame[y, x, 1] + frame[y, x, 2] != 0:
                cnt[0] += y
                cnt[1] += x
                nonzero += 1
    if nonzero != 0:
        cnt[0] /= nonzero
        cnt[1] /= nonzero
    return cnt, nonzero