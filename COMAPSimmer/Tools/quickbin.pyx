import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, abs, sin, cos, tan
from cpython cimport array
import array


def cube2TOD(long[:] pixels, long nsbs, long nchans, double[:,:,:] cube, double[:,:,:] tod):

    cdef int i,j,k  
    cdef int npix = cube.shape[0]
    cdef int nsamples = pixels.size
    for i in range(nsamples):
        for j in range(nsbs):
            for k in range(nchans):
                if (pixels[i] > 0) & (pixels[i] < npix):
                    tod[j,k,i] = cube[pixels[i],j,k]
