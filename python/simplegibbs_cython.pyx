'''
Gibbs sampler for function:

f(x,y) = x x^2 \exp(-xy^2 - y^2 + 2y - 4x)

using conditional distributions:

x|y \sim Gamma(3, y^2 +4)
y|x \sim Normal(\frac{1}{1+x}, \frac{1}{2(1+x)})
'''
cimport cython
import numpy as np
from numpy cimport *

cdef extern from "math.h":
    double sqrt(double)
  
cdef extern from "gsl/gsl_rng.h":
    ctypedef struct gsl_rng_type
    ctypedef struct gsl_rng

    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T) nogil
  
cdef extern from "gsl/gsl_randist.h":
    double gamma "gsl_ran_gamma"(gsl_rng * r,double,double)
    double gaussian "gsl_ran_gaussian"(gsl_rng * r,double)
  
cdef gsl_rng *r = gsl_rng_alloc(gsl_rng_mt19937)

@cython.wraparound(False)
@cython.boundscheck(False)
def gibbs(int N=20000,int thin=500):
    cdef: 
        double x=0
        double y=0
        int i, j
        ndarray[float64_t, ndim=2] samples

    samples = np.empty((N,thin))
    for i from 0 <= i < N:
        for j from 0 <= j < thin:
            x = gamma(r,3,1.0/(y*y+4))
            y = gaussian(r,1.0/sqrt(x+1))
        samples[i,0] = x
        samples[i,1] = y
    return samples
