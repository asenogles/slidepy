# distutils: language = c
# distutils: sources = augment/augment.c, com/com.c, mp_math/mp_math.c

import numpy as np
cimport cython
cimport numpy as np
from cython.parallel cimport prange
from scipy.ndimage import zoom

##############################################################
################### openMP Math Operations ###################
##############################################################
cdef extern from "mp_math/mp_math.h":
    void init_random(const unsigned int INIT)
    void filter_min_max(float* arr, const int SIZE, const float min, const float max, const int NUMT)
    void randomize_elements(float* arr, const int SIZE, const float min, const float max)
    void copy_arr(float *cpy, const float *src, const int SIZE, const int NUMT)

def cy_copy_to_first(float[:,:,::1] cpy, const float[:,::1] src, const int numt=1):
    cdef int size = src.size
    copy_arr(&cpy[0,0,0], &src[0,0], size, numt)

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_init_random(const unsigned int init):
    init_random(init)

@cython.boundscheck(False)
@cython.wraparound(False)
def cy_generate_vel(vel, const float min, const float max, const int nrow_vals, const int ncol_vals, const int numt=1):

    cdef int numv = int(vel.shape[0])
    cdef int rows = int(vel.shape[1])
    cdef int cols = int(vel.shape[2])

    zoom_factor = (rows / nrow_vals, cols / ncol_vals)

    # create numpy objects to store data
    coarse_vel = np.empty((numv, nrow_vals, ncol_vals), dtype=np.float32)
    cdef float[:,:,::1] coarse_ptr = coarse_vel
    cdef float[:,:,::1] vel_ptr = vel

    # fill the coarse grid
    randomize_elements(&coarse_ptr[0,0,0], numv * nrow_vals * ncol_vals, min, max)
    
    # iterpolate dense grid
    cdef Py_ssize_t i
    for i in prange(numv, nogil=True):
        with gil:
            vel[i] = zoom(coarse_vel[i], zoom_factor)

    # make sure all data is between min and max
    filter_min_max(&vel_ptr[0,0,0], numv * rows * cols, min, max, numt)

##############################################################
####################### AUG Operations #######################
##############################################################
cdef extern from "augment/augment.h":
    void AUGMENT_SLIP_MP(float *aug, const float *DEM, const float *SLIP, const float SCALE, const int SIZE, const int NUMT)

def cy_augment_slip(const float [:,::1] dem, const float[:,::1] ssem, const float scale, const int numt=1):

    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])
    cdef int size = rows * cols

    # create empty object to store vals
    aug_slip = np.empty((rows, cols), dtype=np.float32, order='C')
    cdef float [:,::1] aug_view = aug_slip

    AUGMENT_SLIP_MP(&aug_view[0,0], &dem[0,0], &ssem[0,0], scale, size, numt)
    return aug_slip

##############################################################
####################### COM Operations #######################
##############################################################
cdef extern from "com/com.h":
    void COMPUTE_DEPTH_MP(float *h, const float *dem, const float *slip, const int SIZE, const int NUMT);
    void COM_MP(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT)
    void COM_SSE(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT)
    void COM_AVX(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT)
    void COM_SSE_MULTI(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT)

def cy_calc_depth(float[:,::1] h, const float[:,::1] dem, const float[:,::1] slip, const int numt=1):
    cdef int size = h.size
    COMPUTE_DEPTH_MP(&h[0,0], &dem[0,0], &slip[0,0], size, numt)

def cy_com_mp(float[:,::1] dem, const float[:,::1] u, const float[:,::1] v, const float[:,::1] slip, const float dl, const int epochs, const int numt=1):
    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])
    cdef int row_stride = dem.strides[0] / dem.itemsize
    cdef int col_stride = dem.strides[1] / dem.itemsize

    # create & compute depth raster
    cdef float[:,::1] h = np.empty((rows, cols), dtype=np.float32, order='C')
    COMPUTE_DEPTH_MP(&h[0,0], &dem[0,0], &slip[0,0], rows*cols, numt)

    COM_MP(&dem[1,1], &h[1,1], &u[1,1], &v[1,1], &slip[1,1], dl, epochs, rows-2, cols-2, row_stride, col_stride, numt)

def cy_com_sse(float[:,::1] dem, const float[:,::1] u, const float[:,::1] v, const float[:,::1] slip, const float dl, const int epochs, const int numt=1):
    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])
    cdef int row_stride = dem.strides[0] / dem.itemsize
    cdef int col_stride = dem.strides[1] / dem.itemsize

    # create & compute depth raster
    cdef float[:,::1] h = np.empty((rows, cols), dtype=np.float32, order='C')
    COMPUTE_DEPTH_MP(&h[0,0], &dem[0,0], &slip[0,0], rows*cols, numt)

    COM_SSE(&dem[1,1], &h[1,1], &u[1,1], &v[1,1], &slip[1,1], dl, epochs, rows-2, cols-2, row_stride, col_stride, numt)

def cy_com_avx(float[:,::1] dem, const float[:,::1] u, const float[:,::1] v, const float[:,::1] slip, const float dl, const int epochs, const int numt=1):
    cdef int rows = int(dem.shape[0])
    cdef int cols = int(dem.shape[1])
    cdef int row_stride = dem.strides[0] / dem.itemsize
    cdef int col_stride = dem.strides[1] / dem.itemsize

    # create & compute depth raster
    cdef float[:,::1] h = np.empty((rows, cols), dtype=np.float32, order='C')
    COMPUTE_DEPTH_MP(&h[0,0], &dem[0,0], &slip[0,0], rows*cols, numt)

    COM_AVX(&dem[1,1], &h[1,1], &u[1,1], &v[1,1], &slip[1,1], dl, epochs, rows-2, cols-2, row_stride, col_stride, numt)

def cy_com_sse_multi(float[:,:,::1] dems, float[:,::1] h, const float[:,::1] u, const float[:,::1] v, const float[:,::1] slip, const float dl, const int numt=1):
    cdef int epochs = int(dems.shape[0])
    cdef int rows = int(dems.shape[1])
    cdef int cols = int(dems.shape[2])
    cdef int row_stride = h.strides[0] / h.itemsize
    cdef int col_stride = h.strides[1] / h.itemsize

    COM_SSE_MULTI(&dems[0,1,1], &h[0,0], &u[1,1], &v[1,1], &slip[1,1], dl, epochs, rows, cols, row_stride, col_stride, numt)
