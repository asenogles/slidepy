#include <omp.h>
#include <immintrin.h>
#include "com.h"

#define SSE_WIDTH 4
#define AVX_WIDTH 8

void COMPUTE_DEPTH_MP(float *h, const float *dem, const float *slip, const int SIZE, const int NUMT) {
    omp_set_num_threads(NUMT);
    int i;
    #pragma omp parallel for default(none) shared(SIZE, h, dem, slip)
    for (i=0; i < SIZE; i++) {
        h[i] = dem[i] - slip[i];
    }
}

void COM_MP(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT) {
    /* Model a landslide using conservation of mass with openMP */

    omp_set_num_threads(NUMT);
    for (int t=0; t < epochs; t++)
    {
        int r;
        // compute new dem elevation
        #pragma omp parallel for default(none) shared(NROWS, NCOLS, row_stride, col_stride, dl, h, dem, u, v, slip)
        for (r = 1; r < NROWS; r++)
        {
            int i = r * row_stride;
            int ip = (r+1) * row_stride;
            int im = (r-1) * row_stride;
            for (int c = 1; c < NCOLS; c++)
            {
                int j = c * col_stride;
                int jp = (c+1) * col_stride;
                int jm = (c-1) * col_stride;
                dem[i+j] -= ((h[i+j] * (u[i+jm] - u[i+jp]) / dl) + (u[i+j] * (h[i+jm] - h[i+jp]) / dl)) + ((h[i+j] * (v[ip+j] - v[im+j]) / dl) + (v[i+j] * (h[ip+j] - h[im+j]) / dl));
            }
        }
        // update landslide depth
        #pragma omp parallel for default(none) shared(NROWS, NCOLS, row_stride, col_stride, h, dem, slip)
        for (r = 1; r < NROWS; r++)
        {
            int i = r * row_stride;
            for (int c = 1; c < NCOLS; c++)
            {
                int j = c * col_stride;
                h[i+j] = dem[i+j] - slip[i+j];
            }
        }
    }
}

void COM_SSE(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT) {
    /* Model a landslide using conservation of mass with openMP and SIMD */

    omp_set_num_threads(NUMT);

    const int col_limit = (NCOLS / SSE_WIDTH) * SSE_WIDTH;
    for (int t=0; t < epochs; t++)
    {
        // add pointer arrays to register
        register float *pdem = dem;
        register float *ph = h;
        register float const *pu = u;
        register float const *pv = v;
        register float const *pslip = slip;

        // load dl into all elements of dst
        const __m128 dl_ps = _mm_load_ps1(&dl);
        int r; // compiler likes r declared prior to omp for
        #pragma omp parallel for default(none) shared(NROWS, NCOLS, row_stride, col_stride, col_limit, dl, dl_ps, dem, h, u, v, pdem, ph, pu, pv)
        for (r = 0; r < NROWS; r++)
        {
            // get all row idex vals
            int i = r * row_stride;
            int ip = (r+1) * row_stride;
            int im = (r-1) * row_stride;
            for (int c = 0; c < col_limit; c+=SSE_WIDTH)
            {
                // get all column index vals
                int j = c * col_stride;
                int jp = (c+1) * col_stride;
                int jm = (c-1) * col_stride;

                // load vals into SSE
                __m128 h_ps = _mm_loadu_ps(&ph[i+j]);
                __m128 u_ps = _mm_loadu_ps(&pu[i+j]);
                __m128 v_ps = _mm_loadu_ps(&pv[i+j]);
                __m128 dem_ps = _mm_loadu_ps(&pdem[i+j]);
                __m128 du_ps = _mm_sub_ps(_mm_loadu_ps(&pu[i+jm]), _mm_loadu_ps(&pu[i+jp]));
                __m128 dv_ps = _mm_sub_ps(_mm_loadu_ps(&pv[ip+j]), _mm_loadu_ps(&pv[im+j]));
                __m128 dh_u_ps = _mm_sub_ps(_mm_loadu_ps(&ph[i+jm]), _mm_loadu_ps(&ph[i+jp]));
                __m128 dh_v_ps = _mm_sub_ps(_mm_loadu_ps(&ph[ip+j]), _mm_loadu_ps(&ph[im+j]));

                // u comp
                __m128 dz_hdu = _mm_div_ps(_mm_mul_ps(h_ps, du_ps), dl_ps);
                __m128 dz_udh = _mm_div_ps(_mm_mul_ps(u_ps, dh_u_ps), dl_ps);

                // v comp
                __m128 dz_hdv = _mm_div_ps(_mm_mul_ps(h_ps, dv_ps), dl_ps);
                __m128 dz_vdh = _mm_div_ps(_mm_mul_ps(v_ps, dh_v_ps), dl_ps);

                __m128 dz_ps = _mm_add_ps(_mm_add_ps(dz_hdu, dz_udh), _mm_add_ps(dz_hdv, dz_vdh));

                // update dem
                dem_ps = _mm_sub_ps(dem_ps, dz_ps);
                // write SSE to array
                _mm_storeu_ps(&dem[i+j], dem_ps);

            }
            for (int c = col_limit; c < NCOLS; c++)
            {
                int j = c * col_stride;
                int jp = (c+1) * col_stride;
                int jm = (c-1) * col_stride;
                dem[i+j] -= ((h[i+j] * (u[i+jm] - u[i+jp]) / dl) + (u[i+j] * (h[i+jm] - h[i+jp]) / dl)) + ((h[i+j] * (v[ip+j] - v[im+j]) / dl) + (v[i+j] * (h[ip+j] - h[im+j]) / dl));
            }
        }
        // Now update depth (h)
        #pragma omp parallel for default(none) shared(NROWS, NCOLS, row_stride, col_stride, col_limit, dem, h, slip, pdem, pslip)
        for (r = 0; r < NROWS; r++)
        {
            // get all row idex vals
            int i = r * row_stride;
            for (int c = 0; c < col_limit; c+=SSE_WIDTH)
            {
                // get all column index vals
                int j = c * col_stride;
                __m128 h_ps = _mm_sub_ps(_mm_loadu_ps(&pdem[i+j]), _mm_loadu_ps(&pslip[i+j])); // update h
                _mm_storeu_ps(&h[i+j], h_ps);
            }
            for (int c = col_limit; c < NCOLS; c++)
            {
                int j = c * col_stride;
                h[i+j] = dem[i+j] - slip[i+j];
            }
        }
    }
}

void COM_AVX(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT) {
    /* Model a landslide using conservation of mass with openMP and AVX */

    const int col_limit = (NCOLS / AVX_WIDTH) * AVX_WIDTH;
    omp_set_num_threads(NUMT);
    for (int t=0; t < epochs; t++)
    {
        // add pointer arrays to register
        register float *ph = h;
        register float *pdem = dem;
        register float const *pu = u;
        register float const *pv = v;
        register float const *pslip = slip;

        // load dl into all elements of dst
        float dls[AVX_WIDTH] = {dl, dl, dl, dl, dl, dl, dl, dl};
        const __m256 dl_ps = _mm256_loadu_ps(&dls[0]);
        int r; // compiler likes r declared prior to omp for
        #pragma omp parallel for default(none) shared(NROWS, NCOLS, row_stride, col_stride, col_limit, dl, dl_ps, h, dem, u, v, ph, pdem, pu, pv)
        for (r = 0; r < NROWS; r++)
        {
            // get all row idex vals
            int i = r * row_stride;
            int ip = (r+1) * row_stride;
            int im = (r-1) * row_stride;
            for (int c = 0; c < col_limit; c+=AVX_WIDTH)
            {
                // get all column index vals
                int j = c * col_stride;
                int jp = (c+1) * col_stride;
                int jm = (c-1) * col_stride;

                // load vals into SSE
                __m256 h_ps = _mm256_loadu_ps(&ph[i+j]);
                __m256 u_ps = _mm256_loadu_ps(&pu[i+j]);
                __m256 v_ps = _mm256_loadu_ps(&pv[i+j]);
                __m256 dem_ps = _mm256_loadu_ps(&pdem[i+j]);
                __m256 du_ps = _mm256_sub_ps(_mm256_loadu_ps(&pu[i+jm]), _mm256_loadu_ps(&pu[i+jp]));
                __m256 dv_ps = _mm256_sub_ps(_mm256_loadu_ps(&pv[ip+j]), _mm256_loadu_ps(&pv[im+j]));
                __m256 dh_u_ps = _mm256_sub_ps(_mm256_loadu_ps(&ph[i+jm]), _mm256_loadu_ps(&ph[i+jp]));
                __m256 dh_v_ps = _mm256_sub_ps(_mm256_loadu_ps(&ph[ip+j]), _mm256_loadu_ps(&ph[im+j]));

                // u comp
                __m256 dz_hdu = _mm256_div_ps(_mm256_mul_ps(h_ps, du_ps), dl_ps);
                __m256 dz_udh = _mm256_div_ps(_mm256_mul_ps(u_ps, dh_u_ps), dl_ps);

                // v comp
                __m256 dz_hdv = _mm256_div_ps(_mm256_mul_ps(h_ps, dv_ps), dl_ps);
                __m256 dz_vdh = _mm256_div_ps(_mm256_mul_ps(v_ps, dh_v_ps), dl_ps);

                __m256 dz_ps = _mm256_add_ps(_mm256_add_ps(dz_hdu, dz_udh), _mm256_add_ps(dz_hdv, dz_vdh));
                dem_ps = _mm256_sub_ps(dem_ps, dz_ps); //update dem
                _mm256_storeu_ps(&dem[i+j], dem_ps);

            }
            for (int c = col_limit; c < NCOLS; c++)
            {
                int j = c * col_stride;
                int jp = (c+1) * col_stride;
                int jm = (c-1) * col_stride;
                dem[i+j] -= ((h[i+j] * (u[i+jm] - u[i+jp]) / dl) + (u[i+j] * (h[i+jm] - h[i+jp]) / dl)) + ((h[i+j] * (v[ip+j] - v[im+j]) / dl) + (v[i+j] * (h[ip+j] - h[im+j]) / dl));
            }
        }
        #pragma omp parallel for default(none) shared(NROWS, NCOLS, row_stride, col_stride, col_limit, h, dem, slip, pdem, pslip)
        for (r = 0; r < NROWS; r++)
        {
            // get all row idex vals
            int i = r * row_stride;
            for (int c = 0; c < col_limit; c+=AVX_WIDTH)
            {
                // get all column index vals
                int j = c * col_stride;
                __m256 h_ps = _mm256_sub_ps(_mm256_loadu_ps(&pdem[i+j]), _mm256_loadu_ps(&pslip[i+j])); // update h
                _mm256_storeu_ps(&h[i+j], h_ps);
            }
            for (int c = col_limit; c < NCOLS; c++)
            {
                int j = c * col_stride;
                h[i+j] = dem[i+j] - slip[i+j];
            }
        }
    }
}

void COM_SSE_MULTI(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT) {
    /* 
     * Model a landslide using conservation of mass with openMP and SIMD,
     * Saving each epoch into it's own band of the DEM,
     * Note: This will not update the boundary pixels of the dem, 
     * they will be left at 0 or whatever they were initialized to.
     */

    omp_set_num_threads(NUMT);

    const int col_limit = ((NCOLS-2) / SSE_WIDTH) * SSE_WIDTH;
    for (int t=0; t < epochs; t++)
    {
        // get time index vals
        const int e = t * NROWS * NCOLS;
        int em = (t-1) * NROWS * NCOLS;

        // don't negative index
        if (t == 0) {
            em = 0;
        }
        // add pointer arrays to register
        register float *pdem = dem;
        register float *ph = h;
        register float const *pu = u;
        register float const *pv = v;
        register float const *pslip = slip;

        // load dl into all elements of dst
        const __m128 dl_ps = _mm_load_ps1(&dl);
        int r; // compiler likes r declared prior to omp for
        #pragma omp parallel for default(none) shared(NROWS, NCOLS, row_stride, col_stride, col_limit, dl, dl_ps, e, dem, h, u, v, em, pdem, ph, pu, pv)
        for (r = 0; r < (NROWS - 2); r++)
        {
            // get all row idex vals
            int i = r * row_stride;
            int ip = (r+1) * row_stride;
            int im = (r-1) * row_stride;
            for (int c = 0; c < col_limit; c+=SSE_WIDTH)
            {
                // get all column index vals
                int j = c * col_stride;
                int jp = (c+1) * col_stride;
                int jm = (c-1) * col_stride;

                // load vals into SSE
                __m128 h_ps = _mm_loadu_ps(&ph[i+j]);
                __m128 u_ps = _mm_loadu_ps(&pu[i+j]);
                __m128 v_ps = _mm_loadu_ps(&pv[i+j]);
                __m128 dem_ps = _mm_loadu_ps(&pdem[em+i+j]);
                __m128 du_ps = _mm_sub_ps(_mm_loadu_ps(&pu[i+jm]), _mm_loadu_ps(&pu[i+jp]));
                __m128 dv_ps = _mm_sub_ps(_mm_loadu_ps(&pv[ip+j]), _mm_loadu_ps(&pv[im+j]));
                __m128 dh_u_ps = _mm_sub_ps(_mm_loadu_ps(&ph[i+jm]), _mm_loadu_ps(&ph[i+jp]));
                __m128 dh_v_ps = _mm_sub_ps(_mm_loadu_ps(&ph[ip+j]), _mm_loadu_ps(&ph[im+j]));

                // u comp
                __m128 dz_hdu = _mm_div_ps(_mm_mul_ps(h_ps, du_ps), dl_ps);
                __m128 dz_udh = _mm_div_ps(_mm_mul_ps(u_ps, dh_u_ps), dl_ps);

                // v comp
                __m128 dz_hdv = _mm_div_ps(_mm_mul_ps(h_ps, dv_ps), dl_ps);
                __m128 dz_vdh = _mm_div_ps(_mm_mul_ps(v_ps, dh_v_ps), dl_ps);

                __m128 dz_ps = _mm_add_ps(_mm_add_ps(dz_hdu, dz_udh), _mm_add_ps(dz_hdv, dz_vdh));

                //update dem
                dem_ps = _mm_sub_ps(dem_ps, dz_ps);
                // write SSE to array
                _mm_storeu_ps(&dem[e+i+j], dem_ps);

            }
            for (int c = col_limit; c < (NCOLS - 2); c++)
            {
                int j = c * col_stride;
                int jp = (c+1) * col_stride;
                int jm = (c-1) * col_stride;
                dem[e+i+j] = dem[em+i+j] - ((h[i+j] * (u[i+jm] - u[i+jp]) / dl) + (u[i+j] * (h[i+jm] - h[i+jp]) / dl)) + ((h[i+j] * (v[ip+j] - v[im+j]) / dl) + (v[i+j] * (h[ip+j] - h[im+j]) / dl));
            }
        }

        // Now update h
        #pragma omp parallel for default(none) shared(NROWS, NCOLS, row_stride, col_stride, col_limit, e, dem, h, slip, pdem, pslip)
        for (r = 0; r < (NROWS-2); r++)
        {
            // get all row idex vals
            int i = r * row_stride;
            for (int c = 0; c < col_limit; c+=SSE_WIDTH)
            {
                // get all column index vals
                int j = c * col_stride;
                __m128 h_ps = _mm_sub_ps(_mm_loadu_ps(&pdem[e+i+j]), _mm_loadu_ps(&pslip[i+j])); // update h
                _mm_storeu_ps(&h[i+j], h_ps);
            }
            for (int c = col_limit; c < (NCOLS-2); c++)
            {
                int j = c * col_stride;
                h[i+j] = dem[e+i+j] - slip[i+j];
            }
        }
    }
}


