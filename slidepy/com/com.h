#pragma once

void COMPUTE_DEPTH_MP(float *h, const float *dem, const float *slip, const int SIZE, const int NUMT);

void COM_MP(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT);

void COM_SSE(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT);

void COM_AVX(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT);

void COM_SSE_MULTI(float *dem, float *h, const float *u, const float *v, const float *slip, const float dl, const int epochs, const int NROWS, const int NCOLS, const int row_stride, const int col_stride, const int NUMT);