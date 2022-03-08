#pragma once

void init_random(const unsigned int INIT);

void filter_min_max(float* arr, const int SIZE, const float min, const float max, const int NUMT);

void randomize_elements(float* arr, const int SIZE, const float min, const float max);

void copy_arr(float *cpy, const float *src, const int SIZE, const int NUMT);