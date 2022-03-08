#include <stdlib.h>
#include <stdint.h>
#include<omp.h>
#include <time.h>

/* Simple helper array based math functions implemented using openMP */

void init_random(const unsigned int INIT) {
    srand(INIT);
}

void filter_min_max(float* arr, const int SIZE, const float min, const float max, const int NUMT) {
    omp_set_num_threads(NUMT);
    int i;
    #pragma omp parallel for default(none) shared(SIZE, min, max, arr)
    for (i=0; i < SIZE; i++) {
        if (arr[i] < min)
            arr[i] = min;
        else if (arr[i] > max)
            arr[i] = max;
    }
}

void randomize_elements(float* arr, const int SIZE, const float min, const float max) {
    for (int i=0; i < SIZE; i++) {
        arr[i] = min + (float)rand() / RAND_MAX * (max - min);
    }
}

void copy_arr(float *cpy, const float *src, const int SIZE, const int NUMT) {
    omp_set_num_threads(NUMT);
    int i;
    #pragma omp parallel for default(none) shared(SIZE, cpy, src)
    for (i=0; i < SIZE; i++) {
        cpy[i] = src[i];
    }
}