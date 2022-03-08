#include <omp.h>


void AUGMENT_SLIP_MP(float *aug, const float *DEM, const float *SLIP, const float SCALE, const int SIZE, const int NUMT) {
    omp_set_num_threads(NUMT);
    int i;
    #pragma omp parallel for default(none) shared(SIZE, aug, DEM, SLIP, SCALE)
    for (i=0; i < SIZE; i++) {
        aug[i] = DEM[i] - (SCALE * (DEM[i] - SLIP[i]));
    } 
}