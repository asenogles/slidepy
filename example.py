import timeit
import numpy as np
import fasterraster as fr
import slidepy as sp
from pathlib import Path

NTESTS = 10

# Load grid files
dir = Path('./test_data/')
dem = fr.read(dir / 'dem.bil')
mask = fr.read(dir / 'mask.bil')
ssem = fr.read(dir / 'ssem.bil')
vel = fr.Flo(dir / 'vel.flo')

# prep velocity grids
fr.multiplyFloMask(vel.raster, mask.raster) # 0 velocity values outslide of landslide extent
u, v = fr.flo_to_u_v(vel.raster)            # split velocity grid into u & v components

# regular python implementation of com function
def py_com(dem, u, v, ssem, cell_size, epochs):

    dem_cpy = dem.copy()

    dl = 2. * cell_size
    rows = dem_cpy.shape[0] - 2
    cols = dem_cpy.shape[1] - 2

    # calculate depth
    h = dem_cpy - ssem

    for i in range(epochs):
        for i in range(1, rows):
            for j in range(1, cols):
                dem_cpy[i,j] -= ((h[i,j] * (u[i,j-1] - u[i,j+1]) / dl) + (u[i,j] * (h[i,j-1] - h[i,j+1]) / dl)) + ((h[i,j] * (v[i+1,j] - v[i-1,j]) / dl) + (v[i,j] * (h[i+1,j] - h[i-1,j]) / dl))
        for i in range(1, rows):
            for j in range(1, cols):
                h[i,j] = dem_cpy[i,j] - ssem[i,j]
    return dem_cpy

# regular numpy implementation of com function
def np_com(dem, u, v, ssem, cell_size, epochs):
    
    dem_cpy = dem.copy()

    # calculate depth
    h = dem_cpy - ssem

    # calculate vel gradients
    du = -1 * np.gradient(u, axis=1) / cell_size
    dv = np.gradient(v, axis=1) / cell_size

    for i in range(epochs):
        # calculate depth gradient
        dh_v, dh_u = np.gradient(h)
        dh_u = -1 * dh_u / cell_size
        dh_v = dh_v / cell_size

        # calculate dz
        dz_u = (h * du) + (u * dh_u)
        dz_v = (h * dv) + (v * dh_v)
        dz = dz_u + dz_v

        # update dem & depth
        dem_cpy = dem_cpy - dz
        h = dem_cpy - ssem
    
    return dem_cpy

# Time Conservation of mass simulation using regular python
time = timeit.timeit(lambda: py_com(dem.raster, u, v, ssem.raster, dem.XDIM, 1), number=1)
print(f'python COM took {time:.3f} seconds')

# Time Conservation of mass simulation using numpy
time = timeit.timeit(lambda: np_com(dem.raster, u, v, ssem.raster, dem.XDIM, 1), number=1)
print(f'numpy COM took {time:.3f} seconds')

# Time Conservation of mass simulation using open-MP and SIMD for numt-threads
num_threads = [1,2,4,8]
for numt in num_threads:
    time = timeit.timeit(lambda: sp.com_mp(dem.raster, u, v, ssem.raster, dem.XDIM, 1, numt), number=NTESTS)
    print(f'MP COM averaged {time/NTESTS:.3f} seconds')

for numt in num_threads:
    time = timeit.timeit(lambda: sp.com_sse(dem.raster, u, v, ssem.raster, dem.XDIM, 1, numt), number=NTESTS)
    print(f'SSE COM averaged {time/NTESTS:.3f} seconds')
