# slidePy

[![pypi](https://img.shields.io/pypi/v/slidepy.svg)](https://pypi.python.org/pypi/slidepy)
[![image](https://img.shields.io/badge/dynamic/json?query=info.requires_python&label=python&url=https%3A%2F%2Fpypi.org%2Fpypi%2Fslidepy%2Fjson )](https://pypi.python.org/pypi/slidepy)
[![License: LGPL v3](https://img.shields.io/badge/License-LGPL_v3-yellow.svg)](https://www.gnu.org/licenses/lgpl-3.0)

***slidepy*** is a fast multi-threaded python library for performing 3D landslide simulation and modeling using [openMP](https://www.openmp.org/), [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) and [numpy](https://numpy.org/) objects.

 - Github repository: https://github.com/asenogles/slidepy
  - PyPI: https://pypi.org/project/slidepy

## Motivation

***slidepy*** was developed to quickly perform landslide simulations, enabling self-supervised learning for landslide analyses. ***slidepy*** provides a cython wrapper for optimized [openMP](https://www.openmp.org/) *c* code with additional [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) support for SSE & AVX instruction sets using [Intrinsics](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html). Data objects are handled by [numpy](https://numpy.org/) allowing for straightforward memory management. Currently only conservation of mass modeling has been fully implemented, however this is open to expansion in the future. All code is still in development and thus it is recommended to test fully before use.

## Installation

***slidepy*** has currently been tested on Linux and Microsoft windows operating systems. You will need python>=3.6 installed. If running ***slidepy*** on non-x86 architecture, you will need to modify the SIMD code in order to compile. It is recommended to install ***slidepy*** within a virtual environment.
### Install using pip

To install ***slidepy*** from PyPI using pip:

```console
pip install slidepy
```
### Install from source

To build ***slidepy*** from source, download this repository and run:
```console
python3 setup.py build_ext --inplace
```
**Note**: You will need to have the required build dependencies installed.

## Example

```python
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
    du = np.gradient(u, axis=1) / cell_size
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

# Time Conservation of mass simulation using open-MP for numt-threads
num_threads = [1,2,4,8]
for numt in num_threads:
    time = timeit.timeit(lambda: sp.com_mp(dem.raster, u, v, ssem.raster, dem.XDIM, 1, numt), number=NTESTS)
    print(f'MP COM averaged {time/NTESTS:.3f} seconds for {numt} threads')

# Time Conservation of mass simulation using open-MP and SIMD for numt-threads
for numt in num_threads:
    time = timeit.timeit(lambda: sp.com_sse(dem.raster, u, v, ssem.raster, dem.XDIM, 1, numt), number=NTESTS)
    print(f'SSE COM averaged {time/NTESTS:.3f} seconds for {numt} threads')
```
Example output:
```console
python COM took 162.632 seconds
numpy COM took 7.911 seconds
MP COM averaged 0.095 seconds for 1 threads
MP COM averaged 0.092 seconds for 2 threads
MP COM averaged 0.091 seconds for 4 threads
MP COM averaged 0.088 seconds for 8 threads
SSE COM averaged 0.048 seconds for 1 threads
SSE COM averaged 0.033 seconds for 2 threads
SSE COM averaged 0.028 seconds for 4 threads
SSE COM averaged 0.030 seconds for 8 threads
```