"""
A collection of functions for landslide simulation and modelling
"""

import numpy as np
from scipy.ndimage import zoom
from scipy.interpolate import LSQBivariateSpline as LSQBS

from ._ext import *

def init_random(init):
    """Initialize the random number generator

    Args:
        init (uint): seed used to initialize the random number generator
    """
    cy_init_random(init)

def augment_mask(mask, scale):
    """Augment a binary mask representing a landslide boundary

    Args:
        mask (2D array): 2D numpy binary grid to augment
        scale (float): scale to scale binary grid by
        method (str): 'edge', 'centroid'

    Returns:
        (2D array): Augmented version of input
    """

    # TODO: Implement in c

    rows = mask.shape[0]
    cols = mask.shape[1]
    mask_extents = np.where(mask==1)

    # get extents of the mask boundary
    y0 = mask_extents[0].min()      # top pixel of mask extent
    y1 = mask_extents[0].max()      # bottom pixel of mask extent
    x0 = mask_extents[1].min()      # left-most pixel of mask extent
    x1 = mask_extents[1].max()      # right-most pixel of mask extent
    cx = x0 + ((x1 - x0) // 2)      # centroid of mask (x)
    cy = y0 + ((y1 - y0) // 2)      # centroid of mask (y)
    #h = y1 - y0                     # height of mask
    #w = x1 - x0                     # width of mask

    # compute zoom raster
    z_mask = zoom(mask, scale, order=0) # Resize raster
    z_mask[z_mask < 1] = 0            # Change inbetween values to 0
    
    z_rows = z_mask.shape[0]
    z_cols = z_mask.shape[1]
    zoom_extents = np.where(z_mask==1)

    # get extents of the new mask boundary
    #z_y0 = zoom_extents[0].min()      # top pixel of new mask extent
    #z_y1 = zoom_extents[0].max()      # bottom pixel of new mask extent
    #z_x0 = zoom_extents[1].min()      # left-most pixel of new mask extent
    #z_x1 = zoom_extents[1].max()      # right-most pixel of new mask extent
    #z_cx = (z_x1 - z_x0) // 2         # centroid of mask (x)
    #z_cy = (z_y1 - z_y0) // 2         # centroid of mask (y)
    #z_h = z_y1 - z_y0                 # height of mask
    #z_w = z_x1 - z_x0                 # width of mask

    # create padding
    unit_row = y0 + (rows-y1)
    unit_col = x0 + (cols-x1)
    pad_row = rows - z_rows
    pad_col = cols - z_cols

    if x0 == 0:
        # boundary at x edge
        px0 = int(round((x0 / unit_col) * pad_col))
        px1 = int(round(((cols-x1) / unit_col) * pad_col))
    else:
        px0 = int(round((cx / cols) * pad_col))
        px1 = int(round(((cols-cx) / cols) * pad_col))

    if y0 == 0:
        # boundary at x edge
        py0 = int(round((y0 / unit_row) * pad_row))
        py1 = int(round(((rows-y1) / unit_row) * pad_row))
    else:
        py0 = int(round((cy / rows) * pad_row))
        py1 = int(round(((rows-cy) / rows) * pad_row))
        
    # return raster to original size (pad or crop)
    if scale < 1:
        z_mask = np.pad(z_mask, ((py0,py1), (px0,px1)), constant_values=0)
    elif scale > 1:
        z_mask = np.delete(z_mask, np.arange(z_cols//2 + pad_col//2, z_cols//2 - pad_col//2), 1)
        z_mask = np.delete(z_mask, np.arange(z_rows//2 + pad_row//2, z_rows//2 - pad_row//2), 0)

    return z_mask

def interp_slip(dem, ctrl, mask, pct_c=0.05, num_k=20, min_offset=5):
    """Interpolate a slip surface from a set of control points 
    and landslide boundary

    Args:
        dem (2D array): Grid of surface elevation values
        ctrl (2D array): Grid of slip surface control pts with slip surface elevation values
        mask (2D array): Gird of landside extents, 1 = landside, 0=no landslide
        pct_c (int, optional): fraction of ctrl pts to use in Spline, Defaults to 0.2.
        num_k (int, optional): number of knot pts in Spline. Defaults to 20.
        min_offset (int, optional): num pixel offset from side for spline fitting. Defaults to 5.

    Raises:
        ValueError: if dem, ctrl & mask do not match

    Returns:
        2D array: grid of interpolated slip surface elevations
    """

    # TODO: Implement in c

    if dem.shape == ctrl.shape == mask.shape:
        rows = dem.shape[0]
        cols = dem.shape[1]
    else:
        raise ValueError(
            f'raster grid dimensions do not match:\n'
            f'DEM: {dem.shape}\n'
            f'CTRL: {ctrl.shape}\n'
            f'MASK: {mask.shape}\n'
        )

    # Create ctrl coordinate arrays
    x, y = np.arange(cols), np.arange(rows) # create grid dim indexes
    X, Y = np.meshgrid(x, y)                # create grid of X/Y values
    Xc = X[~np.isnan(ctrl)]                 # Remove X where ctrl == nan (control pts)
    Yc = Y[~np.isnan(ctrl)]                 # Remove Y where ctrl == nan (control pts)
    Zc = ctrl[~np.isnan(ctrl)]              # Remove Z where ctrl == nan (control pts)

    # reduce ctrl coordinate array
    idx = np.round(np.linspace(0, len(Zc) - 1, int(len(Zc) * pct_c))).astype(int)
    Xc = Xc[idx]
    Yc = Yc[idx]
    Zc = Zc[idx]

    # Create knot coordinates ~ pts where slope can change
    xo = int((cols % num_k) // 2)
    if xo < min_offset:
        xo = min_offset
    yo = int((rows % num_k) // 2)
    if yo < min_offset:
        yo = min_offset
    xk = np.linspace(xo, cols-xo, num_k, dtype=np.float32)    # x knot pts
    yk = np.linspace(yo, rows-yo, num_k, dtype=np.float32)    # Y knot pts
    
    # Create spline func
    spline = LSQBS(Yc,Xc,Zc,yk,xk)          # Create lsq spline func
    ssem = spline(y, x).astype(np.float32)  # interpolate slip surface

    # Set points outside of landslide area back to the DEM value
    return np.where(mask==1, ssem, dem)

def generate_slip(mask, dem, ssem, slip_area):
    """Generates slip surface over mask based off large slip surface

    Args:
        mask (2D array): Binary array where 1==landslide, 0==non-landslide
        dem (2D array): Digital elevation model of terrain
        ssem (2D array): Digital elevation model of slip surface
        slip_area (float): Ratio of slip area to use as control

    Returns:
        2D array: Digital elevation model of new slip surface
    """

    # TODO: Implement in c
 
    inner_mask = augment_mask(mask, slip_area)
    slip_ctrl = np.where(inner_mask==1, ssem, np.where(mask==0, dem, np.nan))
    return interp_slip(dem, slip_ctrl, mask, num_k=10)

def augment_slip(dem, ssem, scale, numt=1):
    """augment the slip surface using a depth scaler

    Args:
        dem (2D array): Digital elevation model of terrain
        ssem (2D array): Digital elevation model of slip surface to augment
        scale (float): Scale used to augment depth
        numt (int): Number of threads to use for openMP operations

    Returns:
        2D array: Digital elevation model of augmented slip surface
    """

    # TODO: Implement in c

    aug_ssem = cy_augment_slip(dem, ssem, scale, numt)
    return aug_ssem

def generate_vel(vel, min, max, nrow_vals=3, ncol_vals=3, numt=1):
    """ Generates velocity raster in place using random spline process between min & max.
    Wrapper function for cy_generate_vel

    Args:
        vel (3D array): empty array to store velocity grid (NUM VELS x NROWS X NCOLS)
        min (float): minimum possible velocity value
        max (float): maximum possible velocity value
        nrow_vals (int, optional): number of ctrl pts to use in row-axis, defaults to 3
        ncol_vals (int, optional): number of ctrl pts to use in col-axis, defaults to 3
        numt (int, optional): number of threads to use for openMP processes, defaults to 1
        
    """

    # TODO: Implement in c

    # make sure vel array is 3D
    assert(len(vel.shape) == 3)
    # make sure vel array is float
    if vel.dtype != np.float32:
        vel = np.empty_like(vel, dtype=np.float32)
    cy_generate_vel(vel, min, max, nrow_vals, ncol_vals, numt)

def calc_depth_mp(dem, ssem, numt=1):
    """ Calculate the depth across a lanslide

    Args:
        dem (2D array): Digital Elevation Model of landslide terrain
        ssem (2D array): Slip Surface Elevation Model of landslide
        numt (int, optional): number of threads to use for openMP operations, defaults to 1
    
    Returns:
        (2D array): landslide depth grid

    """
    assert (dem.shape == ssem.shape and len(dem.shape) == 2)
    rows = dem.shape[0]
    cols = dem.shape[0]
    h = np.empty((rows, cols), dtype=np.float32, order='C')
    cy_calc_depth(h, dem, ssem, numt)
    return h

def com_mp(dem, u, v, ssem, cell_size, epochs=1, numt=1):
    """ Compute a conservation of mass simulation across the landslide using openMP

    Args:
        dem (2D array): Digital Elevation Model of landslide terrain
        u (2D array): x-component of landslide velocity (units/epoch)
        v (2D array): y-component of landslide velocity (units/epoch)
        ssem (2D array): Slip Surface Elevation Model of landslide
        cell_size (float): cellsize of the rasters
        epochs (int, optional): number of simulation epochs to compute, defaults to 1
        numt (int, optional): number of threads to use for openMP operations, defaults to 1
    
    Returns:
        (2D array): DEM of the post simulation landslide

    """
    assert (dem.shape == u.shape == v.shape == ssem.shape and len(dem.shape) == 2)
    dem_cpy = dem.copy()
    cy_com_mp(dem_cpy, u, v, ssem, cell_size*2., epochs, numt)
    return dem_cpy

def com_sse(dem, u, v, ssem, cell_size, epochs=1, numt=1):
    """ Compute a conservation of mass simulation across the landslide using openMP and SIMD

    Args:
        dem (2D array): Digital Elevation Model of landslide terrain
        u (2D array): x-component of landslide velocity (units/epoch)
        v (2D array): y-component of landslide velocity (units/epoch)
        ssem (2D array): Slip Surface Elevation Model of landslide
        cell_size (float): cellsize of the rasters
        epochs (int, optional): number of simulation epochs to compute, defaults to 1
        numt (int, optional): number of threads to use for openMP operations, defaults to 1
    
    Returns:
        (2D array): DEM of the post simulation landslide

    """
    dem_cpy = dem.copy()
    assert (dem.shape == u.shape == v.shape == ssem.shape and len(dem.shape) == 2)
    cy_com_sse(dem_cpy, u, v, ssem, cell_size*2, epochs, numt)
    return dem_cpy

def com_avx(dem, u, v, ssem, cell_size, epochs=1, numt=1):
    """ Compute a conservation of mass simulation across the landslide using openMP and AVX

    Args:
        dem (2D array): Digital Elevation Model of landslide terrain
        u (2D array): x-component of landslide velocity (units/epoch)
        v (2D array): y-component of landslide velocity (units/epoch)
        ssem (2D array): Slip Surface Elevation Model of landslide
        cell_size (float): cellsize of the rasters
        epochs (int, optional): number of simulation epochs to compute, defaults to 1
        numt (int, optional): number of threads to use for openMP operations, defaults to 1
    
    Returns:
        (2D array): DEM of the post simulation landslide

    """
    assert (dem.shape == u.shape == v.shape == ssem.shape and len(dem.shape) == 2)
    dem_cpy = dem.copy()
    cy_com_avx(dem_cpy, u, v, ssem, cell_size*2, epochs, numt)
    return dem_cpy

def com_sse_multi(dem, u, v, ssem, cell_size, epochs=1, numt=1):
    """ Compute a conservation of mass simulation across the landslide using openMP and SIMD

    Args:
        dem (2D array): Digital Elevation Model of landslide terrain
        u (2D array): x-component of landslide velocity (units/epoch)
        v (2D array): y-component of landslide velocity (units/epoch)
        ssem (2D array): Slip Surface Elevation Model of landslide
        cell_size (float): cellsize of the rasters
        epochs (int, optional): number of simulation epochs to compute, defaults to 1
        numt (int, optional): number of threads to use for openMP operations, defaults to 1
    
    Returns:
        (3D array): (NxNROWSxNCOLS) DEM of the post simulation landslide, each band contains the DEM at sim epoch N

    """
    assert (dem.shape == u.shape == v.shape == ssem.shape and len(dem.shape) == 2)
    
    rows = dem.shape[0]
    cols = dem.shape[1]

    # copy dem to multi-band version
    dems = np.empty((epochs, rows, cols), dtype=np.float32)
    cy_copy_to_first(dems, dem, numt)

    # calculate depth
    h = np.empty((rows, cols), dtype=np.float32, order='C')
    cy_calc_depth(h, dem, ssem, numt)

    # compute sim using conservation of mass
    cy_com_sse_multi(dems, h, u, v, ssem, cell_size*2, numt)
    return dems
