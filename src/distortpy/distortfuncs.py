# -*- coding: utf-8 -*-
"""
Main script with the function definitions for distortion of images calculation.

@author: Sergei Klykov, @year: 2023 \n
@licence: MIT \n

"""
# %% Global imports
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian
# from skimage.util import img_as_uint
# from scipy.fft import fft2, fftshift
# from matplotlib.colors import LogNorm
from pathlib import Path
import warnings
import time

# %% Local imports
# absolute import as the main module for various types of module calls
if __name__ == "__main__" or __name__ == "__mp_main__" or __name__ == Path(__file__).stem:
    from gen_samples.specific_samples import grid_points, vertical_fringes, horizontal_fringes
else:
    from .gen_samples.specific_samples import grid_points, vertical_fringes, horizontal_fringes

# %% Samples generation parameters
use_vertical_fringes = True; k_barrel = -0.5; k_pincushion = 0.5
perf_iterations = 10


# %% Distortion
def radial_distort(image: np.ndarray, k1: float, smooth_output: bool = True,
                   smooth_sigma: float = 0.95, crop_dist_img: bool = True) -> np.ndarray:
    # Check input image to be appropriate class and shape
    if not (isinstance(image, np.ndarray) and len(image.shape) == 2):
        raise ValueError("Provided image isn't instance of numpy.ndarray class or isn't an 2D image")

    h, w = image.shape; radius = np.min((h, w)); norm_radius = 1.0 / radius
    # Check the input image sizes
    if h <= 9 or w <= 9:
        raise ValueError(f"Input image {w}x{h} is considered too small for distortion, provide at least 10x10 image")
    # Check the input distortion parameter k1
    if k1 == 0.0:
        return image
    elif abs(k1*radius*radius) < 0.5:
        _warn_message_ = "The distortion parameter is too small to make distortion conversion of an input image"
        warnings.warn(_warn_message_)
        return image

    # Calculate the center of an image
    i_center = h / 2; j_center = w / 2
    # Initialize empty image for distortion representation
    distorted_image = np.zeros((h, w), dtype=image.dtype)

    # Below calculation performed depending on the provided coefficient, applying specific rules for barel / pincushion types

    # Vectorized calculation of transformed pixel coordinates of the initial image - independent of the distortion type
    # Using sparce meshgrids for potential saving time of calculation
    ii_undist, jj_undist = np.meshgrid(np.arange(start=0, stop=h, step=1), np.arange(start=0, stop=w, step=1), sparse=True)
    i_coord_sq_d = np.arange(start=0, stop=h, step=1); j_coord_sq_d = np.arange(start=0, stop=w, step=1)
    i_coord_sq_d = np.power(i_coord_sq_d - i_center, 2); j_coord_sq_d = np.power(j_coord_sq_d - j_center, 2)
    ii_coord_sq_d, jj_coord_sq_d = np.meshgrid(i_coord_sq_d, j_coord_sq_d, sparse=True)
    radii_mesh_sq_d = np.power(np.sqrt(ii_coord_sq_d + jj_coord_sq_d)*norm_radius, 2)

    # Calculate distorted coordinates, note that transposing happen on meshgrids with coordinates
    ii_dist = np.int16(np.round(i_center + ((ii_undist - i_center)*(1.0 + k1*radii_mesh_sq_d)), 0)).T
    jj_dist = np.int16(np.round(j_center + ((jj_undist - j_center)*(1.0 + k1*radii_mesh_sq_d)), 0)).T

    # Barrel distortion
    if k1 < 0.0:
        # Re-assign original image pixels to the distorted ones to form the distorted images
        distorted_image[ii_dist, jj_dist] = image

        # Crop out not assigned coordinates
        if crop_dist_img:
            i_top = np.min(ii_dist[:, 0]); i_bottom = np.max(ii_dist[:, 0])
            j_left = np.min(jj_dist[0, :]); j_right = np.max(jj_dist[0, :])
            distorted_image = distorted_image[i_top:i_bottom, j_left:j_right]

        # Smooth the distorted image to remove a bit the artifacts
        if smooth_output:
            distorted_image = gaussian(distorted_image, smooth_sigma)

    # Pincushion distortion
    else:
        # Filter out the pixel coordinates laying outside the original image size
        # Slow implementation, involving 2 for loops - as the reference for getting the result, partially speeded up
        ii_dist_log_mask = np.logical_and(0 <= ii_dist, ii_dist < h)
        jj_dist_log_mask = np.logical_and(0 <= jj_dist, jj_dist < w)
        for i in range(h):
            for j in range(w):
                if ii_dist_log_mask[i, j]:
                    if jj_dist_log_mask[i, j]:
                        distorted_image[ii_dist[i, j], jj_dist[i, j]] = image[i, j]

    return distorted_image, ii_dist, jj_dist


def radial_distort2():
    """
    Holder for possible implementation of 2nd and 4th degree dependent on radius distortion.

    Returns
    -------
    None.

    """
    pass


def test_performance(image, k):
    """
    Measuring the average time that calculation of a distorted image takes.

    Print out the measurement result.

    Parameters
    ----------
    image : numpy.ndarray
        Image for tests.
    k : distortion coefficient
        Used for distortion calculation.

    Returns
    -------
    None.

    """
    h, w = image.shape
    t_mean_ms = 0.0
    for i in range(perf_iterations):
        t1 = time.perf_counter()
        radial_distort(image, k)
        t2 = time.perf_counter()
        t_mean_ms += 1000.0*(t2-t1)
    t_mean_ms = int(np.round(t_mean_ms / perf_iterations, 0))
    print(f"Distortion with {k} of an image {w}x{h} takes ms:", t_mean_ms)


# %% Testing as the script
if __name__ == "__main__":
    plt.close("all"); make_crop = True
    # points = generate_grid_points(); plt.figure(); plt.imshow(points); plt.tight_layout()
    if use_vertical_fringes:
        fringes_image = vertical_fringes()
    else:
        fringes_image = horizontal_fringes()
    plt.figure(); plt.imshow(fringes_image); plt.tight_layout()
    # distorted_fringes = radial_distort(fringes_image, k_barrel, crop_dist_img=make_crop)
    # plt.figure(); plt.imshow(distorted_fringes); plt.tight_layout()
    distorted_fringes2, ii_dist, jj_dist = radial_distort(fringes_image, k_pincushion)
    plt.figure(); plt.imshow(distorted_fringes2); plt.tight_layout()

    # Testing the performance of implementations
    test_performance(fringes_image, k_barrel)
    test_performance(fringes_image, k_pincushion)
