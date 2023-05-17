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
# from skimage.filters.rank import median, mean
# from skimage.morphology import square, disk, diamond
# from skimage.util import img_as_uint
# from scipy.fft import fft2, fftshift
# from matplotlib.colors import LogNorm
from pathlib import Path
import warnings
import time

# %% Local imports
# absolute import as the main module for various types of module calls
if __name__ == "__main__" or __name__ == "__mp_main__" or __name__ == Path(__file__).stem:
    from gen_samples.specific_samples import vertical_fringes, horizontal_fringes
else:
    from .gen_samples.specific_samples import vertical_fringes, horizontal_fringes

# %% Sample generation and performance evaluation parameters
k_barrel = -0.5; k_pincushion = 0.5; perf_iterations = 10


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

    # Below calculation performed depending on the provided coefficient, applying specific rules for barrel / pincushion types

    # Vectorized calculation of transformed pixel coordinates of the initial image - independent of the distortion type
    # Using sparce mesh-grids for potential saving time of calculation
    ii_undist, jj_undist = np.meshgrid(np.arange(start=0, stop=h, step=1), np.arange(start=0, stop=w, step=1), sparse=True)
    i_coord_sq_d = np.arange(start=0, stop=h, step=1); j_coord_sq_d = np.arange(start=0, stop=w, step=1)
    i_coord_sq_d = np.power(i_coord_sq_d - i_center, 2); j_coord_sq_d = np.power(j_coord_sq_d - j_center, 2)
    ii_coord_sq_d, jj_coord_sq_d = np.meshgrid(i_coord_sq_d, j_coord_sq_d, sparse=True)
    radii_mesh_sq_d = np.power(np.sqrt(ii_coord_sq_d + jj_coord_sq_d)*norm_radius, 2)

    # Calculate distorted coordinates, note that transposing happen on mesh-grids with coordinates
    ii_dist = np.int16(np.round(i_center + ((ii_undist - i_center)*(1.0 + k1*radii_mesh_sq_d)), 0)).T
    jj_dist = np.int16(np.round(j_center + ((jj_undist - j_center)*(1.0 + k1*radii_mesh_sq_d)), 0)).T

    # Barrel distortion
    if k1 < 0.0:
        # Initialize empty image for distortion representation
        distorted_image = np.zeros((h, w), dtype=image.dtype)

        # Re-assign original image pixels to the distorted ones to form the distorted images
        distorted_image[ii_dist, jj_dist] = image

        # Crop out not assigned coordinates
        if crop_dist_img:
            i_top = np.min(ii_dist[:, 0]); i_bottom = np.max(ii_dist[:, 0])
            j_left = np.min(jj_dist[0, :]); j_right = np.max(jj_dist[0, :])
            distorted_image = distorted_image[i_top:i_bottom, j_left:j_right]

    # Pincushion distortion
    else:
        # Distorted image calculation avoiding 2 nested loops as for direct transform
        # Avoiding 2 nested for loops by the artificial elongation of the transformed image and cropping it to the initial size
        # Define elongated image size parameters
        ii_min = ii_dist[0, 0]; ii_max = ii_dist[h-1, w-1]
        jj_min = jj_dist[0, 0]; jj_max = jj_dist[h-1, w-1]
        distorted_image = np.zeros((ii_max + abs(ii_min) + 2, jj_max + abs(jj_min) + 2))  # elongated image for storing converted pixels
        ii_dist += abs(ii_min); jj_dist += abs(jj_min)  # making distorted coordinates non-negative
        distorted_image[ii_dist, jj_dist] = image  # conversion pixel values
        h2, w2 = distorted_image.shape  # get the new image size

        # Cropped the elongated image back to the initial image sizes
        if crop_dist_img:
            distorted_image = distorted_image[(h2-h)//2: h2 - (h2-h)//2, (w2-w)//2: w2 - (w2-w)//2]  # cropping out the central part related to input image
            h2, w2 = distorted_image.shape  # update image size
            # Applying additional crop if images have different sizes
            if h2 != h and w2 != w:
                distorted_image = distorted_image[0:h, :]
                distorted_image = distorted_image[:, 0:w]
            elif h2 != h and w2 == w:
                distorted_image = distorted_image[0:h, :]
            elif w2 != w and h2 == h:
                distorted_image = distorted_image[:, 0:w]
        else:
            h, w = h2, w2

        # Initialize empty image for distortion representation
        # distorted_image = np.zeros((h, w), dtype=image.dtype)

        # Below - direct implementation for the reference. Also, the code for comparing faster implementation
        # Filter out the pixel coordinates laying outside the original image size - used for the nested 2 loops
        # Slow implementation, involving 2 for loops - as the reference for getting the result
        # ii_dist_log_mask = np.logical_and(0 <= ii_dist, ii_dist < h)
        # jj_dist_log_mask = np.logical_and(0 <= jj_dist, jj_dist < w)

        # # Direct implementation, slow because of the 2 nested for loops and each pixel checking and transformation
        # for i in range(h):
        #     for j in range(w):
        #         if ii_dist_log_mask[i, j]:
        #             if jj_dist_log_mask[i, j]:
        #                 distorted_image[ii_dist[i, j], jj_dist[i, j]] = image[i, j]

        # Faster implementation and comparison with the exact direct implementation
        # ii_min = ii_dist[0, 0]; ii_max = ii_dist[h-1, w-1]
        # jj_min = jj_dist[0, 0]; jj_max = jj_dist[h-1, w-1]
        # distorted_image2 = np.zeros((ii_max + abs(ii_min) + 2, jj_max + abs(jj_min) + 2))
        # ii_dist += abs(ii_min); jj_dist += abs(jj_min)
        # distorted_image2[ii_dist, jj_dist] = image
        # h2, w2 = distorted_image2.shape
        # distorted_image2 = distorted_image2[(h2-h)//2: h2 - (h2-h)//2, (w2-w)//2: w2 - (w2-w)//2]
        # h2, w2 = distorted_image2.shape
        # # Applying additional crop if images have different sizes
        # if h2 != h and w2 != w:
        #     distorted_image2 = distorted_image2[0:h, :]
        #     distorted_image2 = distorted_image2[:, 0:w]
        # elif h2 != h and w2 == w:
        #     distorted_image2 = distorted_image2[0:h, :]
        # elif w2 != w and h2 == h:
        #     distorted_image2 = distorted_image2[:, 0:w]
        # distorted_image = distorted_image - distorted_image2

        # Interpolation of pixel values not involved in the distortion transform (non-transformed pixels), makes sence only for a cropped image
        if crop_dist_img:
            interpolation_sum_coeffs = np.asarray([1.0, 0.5, 1/3, 0.25, 0.2, 1/6, 1/7, 0.125])
            zero_ii, zero_jj = np.nonzero(distorted_image[1:h-1, 1:w-1] < 1E-9)  # controversially, returns indices of zero pixels
            if zero_ii.shape[0] > 0:
                for zero_index in range(zero_ii.shape[0]):
                    i, j = zero_ii[zero_index] + 1, zero_jj[zero_index] + 1  # because borders of the initial image removed
                    # Define mask coordinates - for calculation interpolation sum
                    i_top = i-1; i_bottom = i+1; j_left = j-1; j_right = j+1
                    # Calculate sum of pixels inside 3x3 mask using the numpy
                    zero_mask_ii, _ = np.nonzero(distorted_image[i_top:i_bottom+1, j_left:j_right+1] > 1E-9)
                    distorted_image[i, j] = (interpolation_sum_coeffs[zero_mask_ii.shape[0]-1]
                                             * np.sum(distorted_image[i_top:i_bottom+1, j_left:j_right+1]))

        # Rough global smooth - not efficiently removing artifacts
        # distorted_image = img_as_uint(distorted_image)
        # distorted_image = mean(distorted_image, disk(2))
        # distorted_image = gaussian(distorted_image, 2)
        # distorted_image = median(distorted_image, disk(8))

    # Smooth the distorted image to remove a bit of the artifacts
    if smooth_output:
        distorted_image = gaussian(distorted_image, smooth_sigma)

    return distorted_image


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
    t_mean_ms = 0.0
    for i in range(perf_iterations):
        t1 = time.perf_counter()
        radial_distort(image, k)
        t2 = time.perf_counter()
        t_mean_ms += 1000.0*(t2-t1)
    t_mean_ms = int(np.round(t_mean_ms / perf_iterations, 0))
    print(f"Distortion with {k} of an image {image.shape[1]}x{image.shape[0]} takes ms:", t_mean_ms)


# %% Testing as the script
if __name__ == "__main__":
    plt.close("all")  # close all opened plots on external windows

    # Testing flags
    use_vertical_fringes = True; make_crop = True; time_performance = True

    # Generation of sample - interferometric fringes
    if use_vertical_fringes:
        fringes_image = vertical_fringes()
    else:
        fringes_image = horizontal_fringes()

    # Testing the performance of implementations - before plotting, maybe last influences the measures
    if time_performance:
        test_performance(fringes_image, k_barrel)
        test_performance(fringes_image, k_pincushion)

    # Plotting the sample and the result of distortion
    plt.figure(); plt.imshow(fringes_image); plt.tight_layout()
    # distorted_fringes = radial_distort(fringes_image, k_barrel, crop_dist_img=make_crop)
    # plt.figure(); plt.imshow(distorted_fringes); plt.tight_layout()
    distorted_fringes2 = radial_distort(fringes_image, k_pincushion)
    plt.figure(); plt.imshow(distorted_fringes2); plt.tight_layout()
    # distorted_fringes2 = radial_distort(fringes_image, k_pincushion, crop_dist_img=False, smooth_output=False)
    # plt.figure(); plt.imshow(distorted_fringes2); plt.tight_layout()
