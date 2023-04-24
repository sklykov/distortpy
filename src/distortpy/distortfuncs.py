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
import time
from pathlib import Path

# %% Local imports
# absolute import as the main module for various types of module calls
if __name__ == "__main__" or __name__ == "__mp_main__" or __name__ == Path(__file__).stem:
    from gen_samples.specific_samples import grid_points, vertical_fringes
else:
    from .gen_samples.specific_samples import grid_points, vertical_fringes

# %% Samples generation
# points = generate_grid_points(); plt.figure(); plt.imshow(points); plt.tight_layout()
fringes_image = vertical_fringes(); plt.figure(); plt.imshow(fringes_image); plt.tight_layout()


# %% Distortion
def radial_distort(image, k1: float) -> np.ndarray:
    pass


def radial_distort2(image, k1: float, k2: float) -> np.ndarray:
    pass


# %% Testing as the script
if __name__ == "__main__":
    pass
