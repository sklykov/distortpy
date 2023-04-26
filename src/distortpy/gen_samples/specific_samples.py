# -*- coding: utf-8 -*-
"""
Functions for sample images generation.

@author: Sergei Klykov, @year: 2023 \n
@licence: MIT \n
"""
# %% Global imports
import numpy as np


# %% Grid points generation
def grid_points(width: int = 500, height: int = 480, vert_step: int = 5,
                hor_step: int = 6) -> np.ndarray:
    img = None
    if vert_step <= 2*width and hor_step <= 2*height and vert_step >= 2 and hor_step >= 2:
        img = np.zeros((height, width), dtype="float")
        for i in range(hor_step, height, hor_step):
            for j in range(vert_step, width, vert_step):
                img[i, j] = 1.0
    else:
        raise ValueError(f"Some of provided parameters {width, height, vert_step, hor_step} inconsistent")
    return img


def vertical_fringes(width: int = 520, height: int = 480, step: int = 15, fringe_width: int = 12) -> np.ndarray:
    img_fringes = None
    if step <= 2*width and step >= 2 and width > 4 and height > 4:
        img_fringes = np.zeros((height, width), dtype="float")
        for j_fringe_center in range(step//4, width, step):
            # img_fringes[i-fringe_step_vert//4:i+fringe_step_vert//4, :] = 1.0  # straight, not blurred fringes
            # below - modelling the fringes as gaussian blurred fringes
            sigma_fringe = fringe_width*0.25  # sigma_fringe = fringe_width / 4.0
            j = j_fringe_center - step // 2
            if j < 0:
                j = 0
            elif j >= width:
                j = width-1
            while j < j_fringe_center + (step // 2):
                if j < width:
                    fringe_profile = np.exp(-np.power(j-j_fringe_center, 2)/np.power(sigma_fringe, 2))
                    img_fringes[:, j] = fringe_profile
                    j += 1
                else:
                    break
    else:
        raise ValueError(f"Some of provided parameters {width, height, step, fringe_width} inconsistent")
    return img_fringes


def horizontal_fringes(width: int = 520, height: int = 480, step: int = 16, fringe_width: int = 12) -> np.ndarray:
    img_fringes = None
    if step <= 2*width and step >= 2 and width > 4 and height > 4:
        img_fringes = np.zeros((height, width), dtype="float")
        for i_fringe_center in range(step//4, height, step):
            # img_fringes[i-fringe_step_vert//4:i+fringe_step_vert//4, :] = 1.0  # straight, not blurred fringes
            # below - modelling the fringes as gaussian blurred fringes
            sigma_fringe = fringe_width*0.25  # sigma_fringe = fringe_width / 4.0
            i = i_fringe_center - step // 2
            if i < 0:
                i = 0
            elif i >= height:
                i = height-1
            while i < i_fringe_center + (step // 2):
                if i < height:
                    fringe_profile = np.exp(-np.power(i-i_fringe_center, 2)/np.power(sigma_fringe, 2))
                    img_fringes[i, :] = fringe_profile
                    i += 1
                else:
                    break
    else:
        raise ValueError(f"Some of provided parameters {width, height, step, fringe_width} inconsistent")
    return img_fringes
