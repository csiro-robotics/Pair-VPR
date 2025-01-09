# Files reusued and modified from https://github.com/gmberton/EigenPlaces, licensed under the MIT License.
#
#


import math
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from glob import glob
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True



def get_angle(focal_point, obs_point):
    obs_e, obs_n = float(obs_point[0]), float(obs_point[1])
    focal_e, focal_n = focal_point
    side1 = focal_e - obs_e
    side2 = focal_n - obs_n
    angle = - math.atan2(side1, side2) / math.pi * 90 * 2
    return angle

def anglediffcalc(ang1, ang2):
    diff = (ang1 - ang2) % 360
    if diff > 180:
        diff = -(360 - diff)
    return diff

def get_eigen_things(utm_coords):
    mu = utm_coords.mean(0)
    norm_data = utm_coords - mu
    eigenvectors, eigenvalues, v = np.linalg.svd(norm_data.T, full_matrices=False)
    return eigenvectors, eigenvalues, mu

def rotate_2d_vector(vector, angle):
    assert vector.shape == (2,)
    theta = np.deg2rad(angle)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
    rotated_point = np.dot(rot_mat, vector)
    return rotated_point

def get_focal_point(utm_coords, meters_from_center=20, angle=0):
    """Return the focal point from a set of utm coords"""
    B, D = utm_coords.shape
    assert D == 2
    eigenvectors, eigenvalues, mu = get_eigen_things(utm_coords)

    direction = rotate_2d_vector(eigenvectors[1], angle)
    focal_point = mu + direction * meters_from_center
    return focal_point

def focal_precompute(utm_coords):
    """Return the focal point from a set of utm coords"""
    B, D = utm_coords.shape
    assert D == 2
    eigenvectors, eigenvalues, mu = get_eigen_things(utm_coords)
    return eigenvectors, mu

def read_images_paths(dataset_folder, get_abs_path=False):
    # acknowledgements: https://github.com/gmberton/EigenPlaces/tree/main
    """Find images within 'dataset_folder' and return their relative paths as a list.
    If there is a file 'dataset_folder'_images_paths.txt, read paths from such file.
    Otherwise, use glob(). Keeping the paths in the file speeds up computation,
    because using glob over large folders can be slow.
    
    Parameters
    ----------
    dataset_folder : str, folder containing JPEG images
    get_abs_path : bool, if True return absolute paths, otherwise remove
        dataset_folder from each path
    
    Returns
    -------
    images_paths : list[str], paths of JPEG images within dataset_folder
    """
    
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Folder {dataset_folder} does not exist")
    
    file_with_paths = dataset_folder + "_images_paths.txt"
    if os.path.exists(file_with_paths):
        logging.debug(f"Reading paths of images within {dataset_folder} from {file_with_paths}")
        with open(file_with_paths, "r") as file:
            images_paths = file.read().splitlines()
        images_paths = [dataset_folder + "/" + path for path in images_paths]
        # Sanity check that paths within the file exist
        if not os.path.exists(images_paths[0]):
            raise FileNotFoundError(f"Image with path {images_paths[0]} "
                                    f"does not exist within {dataset_folder}. It is likely "
                                    f"that the content of {file_with_paths} is wrong.")
    else:
        logging.debug(f"Searching images in {dataset_folder} with glob()")
        images_paths = sorted(glob(f"{dataset_folder}/**/*.jpg", recursive=True))
        if len(images_paths) == 0:
            raise FileNotFoundError(f"Directory {dataset_folder} does not contain any JPEG images")
    
    if not get_abs_path:  # Remove dataset_folder from the path
        images_paths = [p[len(dataset_folder) + 1:] for p in images_paths]
    
    return images_paths

