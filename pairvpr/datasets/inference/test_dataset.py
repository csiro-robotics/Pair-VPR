# Modified from EigenPlaces, acknowledgements and thanks: https://github.com/gmberton/EigenPlaces (MIT License)
# 
# 


import os
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as T
from sklearn.neighbors import NearestNeighbors
import logging
from glob import glob
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


def read_images_paths(dataset_folder: str, get_abs_path: bool=False):
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


class TestDataset(data.Dataset):
    def __init__(self, dataset_folder: str, src_root_dir: str, database_folder: str="database",
                 queries_folder: str="queries", positive_dist_threshold: int=25,
                 input_transform=None):
        """Dataset with images from database and queries, used for validation and test.
        Parameters
        ----------
        dataset_folder : str, should contain the path to the val or test set,
            which contains the folders {database_folder} and {queries_folder}.
        src_root_dir : str, rootdir of this codebase. Currently unused.
        database_folder : str, name of folder with the database.
        queries_folder : str, name of folder with the queries.
        positive_dist_threshold : int, distance in meters for a prediction to
            be considered a positive.
        """
        super().__init__()
        
        self.database_folder = os.path.join(dataset_folder, database_folder)
        self.queries_folder = os.path.join(dataset_folder, queries_folder)
        self.database_paths = read_images_paths(self.database_folder, get_abs_path=True)
        self.queries_paths = read_images_paths(self.queries_folder, get_abs_path=True)
        
        self.dataset_name = os.path.basename(dataset_folder)
        
        #### Read paths and UTM coordinates for all images.
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.database_paths]).astype(float)
        self.queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in self.queries_paths]).astype(float)
        
        # Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
        knn = NearestNeighbors(n_jobs=-1)
        knn.fit(self.database_utms)
        self.ground_truth = knn.radius_neighbors(
            self.queries_utms, radius=positive_dist_threshold, return_distance=False
        )
        
        self.images_paths = self.database_paths + self.queries_paths
        self.images = self.images_paths
        
        self.num_references = len(self.database_paths)
        self.num_queries = len(self.queries_paths)

        if input_transform is not None:
            self.base_transform = input_transform
        else:
            self.base_transform = T.Compose([
                T.Resize((322, 322), interpolation=T.InterpolationMode.BILINEAR),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        
    def __getitem__(self, index):
        image_path = self.images_paths[index]
        pil_img = Image.open(image_path)
        normalized_img = self.base_transform(pil_img)
        return normalized_img, index
    
    def __len__(self):
        return len(self.images_paths)
    
    def __repr__(self):
        return f"< {self.dataset_name} - #q: {self.num_queries}; #db: {self.num_references} >"
    
    def get_positives(self):
        return self.ground_truth
