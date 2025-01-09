# Modified from GSV-Cities, acknowledgements and thanks: https://github.com/amaralibey/gsv-cities
#
#


import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class PittsburghDataset(Dataset):
    def __init__(self, dataset_root: str, src_root_dir: str, which_ds: str = 'pitts30k_test', input_transform = None):
        """
        Dataset class for the Pittsburgh dataset.
        ---
        dataset_root: str,
        src_root_dir: str,
        which_ds: str ; choices=[pitts30k_val, pitts30k_test, pitts250k_test],
        input_transform
        """
        
        assert which_ds.lower() in ['pitts30k_val', 'pitts30k_test', 'pitts250k_test']

        self.dataset_root = dataset_root
        
        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/' + f'pitts/{which_ds}_dbImages.npy'))
        
        # query images names
        self.qImages = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/' + f'pitts/{which_ds}_qImages.npy'))
        
        # ground truth
        self.ground_truth = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/' + f'pitts/{which_ds}_gt.npy'), allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset_root, self.images[index]))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)