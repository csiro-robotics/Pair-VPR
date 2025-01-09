# Modified from GSV-Cities, acknowledgements and thanks: https://github.com/amaralibey/gsv-cities
#
#


import os
from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class MapillaryValDataset(Dataset):
    def __init__(self, dataset_root: str, src_root_dir: str, input_transform = None):
        """
        Dataset class for the Mapillary Validation set.
        ---
        dataset_root: str,
        src_root_dir: str,
        input_transform
        """
        
        self.dataset_root = dataset_root

        self.input_transform = input_transform
        
        # reference image names
        self.dbImages = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/msls_val/msls_val_dbImages.npy'))
        
        # query image names.
        self.qImages = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/msls_val/msls_val_qImages.npy'))
        
        # index of query images
        self.qIdx = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/msls_val/msls_val_qIdx.npy'))
        
        # ground truth (correspondence between each query and its matches)
        self.pIdx = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/msls_val/msls_val_pIdx.npy'), allow_pickle=True)
        self.ground_truth = self.pIdx
        
        # concatenate reference images then query images so we only have one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))
        
        self.num_references = len(self.dbImages)

        self.num_queries = len(self.qImages[self.qIdx])
    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.dataset_root, self.images[index]))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)
    
    # used for submissions to challenge servers
    def save_predictions(self, preds, path):
        with open(path, 'w') as f:
            for i in range(len(preds)):
                q = Path(self.qImages[self.qIdx[i]]).stem
                db = ' '.join([Path(self.dbImages[j]).stem for j in preds[i]])
                f.write(f"{q} {db}\n")