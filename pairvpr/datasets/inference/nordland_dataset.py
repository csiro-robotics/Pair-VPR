# Modified from GSV-Cities, acknowledgements and thanks: https://github.com/amaralibey/gsv-cities
#
#


import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset

# Note this is the VPR Bench version of the Nordland dataset.
# You need to download the Nordland dataset from  https://surfdrive.surf.nl/files/index.php/s/sbZRXzYe3l0v67W
# this link is shared and maintained by the authors of VPR_Bench: https://github.com/MubarizZaffar/VPR-Bench


class NordlandDataset(Dataset):
    def __init__(self, dataset_root: str, src_root_dir: str, input_transform = None):
        """
        Dataset class for the Nordland dataset (VPR Bench version).
        ---
        dataset_root: str,
        src_root_dir: str,
        input_transform
        """

        self.dataset_root = dataset_root

        self.input_transform = input_transform

        # reference images names
        self.dbImages = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/nord/Nordland_dbImages.npy'))
        
        # query images names
        self.qImages = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/nord/Nordland_qImages.npy'))
        
        # ground truth
        self.ground_truth = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/nord/Nordland_gt.npy'), allow_pickle=True)
        
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
    
    def save_predictions(self, preds, path):
        with open(path, 'w') as f:
            for i in range(len(preds)):
                q = Path(self.qImages[i]).stem
                db = ' '.join([Path(self.dbImages[j]).stem for j in preds[i]])
                f.write(f"{q} {db}\n")