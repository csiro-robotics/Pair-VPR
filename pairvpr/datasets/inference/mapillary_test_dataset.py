# Copyright 2024-present CSIRO.
# Licensed under CSIRO BSD-3-Clause-Clear (Non-Commercial)
#


import os
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from pathlib import Path


class MapillaryTestDataset(Dataset):
    def __init__(self, dataset_root: str, src_root_dir: str, input_transform = None):
        """
        Dataset class for the Mapillary Test (challenge) set.
        ---
        dataset_root: str,
        src_root_dir: str,
        input_transform
        """

        self.dataset_root = dataset_root

        self.input_transform = input_transform

        self.dbImages = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/msls_test/msls_test_dbImages.npy'), allow_pickle=True)

        self.qImages = np.load(os.path.join(src_root_dir, 'pairvpr/datasets/datasetfiles/msls_test/msls_test_qImages.npy'), allow_pickle=True)
        
        # reference images then query images
        self.images = np.concatenate((self.dbImages, self.qImages))
        self.num_references = len(self.dbImages)
        self.num_queries = len(self.qImages)
        self.ground_truth = None # send results to challenge server for evaluation
    
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