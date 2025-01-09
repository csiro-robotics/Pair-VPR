# Copyright 2024-present CSIRO.
# Licensed under CSIRO BSD-3-Clause-Clear (Non-Commercial)
#


from pathlib import Path
import os

import torchvision.transforms as T

from pairvpr.datasets.inference.test_dataset import TestDataset
from pairvpr.datasets.inference.pittsburg_dataset import PittsburghDataset
from pairvpr.datasets.inference.mapillary_val_dataset import MapillaryValDataset
from pairvpr.datasets.inference.nordland_dataset import NordlandDataset
from pairvpr.datasets.inference.mapillary_test_dataset import MapillaryTestDataset


IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406],
                     'std': [0.229, 0.224, 0.225]}


def get_test_dataset(cfg, dsetroot: str, src_root_dir: Path, dataset_name: str, image_size=322):
    mean_dataset = IMAGENET_MEAN_STD['mean']
    std_dataset = IMAGENET_MEAN_STD['std']

    dataset_name = dataset_name.lower()

    transform = T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=mean_dataset, std=std_dataset)])

    if 'sf-xl_testv1' in dataset_name:
        # support for EigenPlaces datasets
        dataset_location = os.path.join(dsetroot, cfg.dataset_locations.sfxl)
        ds = TestDataset(dataset_location, str(src_root_dir), queries_folder="queries_v1",
                              positive_dist_threshold=25,
                              input_transform=transform)

    elif 'sf-xl_testv2' in dataset_name:
        dataset_location = os.path.join(dsetroot, cfg.dataset_locations.sfxl)
        ds = TestDataset(dataset_location, str(src_root_dir), queries_folder="queries_v2",
                              positive_dist_threshold=25,
                              input_transform=transform)

    elif 'nordland' in dataset_name:
        dataset_location = os.path.join(dsetroot, cfg.dataset_locations.nord)
        ds = NordlandDataset(dataset_location, str(src_root_dir), input_transform=transform)

    elif 'msls_test' in dataset_name:
        dataset_location = os.path.join(dsetroot, cfg.dataset_locations.msls_test)
        ds = MapillaryTestDataset(dataset_location, str(src_root_dir), input_transform=transform)

    elif 'msls_val' in dataset_name:
        dataset_location = os.path.join(dsetroot, cfg.dataset_locations.msls)
        ds = MapillaryValDataset(dataset_location, str(src_root_dir), input_transform=transform)

    elif 'pitts' in dataset_name:
        dataset_location = os.path.join(dsetroot, cfg.dataset_locations.pitts)
        ds = PittsburghDataset(dataset_location, str(src_root_dir), which_ds=dataset_name, input_transform=transform)

    elif 'tokyo247' in dataset_name:
        dataset_location = os.path.join(dsetroot, cfg.dataset_locations.tokyo)
        ds = TestDataset(dataset_location, str(src_root_dir), queries_folder="queries",
                              positive_dist_threshold=25,
                              input_transform=transform)

    else:
        raise ValueError

    num_references = ds.num_references
    num_queries = ds.num_queries
    ground_truth = ds.ground_truth
    return ds, num_references, num_queries, ground_truth
