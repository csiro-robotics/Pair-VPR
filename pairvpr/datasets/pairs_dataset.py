# Modified from EigenPlaces, acknowledgements and thanks: https://github.com/gmberton/EigenPlaces (MIT License)
# Including additional modifications from GSV-Cities: https://github.com/amaralibey/gsv-cities
# Modifications are Copyright CSIRO 2024


import os
import time
from typing import List
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image, ImageFile, UnidentifiedImageError
from collections import defaultdict

import pairvpr.datasets.dataset_utils as dataset_utils
from pairvpr.datasets.pairtransforms import get_pair_transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

PANO_WIDTH = int(512 * 6.5)

TRAIN_CITIES_GSV = [
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity',
    'OSL',
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT',
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG',
    'WashingtonDC',
    'Brussels',
    'London',
    'Melbourne',
    'Osaka',
    'PRS',
]


class PairsDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, root_dir: str, datasets_used: List[str],
                 dataset_folder_sfxl: str, dataset_folder_gsv: str,
                 dataset_folder_gldv2: str,
                 M: int=15, N: int=1, focal_dist: int=10, focal_dist_max: int=20, current_group: int=0,
                 min_images_per_class: int=10, angle: int=0,
                 dynamicmode: bool=True, angcheck: bool=True, rank: int=0):
        """
        Parameters - EigenPlaces.
        
        dataset_folder_sfxl : str, the path of the folder with the train images (SF-XL).
        M : int, the length of the side of each cell in meters.
        N : int, distance (M-wise) between two classes of the same group.
        focal_dist : int, distance (in meters) between the center of the class and
            the focal point. The center of the class is computed as the
            mean of the positions of the images within the class.
        current_group : int, which one of the groups to consider.
        min_images_per_class : int, minimum number of image in a class.
        angle : int, the angle formed between the line of the first principal
            component, and the line that connects the center of gravity of the
            images to the focal point.
        ----------
        Parameters - PairVPR.

        root_dir: str, directory of this codebase.
        dataset_folder_gsv: str, the path of the folder with the GSV-Cities images.
        dataset_folder_gldv2: str, path to the google landmarks v2 images.
        dynamicmode: bool, flag to activate dynamic focal length calculations. Setting to false reverts to EigenPlaces.
        angcheck: bool, turn off to allow almost *any* pair combinations. May cause poor training performance due to extreme viewpoint shifts.
        rank: int, current rank of this process. Used to prevent two processes loading the same file simultaneously.
        """
        super().__init__()

        # common init
        self.transforms = get_pair_transforms(transform_str=cfg.augmentation.transform, totensor=True, normalize=True)

        # gld init
        # need to load csv_clean
        # then pre-process to remove classes with less than N images
        # then compile data together for training
        self.classes_ids_gldv2 = []
        if "gldv2" in datasets_used:
            self.min_img_per_landmark = 4
            gld_df = pd.read_csv(os.path.join(root_dir, 'pairvpr', 'datasets', 'datasetfiles', 'gldv2', 'train_clean.csv'))
            keep_indices = []
            for iter, row in gld_df.iterrows():
                imglist = row.images.split(' ')
                if len(imglist) >= self.min_img_per_landmark:
                    keep_indices.append(iter)
            self.gld_df_clean = gld_df.iloc[keep_indices]

            self.classes_ids_gldv2 = pd.unique(self.gld_df_clean.index)
            self.dataset_folder_gldv2 = dataset_folder_gldv2

        # now we have 72,322 landmarks to add to our training process

        # -------------------------------------------------------
        # gsv cities init
        self.classes_ids_gsv = []
        if "gsv" in datasets_used:
            self.base_path_gsv = dataset_folder_gsv
            self.cities_gsv = TRAIN_CITIES_GSV

            self.img_per_place_gsv = 2
            self.min_img_per_place_gsv = 4
            self.random_sample_from_each_place_gsv = True

            # generate the dataframe contraining images metadata
            self.dataframe_gsv = self._getdataframes_gsv()

            # get all unique place ids
            self.classes_ids_gsv = pd.unique(self.dataframe_gsv.index)
            self.total_nb_images_gsv = len(self.dataframe_gsv)

        # -------------------------------------------------------
        # sfxl init
        self.classes_ids_sfxl = []
        if "sf" in datasets_used:
            self.M = M
            self.N = N
            self.focal_dist = focal_dist
            self.focal_dist_max = focal_dist_max
            self.current_group = current_group
            self.dataset_folder_sfxl = dataset_folder_sfxl
            self.dynamic_mode = dynamicmode
            self.angcheck = angcheck

            filename = os.path.join(root_dir, 'pairvpr', 'datasets', 'datasetfiles', 'sfxl',
                                    f"sfxl_M{M}_N{N}_mipc{min_images_per_class}.torch")

            if not os.path.exists(filename):
                os.makedirs(os.path.join(root_dir, "cache"), exist_ok=True)
                print(f"Cached dataset {filename} does not exist, I'll create it now.")
                self.initialize(dataset_folder_sfxl, M, N, min_images_per_class, filename)
            elif current_group == 0:
                print(f"Using cached dataset {filename}")

            waittime = rank*5
            time.sleep(waittime) # only load on one thread at a time (otherwise can get an error)
            classes_per_group, self.images_per_class = torch.load(filename)
            if current_group >= len(classes_per_group):
                raise ValueError(f"With this configuration there are only {len(classes_per_group)} " +
                                 f"groups, therefore I can't create the {current_group}th group. " +
                                 "You should reduce the number of groups by setting for example " +
                                 f"'--groups_num {current_group}'")
            self.classes_ids = classes_per_group[current_group]

            new_classes_ids = []
            self.focal_point_per_class = {}
            for class_id in self.classes_ids:
                paths = self.images_per_class[class_id]
                u_coords = np.array([p.split("@")[1:3] for p in paths]).astype(float)
                if dynamicmode:
                    eigenvectors, mu = dataset_utils.focal_precompute(u_coords)
                    self.focal_point_per_class[class_id] = (eigenvectors, mu)
                else:
                    focal_point = dataset_utils.get_focal_point(u_coords, focal_dist, angle=angle)
                    self.focal_point_per_class[class_id] = focal_point
                new_classes_ids.append(class_id)

            self.classes_ids_sfxl = new_classes_ids

        # merge
        self.allclasses = self.classes_ids_sfxl + list(self.classes_ids_gldv2) + list(self.classes_ids_gsv)
        self.dset_cat = [2]*len(self.classes_ids_sfxl) + [0]*len(self.classes_ids_gldv2) + [1]*len(self.classes_ids_gsv)  # a mask denoting which dataset a given index came from

    @staticmethod
    def get_crop(pano_path, focal_point):
        obs_point = pano_path.split("@")[1:3]
        angle = - dataset_utils.get_angle(focal_point, obs_point) % 360
        crop_offset = int((angle / 360 * PANO_WIDTH) % PANO_WIDTH)
        yaw = int(pano_path.split("@")[9])
        north_yaw_in_degrees = (180 - yaw) % 360
        yaw_offset = int((north_yaw_in_degrees / 360) * PANO_WIDTH)
        offset = (yaw_offset + crop_offset - 256) % PANO_WIDTH
        pano_pil = Image.open(pano_path)
        if offset + 512 <= PANO_WIDTH:
            pil_crop = pano_pil.crop((offset, 0, offset + 512, 512))
        else:
            crop1 = pano_pil.crop((offset, 0, PANO_WIDTH, 512))
            crop2 = pano_pil.crop((0, 0, 512 - (PANO_WIDTH - offset), 512))
            pil_crop = Image.new('RGB', (512, 512))
            pil_crop.paste(crop1, (0, 0))
            pil_crop.paste(crop2, (crop1.size[0], 0))

        return pil_crop, angle

    def __getitem__(self, class_num):
        class_id = self.allclasses[class_num]
        dsource = self.dset_cat[class_num]

        if dsource == 0: # gldv2
            landmark = self.gld_df_clean.loc[class_id]
            img_list = landmark.images.split(' ')
            pair = random.sample(img_list, 2)
            imgpath1 = self.make_gld_fullpath(pair[0])
            imgpath2 = self.make_gld_fullpath(pair[1])
            crop1 = self.image_loader(imgpath1)
            crop2 = self.image_loader(imgpath2)
            crop1, crop2 = self.transforms(crop1, crop2)
        elif dsource == 1: # gsv-cities
            place = self.dataframe_gsv.loc[class_id]

            if self.random_sample_from_each_place_gsv:
                place = place.sample(self.img_per_place_gsv)
            else:
                place = place.sort_values(
                    by=['year', 'month', 'lat'], ascending=False)
                place = place[: self.img_per_place_gsv]
            imgs = []
            for i, row in place.iterrows():
                img_name = self.get_img_name(row)
                img_path = os.path.join(self.base_path_gsv, 'Images',
                                        row['city_id'], img_name)
                img = self.image_loader(img_path)

                imgs.append(img)

            crop1, crop2 = imgs[0], imgs[1]
            crop1, crop2 = self.transforms(crop1, crop2)
        else: # sf-xl
            # This function takes as input the class_num instead of the index of
            # the image. This way each class is equally represented during training.
            if self.dynamic_mode:
                eigenvectors, mu = self.focal_point_per_class[class_id]
                meters_from_center = np.random.randint(self.focal_dist, self.focal_dist_max+1)  # random focal length between 10 and 20m
                angle = np.random.randint(0, 360)  # random angle from 360 deg panorama
                dir = dataset_utils.rotate_2d_vector(eigenvectors[1], angle)
                focal_point = mu + dir * meters_from_center
            else:
                focal_point = self.focal_point_per_class[class_id]

            # for pre-training, grab random pairs from a class viewing the same focal point

            pano_path1 = self.dataset_folder_sfxl + "/" + random.choice(self.images_per_class[class_id])
            crop1, ang1 = self.get_crop(pano_path1, focal_point)
            if self.angcheck:
                angdiff = 1000
                earlyexit = 0
                while (angdiff > 50 or angdiff < 3) and earlyexit < 10:
                    # we don't want trivial (less than 3 degrees) or too hard (greater than 50 degrees) pairs
                    # excessive sampling is slow, therefore exit after 5 attempts, but make sure the last pair is not identical (angdiff = 0)
                    # however, there may be some classes which simply don't have sufficient images (edge cases) so terminate after 10 attempts
                    pano_path2 = self.dataset_folder_sfxl + "/" + random.choice(self.images_per_class[class_id])
                    crop2, ang2 = self.get_crop(pano_path2, focal_point)
                    angdiff = np.abs(dataset_utils.anglediffcalc(ang1, ang2))
                    earlyexit += 1
                    if (earlyexit >= 5 and angdiff > 0.1):
                        break
            else:
                pano_path2 = self.dataset_folder_sfxl + "/" + random.choice(self.images_per_class[class_id])
                crop2, ang2 = self.get_crop(pano_path2, focal_point)

            crop1, crop2 = self.transforms(crop1, crop2)

        return crop1, crop2, dsource

    def get_images_num(self):
        """Return the number of images within this group."""
        return sum([len(self.images_per_class[c]) for c in self.classes_ids])

    def __len__(self):
        """Return the number of classes within this group."""
        return len(self.allclasses)

    def _getdataframes_gsv(self):
        # acknowledgements: https://github.com/amaralibey/gsv-cities
        '''
            Return one dataframe containing
            all info about the images from all cities

            This requieres DataFrame files to be in a folder
            named Dataframes, containing a DataFrame
            for each city in self.cities
        '''
        # read the first city dataframe
        df = pd.read_csv(os.path.join(self.base_path_gsv, 'Dataframes', f'{self.cities_gsv[0]}.csv'))
        df = df.sample(frac=1)  # shuffle the city dataframe

        # append other cities one by one
        for i in range(1, len(self.cities_gsv)):
            tmp_df = pd.read_csv(
                os.path.join(self.base_path_gsv, 'Dataframes', f'{self.cities_gsv[i]}.csv'))

            # Now we add a prefix to place_id, so that we
            # don't confuse, say, place number 13 of NewYork
            # with place number 13 of London ==> (0000013 and 0500013)
            # We suppose that there is no city with more than
            # 99999 images and there won't be more than 99 cities
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10 ** 5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe

            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place_gsv]
        return res.set_index('place_id')

    def make_gld_fullpath(self, imgcode):
        imgpath = os.path.join(imgcode[0], imgcode[1], imgcode[2], imgcode)
        fullpath = os.path.join(self.dataset_folder_gldv2, imgpath + '.jpg')
        return fullpath

    @staticmethod
    def image_loader(path):
        try:
            return Image.open(path).convert('RGB')
        except UnidentifiedImageError:
            print(f'Image {path} could not be loaded')
            return Image.new('RGB', (224, 224))

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name

        city = row['city_id']

        # now remove the two digit we added to the id
        # they are superficially added to make ids different
        # for different cities
        pl_id = row.name % 10 ** 5  # row.name is the index of the row, not to be confused with image name
        pl_id = str(pl_id).zfill(7)

        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city + '_' + pl_id + '_' + year + '_' + month + '_' + \
               northdeg + '_' + lat + '_' + lon + '_' + panoid + '.jpg'
        return name


    @staticmethod
    def initialize(dataset_folder, M, N, min_images_per_class, filename):
        # acknowledgements: https://github.com/gmberton/EigenPlaces
        print(f"Searching training images in {dataset_folder}")

        images_paths = dataset_utils.read_images_paths(dataset_folder)
        print(f"Found {len(images_paths)} images")

        print("For each image, get its UTM east, UTM north from its path")
        images_metadatas = [p.split("@") for p in images_paths]
        # field 1 is UTM east, field 2 is UTM north
        utmeast_utmnorth = [(m[1], m[2]) for m in images_metadatas]
        utmeast_utmnorth = np.array(utmeast_utmnorth).astype(float)

        print("For each image, get class and group to which it belongs")
        class_id__group_id = [PairsDataset.get__class_id__group_id(*m, M, N)
                              for m in utmeast_utmnorth]

        print("Group together images belonging to the same class")
        images_per_class = defaultdict(list)
        for image_path, (class_id, _) in zip(images_paths, class_id__group_id):
            images_per_class[class_id].append(image_path)

        # Images_per_class is a dict where the key is class_id, and the value
        # is a list with the paths of images within that class.
        images_per_class = {k: v for k, v in images_per_class.items() if len(v) >= min_images_per_class}

        print("Group together classes belonging to the same group")
        # Classes_per_group is a dict where the key is group_id, and the value
        # is a list with the class_ids belonging to that group.
        classes_per_group = defaultdict(set)
        for class_id, group_id in class_id__group_id:
            if class_id not in images_per_class:
                continue  # Skip classes with too few images
            classes_per_group[group_id].add(class_id)

        # Convert classes_per_group to a list of lists.
        # Each sublist represents the classes within a group.
        classes_per_group = [list(c) for c in classes_per_group.values()]

        torch.save((classes_per_group, images_per_class), filename)

    @staticmethod
    def get__class_id__group_id(utm_east, utm_north, M, N):
        # acknowledgements: https://github.com/gmberton/EigenPlaces
        """Return class_id and group_id for a given point.
            The class_id is a triplet (tuple) of UTM_east, UTM_north
            (e.g. (396520, 4983800)).
            The group_id represents the group to which the class belongs
            (e.g. (0, 1)), and it is between (0, 0) and (N, N).
        """
        rounded_utm_east = int(utm_east // M * M)  # Rounded to nearest lower multiple of M
        rounded_utm_north = int(utm_north // M * M)

        class_id = (rounded_utm_east, rounded_utm_north)
        # group_id goes from (0, 0) to (N, N)
        group_id = (rounded_utm_east % (M * N) // M,
                    rounded_utm_north % (M * N) // M)
        return class_id, group_id
