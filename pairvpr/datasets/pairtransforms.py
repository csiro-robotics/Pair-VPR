# This file defines a collection of pair-augmentations


import torch
import torchvision.transforms
import torchvision.transforms.functional as F


class ComposePair(torchvision.transforms.Compose):
    def __call__(self, img1, img2):
        for transform in self.transforms:
            img1, img2 = transform(img1, img2)
        return img1, img2


class RandomResizedCropPair(torchvision.transforms.RandomResizedCrop):
    # the transform (a crop) will be different for the two images with this class
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2
    

class NormalizePair(torchvision.transforms.Normalize):
    def forward(self, img1, img2):
        img1 = super().forward(img1)
        img2 = super().forward(img2)
        return img1, img2


class ToTensorPair(torchvision.transforms.ToTensor):
    def __call__(self, img1, img2):
        img1 = super().__call__(img1)
        img2 = super().__call__(img2)
        return img1, img2


class ColorJitterPair(torchvision.transforms.ColorJitter): 
    # can either apply same augment to both images in a pair, or different augments per image

    def __init__(self, diffaug_prob, **kwargs):
        super().__init__(**kwargs)
        self.diffaug_prob = diffaug_prob
    
    def jitter_fn(self, img, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor):
        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = F.adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = F.adjust_contrast(img, contrast_factor)
            elif fn_id == 2 and saturation_factor is not None:
                img = F.adjust_saturation(img, saturation_factor)
            elif fn_id == 3 and hue_factor is not None:
                img = F.adjust_hue(img, hue_factor)
        return img
        
    def forward(self, img1, img2):
        # get_params is a method from the super class
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
            self.brightness, self.contrast, self.saturation, self.hue
        )
        img1 = self.jitter_fn(img1, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)

        if torch.rand(1) < self.diffaug_prob: # change params for img2:
            fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        img2 = self.jitter_fn(img2, fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor)
        return img1, img2


def get_pair_transforms(transform_str, totensor=True, normalize=True):
    # transform_str is in the format: "crop224,color" (augment1, augment2, etc)
    transforms_list = []
    for s in transform_str.split(','):
        if s.startswith('crop'):
            size = int(s[len('crop'):])
            transforms_list.append(RandomResizedCropPair(size, scale=(0.7, 1), antialias=True))
        elif s=='acolor':
            transforms_list.append(ColorJitterPair(diffaug_prob=1.0, brightness=(0.6, 1.4), contrast=(0.6, 1.4), saturation=(0.6, 1.4), hue=0.0))
        elif s=='': # if transform_str was ""
            pass
        elif s=='ablation':
            transforms_list.append(ColorJitterPair(diffaug_prob=1.0, brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=0.1))
        else:
            raise NotImplementedError('Unknown augmentation: '+s)
            
    if totensor:
        transforms_list.append(ToTensorPair())
    if normalize:
        transforms_list.append(NormalizePair(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    if len(transforms_list)==0:
        return None
    elif len(transforms_list)==1:
        return transforms_list
    else:
        return ComposePair(transforms_list)
        

class UnNormalize(object):
    '''
    For visalizations
    '''
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor