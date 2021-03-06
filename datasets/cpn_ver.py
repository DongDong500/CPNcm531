import os
import sys
import tarfile
import collections
import torch.utils.data as data
import shutil
import numpy as np

from collections import namedtuple
from PIL import Image
#from .splits import split_dataset
'''
if __package__ is None:
    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from splits import split_dataset
else:
    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from .splits import split_dataset
'''

def cpn_cmap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap

CpnDataDir = 'CPN_all'

class CPNver(data.Dataset):
    """
    Args:6
        root (string): Root directory of the VOC Dataset.
        datatype (string): Dataset type 
        image_set (string): Select the image_set to use, ``train`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    
    CpnSixClass = namedtuple('CpnSixClass', ['name', 'id', 'train_id', 'category', 'category_id',
                                                'has_instances', 'ignore_in_eval', 'color'])
    classes = [
        CpnSixClass('background', 0, 0, 'void', 0, False, True, (0, 0, 0)),
        CpnSixClass('nerve', 1, 1, 'void', 0, False, True, (0, 0, 255))
    ]

    train_id_to_color = [c.color for c in classes if (c.train_id != -1 and c.train_id != 255)]
    train_id_to_color.append([0, 0, 0])
    train_id_to_color = np.array(train_id_to_color)
    id_to_train_id = np.array([c.train_id for c in classes])
    
    cmap = cpn_cmap()

    def __init__(self, root, datatype='CPN_all_ver01', image_set='train', transform=None, is_rgb=True):
        
        is_aug = True

        self.root = os.path.expanduser(root)
        self.datafolder = datatype
        self.image_set = image_set
        self.transform = transform
        self.is_rgb = is_rgb

        cpn_root = os.path.join(self.root, datatype)
        image_dir = os.path.join(self.root, CpnDataDir, 'Images')
        rHE_image_dir = os.path.join(self.root, 'CPN_all_rHE', 'Images')
        HE_image_dir = os.path.join(self.root, 'CPN_all_HE', 'Images')
        mask_dir = os.path.join(self.root, CpnDataDir,'Masks')

        if not os.path.exists(cpn_root):
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')
        
        if is_aug and image_set=='train':
            splits_dir = os.path.join(cpn_root, 'splits')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')
        else:
            splits_dir = os.path.join(cpn_root, 'splits')
            split_f = os.path.join(splits_dir, image_set.rstrip('\n') + '.txt')

        if not os.path.exists(splits_dir):
            split_dataset(splits_dir=splits_dir, data_dir=image_dir)

        if not os.path.exists(split_f):
            raise ValueError('Wrong image_set entered!' 
                             'Please use image_set="train" or image_set="val"')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        if is_aug and image_set=='train':
            self.images = [os.path.join(image_dir, x + ".bmp") for x in file_names]
            self.rHE_images = [os.path.join(rHE_image_dir, x + ".bmp") for x in file_names]
            self.HE_images = [os.path.join(HE_image_dir, x + ".bmp") for x in file_names]
            self.masks = [os.path.join(mask_dir, x + "_mask.bmp") for x in file_names]
            self.rHE_masks = [os.path.join(mask_dir, x + "_mask.bmp") for x in file_names]
            self.HE_masks = [os.path.join(mask_dir, x + "_mask.bmp") for x in file_names]
            self.images.extend(self.rHE_images)
            self.images.extend(self.HE_images)
            self.masks.extend(self.rHE_masks)
            self.masks.extend(self.HE_masks)
        else:
            self.images = [os.path.join(image_dir, x + ".bmp") for x in file_names]
            self.masks = [os.path.join(mask_dir, x + "_mask.bmp") for x in file_names]

        assert (len(self.images) == len(self.masks))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the image segmentation.
        """
        if not os.path.exists(self.images[index]):
            raise FileNotFoundError("Error: ", self.images[index])
        if not os.path.exists(self.masks[index]):
            raise FileNotFoundError("Error: ", self.masks[index])
        
        if self.is_rgb:
            img = Image.open(self.images[index]).convert('RGB')
            target = Image.open(self.masks[index]).convert('L')
        else:
            img = Image.open(self.images[index]).convert('L')
            target = Image.open(self.masks[index]).convert('L')

        if self.transform is not None:
            img, target = self.transform(img, target)

        return img, target


    def __len__(self):
        return len(self.images)

    @classmethod
    def decode_target(cls, mask):
        """decode semantic mask to RGB image"""
        return cls.cmap[mask]

if __name__ == "__main__":

    print(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    sys.path.append(os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ))
    from utils import ext_transforms as et
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from splits import split_dataset

    transform = et.ExtCompose([
            et.ExtRandomCrop(size=(512, 512), pad_if_needed=True),
            et.GaussianBlur(kernel_size=(5, 5)),
            et.ExtScale(scale=0.5),
            et.ExtToTensor(),
            et.ExtNormalize(mean=0.485, std=0.229)
            ])
    
    dlist = ['CPN_all_ver01']

    for j in dlist:
            
        dst = CPNver(root='/mnt/server5/sdi/datasets', datatype=j, image_set='val',
                                    transform=transform, is_rgb=True)
        train_loader = DataLoader(dst, batch_size=1,
                                    shuffle=True, num_workers=2, drop_last=True)
        
        for i, (ims, lbls) in tqdm(enumerate(train_loader)):
            print(ims.shape)
            print(lbls.shape)
            print(lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
            print(1 - lbls.numpy().sum()/(lbls.shape[0] * lbls.shape[1] * lbls.shape[2]))
            if i > 1:
                break
        