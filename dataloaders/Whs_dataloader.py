from __future__ import print_function, division
import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from glob import glob
import random
import nibabel as nib

class WhsSegmentation(Dataset):

    def __init__(self,
                 base_dir,
                 dataset='whs',
                 split='train',
                 geshi='png',
                 testid=None,
                 transform=None,
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.geshi = geshi

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []


        if geshi == 'png':
            self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
            print(self._image_dir)
            imagelist = glob(self._image_dir + "/*.png")
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
        if geshi == 'npy':
            self._image_dir = os.path.join(self._base_dir, dataset, split, 'IMG')
            print(self._image_dir)
            imagelist = glob(self._image_dir + "/*.npy")
            for image_path in imagelist:
                gt_path = image_path.replace('IMG', 'GT')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
        if geshi == 'nii':
            self._image_dir = os.path.join(self._base_dir, dataset, split, 'IMG')
            print(self._image_dir)
            imagelist = glob(self._image_dir + "/*.nii")
            for image_path in imagelist:
                gt_path = image_path.replace('IMG', 'GT')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

        self.transform = transform
        # self._read_img_into_memory()
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if self.geshi == 'png':
            _img = Image.open(self.image_list[index]['image']).convert('RGB')
            _target = Image.open(self.image_list[index]['label'])
        if self.geshi == 'npy':
            _img = np.load(self.image_list[index]['image'])
            _target = np.load(self.image_list[index]['label'])

            _target = _target.squeeze()
            _target = (_target).astype(np.uint8)
            _target = Image.fromarray(_target, 'L')

            _img = _img[:,:,0]
            _img_min = np.min(_img)
            denominator = _img.max() - _img_min
            if denominator == 0:
                denominator = 1
            _img = 255.0 * ((_img - _img_min) / denominator)
            _img = np.repeat(_img[..., np.newaxis], 3, axis=-1)
            _img = _img.astype(np.uint8)
            _img = Image.fromarray(_img, 'RGB')

        if self.geshi == 'nii':
            _img = nib.load(self.image_list[index]['image'])
            _img = _img.get_fdata()
            _target = nib.load(self.image_list[index]['label'])
            _target = _target.get_fdata()
            _target = _target[0,:,:]
            _target = (_target).astype(np.uint8)
            _target = Image.fromarray(_target, 'L')

            _img = _img[0,:,:]
            _img_min = np.min(_img)
            denominator = _img.max() - _img_min
            if denominator == 0:
                denominator = 1
            _img_range = 255.0 * ((_img - _img_min) / denominator)

            _img = np.repeat(_img_range[..., np.newaxis], 3, axis=-1)
            _img = _img.astype(np.uint8)
            _img = Image.fromarray(_img, 'RGB')

        if _target.mode is 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        if self.transform is not None:
            anco_sample = self.transform(anco_sample)

        return anco_sample

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Fundus(split=' + str(self.split) + ')'


class WhsSegmentation_2transform(Dataset):
    def __init__(self,
                 base_dir,
                 dataset='refuge',
                 split='train',
                 geshi='png',
                 testid=None,
                 transform_weak=None,
                 transform_strong=None
                 ):
        """
        :param base_dir: path to VOC dataset directory
        :param split: train/val
        :param transform: transform to apply
        """
        # super().__init__()
        self._base_dir = base_dir
        self.image_list = []
        self.split = split

        self.geshi = geshi

        self.image_pool = []
        self.label_pool = []
        self.img_name_pool = []
        ###png
        if geshi == 'png':
            self._image_dir = os.path.join(self._base_dir, dataset, split, 'image')
            print(self._image_dir)
            imagelist = glob(self._image_dir + "/*.png")
            for image_path in imagelist:
                gt_path = image_path.replace('image', 'mask')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
        if geshi == 'npy':
            self._image_dir = os.path.join(self._base_dir, dataset, split, 'IMG')
            print(self._image_dir)
            imagelist = glob(self._image_dir + "/*.npy")
            for image_path in imagelist:
                gt_path = image_path.replace('IMG', 'GT')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})
        if geshi == 'nii':
            self._image_dir = os.path.join(self._base_dir, dataset, split, 'IMG')
            print(self._image_dir)
            imagelist = glob(self._image_dir + "/*.nii")
            for image_path in imagelist:
                gt_path = image_path.replace('IMG', 'GT')
                self.image_list.append({'image': image_path, 'label': gt_path, 'id': testid})

        self.transform_weak = transform_weak
        self.transform_strong = transform_strong
        # self._read_img_into_memory()
        # Display stats
        print('Number of images in {}: {:d}'.format(split, len(self.image_list)))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if self.geshi == 'png':
            _img = Image.open(self.image_list[index]['image']).convert('RGB')
            _target = Image.open(self.image_list[index]['label'])
        if self.geshi == 'npy':
            _img = np.load(self.image_list[index]['image'])
            _target = np.load(self.image_list[index]['label'])
            _target = _target.squeeze()
            _target = (_target).astype(np.uint8)
            _target = Image.fromarray(_target, 'L')

            _img = _img[:, :, 0]
            _img_min = np.min(_img)
            denominator = _img.max() - _img_min
            if denominator == 0:
                denominator = 1
            _img = 255.0 * ((_img - _img_min) / denominator)
            _img = np.repeat(_img[..., np.newaxis], 3, axis=-1)
            _img = _img.astype(np.uint8)
            _img = Image.fromarray(_img, 'RGB')
        if self.geshi == 'nii':
            _img = nib.load(self.image_list[index]['image'])
            _img = _img.get_fdata()
            _target = nib.load(self.image_list[index]['label'])
            _target = _target.get_fdata()
            _target = _target[0,:,:]
            _target = (_target).astype(np.uint8)
            _target = Image.fromarray(_target, 'L')

            _img = _img[0,:,:]
            _img_min = np.min(_img)
            _img_range = 255.0 * ((_img - _img_min) / (_img.max() - _img_min))
            _img = np.repeat(_img_range[..., np.newaxis], 3, axis=-1)
            _img = _img.astype(np.uint8)
            _img = Image.fromarray(_img, 'RGB')


        if _target.mode is 'RGB':
            _target = _target.convert('L')
        _img_name = self.image_list[index]['image'].split('/')[-1]

        # _img = self.image_pool[index]
        # _target = self.label_pool[index]
        # _img_name = self.img_name_pool[index]
        anco_sample = {'image': _img, 'label': _target, 'img_name': _img_name}

        anco_sample_weak_aug = self.transform_weak(anco_sample)

        anco_sample_strong_aug = self.transform_strong(anco_sample)

        return anco_sample_weak_aug, anco_sample_strong_aug

    def _read_img_into_memory(self):

        img_num = len(self.image_list)
        for index in range(img_num):
            self.image_pool.append(Image.open(self.image_list[index]['image']).convert('RGB'))
            _target = Image.open(self.image_list[index]['label'])
            if _target.mode is 'RGB':
                _target = _target.convert('L')
            self.label_pool.append(_target)
            _img_name = self.image_list[index]['image'].split('/')[-1]
            self.img_name_pool.append(_img_name)


    def __str__(self):
        return 'Whs(split=' + str(self.split) + ')'


