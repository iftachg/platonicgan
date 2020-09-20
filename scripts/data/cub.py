from torch.utils.data import Dataset
import os
import numpy as np
import scripts.renderer.transform as dt
import cv2
from scripts.utils.utils import normalize
from scipy.io import loadmat
from PIL import Image
import scripts.utils.io as dh
from scripts.data.image import ImageDataset


class CUBDataset(ImageDataset):

    def __init__(self, param, mode):
        self.param = param
        self.transform = dt.Transform(self.param)
        print('[INFO] Setup dataset {}'.format(mode))

        self.metadata_path = os.path.join('/media/john/D/projects/platonicgan', param.data.path_dir)
        self.path_dir = os.path.join('/media/john/D/projects/platonicgan', param.data.path_dir, self.param.data['image_folder'])
        # self.path_dir = os.path.join(os.path.expanduser('~'), param.data.path_dir)
        self.mode = mode
        self.cube_len = param.data.cube_len
        self.n_channel = param.data.n_channel_in

        if mode == "test":
            test_image_file = open('{}/lists/new_test.txt'.format(self.metadata_path), 'r')
            file_names = list(test_image_file)
        elif mode == "val":
            test_image_file = open('{}/lists/new_val.txt'.format(self.metadata_path), 'r')
            file_names = list(test_image_file)
        elif mode == "train":
            train_image_file = open('{}/lists/train.txt'.format(self.metadata_path), 'r')
            file_names = list(train_image_file)
        else:
            raise ValueError('Unknown mode: {}'.format(mode))

        self.file_names = []
        for file_name in file_names:
            self.file_names.append(file_name.rstrip().replace('.jpg', '.png'))

        print('Got {} samples for mode {}'.format(len(self.file_names), mode))
        self.dataset_length = len(self.file_names)

    def __len__(self):
        return self.dataset_length