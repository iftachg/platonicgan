import os
import scripts.renderer.transform as dt
from scripts.data.image import ImageDataset
from sklearn.model_selection import train_test_split


class BaseCUBDataset(ImageDataset):
    """
    Base parent for CUB dataset classes
    """

    def __init__(self, param, mode):
        self.param = param
        self.transform = dt.Transform(self.param)
        print('[INFO] Setup dataset {}'.format(mode))

        self.metadata_path = os.path.join('/media/john/D/projects/platonicgan', param.data.path_dir)

        self.mode = mode
        self.cube_len = param.data.cube_len
        self.n_channel = param.data.n_channel_in

        if hasattr(self.param.data, 'augment'):
            self.augment = self.param.data.augment
        else:
            self.augment = False

        if self.mode != 'train':
            self.augment = False

    def __len__(self):
        return self.dataset_length

    def load_split_files(self, mode):
        if mode == "test":
            test_image_file = open('{}/test.txt'.format(self.list_path), 'r')
            file_names = list(test_image_file)
        elif mode == "val":
            test_image_file = open('{}/val.txt'.format(self.list_path), 'r')
            file_names = list(test_image_file)
        elif mode == "train":
            train_image_file = open('{}/train.txt'.format(self.list_path), 'r')
            file_names = list(train_image_file)
        else:
            raise ValueError('Unknown mode: {}'.format(mode))

        return file_names


    def create_splits(self):
        '''
        Dynamically create train-val-test splits if non exist for given combination of dataset
        '''
        os.makedirs(self.list_path, exist_ok=True)

        files = self.find_all_files()
        train, other = train_test_split(files, test_size=0.3)
        test, val = train_test_split(other, test_size=0.5)

        self.write_split(train, 'train')
        self.write_split(test, 'test')
        self.write_split(val, 'val')

    def find_all_files(self):
        files = []
        for class_name in next(os.walk(self.path_dir))[1]:
            for filename in next(os.walk(os.path.join(self.path_dir, class_name)))[2]:
                files.append('{}/{}'.format(class_name, filename))

        return files

    def write_split(self, files, mode='train'):
        target_path = os.path.join(self.list_path, '{}.txt'.format(mode))
        with open(target_path, 'w') as f:
            for filename in files:
                f.write('{}\n'.format(filename))
