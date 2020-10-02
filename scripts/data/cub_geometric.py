import os
from scripts.data.base_cub_dataset import BaseCUBDataset


class CUBGeometricDataset(BaseCUBDataset):
    """
    CUB dataset for geometric filtered data
    """

    def __init__(self, param, mode):
        super(CUBGeometricDataset, self).__init__(param, mode)

        self.path_dir = os.path.join('/media/john/D/projects/platonicgan', param.data.path_dir, self.param.data['image_folder'])

        if mode == "test":
            test_image_file = open('{}/lists/geometric_test.txt'.format(self.metadata_path), 'r')
            file_names = list(test_image_file)
        elif mode == "val":
            test_image_file = open('{}/lists/geometric_val.txt'.format(self.metadata_path), 'r')
            file_names = list(test_image_file)
        elif mode == "train":
            train_image_file = open('{}/lists/geometric_train.txt'.format(self.metadata_path), 'r')
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