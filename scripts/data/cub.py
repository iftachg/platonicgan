import os
from scripts.data.base_cub_dataset import BaseCUBDataset


class CUBDataset(BaseCUBDataset):
    """
    Basic dataset for CUB
    """

    def __init__(self, param, mode):
        super(CUBDataset, self).__init__(param, mode)

        self.path_dir = os.path.join('/media/john/D/projects/platonicgan', param.data.path_dir, self.param.data['image_folder'])

        self.list_path = os.path.join('{}/lists/general_sets/{}/'.format(self.metadata_path, self.param.data['image_folder']))
        if not os.path.exists(os.path.join(self.list_path, 'train.txt')):
            self.create_splits()

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

        self.file_names = []
        for file_name in file_names:
            file_path = os.path.join(self.path_dir, file_name).rstrip().replace('.jpg', '.png')
            if os.path.exists(file_path):
                self.file_names.append(file_name.rstrip().replace('.jpg', '.png'))

        print('Got {} samples for mode {}'.format(len(self.file_names), mode))
        self.dataset_length = len(self.file_names)

    def __len__(self):
        return self.dataset_length

