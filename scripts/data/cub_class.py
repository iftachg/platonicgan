import os
from scripts.data.base_cub_dataset import BaseCUBDataset


class CUBClassDataset(BaseCUBDataset):
    """
    Dataset for single bird class in CUB
    """

    def __init__(self, param, mode):
        super(CUBClassDataset, self).__init__(param, mode)

        self.used_class = self.param.data['use_classes'][0]
        self.path_dir = os.path.join('/media/john/D/projects/platonicgan', param.data.path_dir, self.param.data['image_folder'])

        self.class_list_path = os.path.join('{}/lists/class_sets/{}/'.format(self.metadata_path, self.used_class))
        if not os.path.exists(os.path.join(self.class_list_path, 'train.txt')):
            self.create_splits()

        if mode == "test":
            test_image_file = open('{}/test.txt'.format(self.class_list_path), 'r')
            file_names = list(test_image_file)
        elif mode == "val":
            test_image_file = open('{}/val.txt'.format(self.class_list_path), 'r')
            file_names = list(test_image_file)
        elif mode == "train":
            train_image_file = open('{}/train.txt'.format(self.class_list_path), 'r')
            file_names = list(train_image_file)
        else:
            raise ValueError('Unknown mode: {}'.format(mode))

        self.file_names = []
        for file_name in file_names:
            if self.used_class in file_name:
                self.file_names.append(file_name.rstrip().replace('.jpg', '.png'))

        print('Got {} samples for mode {}'.format(len(self.file_names), mode))
        self.dataset_length = len(self.file_names)

