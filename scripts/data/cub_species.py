import os
from scripts.data.base_cub_dataset import BaseCUBDataset


class CUBSpeciesDataset(BaseCUBDataset):

    def __init__(self, param, mode):
        super(CUBSpeciesDataset, self).__init__(param, mode)
        self.used_class = self.param.data['use_classes'][0]
        self.path_dir = os.path.join('/media/john/D/projects/platonicgan', param.data.path_dir, self.param.data['image_folder'])

        self.species_list_path = os.path.join('{}/lists/species_sets/{}/'.format(self.metadata_path, self.used_class))
        if not os.path.exists(os.path.join(self.species_list_path, 'train.txt')):
            self.create_splits()

        file_names = self.load_split_files(mode)

        self.file_names = []
        for file_name in file_names:
            if '_{}'.format(self.used_class) in file_name.lower():
                self.file_names.append(file_name.rstrip().replace('.jpg', '.png'))

        print('Got {} samples for mode {}'.format(len(self.file_names), mode))
        self.dataset_length = len(self.file_names)

    def __len__(self):
        return self.dataset_length
