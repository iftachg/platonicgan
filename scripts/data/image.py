from torch.utils.data import Dataset
import os
import scripts.utils.io as dh


class ImageDataset(Dataset):

    def __init__(self, param, mode):
        """
        Args:
            path_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.param = param

        if hasattr(self.param.data, 'augment'):
            self.augment = self.param.data.augment
        else:
            self.augment = False

        if self.mode != 'train':
            self.augment = False

        self.path_dir = os.path.join(os.getcwd(), param.data.path_dir, mode)

        self.cube_len = param.data.cube_len
        self.file_names = [name for name in os.listdir(self.path_dir) if
                           os.path.isfile(os.path.join(self.path_dir, name)) and name.endswith('.png')]

        self.dataset_length = len(self.file_names)

    def __len__(self):
        return self.dataset_length - 1

    def __getitem__(self, idx):
        if not hasattr(self, 'augment'):
            self.augment = False

        if self.mode != 'train':
            self.augment = False

        image_path = os.path.join(self.path_dir, self.file_names[idx])
        image = dh.read_image(image_path, self.cube_len, self.augment)

        if not image.shape[0] == 1 and (self.param.renderer.type == 'visual_hull' or self.param.renderer.type == 'absorption_only'):
            image = image[[3]]

        elif self.param.renderer.type == 'emission_absorption':
            image[0:3] = image[0:3] * image[3]

        if self.param.renderer.type == 'absorption_only':
            image = image[[0]]

        return image, 0.0, 0.0, 0.0, idx, 0.0, 0.0
