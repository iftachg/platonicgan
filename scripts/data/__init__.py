from scripts.data import image, caltech_ucsd_birds, cub_class, cub

dataset_dict = {
    'image': image.ImageDataset,
    'ucb':  caltech_ucsd_birds.UCBDataset,
    'cub_class':  cub_class.CUBClassDataset,
    'cub':  cub.CUBDataset
}