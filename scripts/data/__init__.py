from scripts.data import image, caltech_ucsd_birds, cub_class, cub, cub_geometric, cub_species

dataset_dict = {
    'image': image.ImageDataset,
    'ucb':  caltech_ucsd_birds.UCBDataset,
    'cub_class':  cub_class.CUBClassDataset,
    'cub':  cub.CUBDataset,
    'cub_geometric':  cub_geometric.CUBGeometricDataset,
    'cub_species':  cub_species.CUBSpeciesDataset
}