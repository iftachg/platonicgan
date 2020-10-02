from scripts.data import image, cub_class, cub, cub_geometric, cub_species

dataset_dict = {
    'image': image.ImageDataset,
    'cub_class':  cub_class.CUBClassDataset,
    'cub':  cub.CUBDataset,
    'cub_geometric':  cub_geometric.CUBGeometricDataset,
    'cub_species':  cub_species.CUBSpeciesDataset
}