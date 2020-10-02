## Training
```test.py``` - script to generate visualizations from a trained model

## Scripts
```stand_alone_scripts/*``` : scripts used to create CUB dataset and caluclate metrics


## Datasets
```scripts/base_cub_dataset.py``` - base class for CUB datasets

```scripts/cub.py``` - simple class for entire CUB dataset

```scripts/cub_geometric.py``` - class for CUB dataset automatically filtered with simple spatial statistics


```scripts / cub_class.py``` - class for a single bird type in CUB dataset

```scripts / cub_species.py``` - class for a species of birds in CUB dataset


## Infrastructure
```image_gan/*``` : infrastructure to train a simple 2D image GAN

```scripts/trainer/trainer_holistic_platonic.py``` - PlatonicGAN variation which considers the entire batch as a single object