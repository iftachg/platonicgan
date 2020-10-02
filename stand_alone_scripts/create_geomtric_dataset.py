import os
from PIL import Image
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_images(base_path):
    image_paths = []
    images = []
    image_alphas = []
    for class_name in tqdm(next(os.walk(base_path))[1], desc='Loading images'):
        class_path = os.path.join(base_path, class_name)

        for image_name in next(os.walk(class_path))[2]:
            image_path = os.path.join(class_path, image_name)
            image_paths.append(image_path)
            image = np.array(Image.open(image_path))
            images.append(image)
            image_alphas.append(image[:, :, 3])

    return image_paths, image_alphas


def gather_image_statistics():
    center_Xs = []
    center_Ys = []
    widths = []
    heights = []
    for image_alpha in tqdm(image_alphas, desc='Processing images'):
        used_indices = np.where(image_alpha > 0)
        max_row = max(used_indices[0])
        min_row = min(used_indices[0])
        max_col = max(used_indices[1])
        min_col = min(used_indices[1])

        height = max_row - min_row
        width = max_col - min_col

        center_X = width // 2
        center_Y = height // 2

        center_Xs.append(int(center_X))
        center_Ys.append(int(center_Y))
        widths.append(int(width))
        heights.append(int(height))

    center_Xs = np.array(center_Xs)
    center_Ys = np.array(center_Ys)
    widths = np.array(widths)
    heights = np.array(heights)
    X_mean = np.mean(center_Xs)
    X_std = np.std(center_Xs)
    Y_mean = np.mean(center_Ys)
    Y_std = np.std(center_Ys)
    width_mean = np.mean(widths)
    width_std = np.std(widths)
    height_mean = np.mean(heights)
    height_std = np.std(heights)

    return center_Xs, center_Ys, widths, heights, X_mean, X_std, Y_mean, Y_std, width_mean, width_std, height_mean, height_std


def find_mean_images(center_Xs, center_Ys, widths, heights, X_mean, X_std, Y_mean, Y_std, width_mean, width_std, height_mean, height_std):
    valid_inds = np.where(
        ((center_Xs < X_mean + X_std) & (center_Xs > X_mean - X_std)) &
        ((center_Ys < Y_mean + Y_std) & (center_Ys > Y_mean - Y_std)) &
        ((widths < width_mean + width_std) & (widths > width_mean - width_std)) &
        ((heights < height_mean + height_std) & (heights > height_mean - height_std))
    )

    valid_images = []
    for valid_ind in valid_inds[0]:
        valid_image = image_paths[valid_ind]
        valid_images.append(valid_image)

    return valid_images


def generate_dataset_splits(valid_images):
    train_set, other = train_test_split(valid_images, test_size=0.5)
    validation_set, test_set = train_test_split(other, test_size=0.5)

    base_target_path = '/media/john/D/projects/platonicgan/datasets/CUB/lists'

    target_geometric_train = os.path.join(base_target_path, 'geometric_train.txt')
    with open(target_geometric_train, 'w') as f:
        for line in train_set:
            f.write('{}\n'.format(line))

    target_geometric_val = os.path.join(base_target_path, 'geometric_val.txt')
    with open(target_geometric_val, 'w') as f:
        for line in validation_set:
            f.write('{}\n'.format(line))

    target_geometric_test = os.path.join(base_target_path, 'geometric_test.txt')
    with open(target_geometric_test, 'w') as f:
        for line in test_set:
            f.write('{}\n'.format(line))


if __name__ == '__main__':
    source_path = '/media/john/D/projects/platonicgan/datasets/CUB/minimal_images'
    image_paths, image_alphas = load_images(source_path)
    center_Xs, center_Ys, widths, heights, X_mean, X_std, Y_mean, Y_std, width_mean, width_std, height_mean, height_std = gather_image_statistics()

    print('X: {}+-{}'.format(X_mean, X_std))
    print('Y: {}+-{}'.format(Y_mean, Y_std))
    print('width: {}+-{}'.format(width_mean, width_std))
    print('height: {}+-{}'.format(height_mean, height_std))

    valid_images = find_mean_images(center_Xs, center_Ys, widths, heights, X_mean, X_std, Y_mean, Y_std, width_mean,
                                    width_std, height_mean, height_std)

    generate_dataset_splits(valid_images)

