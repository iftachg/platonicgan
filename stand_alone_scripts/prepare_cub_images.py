'''
Generates masked CUB images
'''
from PIL import Image
import numpy as np
import os
from scipy.io import loadmat
from tqdm import tqdm


def load_img_parts(path_dir, annotations_path, image_name):
    image_name = image_name.rstrip()

    image_path = os.path.join(path_dir, image_name)
    image = Image.open(image_path)

    annotation_data = loadmat(annotations_path)
    segmentation_image = annotation_data['seg']

    bounding_box_mat = annotation_data['bbox']
    x = bounding_box_mat[0, 0][0][0, 0]
    y = bounding_box_mat[0, 0][1][0, 0]
    width = bounding_box_mat[0, 0][2][0, 0]
    height = bounding_box_mat[0, 0][3][0, 0]

    bounding_box = (x, y, width, height)

    return image, bounding_box, segmentation_image


def prepare_img(image_original, bounding_box, segmentation_image):

    x, y, width, height = bounding_box

    x = int(float(x))
    y = int(float(y))
    width = int(float(width))
    height = int(float(height))

    original_image_data = np.array(image_original)
    segmented_image_data = np.concatenate((original_image_data, np.expand_dims(segmentation_image, 2) * 255), 2)

    image = segmented_image_data

    # fixing annotation bugs
    if original_image_data.shape[0] < y + height:
        height += original_image_data.shape[0] - (y + height)
    if original_image_data.shape[1] < x + width:
        width += original_image_data.shape[1] - (x + width)

    image = image[y:y + height, x:x + width, :]

    if height > width:
        image_square = np.zeros((height, height, 4))

        x_start = int((height - width) / 2.0)

        image_square[:, x_start:x_start + width, :] = image
    else:
        image_square = np.zeros((width, width, 4))
        y_start = int((width - height) / 2.0)
        image_square[y_start:y_start + height, :, :] = image


    used_indices = np.where(image_square[:, :, 3] > 0)
    max_row = max(used_indices[0])
    min_row = min(used_indices[0])
    max_col = max(used_indices[1])
    min_col = min(used_indices[1])

    # Make sure image is square
    min_ind = min(min_row, min_col)
    max_ind = max(max_row, max_col)

    minimal_image = np.zeros((max_ind, max_ind, 4))
    minimal_image[min_ind:max_ind, min_ind:max_ind, :] = image_square[min_ind:max_ind, min_ind:max_ind, :]
    minimal_image = image_square[min_ind:max_ind, min_ind:max_ind, :]

    result_image = Image.fromarray(minimal_image.astype(np.uint8))
    return result_image


if __name__ == '__main__':
    data_base_path = '/media/john/D/projects/platonicgan/datasets/CUB'
    images_base_path = os.path.join(data_base_path, 'images')
    annotations_base_path = os.path.join(data_base_path, 'annotations-mat')
    output_base_path = os.path.join(data_base_path, 'minimal_images')

    for class_name in tqdm(next(os.walk(images_base_path))[1]):
        class_path = os.path.join(images_base_path, class_name)

        target_class_path = os.path.join(output_base_path, class_name)

        if not os.path.exists(target_class_path):
            os.mkdir(target_class_path)

        for image_name in next(os.walk(class_path))[2]:
            if image_name[0] == '.':
                continue

            annotations_path = os.path.join(annotations_base_path, class_name, image_name.replace('.jpg', '.mat'))
            image, bounding_box, segmentation_image = load_img_parts(class_path, annotations_path, image_name)
            prepared_image = prepare_img(image, bounding_box, segmentation_image)

            output_path = os.path.join(target_class_path, image_name.replace('.jpg', '.png'))
            prepared_image.save(output_path)