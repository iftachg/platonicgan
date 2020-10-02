import os
from glob import glob
from tqdm import tqdm
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import numpy as np
from scipy import signal
from skimage.registration import phase_cross_correlation as cross_correlation


def get_ssim_list(real_images, random_images, maxed_images, sampled_images):
    random_ssim_list = []
    maxed_ssim_list = []
    sampled_ssim_list = []

    for real_image_path, random_image_path, maxed_image_path, sampled_image_path in tqdm(
            zip(real_images, random_images, maxed_images, sampled_images)):
        real_image = np.array(Image.open(real_image_path))
        random_image = np.array(Image.open(random_image_path))
        maxed_image = np.array(Image.open(maxed_image_path))
        sampled_image = np.array(Image.open(sampled_image_path))

        random_ssim = ssim(real_image, random_image, data_range=real_image.max() - real_image.min())
        maxed_ssim = ssim(real_image, maxed_image, data_range=real_image.max() - real_image.min())
        sampled_ssim = ssim(real_image, sampled_image, data_range=real_image.max() - real_image.min())

        random_ssim_list.append(random_ssim)
        maxed_ssim_list.append(maxed_ssim)
        sampled_ssim_list.append(sampled_ssim)

    return random_ssim_list, maxed_ssim_list, sampled_ssim_list


def get_frontal_images(views_path):
    real_images = glob(os.path.join(views_path, '*original*real*'))
    random_images = glob(os.path.join(views_path, '*front*random*'))
    maxed_images = glob(os.path.join(views_path, '*front*maxed*'))
    sampled_images = glob(os.path.join(views_path, '*front*sampled*'))
    real_images.sort()
    random_images.sort()
    maxed_images.sort()
    sampled_images.sort()

    return real_images, random_images, maxed_images, sampled_images


def get_images(views_path):
    real_images = glob(os.path.join(views_path, '*original*real*'))
    random_images = glob(os.path.join(views_path, '*_*_*_*random*'))
    maxed_images = glob(os.path.join(views_path, '*_*_*_*maxed*'))
    sampled_images = glob(os.path.join(views_path, '*_*_*_*sampled*'))
    real_images.sort()
    random_images.sort()
    maxed_images.sort()
    sampled_images.sort()

    return real_images, random_images, maxed_images, sampled_images


if __name__ == '__main__':
    base_path = '/media/john/D/projects/platonicgan/output'
    experiments_base = 'CUB_manual_base'
    experiment_names = [
                            '1_**_manual_class-downy_woodpecker_**_platonic_reconstruction_192.Downy_Woodpecker_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentFalse_28:09-16:38',
                            '1_**_manual_class-downy_woodpecker_**_platonic_reconstruction_192.Downy_Woodpecker_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentTrue_28:09-17:02',
                            '1_**_manual_species-tern_**_platonic_reconstruction_tern_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentFalse_25:09-16:19',
                            '1_**_manual_wings_**_platonic_reconstruction_wings_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentFalse_28:09-19:13',
                            '1_**_manual_wings_**_platonic_reconstruction_wings_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentTrue_28:09-22:54',
                        ]

    epochs = ['latest']

    for experiment_name in experiment_names:
        for epoch in epochs:
            eval_path = os.path.join(base_path, experiments_base, experiment_name, 'eval')
            views_path = os.path.join(eval_path, 'views3_{}'.format(epoch), 'list')
            metrics_path = os.path.join(eval_path, 'metrics')

            if not os.path.exists(metrics_path):
                os.mkdir(metrics_path)

            frontal_real_images, frontal_random_images, frontal_maxed_images, frontal_sampled_images = get_frontal_images(views_path)
            real_images, random_images, maxed_images, sampled_images = get_images(views_path)

            front_random_ssim_list, front_maxed_ssim_list, front_sampled_ssim_list = get_ssim_list(frontal_real_images, frontal_random_images, frontal_maxed_images, frontal_sampled_images)
            random_ssim_list, maxed_ssim_list, sampled_ssim_list = get_ssim_list(real_images, random_images, maxed_images, sampled_images)

            with open(os.path.join(metrics_path, 'metrics_{}.txt'.format(epoch)), 'w') as f:
                f.write('Average SSIM for frontal images: {} \n'.format(np.mean(front_random_ssim_list)))
                f.write('\n')
                f.write('Average SSIM for random view-points: {} \n'.format(np.mean(random_ssim_list)))
                f.write('Average SSIM for maxed view-points: {} \n'.format(np.mean(maxed_ssim_list)))
                f.write('Average SSIM for sampled view-points: {} \n'.format(np.mean(sampled_ssim_list)))
