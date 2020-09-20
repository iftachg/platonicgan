import torch
from scripts.utils.config import load_config, make_dirs
import os, sys
import cv2
import scripts.renderer.transform as dt
import scripts.utils.io as dh
import numpy as np
import scripts.utils.utils as utils
import glob
import scripts.utils.logger as log
from scripts.trainer import TrainerPlatonic, Trainer3D
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from torch.distributions.multinomial import Multinomial

num_views = 5

# TODO: change to output path
result_path = os.path.join(os.path.expanduser('~'), 'share', 'tmp')
# result_path = os.path.join(os.path.expanduser('~'), 'share', 'tmp')
result_path = '/media/john/D/projects/platonicgan/output/1_Tree_platonic_reconstruction_tree_emission_absorption_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views1_lr_g0.0025_lr_d1e-05_bs8_random'
# result_path = '/media/john/D/projects/platonicgan/output/1_Bird_platonic_reconstruction_001.Black_footed_Albatross_emission_absorption_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random'
training_configs = glob.glob('{}/**/config.txt'.format(result_path), recursive=True)


def get_random_views(trainer, output_volume,x_input, reconstructions):
    views = [x_input, reconstructions]
    for i in range(num_views):
        view = trainer.renderer.render(trainer.transform.rotate_random(output_volume))
        views.append(view)

    random_views_tensor = torch.stack(views, 1)

    return random_views_tensor


def get_sampled_views(trainer, output_volume,x_input, reconstructions):
    views = []


    # Sample 100 random views and record their discriminator scores
    unsampled_views = []
    view_scores = []
    for i in range(100):
        view = trainer.renderer.render(trainer.transform.rotate_random(output_volume))
        discriminator_score, _ = trainer.discriminator(view)
        unsampled_views.append(view)
        view_scores.append(discriminator_score)

    view_scores = torch.cat(view_scores, 1)

    # For every sample
    for in_batch_ind in range(view_scores.size(0)):
        sample_views = [x_input[in_batch_ind], reconstructions[in_batch_ind]]

        sample_scores = view_scores[0]
        normalized_sample_scores = (sample_scores - sample_scores.min()) / max(1, (sample_scores.max() - sample_scores.min()))  # Normalize to distrbution

        samples = torch.multinomial(normalized_sample_scores.squeeze(), num_views, replacement=False)  # Sample views based on score distribution
        for i, sample_ind in enumerate(samples):
            view = unsampled_views[sample_ind][in_batch_ind]
            sample_views.append(view)

        sample_views = torch.stack(sample_views).unsqueeze(0)
        views.append(sample_views)

    views = torch.cat(views)

    return views


def get_maxed_views(trainer, output_volume,x_input, reconstructions):
    views = []

    # Sample 100 random views and record their discriminator scores
    unsampled_views = []
    view_scores = []
    for i in range(100):
        view = trainer.renderer.render(trainer.transform.rotate_random(output_volume))
        discriminator_score, _ = trainer.discriminator(view)
        unsampled_views.append(view)
        view_scores.append(discriminator_score)

    view_scores = torch.cat(view_scores, 1).squeeze()

    # Select top-k views
    max_inds = view_scores.topk(num_views, dim=-1).indices
    for in_batch_ind in range(view_scores.size(0)):
        sample_views = [x_input[in_batch_ind], reconstructions[in_batch_ind]]

        sample_topk = max_inds[in_batch_ind]
        for i, sample_ind in enumerate(range(num_views)):
            view = unsampled_views[sample_topk[sample_ind]][in_batch_ind]
            sample_views.append(view)

        sample_views = torch.stack(sample_views).unsqueeze(0)
        views.append(sample_views)

    views = torch.cat(views)
    return views


def save_views(views_dir, idx, views_tensor, postfix):
    for sample_id in range(views_tensor.size(0)):
        sample_results = views_tensor[sample_id] * 255
        sample_grid = make_grid(sample_results)
        result_image = Image.fromarray(sample_grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
        target_path = os.path.join(views_dir, '{}_{}_{}.png'.format(idx, sample_id, postfix))

        result_image.save(target_path)


for idx_configs, config_path in enumerate(training_configs):

    print('{}/{}'.format(idx_configs + 1, len(training_configs)), config_path)

    param = load_config(config_path=config_path)

    config_dir = os.path.dirname(config_path)

    eval_dir = config_dir.replace('stats', 'eval')
    views_dir = os.path.join(eval_dir, 'views')

    # if os.path.exists(eval_dir):
    #     print('Skip')
    #     continue

    os.makedirs(eval_dir, exist_ok=True)
    os.makedirs(views_dir, exist_ok=True)

    # training trainer
    if param.mode == 'platonic':
        trainer = TrainerPlatonic(param, None, True)
    elif param.mode == 'platonic_3D':
        trainer = TrainerPlatonic3D(param, None, True)
    elif param.mode == '3D':
        trainer = Trainer3D(param, None, True)
    else:
        raise NotImplementedError

    checkpoint = torch.load('{}/checkpoint.pkl'.format(config_dir))

    for idx, model in enumerate(trainer.models):
        if model is not None:
            model.load_state_dict(checkpoint['model_{}'.format(str(idx))])

    encoder = trainer.models[0]
    generator = trainer.models[1]

    n_outputs_max = 400
    n_outputs_counter = 0

    object_list = []

    with torch.no_grad():
        for idx, (image, volume, vector, base_path, object_id, view_index, class_name) in enumerate(
                trainer.data_loader_test):

            print('{}/{}: {}/{}'.format(idx_configs + 1, len(training_configs), n_outputs_counter + 1,
                                        int(trainer.dataset_test.__len__() / param.training.batch_size)))

            x_input = image.to(param.device)
            volume = volume.to(param.device)
            vector = vector.to(param.device)

            if not param.task == 'reconstruction':
                z = encoder(x_input)
            else:
                z = utils.generate_z(param, param.training.z_size)

            output_volume = generator(z)

            reconstructions = trainer.renderer.render(output_volume)

            random_views_tensor = get_random_views(trainer, output_volume, x_input, reconstructions)
            sampled_views_tensor = get_sampled_views(trainer, output_volume, x_input, reconstructions)
            maxed_views_tensor = get_maxed_views(trainer, output_volume, x_input, reconstructions)

            save_views(views_dir, idx, random_views_tensor, postfix='random')
            save_views(views_dir, idx, sampled_views_tensor, postfix='sampled')
            save_views(views_dir, idx, maxed_views_tensor, postfix='maxed')

            a = 5
            # threshed_volume = output_volume.gt(0.5).int()
            # fig = plt.figure()
            #
            # ax = fig.gca(projection='3d')
            # ax.voxels(threshed_volume[0][3], facecolors='y', edgecolor='k')
            # plt.show()
            # plt.close()
            # a = 5