import torch
from scripts.utils.config import load_config
import os
import numpy as np
import scripts.utils.utils as utils
import glob
from scripts.trainer import TrainerPlatonic, Trainer3D, TrainerHolisticPlatonic
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

num_views = 5

results_path_base = '/media/john/D/projects/platonicgan/output/CUB_manual_base'
experiment_names = [
    '1_**_manual_class-downy_woodpecker_**_holistic_reconstruction_192.Downy_Woodpecker_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentFalse_02:10-12:47',
                        # '1_**_manual_class-downy_woodpecker_**_platonic_reconstruction_192.Downy_Woodpecker_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentFalse_28:09-16:38',
                        # '1_**_manual_class-downy_woodpecker_**_platonic_reconstruction_192.Downy_Woodpecker_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentTrue_28:09-17:02',
                        # '1_**_manual_species-tern_**_platonic_reconstruction_tern_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentFalse_25:09-16:19',
                        # '1_**_manual_wings_**_platonic_reconstruction_wings_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentFalse_28:09-19:13',
                        # '1_**_manual_wings_**_platonic_reconstruction_wings_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentTrue_28:09-22:54',
                    ]
epochs = ['500', '1000', 'latest']
epochs = ['latest']


def get_random_views(trainer, output_volume, x_input, reconstructions):
    views = [x_input, reconstructions]
    volumes = [torch.zeros(output_volume.size()).to(output_volume.device), output_volume]

    for i in range(num_views):
        volume = trainer.transform.rotate_random(output_volume)
        view = trainer.renderer.render(volume)
        views.append(view)
        volumes.append(volume)

    random_views_tensor = torch.stack(views, 1)
    volumes = torch.stack(volumes, 1)

    return random_views_tensor, volumes


def get_sampled_views(trainer, output_volume,x_input, reconstructions):
    views = []
    volumes = []

    # Sample 100 random views and record their discriminator scores
    unsampled_volumes = []
    unsampled_views = []
    view_scores = []
    for i in range(100):
        volume = trainer.transform.rotate_random(output_volume)
        view = trainer.renderer.render(volume)
        discriminator_score, _ = trainer.discriminator(view)
        unsampled_views.append(view)
        view_scores.append(discriminator_score)
        unsampled_volumes.append(volume)

    view_scores = torch.cat(view_scores, 1)

    # For every sample
    for in_batch_ind in range(view_scores.size(0)):
        sample_views = [x_input[in_batch_ind], reconstructions[in_batch_ind]]
        sampled_volumes = [torch.zeros(output_volume[in_batch_ind].size()).to(output_volume[in_batch_ind].device), output_volume[in_batch_ind]]

        sample_scores = view_scores[0]
        normalized_sample_scores = (sample_scores - sample_scores.min()) / max(1, (sample_scores.max() - sample_scores.min()))  # Normalize to [0, 1]

        samples = torch.multinomial(normalized_sample_scores.squeeze(), num_views, replacement=False)  # Sample views based on score distribution
        for i, sample_ind in enumerate(samples):
            view = unsampled_views[sample_ind][in_batch_ind]
            sample_views.append(view)
            sampled_volumes.append(unsampled_volumes[sample_ind][in_batch_ind])

        sample_views = torch.stack(sample_views).unsqueeze(0)
        sampled_volumes = torch.stack(sampled_volumes).unsqueeze(0)
        views.append(sample_views)
        volumes.append(sampled_volumes)

    views = torch.cat(views)
    volumes = torch.cat(volumes)

    return views, volumes


def get_maxed_views(trainer, output_volume,x_input, reconstructions):
    views = []
    volumes = []

    # Sample 100 random views and record their discriminator scores
    unsampled_views = []
    unsampled_volumes = []
    view_scores = []
    for i in range(100):
        volume = trainer.transform.rotate_random(output_volume)
        view = trainer.renderer.render(volume)
        discriminator_score, _ = trainer.discriminator(view)
        unsampled_views.append(view)
        view_scores.append(discriminator_score)
        unsampled_volumes.append(volume)

    view_scores = torch.cat(view_scores, 1).squeeze()

    # Select top-k views
    max_inds = view_scores.topk(num_views, dim=-1).indices
    for in_batch_ind in range(view_scores.size(0)):
        sample_views = [x_input[in_batch_ind], reconstructions[in_batch_ind]]
        sampled_volumes = [torch.zeros(output_volume[in_batch_ind].size()).to(output_volume[in_batch_ind].device), output_volume[in_batch_ind]]

        sample_topk = max_inds[in_batch_ind]
        for i, sample_ind in enumerate(range(num_views)):
            view = unsampled_views[sample_topk[sample_ind]][in_batch_ind]
            sample_views.append(view)
            sampled_volumes.append(unsampled_volumes[sample_topk[sample_ind]][in_batch_ind])

        sample_views = torch.stack(sample_views).unsqueeze(0)
        sampled_volumes = torch.stack(sampled_volumes).unsqueeze(0)
        views.append(sample_views)
        volumes.append(sampled_volumes)

    views = torch.cat(views)
    volumes = torch.cat(volumes)

    return views, volumes


def save_views(views_dir, idx, views_tensor, volumes, postfix):
    print('')
    print('Saving views for {}'.format(postfix))
    print('')

    for batch_ind in range(views_tensor.size(0)):
        sample_results = views_tensor[batch_ind] * 255

        if volumes is None:
            sample_volume = None
        else:
            sample_volume = volumes[batch_ind]

        save_grid(idx, postfix, batch_ind, sample_results, views_dir)
        save_list(idx, postfix, batch_ind, sample_results, sample_volume, views_dir)


def save_grid(idx, postfix, sample_id, sample_results, views_dir):
    sample_grid = make_grid(sample_results)
    result_image = Image.fromarray(sample_grid.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    target_path = os.path.join(views_dir, 'grid/{}_{}_{}.png'.format(idx, sample_id, postfix))
    result_image.save(target_path)


def save_list(idx, postfix, sample_id, sample_results, sample_volume, views_dir):
    for image_ind in range(sample_results.size(0)):
        print('Saving image {}/{}'.format(image_ind, sample_results.size(0)))
        if image_ind == 0:
            title = 'original'
        elif image_ind == 1:
            title = 'front'
        else:
            title = str(image_ind)

        image = sample_results[image_ind].squeeze()

        if len(image.size()) == 3:
            image = image.permute(1, 2, 0)

        image_matrix = image.cpu().numpy().astype(np.uint8)

        result_image = Image.fromarray(image_matrix)
        image_path = os.path.join(views_dir, 'list/{}_{}_{}_{}.png'.format(idx, sample_id, title, postfix))
        result_image.save(image_path)

        if sample_volume is None:
            return

        volume = sample_volume[image_ind]
        volume_path = os.path.join(views_dir, 'list/{}_{}_{}_{}_volume.png'.format(idx, sample_id, title, postfix))
        aligned_volume_path = os.path.join(views_dir, 'list/{}_{}_{}_{}_volume_aligned.png'.format(idx, sample_id, title, postfix))
        save_volume_view(volume, volume_path)
        save_volume_view(volume, aligned_volume_path, elev=90, azim=135)


def save_volume_view(volume, path, elev=0, azim=0):

    threshed_volume = volume.gt(0.5).int()

    fig = plt.figure()
    plt.xticks(rotation=elev)
    plt.yticks(rotation=azim)
    ax = fig.gca(projection='3d')
    ax.view_init(elev=elev, azim=azim)
    plt.axis('off')
    ax.voxels(threshed_volume[-1], facecolors='y', edgecolor='k')
    plt.savefig(path, bbox_inches='tight')
    plt.close()


for experiment_ind, experiment_name in enumerate(experiment_names):
    result_path = os.path.join(results_path_base, experiment_name)
    training_configs = glob.glob('{}/**/config.txt'.format(result_path), recursive=True)

    print('Experiment {}/{}'.format(experiment_ind, len(experiment_names)))
    print('Testing for experient {}'.format(experiment_name))
    for load_epoch in epochs:
        print('Testing epoch {}'.format(load_epoch))
        for idx_configs, config_path in enumerate(training_configs):

            print('config file {}/{}'.format(idx_configs + 1, len(training_configs)), config_path)

            param = load_config(config_path=config_path)

            config_dir = os.path.dirname(config_path)

            eval_dir = config_dir.replace('stats', 'eval')
            views_dir = os.path.join(eval_dir, 'views_{}'.format(load_epoch))

            os.makedirs(eval_dir, exist_ok=True)
            os.makedirs(views_dir, exist_ok=True)
            os.makedirs(os.path.join(views_dir, 'grid'), exist_ok=True)
            os.makedirs(os.path.join(views_dir, 'list'), exist_ok=True)

            # training trainer
            if param.mode == 'platonic':
                trainer = TrainerPlatonic(param, None, True)
            elif param.mode == 'platonic_3D':
                trainer = TrainerPlatonic3D(param, None, True)
            elif param.mode == '3D':
                trainer = Trainer3D(param, None, True)
            elif param.mode == 'holistic':
                trainer = TrainerHolisticPlatonic(param, None, True)
            else:
                raise NotImplementedError

            checkpoint = torch.load('{}/checkpoint_{}.pkl'.format(config_dir, load_epoch))

            for idx, model in enumerate(trainer.models):
                if model is not None:
                    model.load_state_dict(checkpoint['model_{}'.format(str(idx))])

            encoder = trainer.models[0]
            generator = trainer.models[1]

            n_outputs_max = min(400, len(trainer.data_loader_test))
            n_outputs_counter = 0
            tqdm_obj = tqdm(total=n_outputs_max)
            object_list = []

            with torch.no_grad():
                for idx, (image, volume, vector, base_path, object_id, view_index, class_name) in enumerate(
                        trainer.data_loader_test):

                    # print('{}/{}: {}/{}'.format(idx_configs + 1, len(training_configs), n_outputs_counter + 1,
                    #                             int(trainer.dataset_test.__len__() / param.training.batch_size)))

                    x_input = image.to(param.device)
                    volume = volume.to(param.device)
                    vector = vector.to(param.device)

                    if not param.task == 'reconstruction':
                        z = encoder(x_input)
                    else:
                        z = utils.generate_z(param, param.training.z_size)

                    output_volume = generator(z)

                    reconstructions = trainer.renderer.render(output_volume)

                    random_views_tensor, random_volumes = get_random_views(trainer, output_volume, x_input, reconstructions)
                    sampled_views_tensor, sampled_volumes = get_sampled_views(trainer, output_volume, x_input, reconstructions)
                    maxed_views_tensor, maxed_volumes = get_maxed_views(trainer, output_volume, x_input, reconstructions)

                    save_views(views_dir, idx, image.unsqueeze(1), volumes=output_volume.unsqueeze(1), postfix='real')
                    save_views(views_dir, idx, random_views_tensor, random_volumes, postfix='random')
                    save_views(views_dir, idx, sampled_views_tensor, sampled_volumes, postfix='sampled')
                    save_views(views_dir, idx, maxed_views_tensor, maxed_volumes, postfix='maxed')
                    a = 5

                    tqdm_obj.update(1)

                    if n_outputs_counter >= n_outputs_max:
                        break
