import torch
from scripts.utils.config import load_config
import os
import numpy as np
import scripts.utils.utils as utils
import glob
from scripts.trainer import TrainerPlatonic, Trainer3D, TrainerHolisticPlatonic, TrainerPlatonic3D
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from PIL import Image
from tqdm import tqdm

results_path_base = '/media/john/D/projects/platonicgan/output/CUB_manual_base'
experiment_names = [
    '1_**_manual_class-downy_woodpecker_**_holistic_reconstruction_192.Downy_Woodpecker_absorption_only_g2d1.0_g3d0.0_rec2d8.0_rec3d0.0_n_views2_lr_g0.0025_lr_d1e-05_bs4_random_augmentFalse_02:10-18:11',
   ]

epochs = ['500', '1000', 'latest']
epochs = ['5000']


class ViewSampler:

    def __init__(self):
        self.num_views = 5
        self.views_sample_pool_size = 100
        pass

    @staticmethod
    def get_a_view(trainer, volume):
        rotated_volume = trainer.transform.rotate_random(volume)
        view = trainer.renderer.render(rotated_volume)
        discriminator_score, _ = trainer.discriminator(view)

        return view, rotated_volume, discriminator_score

    def get_random_views(self, trainer, volume):
        views = []
        volumes = []

        for i in range(self.num_views):
            view, rotated_volume, _ = self.get_a_view(trainer, volume)
            views.append(view)
            volumes.append(rotated_volume)

        random_views_tensor = torch.stack(views, 1)
        volumes = torch.stack(volumes, 1)

        return random_views_tensor, volumes

    def score_views(self, trainer, volume):
        """
        Sample random views and record their discriminator scores
        """
        volumes_pool = []
        views_pool = []
        view_scores = []

        for i in range(self.views_sample_pool_size):
            view, rotated_volume, discriminator_score = self.get_a_view(trainer, volume)
            views_pool.append(view)
            view_scores.append(discriminator_score)
            volumes_pool.append(rotated_volume)

        view_scores = torch.cat(view_scores, 1)
        volumes_pool = torch.stack(volumes_pool, 1)
        views_pool = torch.stack(views_pool, 1)

        return volumes_pool, views_pool, view_scores

    def view_sampling(self, sample_scores):
        '''
        Sample indices based on sample_scores as weights
        '''
        normalized_sample_scores = (sample_scores - sample_scores.min()) / max(1, (sample_scores.max() - sample_scores.min()))  # Normalize to [0, 1]
        samples = torch.multinomial(normalized_sample_scores.squeeze(), self.num_views, replacement=False)  # Sample views based on score distribution

        return samples

    def view_max_sampling(self, sample_scores):
        '''
        Select top-k views
        '''

        sample_inds = sample_scores.squeeze().topk(self.num_views, dim=-1).indices
        return sample_inds

    @staticmethod
    def select_from_pools(views_pool, volumes_pool, inds):
        views = []
        volumes = []

        for batch_ind in range(inds.size(0)):
            current_sampled_views = []
            current_sampled_volumes = []

            for i, sample_ind in enumerate(inds[batch_ind]):
                sampled_view = views_pool[batch_ind][sample_ind]
                sampled_volume = volumes_pool[batch_ind][sample_ind]
                current_sampled_views.append(sampled_view)
                current_sampled_volumes.append(sampled_volume)

            sampled_views = torch.stack(current_sampled_views).unsqueeze(0)
            sampled_volumes = torch.stack(current_sampled_volumes).unsqueeze(0)
            views.append(sampled_views)
            volumes.append(sampled_volumes)

        views = torch.cat(views)
        volumes = torch.cat(volumes)

        return views, volumes

    def get_sampled_views(self, trainer, volume):
        volumes_pool, views_pool, view_scores = self.score_views(trainer, volume)
        sampled_inds = self.view_sampling(view_scores)
        views, volumes = self.select_from_pools(views_pool, volumes_pool, sampled_inds)

        return views, volumes

    def get_maxed_views(self, trainer, volume):
        volumes_pool, views_pool, view_scores = self.score_views(trainer, volume)
        sampled_inds = self.view_max_sampling(view_scores)
        views, volumes = self.select_from_pools(views_pool, volumes_pool, sampled_inds)

        return views, volumes


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


class TestManager:

    def __init__(self, config_file_path, epoch_name):
        self.param = load_config(config_path=config_file_path)  # Load params
        self.epoch_name = epoch_name

        self.config_dir = os.path.dirname(config_path)
        self.views_dir = self.prepare_result_dirs()  # Prepare folders for results

        self.load_trainer()
        self.view_sampler = ViewSampler()

    def prepare_result_dirs(self):
        eval_dir = self.config_dir.replace('stats', 'eval')
        views_dir = os.path.join(eval_dir, 'views_{}'.format(self.epoch_name))

        os.makedirs(views_dir, exist_ok=True)
        os.makedirs(os.path.join(views_dir, 'grid'), exist_ok=True)
        os.makedirs(os.path.join(views_dir, 'list'), exist_ok=True)

        return views_dir

    def trainer_selection(self):
        if self.param.mode == 'platonic':
            trainer = TrainerPlatonic(self.param, None, test=True)
        elif self.param.mode == 'platonic_3D':
            trainer = TrainerPlatonic3D(self.param, None, test=True)
        elif self.param.mode == '3D':
            trainer = Trainer3D(self.param, None, test=True)
        elif self.param.mode == 'holistic':
            trainer = TrainerHolisticPlatonic(self.param, None, test=True)
        else:
            raise NotImplementedError

        return trainer

    def load_trainer(self):
        self.trainer = self.trainer_selection()

        checkpoint = torch.load('{}/checkpoint_{}.pkl'.format(self.config_dir, self.epoch_name))

        for idx, model in enumerate(self.trainer.models):
            if model is not None:
                model.load_state_dict(checkpoint['model_{}'.format(str(idx))])

    def load_model_parts(self):
        self.load_trainer()

        self.encoder = self.trainer.models[0]
        self.generator = self.trainer.models[1]

    @staticmethod
    def prepend_view_parts(tensor, dim, *parts):
        for part in parts[::-1]:
            tensor = torch.cat((part.unsqueeze(dim), tensor), dim)

        return tensor

    def test(self):
        max_batches = min(400, len(self.trainer.data_loader_test))
        tqdm_obj = tqdm(total=max_batches)

        with torch.no_grad():
            for idx, (image, volume, vector, base_path, object_id, view_index, class_name) in enumerate(
                    self.trainer.data_loader_test):

                image_input = image.to(self.param.device)

                if not self.param.task == 'reconstruction':
                    z = self.trainer.encoder(image_input)
                else:
                    z = utils.generate_z(self.param, self.param.training.z_size)

                generated_volume = self.trainer.generator(z)

                frontal_view = self.trainer.renderer.render(generated_volume)

                # Generate views
                random_views_tensor, random_volumes = self.view_sampler.get_random_views(self.trainer, generated_volume)
                sampled_views_tensor, sampled_volumes = self.view_sampler.get_sampled_views(self.trainer, generated_volume)
                maxed_views_tensor, maxed_volumes = self.view_sampler.get_maxed_views(self.trainer, generated_volume)

                # Prepend real and frontal views
                random_views_tensor = self.prepend_view_parts(random_views_tensor, 1, image_input, frontal_view)
                sampled_views_tensor = self.prepend_view_parts(sampled_views_tensor, 1, image_input, frontal_view)
                maxed_views_tensor = self.prepend_view_parts(maxed_views_tensor, 1, image_input, frontal_view)

                empty_volume = torch.zeros(generated_volume.size()).to(generated_volume.device)
                random_volumes = self.prepend_view_parts(random_volumes, 1, empty_volume, generated_volume)
                sampled_volumes = self.prepend_view_parts(sampled_volumes, 1, empty_volume, generated_volume)
                maxed_volumes = self.prepend_view_parts(maxed_volumes, 1, empty_volume, generated_volume)

                save_views(self.views_dir, idx, image.unsqueeze(1), volumes=generated_volume.unsqueeze(1), postfix='real')
                save_views(self.views_dir, idx, random_views_tensor, random_volumes, postfix='random')
                save_views(self.views_dir, idx, sampled_views_tensor, sampled_volumes, postfix='sampled')
                save_views(self.views_dir, idx, maxed_views_tensor, maxed_volumes, postfix='maxed')

                tqdm_obj.update(1)

                if idx >= max_batches:
                    break


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


if __name__ == '__main__':
    for experiment_ind, experiment_name in enumerate(experiment_names):
        result_path = os.path.join(results_path_base, experiment_name)
        training_configs = glob.glob('{}/**/config.txt'.format(result_path), recursive=True)

        print('Experiment {}/{}'.format(experiment_ind, len(experiment_names)))
        print('Testing for experient {}'.format(experiment_name))
        for epoch_name in epochs:
            print('Testing epoch {}'.format(epoch_name))
            for idx_configs, config_path in enumerate(training_configs):

                print('config file {}/{}'.format(idx_configs + 1, len(training_configs)), config_path)

                test_manager = TestManager(config_path, epoch_name)


                test_manager.test()

