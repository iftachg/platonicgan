import torch
from scripts.trainer.trainer_platonic import TrainerPlatonic
import itertools


class TrainerHolisticPlatonic(TrainerPlatonic):

    def __init__(self, param, dirs, test=False, init=True):
        super(TrainerHolisticPlatonic, self).__init__(param, dirs, test, init)

    def process_views(self, volume, image_real, target):

        views = []
        losses = []
        d_fakes = []
        gradient_penalties = []

        n_views = self.param.training.n_views

        if self.param.task == 'reconstruction':
            n_views += 1

        for idx in range(n_views):
            if idx == 0 and self.param.task == 'reconstruction':
                view = self.renderer.render(volume)
            else:
                view = self.renderer.render(self.transform.rotate_random(volume))
            d_fake, _ = self.discriminator(view)
            loss = self._compute_adversarial_loss(d_fake, target, self.param.training.adversarial_term_lambda_2d)
            holistic_losses = self.holistic_losses(d_fake, target)
            gp = self.gan_loss.gradient_penalty(image_real, view, self.discriminator)
            self.d_optimizer.zero_grad()

            self.logger.log_images('{}_{}'.format('view_output', idx), view)

            views.append(view)
            losses.append(loss)
            losses.extend(holistic_losses)
            gradient_penalties.append(gp)
            d_fakes.append(d_fake)

        return views, d_fakes, losses, gradient_penalties

    def holistic_losses(self, d_fake, target):
        target_permutations = torch.IntTensor(list(itertools.permutations(list(range(target.size(0))))))
        losses = []

        for permutated_inds in target_permutations:
            loss = self._compute_adversarial_loss(d_fake, target[permutated_inds], self.param.training.holistic_adversarial_term)
            losses.append(loss)

        losses = torch.cat(losses, dim=1).min(1)  # Take the minimal loss per generated sample

        return losses

