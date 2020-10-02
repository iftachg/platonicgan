import torch
from scripts.trainer.trainer_platonic import TrainerPlatonic
import itertools


class TrainerHolisticPlatonic(TrainerPlatonic):

    def __init__(self, param, dirs, test=False, init=True):
        print("[INFO] setup TrainerPlatonic")
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
            gp = self.gan_loss.gradient_penalty(image_real, view, self.discriminator)
            holistic_loss = self.holistic_losses(image_real, view)
            self.d_optimizer.zero_grad()

            self.logger.log_images('{}_{}'.format('view_output', idx), view)

            views.append(view)
            losses.append(loss)
            gradient_penalties.append(holistic_loss)
            gradient_penalties.append(gp)
            d_fakes.append(d_fake)

        return views, d_fakes, losses, gradient_penalties

    def holistic_losses(self, image_real, view):
        target_permutations = torch.LongTensor(list(itertools.permutations(list(range(image_real.size(0))))))
        losses = []

        for permutated_inds in target_permutations:
            loss = self.gan_loss.gradient_penalty(image_real[permutated_inds], view, self.discriminator)
            losses.append(loss)

        loss = torch.min(torch.stack(losses)) * self.param.training.holistic_adversarial_term  #  Take the minimal loss per generated sample
        return loss

