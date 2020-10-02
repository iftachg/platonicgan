import torch
import torch.utils.data
import torch.utils.data.distributed
from torch import autograd
import numpy as np
import scripts.utils.logger as log
from torch import optim
from scripts.data import dataset_dict
from scripts.renderer import renderer_dict
from scripts.models import generator_dict, discriminator_dict, encoder_dict
import scripts.renderer.transform as dt
from scripts.models.default import Discriminator, Encoder, Generator2D
from scripts.trainer.trainer import GANLoss
import scripts.utils.logger as log


class ImageGAN:

    def __init__(self, param, dirs, test=False, init=True):

        self.param = param
        self.iteration = 0

        if init:
            self.datal_loader_train, self.dataset_train = self.setup_dataset('train')
            self.data_loader_val, self.dataset_val = self.setup_dataset('val')
            self.data_loader_test, self.dataset_test = self.setup_dataset('test')

            self.setup_models()

            self.transform = dt.Transform(param)

            if not test:
                self.setup_optimizers()
                self.logger = log.Logger(dirs, param, None, self.transform, self.models, self.optimizers)
                self.gan_loss = GANLoss(param, param.training.loss)
                self.criterion_data_term = torch.nn.MSELoss()

                self.dirs = dirs

    def sample_noise(self, batch_size):
        point = torch.randn(batch_size,  self.param.training.z_size)
        point = (point / point.norm()) * (self.param.training.z_size ** 0.5)

        return point

    def setup_models(self):
        self.encoder = Encoder(self.param)
        self.generator = Generator2D(self.param)
        self.discriminator_2d = self.discriminator_2d(self.param)

        self.encoder = self.encoder.to(self.param.device)
        self.generator = self.generator.to(self.param.device)
        self.discriminator_2d = self.discriminator_2d.to(self.param.device)

        self.models = [self.encoder, self.generator, self.discriminator_2d]

    def setup_optimizers(self):
        print("[INFO] generate optimizers")

        optimizer_g = self.param.training.optimizer_g
        optimizer_d = self.param.training.optimizer_d
        lr_g = self.param.training.lr_g
        lr_d = self.param.training.lr_d

        if self.param.task == 'reconstruction':
            g_params = list(self.encoder.parameters()) + list(self.generator.parameters())
        else:
            g_params = list(self.generator.parameters())

        d_params_2d = filter(lambda p: p.requires_grad, self.models[2].parameters())
        if optimizer_d == 'rmsprop':
            d_optimizer_2d = optim.RMSprop(d_params_2d, lr=lr_d, alpha=0.99, eps=1e-8)
        elif optimizer_d == 'adam':
            d_optimizer_2d = optim.Adam(d_params_2d, lr=lr_d, betas=(0.5, 0.9))
        elif optimizer_d == 'sgd':
            d_optimizer_2d = optim.SGD(d_params_2d, lr=lr_d)
        else:
            raise NotImplementedError

        if optimizer_g == 'rmsprop':
            g_optimizer = optim.RMSprop(g_params, lr=lr_g, alpha=0.99, eps=1e-8)
        elif optimizer_g == 'adam':
            g_optimizer = optim.Adam(g_params, lr=lr_g, betas=(0.5, 0.9))
        elif optimizer_g == 'sgd':
            g_optimizer = optim.SGD(g_params, lr=lr_g)
        else:
            raise NotImplementedError

        self.optimizers = [g_optimizer, d_optimizer_2d]
        self.g_optimizer = g_optimizer
        self.d_optimizer_2d = d_optimizer_2d

    def encoder_train(self, x):
        self.encoder.train()
        z = self.encoder(x)

        return z

    def set_iteration(self, iteration):
        self.iteration = iteration

    def _compute_adversarial_loss(self, output, target, weight):
        return self.gan_loss(output, target) * weight

    def _compute_data_term_loss(self, output, target, weight):
        return self.criterion_data_term(output, target) * weight

    def generator_train(self, image, z):
        self.generator.train()
        self.g_optimizer.zero_grad()

        data_loss = torch.tensor(0.0).to(self.param.device)

        fake_image = self.generator(z)

        if self.param.task == 'reconstruction':
            data_loss += self._compute_data_term_loss(fake_image, image, self.param.training.data_term_lambda_2d)

        g_loss = data_loss

        g_loss.backward()
        self.g_optimizer.step()

        ### log
        # print("  Generator loss 2d: {:2.4}".format(g_loss))
        self.logger.log_scalar('g_2d_loss', g_loss.item())
        self.logger.log_scalar('g_2d_rec_loss', data_loss.item())
        self.logger.log_images('image_input', image)

    def discriminator_train(self, image, z):
        self.discriminator_2d.train()
        self.d_optimizer_2d.zero_grad()

        d_real, _ = self.discriminator_2d(image)
        d_real_loss = self._compute_adversarial_loss(d_real, 1, self.param.training.adversarial_term_lambda_2d)

        with torch.no_grad():
            fake_image = self.generator(z)

        d_fake, _ = self.discriminator_2d(fake_image)
        loss = self._compute_adversarial_loss(d_fake, 0, self.param.training.adversarial_term_lambda_2d)
        gradient_penalty = self.gan_loss.gradient_penalty(image, fake_image, self.discriminator_2d)
        self.d_optimizer_2d.zero_grad()

        # gradient_penalty = torch.mean(torch.stack(gradient_penalties))
        d_fake_loss = loss + gradient_penalty
        d_loss = d_real_loss + d_fake_loss

        if self.param.training.loss == 'vanilla':
            d_real = torch.sigmoid(d_real)
            d_fakes = list(map(torch.sigmoid, [d_fake]))

            d_real_accuracy = torch.mean(d_real)
            d_fake_accuracy = 1.0 - torch.mean(torch.stack(d_fakes))
            d_accuracy = ((d_real_accuracy + d_fake_accuracy) / 2.0).item()
        else:
            d_accuracy = 0.0

        # only update discriminator if accuracy <= d_thresh
        if d_accuracy <= self.param.training.d_thresh or self.param.training.loss == 'wgangp':
            d_loss.backward()
            self.d_optimizer_2d.step()
            # print("  *Discriminator 2d update*")

        ### log
        # print("  Discriminator2d loss: {:2.4}".format(d_loss))
        self.logger.log_scalar('d_2d_loss', d_loss.item())
        self.logger.log_scalar('d_2d_real', torch.mean(d_real).item())
        self.logger.log_scalar('d_2d_fake', torch.mean(torch.stack([d_fake])).item())
        self.logger.log_scalar('d_2d_real_loss', d_real_loss.item())
        self.logger.log_scalar('d_2d_fake_loss', d_fake_loss.item())
        self.logger.log_scalar('d_2d_accuracy', d_accuracy)
        self.logger.log_scalar('d_2d_gradient_penalty', gradient_penalty.item())
