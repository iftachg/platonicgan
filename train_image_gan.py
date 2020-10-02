import time
from image_gan.image_gan import ImageGAN
from scripts.utils.config import load_config, make_dirs

from tqdm import tqdm

# training setup
param = load_config()
dirs = make_dirs(param)

image_gan = ImageGAN(param, dirs)

print('[Training] training start:')

for epoch in tqdm(range(1, param.training.n_epochs + 1), desc='Training epoch'):
    start_time = time.time()

    tqdm_obj = tqdm(total=len(image_gan.datal_loader_train), desc='Inner epoch {}'.format(epoch))
    for idx, (image, volume, vector, base_path, object_id, view_index, class_name) in enumerate(image_gan.datal_loader_train):

        image = image.to(param.device)
        noise = image_gan.sample_noise()

        image_gan.generator_train(image, noise)
        image_gan.discriminator_train(image, noise)

        image_gan.logger.log_checkpoint(image_gan.models, image_gan.optimizers)
        if epoch % 50 == 0:
            image_gan.logger.log_checkpoint(image_gan.models, image_gan.optimizers, epoch=epoch)

        for model in image_gan.models:
            image_gan.logger.log_gradients(model)

        image_gan.logger.step()
        tqdm_obj.update(1)

print('Training finished!...')
