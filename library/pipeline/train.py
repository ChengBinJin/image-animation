import time
import torch
import numpy as np

from tqdm import trange
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader

from library.utils.logger.logger import Logger, Logger2
from library.modules.model import GeneratorFullModel, DiscriminatorFullModel, GeneratorFullModel2, DiscriminatorFullModel2
from library.third_partys.sync_batchnorm import DataParallelWithCallback
from library.utils.loss import generator_loss_names, discriminator_loss_names
from library.dataset.augmentation import worker_init_fn
from library.dataset.frames_dataset import DatasetRepeater


def train2(config, kp_detector, generator, discriminator, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']

    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr_kp_detector'],
                                             betas=(0.5, 0.999))
    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr_generator'],
                                           betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr_discriminator'],
                                               betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch = Logger2.load_cpk(checkpoint, kp_detector, generator, discriminator,
                                       None if train_params['lr_kp_detector'] == 0 else optimizer_kp_detector,
                                       optimizer_generator, optimizer_discriminator)
    else:
        start_epoch = 0

    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=-1 + start_epoch * (train_params['lr_kp_detector'] != 0))
    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch - 1)
    scheduler_discrimiantor = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch - 1)

    if 'num_repeats' in train_params or train_params['num_repeats'] != 1:
        dataset = DatasetRepeater(dataset, train_params['num_repeats'])
    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=6, drop_last=True)

    generator_full = GeneratorFullModel2(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel2(kp_detector, generator, discriminator, train_params)

    if torch.cuda.is_available():
        generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
        discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger2(log_dir=log_dir, visualizer_params=config['visualizer_params'],
                 checkpoint_freq=train_params['checkpoint_freq']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            for x in dataloader:
                losses_generator, generated = generator_full(x)

                loss_values = [val.mean() for val in losses_generator.values()]
                loss = sum(loss_values)

                loss.backward()
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_kp_detector.step()
                optimizer_kp_detector.zero_grad()

                if train_params['loss_weights']['generator_gan'] != 0:
                    optimizer_discriminator.zero_grad()
                    losses_discriminator = discriminator_full(x, generated)
                    loss_values = [val.mean() for val in losses_discriminator.values()]
                    loss = sum(loss_values)

                    loss.backward()
                    optimizer_discriminator.step()
                    optimizer_discriminator.zero_grad()
                else:
                    losses_discrimiantor = {}

                losses_generator.update(losses_discriminator)
                losses = {key: value.mean().detach().data.cpu().numpy() for key, value in losses_generator.items()}
                logger.log_iter(losses=losses)

            scheduler_kp_detector.step()
            scheduler_generator.step()
            scheduler_discrimiantor.step()

            logger.log_epoch(epoch, {
                'kp_detector': kp_detector,
                'generator': generator,
                'discriminator': discriminator,
                'optimizer_kp_detector': optimizer_kp_detector,
                'optimizer_generator': optimizer_generator,
                'optimizer_discriminator': optimizer_discriminator
            }, inp=x, out=generated)


def train(config, kp_detector, generator, discriminator, checkpoint, log_dir, dataset, device_ids):
    train_params = config['train_params']
    initial_seed = int(time.time())

    optimizer_generator = torch.optim.Adam(generator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
    optimizer_kp_detector = torch.optim.Adam(kp_detector.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

    if checkpoint is not None:
        start_epoch, it = Logger.load_cpk(checkpoint, kp_detector, generator, discriminator, optimizer_kp_detector,
                                          optimizer_generator, optimizer_discriminator)
    else:
        start_epoch, it = 0, 0

    scheduler_generator = MultiStepLR(optimizer_generator, train_params['epoch_milestones'], gamma=0.1,
                                      last_epoch=start_epoch-1)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, train_params['epoch_milestones'], gamma=0.1,
                                          last_epoch=start_epoch-1)
    scheduler_kp_detector = MultiStepLR(optimizer_kp_detector, train_params['epoch_milestones'], gamma=0.1,
                                        last_epoch=start_epoch-1)

    dataloader = DataLoader(dataset, batch_size=train_params['batch_size'], shuffle=True, num_workers=4,
                            worker_init_fn=worker_init_fn, drop_last=True)

    generator_full = GeneratorFullModel(kp_detector, generator, discriminator, train_params)
    discriminator_full = DiscriminatorFullModel(kp_detector, generator, discriminator, train_params)

    generator_full = DataParallelWithCallback(generator_full, device_ids=device_ids)
    discriminator_full = DataParallelWithCallback(discriminator_full, device_ids=device_ids)

    with Logger(log_dir=log_dir, visualizer_params=config['visualizer_params'], **train_params['log_params']) as logger:
        for epoch in trange(start_epoch, train_params['num_epochs']):
            np.random.seed(initial_seed + epoch)

            for x in dataloader:
                out = generator_full(x)
                loss_values = out[:-2]
                generated = out[-2]
                kp_joined = out[-1]
                loss_values = [val.mean() for val in loss_values]
                loss = sum(loss_values)

                loss.backward(retain_graph=not train_params['detach_kp_discriminator'])
                optimizer_generator.step()
                optimizer_generator.zero_grad()
                optimizer_discriminator.zero_grad()
                if train_params['detach_kp_discriminator']:
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()

                generator_loss_values = [val.detach().cpu().numpy() for val in loss_values]

                loss_values = discriminator_full(x, kp_joined, generated)
                loss_values = [val.mean() for val in loss_values]
                loss = sum(loss_values)

                loss.backward()
                optimizer_discriminator.step()
                optimizer_discriminator.zero_grad()
                if not train_params['detach_kp_discriminator']:
                    optimizer_kp_detector.step()
                    optimizer_kp_detector.zero_grad()

                discriminator_loss_values = [val.detach().cpu().numpy() for val in loss_values]

                logger.log_iter(it,
                                names=generator_loss_names(train_params['loss_weights']) + discriminator_loss_names(),
                                values=generator_loss_values + discriminator_loss_values, inp=x, out=generated)

                it += 1

            scheduler_kp_detector.step()
            scheduler_generator.step()
            scheduler_discriminator.step()

            logger.log_epoch(epoch, {
                'kp_detector': kp_detector,
                'generator': generator,
                'discriminator': discriminator,
                'optimizer_kp_detector': optimizer_kp_detector,
                'optimizer_generator': optimizer_generator,
                'optimizer_discriminator': optimizer_discriminator,
            })
