import torch


def mean_batch(val):
    out = val.view(val.shape[0], -1).mean(-1)
    return out


def reconstruction_loss(prediction, target, weight):
    if weight == 0:
        out = 0
    else:
        out = weight * mean_batch(torch.abs(prediction - target))
    return out


def generator_gan_loss(discriminator_maps_generated, weight):
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_generated) ** 2
    score = weight * mean_batch(score)
    return score


def discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real, weight):
    scores_real = discriminator_maps_real[-1]
    scores_generated = discriminator_maps_generated[-1]
    score = (1 - scores_real) ** 2 + scores_generated ** 2  # lsgan
    score = weight * mean_batch(score)
    return score


def generator_loss_names(loss_weights):
    loss_names = list()
    if loss_weights['reconstruction'] != 0:
        loss_names.append("rec_def")

    if loss_weights['feature_matching'] is not None:
        for i, _ in enumerate(loss_weights['feature_matching']):
            if loss_weights['feature_matching'][i] == 0:
                continue
            loss_names.append(f"layer-{str(i).zfill(2)}_feature_matching")
    loss_names.append("gen_gan")

    return loss_names


def discriminator_loss_names():
    return ['disc_gan']


def generator_loss(discriminator_maps_generated, discriminator_maps_real, video_deformed, loss_weights):
    loss_values = list()

    if loss_weights['reconstruction'] != 0:
        loss_values.append(
            reconstruction_loss(discriminator_maps_real[0], video_deformed, loss_weights['reconstruction']))

    if loss_weights['feature_matching'] != 0:
        for i, (a, b) in enumerate(zip(discriminator_maps_real[:-1], discriminator_maps_generated[:-1])):
            if loss_weights['feature_matching'][i] == 0:
                continue
            else:
                loss_values.append(reconstruction_loss(b, a, weight=loss_weights['feature_matching'][i]))

    loss_values.append(generator_gan_loss(discriminator_maps_generated, weight=loss_weights['generator_gan']))

    return loss_values


def discriminator_loss(discriminator_maps_generated, discriminator_maps_real, loss_weights):
    loss_values = [discriminator_gan_loss(discriminator_maps_generated, discriminator_maps_real,
                                          loss_weights['discriminator_gan'])]
    return loss_values
