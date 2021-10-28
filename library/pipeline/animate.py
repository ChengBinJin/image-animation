import os
import imageio
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from library.dataset.frames_dataset import PairedDataset
from library.utils.logger.logger import Logger, Logger2, Visualizer, Visualizer2
from library.third_partys.sync_batchnorm import DataParallelWithCallback
from library.utils.keypoint import cat_dict, normalize_kp, normalize_kp2
from library.utils.files import get_name


def transfer_one(kp_detector, generator, source_image, driving_video, animate_params):
    # source_image:     (1, 3, 1, H, W)
    # driving_video:    (1, 3, n, H, W)
    num_frames = driving_video.shape[2]
    kp_driving = cat_dict([kp_detector(driving_video[:, :, i:(i + 1)]) for i in range(num_frames)], dim=1)
    # kp_driving    -   mean:   (1, n, num_kp, 2)
    #               -   var:    (1, n, num_kp, 2, 2)
    kp_source = kp_detector(source_image)
    # kp_source     - mean:     (1, 1, num_kp, 2)
    #               - var:      (1, 1, num_kp, 2, 2)

    kp_driving_norm = normalize_kp(kp_source, kp_driving, **animate_params['normalization_params'])
    # kp_driving_norm   - mean: (1, n, num_kp, 2)
    #                   - var:  (1, n, num_kp, 2, 2)
    kp_video_list = [{k: v[:, i:(i + 1)] for k, v in kp_driving_norm.items()} for i in range(num_frames)]

    out = cat_dict([generator(
        source_image=source_image, kp_source=kp_source, kp_driving=kp) for kp in kp_video_list], dim=2)
    # out   -   video_prediciton:   (1, 3, n, H, W)
    #       -   video_deformed:     (1, 3, n, H, W)
    out['kp_source'] = kp_source
    out['kp_driving'] = kp_driving
    out['kp_norm'] = kp_driving_norm

    return out


def animate2(config, kp_detector, generator, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'animate')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    animate_params = config['animate_params']
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger2.load_cpk(checkpoint, kp_detector=kp_detector, generator=generator)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if torch.cuda.is_available():
        kp_detector = DataParallelWithCallback(kp_detector)
        generator = DataParallelWithCallback(generator)

    kp_detector.eval()
    generator.eval()

    size = config['dataset_params']['frame_shape']
    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            predictions = []
            visualizations = []

            driving_video = x['driving_video']          # (1, 3, num_frames, H, W)
            source_frame = F.interpolate(x['source_img'][:, :, 0, :, :], size=size[0:2])

            kp_source = kp_detector(source_frame)
            driving_initial = F.interpolate(driving_video[:, :, 0], size=size[0:2])
            kp_driving_initial = kp_detector(driving_initial)

            num_frames = driving_video.shape[2]
            for frame_idx in range(num_frames):
                driving_frame = F.interpolate(driving_video[:, :, frame_idx], size=size[0:2])
                kp_driving = kp_detector(driving_frame)
                kp_norm = normalize_kp2(kp_source=kp_source, kp_driving=kp_driving,
                                        kp_driving_initial=kp_driving_initial, **animate_params['normalization_params'])
                out = generator(source_frame, kp_source=kp_source, kp_driving=kp_norm)

                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                out['kp_norm'] = kp_norm
                del out['sparse_deformed']

                predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                visualization = Visualizer2(**config['visualizer_params']).visualize(
                    source=source_frame, driving=driving_frame, out=out)
                visualizations.append(visualization)

            result_name = "-".join([get_name(x['driving_name'][0]), get_name(x['source_name'][0])])
            image_name = result_name + animate_params['format']
            fps = x['driving_fps'].data.cpu().numpy()[0]
            imageio.mimsave(os.path.join(log_dir, image_name), visualizations, fps=fps)


def animate(config, kp_detector, generator, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'animate')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    animate_params = config['animate_params']
    dataset = PairedDataset(initial_dataset=dataset, number_of_pairs=animate_params['num_pairs'])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, kp_detector=kp_detector, generator=generator)
    else:
        raise AttributeError("Checkpoint should be specified for mode='animate'.")

    if torch.cuda.is_available():
        kp_detector = DataParallelWithCallback(kp_detector)
        generator = DataParallelWithCallback(generator)

    kp_detector.eval()
    generator.eval()

    for it, x in tqdm(enumerate(dataloader)):
        with torch.no_grad():
            # x - source_video:     (1, 3, n, H, W)
            # x - source_name:      [string]
            # x - driving_video:    (1, 3, n, H, W)
            # x - driving_name:     [string]

            # x = {key: value if not hasattr(value, 'cuda') else value.cuda() for key, value in x.items()}
            driving_video = x['driving_video']  # (1, 3, n, H, W)
            source_image = x['source_video'][:, :, :1, :, :]  # (1, 3, 1, H, W)
            out = transfer_one(kp_detector, generator, source_image, driving_video, animate_params)
            # out - video_prediction:       (1, 3, n, H, W)
            #     - video_deformed:         (1, 3, n, H, W)
            #     - kp_source   - mean:     (1, 1, num_kp, 2)
            #                   - var:      (1, 1, num_kp, 2, 2)
            #     - kp_driving  - mean:     (1, n, num_kp, 2)
            #                   - var:      (1, n, num_kp, 2, 2)
            #     - kp_norm     - mean:     (1, n, num_kp, 2)
            #                   - var:      (1, n, num_kp, 2, 2)
            img_name = "-".join([get_name(x['source_name'][0]), get_name(x['driving_name'][0])])

            image = Visualizer(**config['visualizer_params']).visualize_animate(
                source_image=source_image, driving_video=driving_video, out=out)
            imageio.mimsave(os.path.join(log_dir, img_name + animate_params['format']), image)
