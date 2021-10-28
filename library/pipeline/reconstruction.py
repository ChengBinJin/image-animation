import os
import imageio
import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

from library.third_partys.sync_batchnorm import DataParallelWithCallback
from library.utils.logger.logger import Logger, Logger2, Visualizer, Visualizer2
from library.utils.loss import reconstruction_loss
from library.utils.files import get_name


def generate(generator, source, kp_source, kp_drivings):
    # source:                   (1, 3, 1, H, W)
    # kp_source     - mean:     (1, 1, num_kp, 2)
    #               - variance: (1, 1, num_kp, 2, 2)
    # kp_drivings   - mean:     (1, n, num_kp, 2)
    # kp_drivings   - variance: (1, n, num_kp, 2, 2)

    out = {'video_prediction': [], 'video_deformed': []}
    for i in range(kp_drivings['mean'].shape[1]):
        kp_target = {k: v[:, i:(i + 1)] for k, v in kp_drivings.items()}
        kp_dict_part = {'kp_driving': kp_target, 'kp_source': kp_source}
        out_part = generator(source, **kp_dict_part)
        out['video_prediction'].append(out_part['video_prediction'])    # [(1, 3, 1, H, W)]
        out['video_deformed'].append(out_part['video_deformed'])        # [(1, 3, 1, H, W)]

    out['video_prediction'] = torch.cat(out['video_prediction'], dim=2)
    out['video_deformed'] = torch.cat(out['video_deformed'], dim=2)
    out['kp_driving'] = kp_drivings
    out['kp_source'] = kp_source
    return out


def reconstruction2(config, kp_detector, generator, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'reconstruction')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if checkpoint is not None:
        Logger2.load_cpk(checkpoint, kp_detector=kp_detector, generator=generator)
    else:
        raise AttributeError("Checkpoint should be specified for mode='reconstruction'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    loss_list = []
    if torch.cuda.is_available():
        kp_detector = DataParallelWithCallback(kp_detector)
        generator = DataParallelWithCallback(generator)

    kp_detector.eval()
    generator.eval()

    size = config['dataset_params']['frame_shape']
    for it, x in tqdm(enumerate(dataloader)):
        print(f"video name: {x['name'][0]}")
        eval_size = x['video'][:, :, 0].shape[-2:]

        with torch.no_grad():
            if torch.cuda.is_available():
                x['video'] = x['video'].cuda()

            source_ori = x['video'][:, :, 0]
            source = F.interpolate(source_ori, size=size[0:2])
            kp_source = kp_detector(source)
            for frame_idx in range(x['video'].shape[2]):
                input_driving = F.interpolate(x['video'][:, :, frame_idx], size=size[0:2])
                eval_driving = x['video'][:, :, frame_idx]
                kp_driving = kp_detector(input_driving)

                out = generator(source, kp_source=kp_source, kp_driving=kp_driving)
                out['kp_source'] = kp_source
                out['kp_driving'] = kp_driving
                del out['sparse_deformed']

                if size != eval_size:
                    eval_out = F.interpolate(out['prediction'], size=eval_size[:2])
                else:
                    eval_out = out['prediction']

                # predictions.append(np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0])
                # visualization = Visualizer2(**config['visualizer_params']).visualize(
                #     source=source, driving=driving, out=out)
                # visualizations.append(visualization)

                loss_list.append(torch.abs(eval_out - eval_driving).mean().cpu().numpy())
                # lpips_list.append(cal_lpips(lpips_net, eval_driving, eval_out))

            # predictions = np.concatenate(predictions, axis=1)

            # image_name = x['name'][0] + config['reconstruction_params']['format']
            # imageio.mimsave(os.path.join(log_dir, image_name), visualizations)

    print(f"Reconstruction loss: {np.mean(loss_list):.3f}")
    # print(f"LPIPS: {np.mean(lpips_list):.3f}")


def reconstruction(config, kp_detector, generator, checkpoint, log_dir, dataset):
    log_dir = os.path.join(log_dir, 'reconstruction')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if checkpoint is not None:
        Logger.load_cpk(checkpoint, kp_detector=kp_detector, generator=generator)
    else:
        raise AttributeError(" [!] Checkpoint should be specified for mode='test'.")
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    loss_list = []
    kp_detector = DataParallelWithCallback(kp_detector)
    generator = DataParallelWithCallback(generator)

    kp_detector.eval()
    generator.eval()

    cat_dict = lambda l, dim: {k: torch.cat([v[k] for v in l], dim=dim) for k in l[0]}
    for it, x in tqdm(enumerate(dataloader)):
        # x video: [1, 3, N, 256, 256]
        #   name: [name]
        with torch.no_grad():
            kp_source = kp_detector(x['video'][:, :, :1])
            # kp_appearance     - mean:     (1, 1, mnum_kp, 2)
            #                   - var:      (1, 1, num_kp, 2, 2)
            num_frames = x['video'].shape[2]
            kp_driving = cat_dict([kp_detector(x['video'][:, :, i:(i+1)]) for i in range(num_frames)], dim=1)
            # kp_video  - mean: (1, n, 10, 2)
            #           - var:  (1, n, 10, 2, 2)

            out = generate(generator, source=x['video'][:, :, :1], kp_source=kp_source,
                           kp_drivings=kp_driving)
            # out   - video_prediction: (1, 3, n, 256, 256)
            #       - video_deformed:   (1, 3, n, 256, 256)
            #       - kp_driving:   - mean: (1, n, 10, 2)
            #                       - var:  (1, n, 10, 2, 2)
            #       - kp_source:   - mean: (1, n, 10, 2)
            #                      - var:  (1, n, 10, 2, 2)
            x['source'] = x['video'][:, :, :1]

            image = Visualizer(**config['visualizer_params']).visualize_reconstruction(x, out)
            image_name = get_name(x['name'][0]) + config['reconstruction_params']['format']
            imageio.mimsave(os.path.join(log_dir, image_name), image)

            loss = reconstruction_loss(out['video_prediction'].cpu(), x['video'].cpu(), 1)
            loss_list.append(loss.data.cpu().numpy())

    print(f"Reconstruction loss: {np.mean(loss_list):.3f}")
