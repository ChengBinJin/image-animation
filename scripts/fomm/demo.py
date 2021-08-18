import os
import sys
import yaml
import imageio
import torch
import numpy as np
from argparse import ArgumentParser
from skimage.transform import resize
from skimage import img_as_ubyte
from tqdm import tqdm

project_folder = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_folder)

from library.third_partys.sync_batchnorm import DataParallelWithCallback
from library.utils.keypoint import normalize_kp2, find_best_frame
from library.modules.keypoint_detector import KPDetector2
from library.modules.generator import OcclusionAwareGenerator


def load_checkpoints(config_path, checkpoint_path):
    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    kp_detector = KPDetector2(**config['model_params']['kp_detector_params'],
                              **config['model_params']['common_params'])
    generator = OcclusionAwareGenerator(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])

    if torch.cuda.is_available():
        kp_detector.cuda()
        generator.cuda()

    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    kp_detector.load_state_dict(checkpoint['kp_detector'])
    generator.load_state_dict(checkpoint['generator'])

    if torch.cuda.is_available():
        kp_detector = DataParallelWithCallback(kp_detector)
        generator = DataParallelWithCallback(generator)

    kp_detector.eval()
    generator.eval()

    print(f'kp_detector:\n {kp_detector}')
    print(f'generator:\n {generator}')

    return kp_detector, generator


def make_animation(source_img, driving_video, kp_detector, generator, relative=True, adapt_movement_scale=True,
                   cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)  # (N, 3, H, W)
        if not cpu:
            source = source.cuda()

        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        # driving shape: (N, 3, num_frames, H, W)
        kp_source = kp_detector(source)
        # kp_source -   value:      (N, num_kp, 2)
        #           -   jacobian:   (N, num_kp, 2, 2)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]  # (1, 3, H, W)
            if not cpu:
                driving_frame = driving_frame.cuda()

            kp_driving = kp_detector(driving_frame)
            # kp_driving    -   value:      (N, num_kp, 2)
            #               -   jacobian:   (N, num_kp, 2, 2)
            kp_norm = normalize_kp2(kp_source=kp_source, kp_driving_initial=kp_driving_initial, kp_driving=kp_driving,
                                    use_relative_movement=relative, use_relative_jacobian=relative,
                                    adapt_movement_scale=adapt_movement_scale)
            # kp_norm       -   value:      (N, num_kp, 2)
            #               -   jacobian:   (N, num_kp, 2, 2)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            # out   -   mask:               (N, num_kp+1, 64, 64)
            #       -   sparse_deformed:    (N, num_kp+1, 3, 64, 64)
            #       -   occlusion_map:      (N, 1, 64, 64)
            #       -   deformed:           (N, 3, 256, 256)
            #       -   prediction:         (N, 3, 256, 256)
            prediction_img = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]  # (256, 256, 3)
            predictions.append(prediction_img)

    return predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default="/workspace/nas-data/Codes/H0088_image-animation/config/fomm/vox-adv-256.yaml", help="path to config")
    parser.add_argument("--checkpoint", default="/workspace/nas-data/Codes/H0088_image-animation/checkpoints/fomm/vox-adv-cpk.pth.tar", help="path to checkpoint to restore")
    parser.add_argument("--source_image", default="/workspace/nas-data/Codes/H0088_image-animation/sup-mat/MrKim.jpg", help="path to source image")
    parser.add_argument("--driving_video", default="/workspace/nas-data/Codes/H0088_image-animation/sup-mat/applepen.MP4", help="path to driving video")
    parser.add_argument("--result_video", default="/workspace/nas-data/Codes/H0088_image-animation/fomm_result.mp4", help="path to output")

    parser.add_argument("--relative", dest="relative", action="store_true",
                        help="use relative or absolute keypoint coordinates")
    parser.add_argument("--adapt_scale", dest="adapt_scale", action="store_true",
                        help="adap movement scale based on convex hull of keypoints")
    parser.add_argument("--find_best_frame", dest="find_best_frame", action="store_true",
                        help="Generate from the frame that is the most alligned with source. "
                             "(Only for faces, requries face_alignment lib")
    parser.add_argument("--img_size", default=256, help="image size fed to network")
    parser.add_argument("--best_frame", dest="best_frame", type=int, default=None, help="Set frame to start from.")

    parser.set_defaults(relative=True)
    parser.set_defaults(adapt_scale=True)
    parser.set_defaults(find_best_frame=True)
    opt = parser.parse_args()

    source_img = imageio.imread(opt.source_image)
    reader = imageio.get_reader(opt.driving_video)
    fps = reader.get_meta_data()['fps']

    driving_video = []
    try:
        for im in reader:
            driving_video.append(im)
    except RuntimeError:
        pass
    reader.close()

    source_img = resize(source_img, (opt.img_size, opt.img_size))[..., :3]
    driving_video = [resize(frame, (opt.img_size, opt.img_size))[..., :3] for frame in driving_video]
    kp_detector, generator = load_checkpoints(config_path=opt.config, checkpoint_path=opt.checkpoint)

    if opt.find_best_frame or opt.best_frame is not None:
        i = opt.best_frame if opt.best_frame is not None else find_best_frame(source_img, driving_video)
        print(f"Best frame: {i}")

        driving_forward = driving_video[i:]
        driving_backward = driving_video[:(i+1)][::-1]
        predictions_forward = make_animation(source_img, driving_forward, kp_detector, generator,
                                             relative=opt.relative, adapt_movement_scale=opt.adapt_scale,
                                             cpu=torch.cuda.is_available())
        predictions_backward = make_animation(source_img, driving_backward, kp_detector, generator,
                                              relative=opt.relative, adapt_movement_scale=opt.adapt_scale,
                                              cpu=torch.cuda.is_available())
        predictions = predictions_backward[::-1] + predictions_forward[1:]
    else:
        predictions = make_animation(source_img, driving_video, kp_detector, generator, relative=opt.relative,
                                     adapt_movement_scale=opt.adapt_scale)
    imageio.mimsave(opt.result_video, [img_as_ubyte(frame) for frame in predictions], fps=fps)
