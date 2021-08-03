import numpy as np
import yaml
import imageio
import torch
from argparse import ArgumentParser
from skimage.transform import resize
from skimage import img_as_ubyte
from tqdm import tqdm

from library.third_partys.sync_batchnorm import DataParallelWithCallback
from library.utils.keypoint import normalize_kp2, find_best_frame
from library.modules.keypoint_detector import KPDetector2


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
    generator.load_state_dict([checkpoint['generator']])

    if torch.cuda.is_available():
        kp_detector = DataParallelWithCallback(kp_detector)
        generator = DataParallelWithCallback(generator)

    kp_detector.eval()
    generator.eval()

    return kp_detector, generator


def make_animation(source_img, driving_video, kp_detector, generator, relative=True, adapt_movement_scale=True,
                   cpu=False):
    with torch.no_grad():
        predictions = []
        source = torch.tensor(source_img[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
        if not cpu:
            source = source.cuda()

        driving = torch.tensor(np.array(driving_video)[np.newaxis].astype(np.float32)).permute(0, 4, 1, 2, 3)
        kp_source = kp_detector(source)
        kp_driving_initial = kp_detector(driving[:, :, 0])

        for frame_idx in tqdm(range(driving.shape[2])):
            driving_frame = driving[:, :, frame_idx]
            if not cpu:
                driving_frame = driving_frame.cuda()

            kp_driving = kp_detector(driving_frame)
            kp_norm = normalize_kp2(kp_source=kp_source, kp_driving_initial=kp_driving_initial, kp_driving=kp_driving,
                                    use_relative_movement=relative, use_relative_jacobian=relative,
                                    adapt_movement_scale=adapt_movement_scale)
            out = generator(source, kp_source=kp_source, kp_driving=kp_norm)
            prediction_img = np.transpose(out['prediction'].data.cpu().numpy(), [0, 2, 3, 1])[0]
            predictions.append(prediction_img)

    return predictions


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, help="path to config")
    parser.add_argument("--checkpoint", default="vox-cpk.pth.tar", help="path to checkpoint to restore")
    parser.add_argument("--source_image", default="sup-mat/source.png", help="path to source image")
    parser.add_argument("--driving_video", default="sup-mat/source.png", help="path to driving video")
    parser.add_argument("--result_video", default="result.mp4", help="path to output")

    parser.add_argument("--relative", dest="relative", action="store_ture",
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
