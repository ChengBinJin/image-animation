import torch
from library.utils.keypoint import normalize_kp


def transfer_one(kp_detector, generator, source_image, driving_video, transfer_params):
    cat_dict = lambda l, dim: {k: torch.cat([v[k] for v in l], dim=dim) for k in l[0]}
    d = driving_video.shape[2]
    kp_driving = cat_dict([kp_detector(driving_video[:, :, i:(i+1)]) for i in range(d)], dim=1)
    kp_source = kp_detector(source_image)

    kp_driving_norm = normalize_kp(kp_driving, kp_source, **transfer_params['normalization_params'])
    kp_video_list = [{k: v[:, i:(i+1)] for k, v in kp_driving_norm.items()} for i in range(d)]
    out = cat_dict([generator(source_image=source_image, kp_driving=kp, kp_source=kp_source)
                    for kp in kp_video_list], dim=2)

    out['kp_driving'] = kp_driving
    out['kp_source'] = kp_source
    out['kp_norm'] = kp_driving_norm

    return out
