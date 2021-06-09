import numpy as np

from skimage import io, img_as_float32
from skimage.color import gray2rgb
from imageio import mimread


def read_video(name, image_shape):
    if name.lower().endswith('.png') or name.lower().endswith('.jpg') or name.lower().endswith('.jpeg'):
        image = io.imread(name)

        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        if image.shape[2] == 4:
            image = image[..., :3]

        image = img_as_float32(image)
        video_array = np.moveaxis(image, 1, 0)
        video_array = video_array.reshape((-1,) + image_shape)
        video_array = np.moveaxis(video_array, 1, 2)
    elif name.lower().endswith('.gif') or name.lower().endswith('.mp4') or name.lower().endswith('.mov'):
        video = np.array(mimread(name, memtest=False))

        if len(video.shape) == 3:
            video = np.array([gray2rgb(frame) for frame in video])
        if video.shape[-1] == 4:
            video = video[..., :3]
        video_array = img_as_float32(video)
    else:
        raise Exception(f"Unknown file extensions {name}")

    return video_array

