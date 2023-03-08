"""
Dveloper: vujadeyoon
Email: vujadeyoon@gmail.com
Github: https://github.com/vujadeyoon/vujade

Title: main_img2vid.py
Description: A main python script to make a video from images.

How to use: python3 main_img2vid.py --path_root_img ${PATH_ROOT_IMG} --path_video ${PATH_VIDEO} --fps ${FPS}
"""


import argparse
import cv2
from tqdm import tqdm
from vujade import vujade_path as path_
from vujade import vujade_imgcv as imgcv_
from vujade import vujade_videocv as videocv_
from vujade.vujade_debug import printd


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Make a video from images.')
    parser.add_argument('--path_root_img', type=str, default='./image')
    parser.add_argument('--path_video', type=str, default='./video.mp4')
    parser.add_argument('--fps', type=float, default=30.0)

    args = parser.parse_args()

    return args


if __name__=='__main__':
    args = get_args()
    spath_root_img = args.path_root_img
    spath_video = args.path_video
    fps = args.fps

    path_root_img = path_.Path(spath_root_img)
    path_video = path_.Path(spath_video)
    path_video.unlink(_missing_ok=True)

    video_writer = None
    list_glob = sorted(list(path_root_img.path.glob('*.png')))
    for _idx, _path_img in enumerate(tqdm(list_glob)):
        if _idx == 0:
            ndarr_img = imgcv_.imread(_filename='{}'.format(_path_img), _flags=cv2.IMREAD_COLOR, _is_bgr2rgb=False)
            img_height, img_width = ndarr_img.shape[0], ndarr_img.shape[1]
            video_writer = videocv_.VideoWriterCV(_spath_video=path_video.str, _size=(img_height, img_width), _fps=fps)

        ndarr_img = imgcv_.imread(_filename='{}'.format(_path_img), _flags=cv2.IMREAD_COLOR, _is_bgr2rgb=False)

        if video_writer is None:
            raise ValueError
        else:
            video_writer.imwrite([ndarr_img])

    if video_writer is None:
        raise ValueError
    else:
        video_writer.close()
