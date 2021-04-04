"""A helper script to watch the portion of video that will be cropped by the specified parameters"""
from cddm.video import crop  # , asmemmaps  # asarrays  # multiply, normalize_video,
from cddm.viewer import VideoViewer
from waterwave_ddm.videoreaders import pyav_single_frames_reader
# import numpy as np


def main():
    import argparse
    import pathlib
    # import json

    parser = argparse.ArgumentParser()
    # parser.add_argument('--method', default='diff', choices=['diff', 'fft', 'corr'])
    parser.add_argument('-p', '--position', nargs=2, default=[0, 0], type=int)
    parser.add_argument('size', type=int)
    parser.add_argument('vid_in')  # Directly passed to PyAV which then pass on to FFmpeg
    params = parser.parse_args()

    # print(params)

    frames_iter = pyav_single_frames_reader(params.vid_in)
    vid_info = next(frames_iter)
    # vid_info.pop('codec_context')

    video = frames_iter
    video = crop(video, (
        (params.position[0], params.position[0] + params.size),
        (params.position[1], params.position[1] + params.size),
    ))

    vid_info['vid_in_path'] = str(pathlib.Path(params.vid_in).resolve())

    print(vid_info)

    viewer = VideoViewer(video, vmin=0, vmax=255, count=vid_info['duration'])
    viewer.show()


if __name__ == '__main__':
    main()
