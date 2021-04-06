""" Calculate 2D Fourier Transform of given frames and save for later analysis """
from cddm.video import crop, asmemmaps  # asarrays  # multiply, normalize_video,
# from cddm.window import blackman
from cddm.fft import rfft2  # , normalize_fft
# from cddm.core import acorr, normalize, stats
# from cddm.multitau import log_average
# import numpy as np
from waterwave_ddm.videoreaders import pyav_single_frames_reader


def fft_save(frames_iter, path_out, size, count=None, kmax=None):
    if not kmax:
        # Ignore features of size smaller than 32 pixels
        # (My avi file has JPEG artifact of size 8 pixels)
        kmax = size // 32

    # These codes are copied from the examples in cddm package

    #: create window for multiplication...
    # window = blackman(SHAPE)

    #: we must create a video of windows for multiplication
    # window_video = ((window,),)*NFRAMES

    #: perform the actual multiplication
    # video = multiply(video_simulator.video, window_video)
    video = frames_iter

    #: if the intesity of light source flickers you can normalize each frame to the intensity of the frame
    # video = normalize_video(video)

    #: perform rfft2 and crop results, to take only first kimax and first kjmax wavenumbers.
    fft = rfft2(video, kimax=kmax, kjmax=kmax)

    #: you can also normalize each frame with respect to the [0,0] component of the fft
    #: this it therefore equivalent to  normalize_video
    # fft = normalize_fft(fft)

    #: load in numpy array
    # fft_array, = asarrays(fft, count)
    # np.save(fft_array, path_out)

    # save into disk
    asmemmaps(path_out, fft, count=count)

    return {
        'kmax': kmax
    }


def main():
    import argparse
    import pathlib
    import json

    parser = argparse.ArgumentParser()
    # parser.add_argument('--method', default='diff', choices=['diff', 'fft', 'corr'])
    parser.add_argument('--kmax', default=None, type=int)
    parser.add_argument('-p', '--position', nargs=2, default=[0, 0], type=int)
    parser.add_argument('-r', '--framerate', type=str, default=None)
    parser.add_argument('-f', '--select', nargs=2, type=int)
    parser.add_argument('size', type=int)
    parser.add_argument('vid_in')  # Directly passed to PyAV which then pass on to FFmpeg
    # parser.add_argument('fft_out', type=argparse.FileType('wb'))  # pass to np.save
    parser.add_argument('fft_out', type=pathlib.Path)
    params = parser.parse_args()

    # print(params)

    try:
        params.fft_out.mkdir(mode=0o755, parents=True, exist_ok=True)
    except FileExistsError:
        parser.error('fft_out must be a directory')

    copt = dict()
    if params.framerate:
        copt['framerate'] = params.framerate
    if params.select:
        copt['start_number'] = str(params.select[0])

    frames_iter = pyav_single_frames_reader(
        params.vid_in,
        container_options=copt,
        frame_count=params.select[1] if params.select else None,
    )
    vid_info = next(frames_iter)
    vid_info.pop('codec_context')

    video = frames_iter
    video = crop(video, (
        (params.position[0], params.position[0] + params.size),
        (params.position[1], params.position[1] + params.size),
    ))

    fft_path = params.fft_out / 'fft_array'

    if params.select:
        frame_count = vid_info['framecount']
    else:
        frame_count = vid_info['duration']
    fft_info = fft_save(video, str(fft_path), params.size, count=frame_count, kmax=params.kmax)

    vid_info['vid_in_path'] = str(pathlib.Path(params.vid_in).resolve())
    fft_info['position'] = params.position
    fft_info['size'] = params.size
    vid_info['fft'] = fft_info

    def fraction_serializer(obj):
        try:
            return [obj.numerator, obj.denominator]
        except AttributeError:
            raise TypeError

    with open(params.fft_out / 'vid_info.json', 'w') as j:
        json.dump(vid_info, j, indent=4, default=fraction_serializer)


if __name__ == '__main__':
    main()
