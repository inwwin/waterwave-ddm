"""Collection of single-frame video reader generators compatible with cddm package"""
import av
import av.container
import av.video.frame


def pyav_single_frames_reader(*args, **kwargs):
    """ Open a video file with PyAV, then firstly yield important informations about the video,
    and then yield frames that are compatible with cddm package

    Parameters
    ===
    All are passed through to `av.open` please see https://pyav.org/docs/develop/api/_globals.html#av.open
    """
    with av.open(*args, **kwargs) as container:
        container: av.container.InputContainer
        # container.streams.video[0].thread_type = 'AUTO'  # Go faster!
        video = container.streams.video[0]
        codec = video.codec_context

        yield {
            'time_base': video.time_base,  # in Fraction
            'average_rate': video.average_rate,
            'base_rate': video.base_rate,
            'guessed_rate': video.guessed_rate,
            'width': codec.width,
            'height': codec.height,
            'framerate': codec.framerate,
            'duration': video.duration,  # in multiple of time_base
            'frames': video.frames,
            'codec_context': codec,
        }

        for frame in container.decode(video=0):
            frame: av.video.frame.VideoFrame
            # gray_frame: av.video.frame.VideoFrame
            # gray_frame = frame.reformat(format='gray')

            yield (frame.to_ndarray(format='gray'),)
