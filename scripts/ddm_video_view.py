"""
A helper script for watching the variation of the entire plane of
an image structure function over different lag time

Important Remark
=====
This script assume a fork version of cddm available [here](https://github.com/inwwin/cddm)
to configure the pause_duration.
"""
import numpy as np
from cddm.viewer import VideoViewer
import json
import argparse
import pathlib
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.animation

parser = argparse.ArgumentParser()
parser.add_argument('-vinfo', '--vid_info_path', type=argparse.FileType('r'), default=False)
parser.add_argument('-c', '--calibration_path', type=argparse.FileType('r'), default=False)
parser.add_argument('-n', '--frames', type=int, default=256)
parser.add_argument('--vmax', type=int, default=False)
parser.add_argument('--kmax', type=int, default=False)
parser.add_argument('ddm_npy_path', type=pathlib.Path)
parser.add_argument('save_html_path', type=pathlib.Path, default=False, nargs='?')
params = parser.parse_args()

wavenumber_factor = 1
wavenumber_unit = r'\mathrm{pixel}^{-1}'

if params.vid_info_path:
    vid_info = json.load(params.vid_info_path)
    frame_interval = vid_info['framerate'][1] / vid_info['framerate'][0]
    if params.calibration_path:
        calibration = json.load(params.calibration_path)
        wavenumber_factor = 2 * np.pi / (vid_info['fft']['size'] * calibration['calibration_factor'])
        wavenumber_unit = '\\mathrm{{rad}}\\,\\mathrm{{{}}}^{{-1}}'.format(calibration['physical_unit'])
        print(f'wavenumber_factor = {wavenumber_factor}')
    print(f'frame_interval = {frame_interval} s')
else:
    frame_interval = 1

ddm_array = np.load(params.ddm_npy_path)
print(ddm_array.shape)
imshow_kwargs = {
    'vmin': 0,
    'vmax': params.vmax if params.vmax else np.amax(ddm_array)
}
interval = 1000 / 3

ddm_array = np.fft.fftshift(ddm_array, axes=0)
if params.kmax:
    kcentre = ddm_array.shape[0] // 2
    ddm_array = ddm_array[kcentre - params.kmax:kcentre + params.kmax + 1,
                          :params.kmax + 1,
                          :]
    imshow_kwargs['extent'] = (
        -0.5 * wavenumber_factor,
        (params.kmax + 0.5) * wavenumber_factor,
        (-params.kmax - 0.5) * wavenumber_factor,
        (params.kmax + 0.5) * wavenumber_factor,
    )
else:
    khalf = ddm_array.shape[0] // 2
    imshow_kwargs['extent'] = (
        -0.5 * wavenumber_factor,
        (ddm_array.shape[1] - 0.5) * wavenumber_factor,
        (-khalf - 0.5) * wavenumber_factor,
        (khalf + 0.5) * wavenumber_factor,
    )
if params.save_html_path:
    fig, ax = plt.subplots()
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    img = ax.imshow(ddm_array[:, :, 0], **imshow_kwargs)
    fig.colorbar(img, ax=ax)
    ti_indicator = fig.text(.02, .98, '$t_i$ = 0 frames\n$t$ = 0', verticalalignment='top')
    ax.set_xlabel(f'$q_x$ (${wavenumber_unit}$)')
    ax.set_ylabel(f'$q_y$ (${wavenumber_unit}$)')

    def change_img(ti):
        img.set_data(ddm_array[:, :, ti])
        ti_indicator.set_text(f"$t_i$ = {ti} frames\n$\\tau$ = {ti*frame_interval:.3f} s")
        return img,

    animation = matplotlib.animation.FuncAnimation(fig, change_img, frames=params.frames, blit=False, interval=interval)
    # plt.show()
    animation.save(params.save_html_path, matplotlib.animation.HTMLWriter(fps=1000 / interval, default_mode='loop'))
else:
    ddm_iter = [ddm_array[:, :, i] for i in range(ddm_array.shape[2])]
    viewer = VideoViewer(ddm_iter, count=params.frames, **imshow_kwargs)
    viewer.pause_duration = interval / 1000
    viewer.show()
