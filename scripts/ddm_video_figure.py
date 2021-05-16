"""Plotting figure consisting of 4x3 axes of the ISF at different tau"""
import numpy as np
import json
import argparse
import pathlib
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
from matplotlib.transforms import Bbox
from matplotlib.patches import FancyBboxPatch

parser = argparse.ArgumentParser()
parser.add_argument('-vinfo', '--vid_info_path', type=argparse.FileType('r'), nargs=2, required=True)
parser.add_argument('-c', '--calibration_path', type=argparse.FileType('r'), required=True)
parser.add_argument('--vmax', type=float, required=True)
parser.add_argument('--kmax', type=int, nargs=4, required=True)
parser.add_argument('ddm_npy_path', type=pathlib.Path, nargs=2)
parser.add_argument('data1a_frame_indices', type=int, nargs=3)
parser.add_argument('data1b_frame_indices', type=int, nargs=3)
parser.add_argument('data2c_frame_indices', type=int, nargs=3)
parser.add_argument('data2d_frame_indices', type=int, nargs=3)
params = parser.parse_args()

calibration = json.load(params.calibration_path)
ddm_data = list()
for vinfo_path, npy_path, kmax in zip(params.vid_info_path, params.ddm_npy_path, params.kmax):
    vid_info = json.load(vinfo_path)
    ddm_datum = {
        'frame_interval': vid_info['framerate'][1] / vid_info['framerate'][0],
        'wavenumber_factor': 2 * np.pi / (vid_info['fft']['size'] * calibration['calibration_factor']),
        'wavenumber_unit': '\\mathrm{{rad}}\\,\\mathrm{{{}}}^{{-1}}'.format(calibration['physical_unit']),
        'ddm_array': np.load(npy_path, mmap_mode='r'),
    }
    ddm_data.append(ddm_datum)
    ddm_data.append(ddm_datum)


matplotlib.rcParams.update({'font.size': 8.5})
fig, axs = plt.subplots(4, 3, figsize=(17 / 2.54, 23.25 / 2.54), dpi=400.,
                        constrained_layout=True, gridspec_kw={'hspace': 2.25 / 23.25})
for axsrow, frame_indices, kmax, ddm_datum, row_name in \
        zip(axs,
            (params.data1a_frame_indices,
             params.data1b_frame_indices,
             params.data2c_frame_indices,
             params.data2d_frame_indices,),
            params.kmax,
            ddm_data,
            ('(a1)', '(a2)', '(b1)', '(b2)')):
    print('row', row_name)
    for j, (ax, frame_index) in enumerate(zip(axsrow, frame_indices)):
        print('cell')
        ddm_array = np.fft.fftshift(ddm_datum['ddm_array'][:, :, frame_index], axes=0)
        kcentre = ddm_array.shape[0] // 2
        ddm_array = ddm_array[kcentre - kmax:kcentre + kmax + 1,
                              :kmax + 1]
        wavenumber_factor = ddm_datum['wavenumber_factor']
        imshow_kwargs = {
            'vmin': 0,
            'vmax': params.vmax,
            'extent': (
                -0.5 * wavenumber_factor,
                (kmax + 0.5) * wavenumber_factor,
                (-kmax - 0.5) * wavenumber_factor,
                (kmax + 0.5) * wavenumber_factor,
            )
        }

        ax: matplotlib.axes.Axes
        img = ax.imshow(ddm_array, **imshow_kwargs)
        time = frame_index * ddm_datum['frame_interval']
        ax.set_title(f'$\\tau$ = {time:.3f} s')
        wavenumber_unit = ddm_datum['wavenumber_unit']
        ax.set_xlabel(f'$q_x$ (${wavenumber_unit}$)')
        ax.set_ylabel(f'$q_y$ (${wavenumber_unit}$)')

        if j < 2:
            bb = Bbox([[1.3, 0.5], [1.55, 0.5]])
            arrow = FancyBboxPatch((bb.xmin, bb.ymin), bb.width, bb.height, 'rarrow, pad=0.07',
                                   transform=ax.transAxes, clip_on=False,
                                   linewidth=0, facecolor='lightgrey')
            fig.add_artist(arrow)
        if j == 0:
            fig.text(-1.17, 1.05, row_name, transform=ax.transAxes)

fig.colorbar(img, ax=axs, shrink=0.35)
fig.savefig('ddm_video_fig.svg')
