import argparse
from pathlib import Path
import json
import numpy as np
# dct: Discrete Cosine Transform https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.dct.html
from scipy.fft import dct
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import plotly.graph_objects as go
from waterwave_ddm.ddm_polar_inspect import map_to_polar


def plot_ddm_osc_slice(ddm_osc, fig: matplotlib.figure.Figure, ax: matplotlib.axes.Axes,
                       wavenumber_factor, wavenumber_unit, freq_factor,
                       min_signal=None, max_signal=None, radial_bin_size=1, with_colorbar=True):
    if min_signal is None:
        min_signal = np.amin(ddm_osc[1:, 1:])
    if max_signal is None:
        max_signal = np.amax(ddm_osc[1:, 1:])

    # We start from index 0
    min_wavenum_axis = -0.5 + 0
    max_wavenum_axis = +0.5 + ddm_osc.shape[0] * radial_bin_size - 1
    min_wavenum_axis *= wavenumber_factor
    max_wavenum_axis *= wavenumber_factor
    min_freq_axis = -0.5 + 0
    max_freq_axis = +0.5 + ddm_osc.shape[1] - 1
    min_freq_axis *= freq_factor
    max_freq_axis *= freq_factor

    axim_ddm_osc = ax.imshow(ddm_osc.T,
                             origin='lower',
                             extent=(min_wavenum_axis, max_wavenum_axis, min_freq_axis, max_freq_axis),
                             vmin=min_signal,
                             vmax=max_signal,
                             aspect='auto',
                             )
    if with_colorbar:
        fig.colorbar(axim_ddm_osc, ax=ax)
    return axim_ddm_osc


def plot_ddm_dct_volume(ddm_dct: np.ndarray, wavenumber_factor, wavenumber_unit, freq_factor):
    orgin_zero = ddm_dct.copy()
    orgin_zero[0, 0, :] = 0
    # min_signal = np.amin(orgin_zero[:, :, 1:])
    max_signal = np.amax(orgin_zero[:, :, 1:])

    qx_len = ddm_dct.shape[1]
    qy_len = ddm_dct.shape[0] // 2
    freq_len = ddm_dct.shape[2]
    qx_axis = np.arange(0, qx_len)
    qy_axis, qx_axis, freq_axis = np.mgrid[-qy_len * wavenumber_factor:qy_len * wavenumber_factor:ddm_dct.shape[0] * 1j,
                                           0:(qx_len - 1) * wavenumber_factor:qx_len * 1j,
                                           0:(freq_len - 1) * freq_factor:freq_len * 1j]

    ddm_dct_shift = np.fft.fftshift(ddm_dct, axes=0)

    vol = go.Volume(
        x=qy_axis.flatten(),
        y=qx_axis.flatten(),
        z=freq_axis.flatten(),
        value=ddm_dct_shift.flatten(),
        isomin=-max_signal,
        isomax=max_signal,
        opacity=0.2,
        surface_count=20,
        opacityscale=[[-max_signal, 1], [-0.05, 0], [0.05, 0], [+max_signal, 1]],
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
    fig = go.Figure(vol)
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vinfo', '--vid_info_path', type=argparse.FileType('r'), default=False)
    parser.add_argument('-c', '--calibration_path', type=argparse.FileType('r'), default=False)
    parser.add_argument('-l', '--limit', nargs=2, type=float, default=[None, None])
    parser.add_argument('-t', '--max_time', type=int, default=512)
    parser.add_argument('-f', '--max_freq_fraction', type=float, default=0.4)
    parser.add_argument('ddm_npy_path', type=Path)
    dimensions_subparsers = parser.add_subparsers(dest='dimensions', required=True)
    d2i_parser = dimensions_subparsers.add_parser('2di', help='2d view, varying j, fix i')
    d2i_parser.add_argument('i', type=int, default=0)
    d2_parser = dimensions_subparsers.add_parser('2d', help='2d view, polar coordinates')
    d2_parser.add_argument('-a', '--angular_bins', type=int, default=18)
    d2_parser.add_argument('-ao', '--angle_offset', type=float, default=0.)
    d2_parser.add_argument('-p', '--radial_bin_size', type=int, default=2)
    d2_parser.add_argument('-s', '--save', type=str, default=False)
    d2_parser.add_argument('angle_index', type=int)
    dimensions_subparsers.add_parser('3d')
    params = parser.parse_args()

    ddm_array = np.load(params.ddm_npy_path, mmap_mode='r')
    print('ddm_array.shape =', ddm_array.shape)

    wavenumber_factor = 1
    wavenumber_unit = r'\mathrm{pixel}^{-1}'
    max_time_index = params.max_time
    # Only view 40% of the available freq data by default
    max_freq_index = int(params.max_freq_fraction * max_time_index)
    ddm_array = ddm_array[..., :max_time_index]
    print('new ddm_array.shape =', ddm_array.shape)

    if params.vid_info_path:
        vid_info = json.load(params.vid_info_path)
        frame_interval = vid_info['framerate'][1] / vid_info['framerate'][0]
        freq_factor = np.pi / ((max_time_index - 1) * frame_interval)
        if params.calibration_path:
            calibration = json.load(params.calibration_path)
            wavenumber_factor = 2 * np.pi / (vid_info['fft']['size'] * calibration['calibration_factor'])
            wavenumber_unit = '\\mathrm{{rad}}\\,\\mathrm{{{}}}^{{-1}}'.format(calibration['physical_unit'])
            print(f'wavenumber_factor = {wavenumber_factor}')
        print(f'frame_interval = {frame_interval} s')
    else:
        frame_interval = 1
        freq_factor = 1

    if params.dimensions == '3d' or params.dimensions == '2di':
        ddm_dct = dct(ddm_array, type=1)
        ddm_dct /= (2 * (max_time_index - 1))
        print(ddm_dct.shape)
        ddm_dct *= -1

        if params.dimensions == '3d':
            fig2 = plot_ddm_dct_volume(ddm_dct[:, :, :max_freq_index], wavenumber_factor, wavenumber_unit, freq_factor)
            fig2.show()

        elif params.dimensions == '2di':
            y = params.i
            if y > ddm_dct.shape[0] // 2:
                y -= ddm_dct.shape[0]
            qy = y * wavenumber_factor

            i = params.i
            if i < 0:
                i += ddm_dct.shape[0]

            ddm_osc = ddm_dct[i, :, :max_freq_index]

            fig1, ax1 = plt.subplots(figsize=(8., 6.))
            fig1: matplotlib.figure.Figure
            ax1: matplotlib.axes.Axes
            plot_ddm_osc_slice(ddm_osc, fig1, ax1, wavenumber_factor, wavenumber_unit, freq_factor,
                               params.limit[0], params.limit[1])
            fig1.suptitle('Heatmap of the inverse of the coefficients of\n'
                          'the cosine decomposition of $I(q,\\tau)$ (i.e. $-C_2(q,\\Omega)$) '
                          f'along $q_y={qy}\\,{wavenumber_unit}$')
            ax1.set_ylabel('$\\Omega$ (rad $\\mathrm{s}^{-1}$)')
            ax1.set_xlabel(f'$q_x$ (${wavenumber_unit}$)')
            # fig.savefig(cdir + '/../plot/try2_multifreq_decomposition.png', dpi=300)
            plt.show()

    elif params.dimensions == '2d':
        ddm_polar, median_angle, _, _, _, \
            lower_angle, _, upper_angle, _, blank_bins = \
            map_to_polar(ddm_array, params.angular_bins, params.radial_bin_size, ddm_array.shape[1] - 1,
                         params.angle_offset, True)
        print('ddm_polar.shape: ', ddm_polar.shape)
        print('blank_bins:', blank_bins)

        ddm_dct = dct(ddm_polar, type=1)
        ddm_dct /= (2 * (max_time_index - 1))
        print(ddm_dct.shape)
        ddm_dct *= -1

        if params.angle_index >= 0:
            ddm_osc = ddm_dct[params.angle_index, :, :max_freq_index]
            print('angle between:', lower_angle[params.angle_index], upper_angle[params.angle_index])
            print('median angle:', median_angle[params.angle_index])

            fig, ax = plt.subplots(figsize=(8., 6.))
            fig: matplotlib.figure.Figure
            ax: matplotlib.axes.Axes
            plot_ddm_osc_slice(ddm_osc, fig, ax, wavenumber_factor, wavenumber_unit, freq_factor,
                               params.limit[0], params.limit[1], params.radial_bin_size)
            fig.suptitle('Radial heatmap of the inverse of the coefficients of '
                         'the cosine decomposition\nof $I(q,\\tau)$ (i.e. $-C_2(q,\\Omega)$) '
                         f'averged within $q_\\theta$ between ${lower_angle[params.angle_index]}\\degree$ and '
                         f'${upper_angle[params.angle_index]}\\degree$')
            ax.set_ylabel('$\\Omega$ (rad $\\mathrm{s}^{-1}$)')
            ax.set_xlabel(f'$q_r$ (${wavenumber_unit}$)')
            if params.save:
                fig.savefig(params.save, dpi=300)
            else:
                plt.show()
        elif params.angle_index == -1:
            fig_path = params.ddm_npy_path.with_suffix('.png')
            fig_path: Path
            for ai in range(ddm_dct.shape[0]):
                ddm_osc = ddm_dct[ai, :, :max_freq_index]

                fig, ax = plt.subplots(figsize=(8., 6.))
                fig: matplotlib.figure.Figure
                ax: matplotlib.axes.Axes
                plot_ddm_osc_slice(ddm_osc, fig, ax, wavenumber_factor, wavenumber_unit, freq_factor,
                                   params.limit[0], params.limit[1], params.radial_bin_size)
                fig.suptitle('Radial heatmap of the inverse of the coefficients of '
                             'the cosine decomposition\nof $I(q,\\tau)$ (i.e. $-C_2(q,\\Omega)$) '
                             f'averged within $q_\\theta$ between ${lower_angle[ai]}\\degree$ and '
                             f'${upper_angle[ai]}\\degree$')
                custom_suffix = ('.' + params.save) if params.save else str()
                ax.set_ylabel('$\\Omega$ (rad $\\mathrm{s}^{-1}$)')
                ax.set_xlabel(f'$q_r$ (${wavenumber_unit}$)')
                fig.savefig(fig_path.with_stem(fig_path.stem + f'{custom_suffix}.polar.freq{ai:02}'), dpi=300)
        elif params.angle_index == -2:
            matplotlib.rcParams.update({'font.size': 8.5})
            fig = plt.figure(figsize=(17 / 2.54, 23.25 / 2.54), dpi=400.,
                             constrained_layout=True)
            gridspec = fig.add_gridspec(5, 4, hspace=2.25 / 23.25)
            # this means we only support 18 angles only
            grid_indices = (0,   1,
                            4,   5,  6,  7,
                            8,   9, 10, 11,
                            12, 13, 14, 15,
                            16, 17, 18, 19)
            axs_flat = [fig.add_subplot(gridspec[i]) for i in grid_indices]
            for ai, ax in zip(range(ddm_dct.shape[0]), axs_flat):
                ddm_osc = ddm_dct[ai, :, :max_freq_index]
                img = plot_ddm_osc_slice(ddm_osc, fig, ax, wavenumber_factor, wavenumber_unit, freq_factor,
                                         params.limit[0], params.limit[1], params.radial_bin_size, False)
                ax.set_title(f'${lower_angle[ai]:.0f}\\degree \\leq q_\\theta < {upper_angle[ai]:.0f}\\degree$',
                             {'fontsize': 'medium'})
            for i in (0, 2, 6, 10, 14):
                axs_flat[i].set_ylabel('$\\Omega$ (rad $\\mathrm{s}^{-1}$)')
            for i in (14, 15, 16, 17):
                axs_flat[i].set_xlabel(f'$q_r$ (${wavenumber_unit}$)')
            fig.colorbar(img, ax=axs_flat, shrink=0.35)
            custom_suffix = ('.' + params.save) if params.save else str()
            fig_path = params.ddm_npy_path.with_name(params.ddm_npy_path.stem + f'{custom_suffix}.polar.allfreq.svg')
            fig.savefig(fig_path)


if __name__ == '__main__':
    main()
