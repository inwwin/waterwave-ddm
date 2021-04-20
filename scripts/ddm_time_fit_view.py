import numpy as np
import argparse
import json
from pathlib import Path
import matplotlib.axes
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-vinfo', '--vid_info_path', type=argparse.FileType('r'), default=False)
    parser.add_argument('-c', '--calibration_path', type=argparse.FileType('r'), default=False)
    parser.add_argument('-s', '--save', action='store_true')
    parser.add_argument('ddm_time_fit_file_prefix', type=Path)
    coord_subparsers = parser.add_subparsers(title='coord', required=True, dest='coord_system')
    coord_subparsers.add_parser('cartesian', aliases=['cart'])
    polar_parser = coord_subparsers.add_parser('polar')
    # angular_bins does not need to be specified
    polar_parser.add_argument('-ao', '--angle_offset', type=float, default=0.)
    polar_parser.add_argument('-p', '--radial_bin_size', type=int, default=2)
    polar_intersect_group = polar_parser.add_mutually_exclusive_group()
    polar_intersect_group.add_argument('-a', '--angle_index', type=int, default=None)
    polar_intersect_group.add_argument('-r', '--radius_index', type=int, default=None)

    params = parser.parse_args()
    if params.coord_system.startswith('cart'):
        params.coord_system = 'cartesian'
    ddm_time_fit_file_prefix = params.ddm_time_fit_file_prefix.with_suffix('.npy')
    ddm_time_fit_file_prefix: Path
    params_fit_path = \
        ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem + '.' + params.coord_system + '.params_fit')
    param_errors_path = \
        ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem + '.' + params.coord_system + '.param_errors')
    param_error_fracs_total_path = \
        ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem +
                                           '.' + params.coord_system + '.param_error_fracs_total')
    params_fit = np.load(params_fit_path)
    param_errors = np.load(param_errors_path)
    param_error_fracs_total = np.load(param_error_fracs_total_path)
    print('params_fit.shape =', params_fit.shape)
    print('param_errors.shape =', param_errors.shape)
    print('param_error_fracs_total.shape =', param_error_fracs_total.shape)
    if params.coord_system == 'polar':
        median_angle_path = \
            ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem +
                                               '.' + params.coord_system + '.median_angle')
        median_radius_path = \
            ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem +
                                               '.' + params.coord_system + '.median_radius')
        # average_angle_path = \
        #     ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem +
        #                                        '.' + params.coord_system + '.average_angle')
        # average_radius_path = \
        #     ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem +
        #                                        '.' + params.coord_system + '.average_radius')
        median_angle = np.load(median_angle_path)
        median_radius = np.load(median_radius_path)
        # average_angle = np.load(average_angle_path)
        # average_radius = np.load(average_radius_path)

    wavenumber_factor = 1
    wavenumber_unit = r'\mathrm{pixel}^{-1}'

    if params.vid_info_path:
        vid_info = json.load(params.vid_info_path)
        frame_interval = vid_info['framerate'][1] / vid_info['framerate'][0]
        # freq_factor = np.pi / ((max_time_index - 1) * frame_interval)
        # freq_factor = 1 / frame_interval
        if params.calibration_path:
            calibration = json.load(params.calibration_path)
            wavenumber_factor = 2 * np.pi / (vid_info['fft']['size'] * calibration['calibration_factor'])
            wavenumber_unit = '\\mathrm{{rad}}\\,\\mathrm{{{}}}^{{-1}}'.format(calibration['physical_unit'])
            print(f'wavenumber_factor = {wavenumber_factor}')
        print(f'frame_interval = {frame_interval} s')
    else:
        frame_interval = 1
        # freq_factor = 1

    params_name = ('$C_1$', '$-C_2$', '$\\Omega$', '$\\tau_2$')
    params_name_with_unit = ('$C_1$', '$-C_2$', '$\\Omega$ (rad/s)', '$\\tau_2$ (s)')
    params_fit[..., 1] *= -1
    params_fit[..., 2] /= frame_interval
    params_fit[..., 3] *= frame_interval
    global_subplots_params = {
        'figsize': (8., 6.),
        'constrained_layout': True
    }
    fig_path_prefix = ddm_time_fit_file_prefix.with_suffix('.png')
    fig_stem_prefix = ddm_time_fit_file_prefix.stem + '.' + params.coord_system + '.param'
    if params.coord_system == 'polar':
        if params.angle_index is not None:
            suptitle = 'Radial dependence of {} parameter as fitted along median ' + \
                f'$q_\\theta={median_angle[params.angle_index]}\\degree$'
            for i, param_name, param_name_with_unit in zip(range(4), params_name, params_name_with_unit):
                fig, ax = plt.subplots(**global_subplots_params)
                ax: matplotlib.axes.Axes
                ax.plot(median_radius[1:] * wavenumber_factor, params_fit[params.angle_index, 1:, i])
                ax.set_xlabel(f'$q_r$ (${wavenumber_unit}$)')
                ax.set_ylabel(param_name_with_unit)
                fig.suptitle(suptitle.format(param_name))
                if params.save:
                    fig.savefig(fig_path_prefix.with_stem(fig_stem_prefix + str(i)), dpi=300)
        elif params.radius_index is not None:
            median_qr = median_radius[params.radius_index] * wavenumber_factor
            suptitle = 'Azimuthal dependence of {} parameter as fitted along median ' + \
                f'$q_r={median_qr}$'
            for i, param_name, param_name_with_unit in zip(range(4), params_name, params_name_with_unit):
                fig, ax = plt.subplots(**global_subplots_params)
                ax: matplotlib.axes.Axes
                ax.plot(median_angle[1:], params_fit[1:, params.radius_index, i])
                ax.set_xlabel('$q_\\theta\\degree$')
                ax.set_ylabel(param_name_with_unit)
                fig.suptitle(suptitle.format(param_name))
                if params.save:
                    fig.savefig(fig_path_prefix.with_stem(fig_stem_prefix + str(i)), dpi=300)
        else:
            global_subplots_params['constrained_layout'] = False
            suptitle = 'Fitted {} parameter'
            angular_bins = params_fit.shape[0]
            angle_space = np.deg2rad(np.linspace(-90.0, 90.0, angular_bins + 1, True) + params.angle_offset)
            radial_bins = params_fit.shape[1]
            max_upper_radius = radial_bins * params.radial_bin_size
            radial_space = np.linspace(0.0, max_upper_radius, radial_bins + 1, True) * wavenumber_factor
            angle, radius = np.meshgrid(angle_space, radial_space, indexing='ij')
            for i, param_name, param_name_with_unit in zip(range(4), params_name, params_name_with_unit):
                fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, **global_subplots_params)
                ax: matplotlib.axes.Axes
                if i == 2:
                    pcm = ax.pcolormesh(angle, radius, params_fit[..., i], vmin=0, vmax=np.pi / frame_interval * 0.4)
                elif i == 3:
                    pcm = ax.pcolormesh(angle, radius, params_fit[..., i], vmin=0, vmax=250 * frame_interval)
                else:
                    pcm = ax.pcolormesh(angle, radius, params_fit[..., i])
                ax.grid(True)
                fig.colorbar(pcm, ax=ax, label=param_name_with_unit)
                fig.suptitle(suptitle.format(param_name))
                if params.save:
                    fig.savefig(fig_path_prefix.with_stem(fig_stem_prefix + str(i)), dpi=300)
    elif params.coord_system == 'cartesian':
        params_fit = np.fft.fftshift(params_fit, axes=0)
        qi_max = params_fit.shape[0] // 2
        qi_space = np.arange(+qi_max, -qi_max - 1, -1) * wavenumber_factor
        qj_space = np.arange(0, params_fit.shape[1]) * wavenumber_factor
        qx_mesh, qy_mesh = np.meshgrid(qj_space, qi_space, indexing='ij')
        for i, param_name in zip(range(4), params_name):
            fig, ax = plt.subplots(**global_subplots_params)
            ax: matplotlib.axes.Axes
            pcm = ax.pcolormesh(qx_mesh, qy_mesh, params_fit[..., i].T, shading='nearest')
            ax.set_xlabel(f'$q_x$ (${wavenumber_unit}$)')
            ax.set_ylabel(f'$q_x$ (${wavenumber_unit}$)')
            ax.set_aspect(1)
            fig.colorbar(pcm, ax=ax)
            fig.suptitle(param_name)
            if params.save:
                fig.savefig(fig_path_prefix.with_stem(fig_stem_prefix + str(i)), dpi=300)

    if not params.save:
        plt.show()


if __name__ == '__main__':
    main()
