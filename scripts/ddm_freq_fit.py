import numpy as np
import argparse
import json
import csv
from pathlib import Path
from scipy.stats import linregress
import scipy.odr as odr
import matplotlib
import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rthreshold', type=float, default=0.85)
    parser.add_argument('-rp', '--regression-plot', dest='regression_plot', action='store_true')
    parser.add_argument('wind_csv', type=Path, default=False)
    params = parser.parse_args()

    matplotlib.rcParams.update({'font.size': 8.5})
    global_subplots_params = {
        'figsize': (4.69, 3.52),
        'constrained_layout': True
    }
    trials_result = dict()
    all_points = list()
    with open(params.wind_csv, 'r', newline='') as wind_csv:
        wind_reader = csv.reader(wind_csv)
        # skip the header row
        next(wind_reader)
        for trial in wind_reader:
            if trial[4].startswith('skip'):
                continue
            print('processing', trial[0])
            ddm_time_fit_file_prefix = Path(trial[1]) / 'auto_correlate_data_lin_diff_diff.npy'
            params_fit_path = \
                ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem + '.polar.params_fit')
            median_angle_path = \
                ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem + '.polar.median_angle')
            median_radius_path = \
                ddm_time_fit_file_prefix.with_stem(ddm_time_fit_file_prefix.stem + '.polar.median_radius')
            params_fit = np.load(params_fit_path)
            median_angle = np.load(median_angle_path)
            median_radius = np.load(median_radius_path)

            with open(trial[3]) as vid_info_file:
                vid_info = json.load(vid_info_file)
            with open(trial[2]) as calibration_file:
                calibration = json.load(calibration_file)
            frame_interval = vid_info['framerate'][1] / vid_info['framerate'][0]
            # freq_factor = np.pi / ((max_time_index - 1) * frame_interval)
            # freq_factor = 1 / frame_interval
            wavenumber_factor = 2 * np.pi / (vid_info['fft']['size'] * calibration['calibration_factor'])
            wavenumber_unit = '\\mathrm{{rad}}\\,\\mathrm{{{}}}^{{-1}}'.format(calibration['physical_unit'])
            # print(f'wavenumber_factor = {wavenumber_factor}')
            # print(f'frame_interval = {frame_interval} s')

            params_fit[..., 2] /= frame_interval
            freq_fit = params_fit[:, 1:, 2]
            angle_indices = (int(i) for i in trial[4].split()) if trial[4] else range(freq_fit.shape[0])
            do_plot = trial[6].lower()[0] == 't' if trial[6] else params.regression_plot
            slope_list = list()
            slopeerr_list = list()
            wavenumber_space = median_radius[1:] * wavenumber_factor
            fig_path_prefix = ddm_time_fit_file_prefix.with_suffix('.png')
            fig_stem_prefix = ddm_time_fit_file_prefix.stem + '.polar.freq_fit'
            for angle_index in angle_indices:
                finite_mark = np.isfinite(freq_fit[angle_index])
                regress = linregress(wavenumber_space[finite_mark], freq_fit[angle_index, finite_mark])
                # print(trial[0], angle_index, regress.slope, regress.intercept, regress.rvalue, regress.stderr)

                if regress.rvalue >= params.rthreshold:
                    print('using', angle_index)
                    slope_list.append(regress.slope)
                    slopeerr_list.append(regress.stderr)
                else:
                    print('skipping', angle_index)
                if do_plot:
                    lin_fit = regress.intercept + wavenumber_space * regress.slope
                    suptitle = 'Radial dependence of $\\Omega$ parameter as fitted along median ' + \
                        f'$q_\\theta={median_angle[angle_index]}\\degree$\n$R={regress.rvalue}$'
                    # fig, ax = plt.subplots(**global_subplots_params)
                    fig = matplotlib.figure.Figure(**global_subplots_params)
                    ax = fig.add_subplot()
                    ax: matplotlib.axes.Axes
                    ax.plot(wavenumber_space, freq_fit[angle_index])
                    ax.plot(wavenumber_space, lin_fit)
                    ax.set_xlabel(f'$q_r$ (${wavenumber_unit}$)')
                    ax.set_ylabel('$\\Omega$ (rad/s)')
                    fig.suptitle(suptitle)
                    fig.savefig(fig_path_prefix.with_stem(fig_stem_prefix + str(angle_index).zfill(2)), dpi=300)

            slopes = np.array(slope_list)
            slope_av = np.mean(slopes)
            if len(slope_list) <= 1:
                slope_std = 0
            else:
                slope_std = np.std(slopes, ddof=1)
            slopes_err = np.array(slopeerr_list)
            slopes_err_av = np.mean(slopes_err)
            print(slope_std, slopes_err_av)
            slope_err = max(slope_std, slopes_err_av)

            entry = (float(trial[-2]), float(trial[-1]), slope_av, slope_err)
            print(trial[-3], *entry)
            trials_list = trials_result.setdefault(trial[-3], list())
            trials_list.append(entry)
            if not (trial[6] and trial[6].lower()[0] == 't'):
                all_points.append(entry)

        all_points = np.array(all_points)
        odr_data = odr.RealData(all_points[:, 0], all_points[:, -2], all_points[:, 1], all_points[:, -1])
        linear_model = odr.polynomial(1)
        odr_obj = odr.ODR(odr_data, linear_model)
        odr_output = odr_obj.run()
        odr_output.pprint()
        fit_x = np.linspace(0, 3)
        fit_y = odr_output.beta[0] + odr_output.beta[1] * fit_x
        # fit_y_min = (odr_output.beta[0] + odr_output.sd_beta[0]) + \
        #     (odr_output.beta[1] - odr_output.sd_beta[1]) * fit_x
        # fit_y_max = (odr_output.beta[0] - odr_output.sd_beta[0]) + \
        #     (odr_output.beta[1] + odr_output.sd_beta[1]) * fit_x

        # regress_result = linregress(all_points[:, 0], all_points[:, -2])
        # fit_x = np.linspace(0, 3)
        # fit_y = regress_result.intercept + regress_result.slope * fit_x

        # colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
        #           '#17becf', '#1a55FF']
        colors = ['maroon', 'indianred', 'royalblue']
        fig, ax = plt.subplots(**global_subplots_params)
        fig = plt.Figure(**global_subplots_params)
        ax = fig.add_subplot()
        ax: matplotlib.axes.Axes
        ax.plot(fit_x, fit_y, color='slategrey', linewidth=.7, linestyle='-', label='Linear fit')
        # ax.plot(fit_x, fit_y_min)
        # ax.plot(fit_x, fit_y_max)
        for c, (wind_type, trials_list) in zip(colors, trials_result.items()):
            trials_array = np.array(trials_list)
            print(c, wind_type)
            ax.errorbar(trials_array[:, 0], trials_array[:, -2], trials_array[:, -1], trials_array[:, 1],
                        linestyle='', ecolor=c, capsize=3, capthick=1, label=wind_type)
        ax.set_xlim(0, 3)
        ax.set_ylim(0)
        ax.set_xlabel(r'Wind speed ($\mathrm{m}\,\mathrm{s}^{-1}$)')
        ax.set_ylabel(r'Wave speed ($\mathrm{m}\,\mathrm{s}^{-1}$)')
        ax.legend()
        fig.savefig('fig.svg', dpi=300)
        # plt.show()


if __name__ == '__main__':
    main()
