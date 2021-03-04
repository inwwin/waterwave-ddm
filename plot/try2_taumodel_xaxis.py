import numpy as np
import json
import os.path as p
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from scipy.optimize import curve_fit
import argparse


def linear_func(x, a, b):
    return a * x + b


def linear_jac(x, a, b):
    jac = np.empty((x.size, 2))
    jac[:, 0] = x
    jac[:, 1] = 1
    return jac


parser = argparse.ArgumentParser()
parser.add_argument('-t2r', '--tau2reciprocal', action='store_true',
                    help='plot reciprocal of tau2 instead of tau2 itself')
parser.add_argument('-s1', '--save1', action='store_true')
parser.add_argument('-s2', '--save2', action='store_true')
cli_params = parser.parse_args()

cdir = p.abspath(p.dirname(__file__))
csv_path = p.abspath(cdir + '/../assets/taumodel_fit_try2.csv')
try2_fit = np.loadtxt(csv_path,
                      delimiter=',',
                      skiprows=1,
                      #        C1, C2, fr, t2, err...
                      usecols=(17, 18, 19, 20, 21, 22, 23, 24,))
try2_config = np.loadtxt(csv_path,
                         dtype=int,
                         delimiter=',',
                         skiprows=1,
                         #        i, j, taurange
                         usecols=(0, 1, 2, 3,))
# print(try2_config)
# print(try2_fit)

vid_info_path = cdir + '/../../data/initial-dev/try2/vid_info.json'
with open(vid_info_path) as vid_info_file:
    vid_info = json.load(vid_info_file)
calibration_path = cdir + '/../../data/initial-dev/try2/spatial_calibration.json'
with open(calibration_path) as calibration_file:
    calibration = json.load(calibration_file)
wavenumber_factor = 2 * np.pi / (vid_info['fft']['size'] * calibration['calibration_factor'])
wavenumber_unit = '\\mathrm{{rad}}\\,\\mathrm{{{}}}^{{-1}}'.format(calibration['physical_unit'])
frame_interval = vid_info['framerate'][1] / vid_info['framerate'][0]
print(f'frame_interval = {frame_interval} s')
print(f'wavenumber_factor = {wavenumber_factor}')

wavenumber_space = wavenumber_factor * try2_config[:, 1]

c1 = try2_fit[:, 0]
c1av = np.mean(c1)
print(f'c1av = {c1av:.4f}')
c2 = try2_fit[:, 1]
c2av = np.mean(c2)
print(f'c2av = {c2av:.4f}')

freq_physical = try2_fit[:, 2] / frame_interval
freq_err_physical = try2_fit[:, 6] / frame_interval
freqpopt, freqpcov = curve_fit(linear_func, wavenumber_space, freq_physical,
                               jac=linear_jac, sigma=freq_err_physical, absolute_sigma=True)
fit_x = np.array([np.min(try2_config[:, 1]) - 0.5, np.max(try2_config[:, 1]) + 0.5])
fit_wavenumber = wavenumber_factor * fit_x
# a is the physical wave speed
speed_physical, b = tuple(freqpopt)
freq_fit = linear_func(fit_wavenumber, speed_physical, b)
print(freqpopt)
print(freqpcov)
print(np.square(np.diagonal(freqpcov)))
speed_logical = speed_physical * frame_interval / calibration['calibration_factor']
print(f'physical speed = {speed_physical} {calibration["physical_unit"]}/s')
print(f'logical speed = {speed_logical:.4} pixels/frame')

tau2_physical = try2_fit[:, 3] * frame_interval
tau2_err_physical = try2_fit[:, 7] * frame_interval
tau2popt, tau2pcov = curve_fit(linear_func, wavenumber_space, tau2_physical,
                               jac=linear_jac, sigma=tau2_err_physical, absolute_sigma=True)
c, d = tuple(tau2popt)
tau2_fit = linear_func(fit_wavenumber, c, d)
print('c, d =', tau2popt)
print(tau2pcov)
print(np.square(np.diagonal(tau2pcov)))

# Try plotting reciprocal of tau2 instead
if cli_params.tau2reciprocal:
    tau2_err_frac = try2_fit[:, 7] / try2_fit[:, 3]
    tau2_reciproc_physical = np.reciprocal(tau2_physical)
    tau2_reciproc_err_physical = tau2_err_frac * tau2_reciproc_physical

figwidth_cm = 22
figwidth_inch = figwidth_cm / 2.54
figheight_inch = figwidth_inch * 7. / 9.
figsize = (figwidth_inch, figheight_inch)

fig1, ((axc1, axc2), (axfreq, axtau2)) = plt.subplots(2, 2, sharex=True, figsize=figsize)
titles = (f'$C_1$ ($C_{{1\\,\\mathrm{{av}}}}={c1av:.4f}$)',
          f'$-C_2$ ($C_{{2\\,\\mathrm{{av}}}}={c2av:.4f}$)',
          f'$\\Omega$ ($\\Omega_\\mathrm{{fit}}='
          f'({speed_physical:.5}\\,\\mathrm{{{calibration["physical_unit"]}}}\\,\\mathrm{{s}}^{{-1}})\\,q_x+'
          f'({b:.4}\\,\\mathrm{{rad}}\\,\\mathrm{{s}}^{{-1}}$)',
          f'$\\tau_2$ ($\\tau_{{2\\,\\mathrm{{fit}}}}={c:.4}\\,q_x+{d:.4}$)')
axs = (axc1, axc2, axfreq, axtau2)
axys = (c1, -c2, freq_physical,
        tau2_physical if not cli_params.tau2reciprocal else tau2_reciproc_physical)
axys_err = (try2_fit[:, 4], try2_fit[:, 5], freq_err_physical,
            tau2_err_physical if not cli_params.tau2reciprocal else tau2_reciproc_err_physical)
axz = zip(titles, axs, axys, axys_err)

for i, (title, ax, axy, axy_err) in enumerate(axz):
    ax: Axes
    ax.errorbar(wavenumber_space, axy, axy_err,
                fmt='_', linestyle='')
    ax.set_title(title)
axc1.sharey(axc2)

fitted_line_kwargs = {
    'color': 'maroon'
}
axc1.axhline(c1av, **fitted_line_kwargs)
axc2.axhline(-c2av, **fitted_line_kwargs)
axfreq.plot(fit_wavenumber, freq_fit, **fitted_line_kwargs)
if not cli_params.tau2reciprocal:
    axtau2.plot(fit_wavenumber, tau2_fit, **fitted_line_kwargs)
for ax in (axfreq, axtau2):
    ax.set_xlabel(f'$q_x$ (${wavenumber_unit}$)')
    secax = ax.secondary_xaxis(1., functions=(lambda x: 200 * np.pi / x, lambda x: 200 * np.pi / x))
    secax.set_xlabel(f'$\\lambda_x$ (c{calibration["physical_unit"]})')
    secax.set_xticks([6, 7, 8, 9, 10, 12, 14, 16, 19, 24, 30, 40])
axfreq.set_ylabel('$\\Omega$ (rad $\\mathrm{s}^{-1}$)')
axtau2.set_ylabel('$\\tau_2$ (s)')
fig1.suptitle('Params fitted along $q_x$ axis at $q_y=0$')
fig1.set_tight_layout(True)

fig2, axtau2s = plt.subplots(2, 2, figsize=figsize)
(axtau2lin, axtau2semilogx), (axtau2semilogy, axtau2loglog) = axtau2s
for axtau2row in axtau2s:
    for axtau2cell in axtau2row:
        axtau2cell.errorbar(wavenumber_space, tau2_physical, tau2_err_physical,
                            fmt='_', linestyle='')
        axtau2cell.set_xlabel(f'$q_x$ (${wavenumber_unit}$)')
        axtau2cell.set_ylabel('$\\tau_2$ (s)')
for axtau2logy in (axtau2semilogy, axtau2loglog):
    axtau2logy.set_yscale('log')
for axtau2logx in (axtau2semilogx, axtau2loglog):
    axtau2logx.set_xscale('log')
axtau2lin.sharey(axtau2semilogx)
axtau2lin.sharex(axtau2semilogy)
axtau2loglog.sharey(axtau2logy)
axtau2loglog.sharex(axtau2logx)
axtau2lin.set_title('$\\tau_2$ linear plot')
axtau2semilogx.set_title('$\\tau_2$ semilog x plot')
axtau2semilogy.set_title('$\\tau_2$ semilog y plot')
fig2.set_tight_layout(True)

axtau2loglog.set_title('$\\tau_2$ log-log plot')
if not (cli_params.save1 or cli_params.save2):
    plt.show()
else:
    if cli_params.save1:
        fig1_path = p.abspath(cdir + '/try2_taumodel_xaxis.png')
        fig1.savefig(fig1_path, dpi=300)
    if cli_params.save2:
        fig2_path = p.abspath(cdir + '/try2_taumodel_xaxis_tau2.png')
        fig2.savefig(fig2_path, dpi=300)
