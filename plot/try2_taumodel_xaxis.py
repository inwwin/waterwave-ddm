import numpy as np
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
parser.add_argument('-s', '--save', action='store_true')
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

c1av = np.mean(try2_fit[:, 0])
print(f'c1av = {c1av:.4f}')
c2av = np.mean(try2_fit[:, 1])
print(f'c2av = {c2av:.4f}')

freqpopt, freqpcov = curve_fit(linear_func, try2_config[:, 1], try2_fit[:, 2],
                               jac=linear_jac, sigma=try2_fit[:, 6], absolute_sigma=True)
fit_x = np.array([0, np.max(try2_config[:, 1]) + 0.5])
a, b = tuple(freqpopt)
freq_fit = linear_func(fit_x, a, b)
print('a, b =', freqpopt)
print(freqpcov)
print(np.square(np.diagonal(freqpcov)))
logical_speed = a * 1024 / (2 * np.pi)
print(f'wave speed = {logical_speed:.4} pixels/frame')

tau2popt, tau2pcov = curve_fit(linear_func, try2_config[:, 1], try2_fit[:, 3],
                               jac=linear_jac, sigma=try2_fit[:, 7], absolute_sigma=True)
c, d = tuple(tau2popt)
tau2_fit = linear_func(fit_x, c, d)
print('c, d =', tau2popt)
print(tau2pcov)
print(np.square(np.diagonal(tau2pcov)))

# Try plotting reciprocal of tau2 instead
if cli_params.tau2reciprocal:
    tau2_err_frac = try2_fit[:, 7] / try2_fit[:, 3]
    tau2_reciproc = np.reciprocal(try2_fit[:, 3])
    tau2_reciproc_err = tau2_err_frac * tau2_reciproc
    try2_fit[:, 3] = tau2_reciproc
    try2_fit[:, 7] = tau2_reciproc_err

try2_fit[:, 1] *= -1
fig, ((axc1, axc2), (axfreq, axtau2)) = plt.subplots(2, 2, sharex=True, figsize=(9., 7.))
params = (f'$C_1$ ($C_{{1\\,av}}={c1av:.4f}$)',
          f'$-C_2$ ($C_{{2\\,av}}={c2av:.4f}$)',
          f'$\\Omega$ ($\\Omega_{{fit}}={a:.5}\\,q_x+{b:.4}$)',
          f'$\\tau_2$ ($\\tau_{{2\\,fit}}={c:.4}\\,q_x+{d:.4}$)')
axs = (axc1, axc2, axfreq, axtau2)
axz = zip(params, axs)

for i, (param, ax) in enumerate(axz):
    ax: Axes
    ax.errorbar(try2_config[:, 1], try2_fit[:, i], try2_fit[:, i + 4],
                fmt='_', linestyle='')
    ax.set_title(param)
axc1.sharey(axc2)

fitted_line_kwargs = {
    'color': 'maroon'
}
axc1.axhline(c1av, **fitted_line_kwargs)
axc2.axhline(-c2av, **fitted_line_kwargs)
axfreq.plot(fit_x, freq_fit, **fitted_line_kwargs)
if not cli_params.tau2reciprocal:
    axtau2.plot(fit_x, tau2_fit, **fitted_line_kwargs)
for ax in (axfreq, axtau2):
    ax.set_xlabel('$q_x$')
fig.suptitle('Params fitted along $q_x$ axis at $q_y=0$')

if cli_params.save:
    fig_path = p.abspath(cdir + '/try2_taumodel_xaxis.png')
    fig.set_tight_layout(True)
    fig.savefig(fig_path, dpi=300)
else:
    plt.show()
