"""
A helper script for investigating the lag time dependence at a particular wavevector of
an image structure function
"""
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.animation
from waterwave_ddm.simplified_models import taumodel_func, taumodel_jac, guess_C1, guess_C2, guess_freq, guess_C2_tau2
from scipy.optimize import curve_fit
import os
import sys


def verbose_print_wrapper(verbose):
    def print_null(*args, **kwargs):
        pass

    def verbose_print(*args, **kwargs):
        kwargs.setdefault('file', sys.stderr)
        return print(*args, **kwargs)

    if verbose:
        return verbose_print
    else:
        return print_null


def parse_ij(ij, ar, xy, array_shape):
    assert array_shape[0] % 2 == 1
    # imax = shape[0] // 2
    if ij:
        i = ij[0]
        j = ij[1]
    else:
        if ar:
            af = ar[0]
            rf = ar[1]
            xf = rf * np.cos(af * np.pi / 180.)
            yf = rf * np.sin(af * np.pi / 180.)
            x = int(np.round(xf))
            y = int(np.round(yf))
        elif xy:
            x = xy[0]
            y = xy[1]
        j = x
        i = -y
    if i < 0:
        i += array_shape[0]
    return i, j


def main():
    # params_names  = ('C1',  'C2', 'freq', 'tau')
    meta_params = ('initial_guess', 'lower_bound', 'upper_bound')
    # initial_guess = [  2.,   -1.,     1.,    1.  ]  # noqa: E201, E202
    # lower_bounds  = [  1.9, -90.,     0.,    0.1 ]  # noqa: E201, E202
    # upper_bounds  = [  2.1,  -0.5,   50.,   20.  ]  # noqa: E201, E202
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    coord_group = parser.add_mutually_exclusive_group(required=True)
    coord_group.add_argument('-ij', type=int,   nargs=2, metavar=('i', 'j'))
    coord_group.add_argument('-xy', type=int,   nargs=2, metavar=('x', 'y'))
    coord_group.add_argument('-ar', type=float, nargs=2, metavar=('angle', 'radius'))
    parser.add_argument('-pt', '--plot-max-time',     dest='plot_max_time',  default=np.inf,   type=int)
    parser.add_argument('-lt', '--lower-time',        dest='fit_lower_time', default=5,      type=int)
    parser.add_argument('-ut', '--upper-time',        dest='fit_upper_time', default=np.inf, type=int)
    parser.add_argument('-tt', '--total-time',        dest='fit_total_time', default=np.inf, type=int)
    parser.add_argument('-glt', '--guess-lower-time', dest='guess_lower_time', default=None, type=int)
    parser.add_argument('-gut', '--guess-upper-time', dest='guess_upper_time', default=None, type=int)
    parser.add_argument('-c1',   nargs=3, default=None, type=float, metavar=meta_params)
    parser.add_argument('-c2',   nargs=3, default=None, type=float, metavar=meta_params)
    parser.add_argument('-freq', nargs=3, default=None, type=float, metavar=meta_params)
    parser.add_argument('-tau2', nargs=3, default=None, type=float, metavar=meta_params)
    plot_parser = parser.add_mutually_exclusive_group()
    plot_parser.add_argument('-s', '--save', nargs='?', default=None, const=False,
                             type=argparse.FileType('wb'), metavar='fig_out')
    plot_parser.add_argument('-np', '--noplot', dest='plot', action='store_false')
    parser.add_argument('ddm_npy_path', type=argparse.FileType('rb'))
    params = parser.parse_args()

    vprint = verbose_print_wrapper(params.verbose)

    ddm_array = np.load(params.ddm_npy_path)
    vprint('array_shape =', ddm_array.shape)

    # ddm_array = np.fft.fftshift(ddm_array, axes=0)
    i, j = parse_ij(params.ij, params.ar, params.xy, ddm_array.shape)
    fit_lower_time = max(0, params.fit_lower_time)
    plot_max_time =  min(ddm_array.shape[-1], params.plot_max_time)
    fit_upper_time = min(ddm_array.shape[-1], params.fit_upper_time)
    fit_total_time = min(ddm_array.shape[-1], params.fit_total_time)
    time_slice = slice(params.fit_lower_time, params.fit_upper_time)
    vprint('fit_time_slice =', time_slice)
    fitting_data = ddm_array[i, j, time_slice]
    time_space = np.arange(fit_lower_time, fit_upper_time)

    guess_lower_time = params.guess_lower_time or fit_lower_time
    guess_upper_time = params.guess_upper_time or fit_total_time
    guess_time_slice = slice(guess_lower_time, guess_upper_time)
    vprint('guess_time_slice =', guess_time_slice)
    guessing_data = ddm_array[i, j, guess_time_slice]

    if params.c1:
        C1_guess, C1_lower_bound, C1_upper_bound = tuple(params.c1)
    else:
        C1_guess, C1_lower_bound, C1_upper_bound = guess_C1(guessing_data)

    if params.freq:
        freq_guess, freq_lower_bound, freq_upper_bound = tuple(params.freq)
    else:
        freq_guess, freq_lower_bound, freq_upper_bound = guess_freq(guessing_data, C1_guess)

    if params.tau2:
        tau2_guess, tau2_lower_bound, tau2_upper_bound = tuple(params.tau2)
        C2_guess, C2_lower_bound, C2_upper_bound = guess_C2(guessing_data)
    else:
        (C2_guess, C2_lower_bound, C2_upper_bound), \
            (tau2_guess, tau2_lower_bound, tau2_upper_bound) = \
            guess_C2_tau2(guessing_data, C1_guess, freq_guess)

    if params.c2:
        C2_guess, C2_lower_bound, C2_upper_bound = tuple(params.c2)

    ddm_samples = fit_total_time - time_space
    ddm_sigma = np.reciprocal(np.sqrt(ddm_samples))
    vprint('sigma =', ddm_sigma)

    kwargs = {
        'ydata': fitting_data,
        'xdata': time_space,
        'f': taumodel_func,
        'jac': taumodel_jac,
        'p0': np.array([C1_guess, C2_guess, freq_guess, tau2_guess]),
        'bounds': (np.array([C1_lower_bound, C2_lower_bound, freq_lower_bound, tau2_lower_bound]),
                   np.array([C1_upper_bound, C2_upper_bound, freq_upper_bound, tau2_upper_bound])),
        'ftol': 5e-16,
        'xtol': 5e-16,
        'gtol': 5e-16,
        'sigma': ddm_sigma,
        'absolute_sigma': True
        # 'method': 'dogbox',
    }
    vprint('initial guess =', kwargs['p0'])
    vprint('lower bounds = ', kwargs['bounds'][0])
    vprint('upper bounds = ', kwargs['bounds'][1])
    popt, pcov = curve_fit(**kwargs)

    print('popt =         ', popt)
    popt = tuple(popt)
    pvar = np.diagonal(pcov, 0, -2, -1)
    perr = np.sqrt(pvar)
    # perr_frac = np.abs(perr / popt)
    perr_frac_total = np.sqrt(np.sum(pvar / np.square(popt)))
    print('perr =         ', perr)
    print(perr_frac_total)
    time_plotting_space = np.linspace(0, plot_max_time + 0.5, 2000)
    fitted_model = taumodel_func(time_plotting_space, *popt)

    if params.plot:
        fig, ax = plt.subplots(figsize=(8., 6.))
        fig: matplotlib.figure.Figure
        ax: matplotlib.axes.Axes
        ax.plot(ddm_array[i, j, 0:plot_max_time])
        ax.plot(time_plotting_space, fitted_model)
        ax.set_xlabel('$\\tau$ (frames)')
        fig.suptitle(f'$q_x={j}$, $q_y={-i}$')
        if params.save is None:
            plt.show()
        else:
            if params.save:
                fig_path = params.save
            else:
                fig_path = os.environ['FIGURE_PATH'] + f'/q_x-{j:02},q_y-{-i:02}.png'
            fig.savefig(fig_path, dpi=300)


if __name__ == '__main__':
    main()
