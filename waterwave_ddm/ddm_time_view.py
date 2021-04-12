"""
A helper script for investigating the lag time dependence at a particular wavevector of
an image structure function
"""
import numpy as np
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.animation
from waterwave_ddm.simplified_models import taumodel_func, taumodel_jac, guess_C1, guess_C2, guess_freq, guess_C2_tau2
from waterwave_ddm.ddm_polar_inspect import map_to_polar
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


class DDMDecayCosineViewer:

    def __init__(self, ddm_array: np.ndarray, plot_max_time=np.inf, fit_lower_time=5, fit_upper_time=np.inf,
                 fit_total_time=np.inf, guess_lower_time=None, guess_upper_time=None, printer=print):
        self.vprint = printer
        self.ddm_array: np.ndarray = ddm_array
        self.vprint('array_shape =', self.ddm_array.shape)
        self.fit_lower_time = max(0, fit_lower_time)
        self.plot_max_time =  min(self.ddm_array.shape[-1], plot_max_time)
        self.fit_upper_time = min(self.ddm_array.shape[-1], fit_upper_time)
        self.fit_total_time = min(self.ddm_array.shape[-1], fit_total_time)
        self.time_space = np.arange(self.fit_lower_time, self.fit_upper_time)
        self.time_slice = slice(self.fit_lower_time, self.fit_upper_time)
        self.vprint('fit_time_slice =', self.time_slice)

        self.guess_lower_time = guess_lower_time or self.fit_lower_time
        self.guess_upper_time = guess_upper_time or self.fit_total_time
        self.guess_time_slice = slice(self.guess_lower_time, self.guess_upper_time)
        self.vprint('guess_time_slice =', self.guess_time_slice)

        self.plot_time_slice = slice(0, self.plot_max_time)
        self.time_plotting_space = np.linspace(0, self.plot_max_time + 0.5, 2000)

        self.coord_system = None
        self.c1 = None
        self.c2 = None
        self.freq = None
        self.tau2 = None

        self.prepare_model()

    def set_coord_cartesian(self, i, j):
        self.coord_system = 'cart'
        self.cart_i = i
        self.cart_j = j

        def update_input_data():
            self.fitting_data = self.ddm_array[self.cart_i, self.cart_j, self.time_slice]
            self.guessing_data = self.ddm_array[self.cart_i, self.cart_j, self.guess_time_slice]
            self.plotting_data = self.ddm_array[self.cart_i, self.cart_j, self.plot_time_slice]

        def handle_key_press(event):
            if event.key == 'h':
                self.cart_j -= 1
            elif event.key == 'l':
                self.cart_j += 1
            elif event.key == 'j':
                self.cart_i += 1
            elif event.key == 'k':
                self.cart_i -= 1

            if event.key in ('h', 'j', 'k', 'l'):
                self.cart_i = self.cart_i % self.ddm_array.shape[0]
                self.cart_j = self.cart_j % self.ddm_array.shape[1]

                self.update_input_data()
                self.fit_and_replot()

        self.update_input_data = update_input_data
        self.handle_key_press = handle_key_press
        update_input_data()

    def set_coord_polar(self, angle_index, radius_index, angular_bins=18, angle_offset=0., radial_bin_size=2):
        self.coord_system = 'polar'
        self.polar_a = radius_index
        self.polar_r = radius_index
        self.ddm_polar, median_angle, median_radius, average_angle, average_radius, \
            lower_angle, lower_radius, upper_angle, upper_radius, blank_bins = \
            map_to_polar(self.ddm_array, angular_bins, radial_bin_size, self.ddm_array.shape[1] - 1,
                         angle_offset, True)
        self.vprint('ddm_polar.shape: ', self.ddm_polar.shape)
        self.vprint('blank_bins:', blank_bins)

        def update_input_data():
            self.vprint('angle between:', lower_angle[self.polar_a], upper_angle[self.polar_a])
            self.vprint('median_angle:', median_angle[self.polar_a])
            self.vprint('radius between:', lower_radius[self.polar_r], upper_radius[self.polar_r])
            self.vprint('median_radius:', median_radius[self.polar_r])
            self.a_av = average_angle[self.polar_a, self.polar_r]
            self.r_av = average_radius[self.polar_a, self.polar_r]
            self.vprint('average_angle:', self.r_av)
            self.vprint('average_radius:', self.a_av)

            self.fitting_data = self.ddm_polar[self.polar_a, self.polar_r, self.time_slice]
            self.guessing_data = self.ddm_polar[self.polar_a, self.polar_r, self.guess_time_slice]
            self.plotting_data = self.ddm_polar[self.polar_a, self.polar_r, self.plot_time_slice]

        def handle_key_press(event):
            if event.key == 'h':
                self.polar_a -= 1
            elif event.key == 'l':
                self.polar_a += 1
            elif event.key == 'j':
                self.polar_r += 1
            elif event.key == 'k':
                self.polar_r -= 1

            if event.key in ('h', 'j', 'k', 'l'):
                self.polar_a = self.polar_a % self.ddm_polar.shape[0]
                self.polar_r = max(0, self.polar_r)
                self.polar_r = min(self.ddm_polar.shape[1] - 1, self.polar_r)

                self.update_input_data()
                self.fit_and_replot()

        self.update_input_data = update_input_data
        self.handle_key_press = handle_key_press
        update_input_data()

    @property
    def suptitle(self):
        if self.coord_system == 'cart':
            x = self.cart_j
            y = -self.cart_i
            ylimit = self.ddm_array.shape[0] // 2
            if y < -ylimit:
                y += self.ddm_array.shape[0]
            return f'$q_x={x}$, $q_y={y}$'
        elif self.coord_system == 'polar':
            return f'$\\langle q_r \\rangle={self.r_av:.1f}$, ' \
                   f'$\\langle q_\\theta \\rangle={self.a_av:.1f}\\degree$'

    @property
    def auto_filename(self):
        if self.coord_system == 'cart':
            x = self.cart_j
            y = -self.cart_i
            ylimit = self.ddm_array.shape[0] // 2
            if y < -ylimit:
                y += self.ddm_array.shape[0]
            return f'/q_x-{x:02},q_y-{y:02}.png'
        elif self.coord_system == 'polar':
            return f'/r_av-{self.r_av:02},a_av-{self.a_av:02}.png'

    def process_guessing(self):
        if self.c1:
            C1_guess, C1_lower_bound, C1_upper_bound = tuple(self.c1)
        else:
            C1_guess, C1_lower_bound, C1_upper_bound = guess_C1(self.guessing_data)

        if self.freq:
            freq_guess, freq_lower_bound, freq_upper_bound = tuple(self.freq)
        else:
            freq_guess, freq_lower_bound, freq_upper_bound = guess_freq(self.guessing_data, C1_guess)

        if self.tau2:
            tau2_guess, tau2_lower_bound, tau2_upper_bound = tuple(self.tau2)
            C2_guess, C2_lower_bound, C2_upper_bound = guess_C2(self.guessing_data)
        else:
            (C2_guess, C2_lower_bound, C2_upper_bound), \
                (tau2_guess, tau2_lower_bound, tau2_upper_bound) = \
                guess_C2_tau2(self.guessing_data, C1_guess, freq_guess)

        if self.c2:
            C2_guess, C2_lower_bound, C2_upper_bound = tuple(self.c2)

        self.fit_kwargs['p0'] = np.array([C1_guess, C2_guess, freq_guess, tau2_guess])
        self.fit_kwargs['bounds'] = (np.array([C1_lower_bound, C2_lower_bound, freq_lower_bound, tau2_lower_bound]),
                                     np.array([C1_upper_bound, C2_upper_bound, freq_upper_bound, tau2_upper_bound]))

        return self.fit_kwargs['p0'], self.fit_kwargs['bounds']

    def prepare_model(self):
        ddm_samples = self.fit_total_time - self.time_space
        ddm_sigma = np.reciprocal(np.sqrt(ddm_samples))
        # self.vprint('sigma =', ddm_sigma)
        self.fit_kwargs = {
            'xdata': self.time_space,
            'f': taumodel_func,
            'jac': taumodel_jac,
            'p0': None,
            'bounds': None,
            'ftol': 5e-16,
            'xtol': 5e-16,
            'gtol': 5e-16,
            'sigma': ddm_sigma,
            'absolute_sigma': True,
            # 'method': 'dogbox',
        }

    def fit_model(self):
        if self.coord_system is None:
            raise ValueError('Please set a coordinate system before fitting model')

        self.process_guessing()
        self.vprint('initial guess =', self.fit_kwargs['p0'])
        self.vprint('lower bounds = ', self.fit_kwargs['bounds'][0])
        self.vprint('upper bounds = ', self.fit_kwargs['bounds'][1])
        popt, pcov = curve_fit(ydata=self.fitting_data, **self.fit_kwargs)

        print('popt =         ', popt)
        self.popt = tuple(popt)
        self.pvar = np.diagonal(pcov, 0, -2, -1)
        self.perr = np.sqrt(self.pvar)
        # perr_frac = np.abs(perr / popt)
        self.perr_frac_total = np.sqrt(np.sum(self.pvar / np.square(popt)))
        print('perr =         ', self.perr)
        print(self.perr_frac_total)
        self.fitted_model = taumodel_func(self.time_plotting_space, *self.popt)

        return popt, pcov, self.perr, self.perr_frac_total

    def plot(self, plot_model=False, fig=None, ax=None):
        if None in (fig, ax):
            self.fig, self.ax = plt.subplots(figsize=(8., 6.))
        else:
            self.fig = fig
            self.ax = ax
        self.fig: matplotlib.figure.Figure
        self.ax: matplotlib.axes.Axes

        self.data_line, = self.ax.plot(self.plotting_data)
        if plot_model:
            self.model_line, = self.ax.plot(self.time_plotting_space, self.fitted_model)
        else:
            self.model_line = None

        self.ax.axvline(self.fit_lower_time, linestyle=':')
        self.ax.axvline(self.fit_upper_time, linestyle=':')
        self.ax.set_xlabel('$\\tau$ (frames)')
        self.fig.suptitle(self.suptitle)
        self.fig.canvas.mpl_connect('key_press_event', lambda event: self.handle_key_press(event))

    def replot(self):
        self.data_line.set_ydata(self.plotting_data)
        if self.model_line is not None and self.fitted_model is not None:
            self.model_line.set_ydata(self.fitted_model)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.suptitle(self.suptitle)
        self.fig.canvas.draw()

    def fit_and_plot(self):
        self.fit_model()
        self.plot(True)

    def fit_and_replot(self):
        self.fit_model()
        self.replot()


def parse_ij(ij, ar, xy, array_shape):
    assert array_shape[0] % 2 == 1
    # imax = shape[0] // 2
    if ij:
        i = ij[0]
        j = ij[1]
        x = j
        y = -i
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
    return i, j, x, y


def main():
    # params_names  = ('C1',  'C2', 'freq', 'tau')
    meta_params = ('initial_guess', 'lower_bound', 'upper_bound')
    # initial_guess = [  2.,   -1.,     1.,    1.  ]  # noqa: E201, E202
    # lower_bounds  = [  1.9, -90.,     0.,    0.1 ]  # noqa: E201, E202
    # upper_bounds  = [  2.1,  -0.5,   50.,   20.  ]  # noqa: E201, E202
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
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
    coord_subparsers = parser.add_subparsers(title='coord', required=True, dest='coord_system')
    cartesian_parser = coord_subparsers.add_parser('cartesian', aliases=['cart'])
    cart_group = cartesian_parser.add_mutually_exclusive_group(required=True)
    cart_group.add_argument('-ij', type=int,   nargs=2, metavar=('i', 'j'))
    cart_group.add_argument('-xy', type=int,   nargs=2, metavar=('x', 'y'))
    cart_group.add_argument('-ar', type=float, nargs=2, metavar=('angle', 'radius'))
    polar_parser = coord_subparsers.add_parser('polar')
    polar_parser.add_argument('-a', '--angular_bins', type=int, default=18)
    polar_parser.add_argument('-ao', '--angle_offset', type=float, default=0.)
    polar_parser.add_argument('-p', '--radial_bin_size', type=int, default=2)
    polar_parser.add_argument('angle_index', type=int)
    polar_parser.add_argument('radius_index', type=int)

    params = parser.parse_args()
    vprint = verbose_print_wrapper(params.verbose)
    ddm_array = np.load(params.ddm_npy_path)
    vprint('array_shape =', ddm_array.shape)

    viewer = DDMDecayCosineViewer(
        ddm_array=ddm_array,
        plot_max_time=params.plot_max_time,
        fit_lower_time=params.fit_lower_time,
        fit_upper_time=params.fit_upper_time,
        fit_total_time=params.fit_total_time,
        guess_lower_time=params.guess_lower_time,
        guess_upper_time=params.guess_upper_time,
        printer=vprint,
    )
    viewer.c1 = params.c1
    viewer.c2 = params.c2
    viewer.freq = params.freq
    viewer.tau2 = params.tau2

    if params.coord_system == 'cart':
        i, j, x, y = parse_ij(params.ij, params.ar, params.xy, ddm_array.shape)
        vprint('initial i, j, x, y =', (i, j, x, y))
        viewer.set_coord_cartesian(i, j)
    elif params.coord_system == 'polar':
        viewer.set_coord_polar(
            angle_index=params.angle_index,
            radius_index=params.radius_index,
            angular_bins=params.angular_bins,
            radial_bin_size=params.radial_bin_size,
            angle_offset=params.angle_offset,
        )

    if params.plot:
        mpl.rcParams['keymap.xscale'] = [',']
        mpl.rcParams['keymap.yscale'] = ['.']
        viewer.fit_and_plot()
        if params.save is None:
            plt.show()
        else:
            if params.save:
                fig_path = params.save
            else:
                fig_path = os.environ['FIGURE_PATH'] + viewer.auto_filename
            viewer.fig.savefig(fig_path, dpi=300)
    else:
        viewer.fit_model()


if __name__ == '__main__':
    main()
