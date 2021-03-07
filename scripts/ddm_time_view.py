"""
A helper script for investigating the lag time dependence at a particular wavevector of
an image structure function
"""
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.animation
from waterwave_ddm.simplified_models import taumodel_func, taumodel_jac
from scipy.optimize import curve_fit

#                 C1     C2    freq    alpha
params_names  = ('C1',  'C2', 'freq', 'alpha')
initial_guess = [  2.,   -1.,     1.,    1.  ]  # noqa: E201, E202
lower_bounds  = [  1.9, -90.,     0.,    0.00001 ]  # noqa: E201, E202
upper_bounds  = [  2.1,  -0.5,   50.,    3.  ]  # noqa: E201, E202
parser = argparse.ArgumentParser()
coord_group = parser.add_mutually_exclusive_group(required=True)
coord_group.add_argument('-ij', type=int, nargs=2, metavar=('i', 'j'))
coord_group.add_argument('-xy', type=int, nargs=2, metavar=('x', 'y'))
coord_group.add_argument('-ar', type=float, nargs=2, metavar=('angle', 'radius'))
parser.add_argument('-tm', '--ti_max', type=int, default=None)
parser.add_argument('-tf', '--fittimerange', dest='timerange',
                    type=int, nargs=2, metavar=('mintime', 'maxtime'), default=(5, 120))
parser.add_argument('-p0', type=float, nargs=4, default=initial_guess, metavar=params_names)
parser.add_argument('-lb', type=float, nargs=4, default=lower_bounds, metavar=params_names, dest='lower_bounds')
parser.add_argument('-ub', type=float, nargs=4, default=upper_bounds, metavar=params_names, dest='upper_bounds')
parser.add_argument('-np', '--noplot', dest='plot', action='store_false')
parser.add_argument('ddm_npy_path', type=pathlib.Path)
# parser.add_argument('save_html_path', type=pathlib.Path, default=False)
params = parser.parse_args()

ddm_array = np.load(params.ddm_npy_path)
print(ddm_array.shape)

# ddm_array = np.fft.fftshift(ddm_array, axes=0)
assert ddm_array.shape[0] % 2 == 1
imax = ddm_array.shape[0] // 2
if params.ij:
    i = params.ij[0]
    j = params.ij[1]
else:
    if params.ar:
        af = params.ar[0]
        rf = params.ar[1]
        xf = rf * np.cos(af * np.pi / 180.)
        yf = rf * np.sin(af * np.pi / 180.)
        x = int(np.round(xf))
        y = int(np.round(yf))
    elif params.xy:
        x = params.xy[0]
        y = params.xy[1]
    j = x
    i = -y
if i < 0:
    i += ddm_array.shape[0]

time_slice = slice(params.timerange[0], params.timerange[1])
fitting_data = ddm_array[i, j, time_slice]
time_space = np.arange(params.timerange[0], params.timerange[1])

kwargs = {
    'ydata': fitting_data,
    'xdata': time_space,
    'f': taumodel_func,
    'jac': taumodel_jac,
    'p0': params.p0,
    'bounds': (params.lower_bounds, params.upper_bounds),
    'ftol': 5e-16,
    'xtol': 5e-16,
    'gtol': 5e-16,
    # 'method': 'dogbox',
}
popt, pcov = curve_fit(**kwargs)

popt = tuple(popt)
pvar = np.diagonal(pcov, 0, -2, -1)
perr = np.sqrt(pvar)
perr_frac = np.abs(perr / popt)
perr_frac_total = np.sqrt(np.sum(pvar / np.square(popt)))
print(popt)
print(perr)
print(perr_frac_total)
time_plotting_space = np.linspace(0, params.ti_max + 0.5, 2000)
fitted_model = taumodel_func(time_plotting_space, *popt)

if params.plot:
    fig, ax = plt.subplots(figsize=(8., 6.))
    fig: matplotlib.figure.Figure
    ax: matplotlib.axes.Axes
    ax.plot(ddm_array[i, j, 0:params.ti_max])
    ax.plot(time_plotting_space, fitted_model)
    ax.set_xlabel('$\\tau$ (frames)')
    plt.show()
