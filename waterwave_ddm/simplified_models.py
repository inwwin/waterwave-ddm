"""
Simplified models ISF function for fitting in single domain

tau refers to lag time
"""
import numpy as np
from scipy.optimize import curve_fit
from typing import Tuple


def taumodel_func(tau, C1, C2, freq, tau2) -> np.ndarray:
    """
    Calculate model function in tau domain as given in
    [this image](https://github.com/inwwin/waterwave-ddm/blob/master/assets/modelv1_simplified.png)

    expect tau to be an 1d array
    """
    phase = freq * tau
    cos = np.cos(phase)
    e2 = np.exp(-tau / tau2)
    return C1 + C2 * e2 * cos


def taumodel_jac(tau, C1, C2, freq, tau2) -> np.ndarray:
    """
    Calculate jacobian for the model function in tau model as given in
    [this image](https://github.com/inwwin/waterwave-ddm/blob/master/assets/modelv1_simplified.png)

    expect tau to be an 1d array
    """
    phase = freq * tau
    cos = np.cos(phase)
    sin = np.sin(phase)
    e2 = np.exp(-tau / tau2)
    jac = np.empty((tau.size, 4))
    jac[:, 0] = 1
    jac[:, 1] =      e2 * cos
    jac[:, 2] = C2 * e2 * sin * -freq
    jac[:, 3] = C2 * e2 * cos * (tau / np.square(tau2))
    return jac


def fit_taumodel_iteratively(data: np.ndarray, progress_report=None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Fit taumodel along tau domain for each data point in data.
    Expect the last dimension in data to be the tau domain.

    Return
    ======
    A tuple of `popt` and `pcov`
    They are ndarrays with the same shape as data except the last dimension which is replaced by the fitted parameters
    """
    # Calculate appropriate shapes
    params_count = 4
    original_data_shape = data.shape
    points_shape = list(data.shape)
    times = points_shape.pop()
    # data_points: int = np.prod(points_shape)
    popt_shape = points_shape.copy()
    popt_shape.append(params_count)
    pcov_shape = popt_shape.copy()
    pcov_shape.append(params_count)
    # interm_data_shape = (data_points, times)
    # interm_popt_shape = (data_points, params_count)
    # interm_pcov_shape = (data_points, params_count, params_count)
    # assert np.prod(interm_data_shape) == data.size
    # assert np.prod(interm_popt_shape) == np.prod(popt_shape)
    # assert np.prod(interm_pcov_shape) == np.prod(pcov_shape)

    # Define returning array with required shapes
    popt = np.empty(popt_shape)
    pcov = np.empty(pcov_shape)
    data.shape = (-1, times)
    popt.shape = (-1, params_count)
    pcov.shape = (-1, params_count, params_count)
    data_points = data.shape[0]
    assert data_points == popt.shape[0]
    assert data_points == pcov.shape[0]

    # To adjust the tau domain used, different xdata should be passed to this function
    kwargs.setdefault('xdata', np.arange(times))
    kwargs.setdefault('f', taumodel_func)
    kwargs.setdefault('jac', taumodel_jac)
    #                C1     C2   freq  tau
    initial_guess = [2.,   -1.,    1.,  1.]
    lower_bounds  = [1.9, -90.,    0.,  0.1]
    upper_bounds  = [2.1,  -0.5,  50., 20.]
    kwargs.setdefault('p0', initial_guess)
    kwargs.setdefault('bounds', (lower_bounds, upper_bounds))
    kwargs.pop('ydata', None)

    if callable(progress_report):
        progress_report(0, data_points)
    for i in range(data_points):
        if not np.isnan(data[i]).any():
            popt[i], pcov[i] = curve_fit(ydata=data[i], **kwargs)
        else:
            popt[i] = np.nan
            pcov[i] = np.nan
        if callable(progress_report):
            progress_report(i + 1, data_points)

    # Reshape back and then return
    data.shape = original_data_shape
    popt.shape = popt_shape
    pcov.shape = pcov_shape
    return popt, pcov


def main():
    import argparse
    import pathlib
    from waterwave_ddm.ddm_polar_inspect import map_to_polar

    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    # TODO: Accept initial_guess and bounds
    parser.add_argument('-t', '--timerange', type=int, nargs=2, metavar=('mintime', 'maxtime'), default=(5, 120))
    parser.add_argument('ddm_npy_path', type=pathlib.Path)
    mode_parsers = parser.add_subparsers(title='mode', dest='mode', required=True)
    bysector_parser = mode_parsers.add_parser('sector')
    bysector_parser.add_argument('-a', '--angular_bins', type=int, default=19)
    bysector_parser.add_argument('-p', '--radial_bin_size', type=int, default=2)
    bysector_parser.add_argument('angle_index', type=int)
    params = parser.parse_args()

    ddm_array: np.ndarray = np.load(params.ddm_npy_path)
    print('ddm_array.shape: ', ddm_array.shape)

    if params.mode == 'sector':
        ddm_polar, median_angle, median_radius, average_angle, average_radius = \
            map_to_polar(ddm_array, params.angular_bins, params.radial_bin_size, ddm_array.shape[1] - 1)
        print('polar_result.shape: ', ddm_polar.shape)

        time_slice = slice(params.timerange[0], params.timerange[1])
        fitting_sector = ddm_polar[params.angle_index, 1:, time_slice]
        time_space = np.arange(params.timerange[0], params.timerange[1])

        popt, pcov = fit_taumodel_iteratively(fitting_sector,
                                              (lambda i, t: print(f'{i}/{t}', end='\r', flush=True))
                                              if params.verbose else None,
                                              xdata=time_space)
        if params.verbose:
            print()

        pvar = np.diagonal(pcov, 0, -2, -1)
        perr = np.sqrt(pvar)
        perr_frac = perr / popt
        perr_frac_total = np.sqrt(np.sum(pvar / np.square(popt)))
        print(popt)
        print(perr_frac)
        print(perr_frac_total)

        # TODO: Then display these in an animated plot


if __name__ == '__main__':
    main()
