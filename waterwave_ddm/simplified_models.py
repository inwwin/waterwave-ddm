"""
Model ISF function for fitting in single tau domain

tau refers to lag time
"""
import warnings
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from typing import Tuple
from numpy.typing import ArrayLike


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
    jac[:, 2] = C2 * e2 * sin * -tau
    jac[:, 3] = C2 * e2 * cos * (tau / np.square(tau2))
    return jac


def guess_C1(ddm_array: np.ndarray, sd_factor: float = 2.5) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Guess C1 parameter and its bound based on the mean and standard deviation of ddm_array

    Return
    ======
    A Tuple of guessed C1 and its lower and upper bound
    """
    C1_guess = np.mean(ddm_array, axis=-1, keepdims=True)
    assert np.all(C1_guess > 0)
    C1_std = np.std(ddm_array, ddof=1, axis=-1, keepdims=True)
    C1_lower_bound = C1_guess - sd_factor * C1_std
    C1_upper_bound = C1_guess + sd_factor * C1_std

    C1_lower_bound_too_low = C1_lower_bound < C1_guess * .5
    C1_lower_bound[C1_lower_bound_too_low] = C1_guess[C1_lower_bound_too_low] * .5

    C1_upper_bound_too_high = C1_upper_bound > C1_guess * 1.5
    C1_upper_bound[C1_upper_bound_too_high] = C1_guess[C1_upper_bound_too_high] * 1.5

    C1_guess = C1_guess.squeeze(axis=-1)
    C1_lower_bound = C1_lower_bound.squeeze(axis=-1)
    C1_upper_bound = C1_upper_bound.squeeze(axis=-1)
    return C1_guess, C1_lower_bound, C1_upper_bound


def guess_C2(C1: ArrayLike) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Preliminarily guess C2 parameter and its bound based on the value of C1

    Return
    ======
    A Tuple of guessed C2 and its lower and upper bound
    """
    return -C1, -1.2 * C1, -.05 * C1


def guess_freq(ddm_array: np.ndarray, C1: ArrayLike, max_observation: int = 3, sd_factor: float = 2.5) \
        -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Guess freq parameter and its bound based on the intercept of ddm_array and C1

    Return
    ======
    A Tuple of guessed freq and its lower and upper bound
    """
    C1 = np.array(C1)
    expect_shape = list(ddm_array.shape)
    expect_shape.pop()
    expect_shape = tuple(expect_shape)
    # expect C1 with broadcastable shape to ddm_array except the last dimension
    C1 = np.broadcast_to(C1, expect_shape)

    it_ddm_array = np.nditer(ddm_array, flags=['external_loop'], order='C')
    # 1st: guess, 2nd: lower_bound, 3rd: upper_bound
    it_params = np.nditer([C1, None, None, None], order='C')
    with it_ddm_array, it_params:
        it_zip = zip(it_ddm_array, it_params)
        for i_ddm_array, \
                (i_C1, i_freq_guess, i_freq_lower_bound, i_freq_upper_bound) \
                in it_zip:
            # Find where ddm_array cross C1
            above_C1 = i_ddm_array >= i_C1
            cross_frames = np.bitwise_xor(above_C1[..., 1:], above_C1[..., :-1]).nonzero()[0]
            periods = cross_frames[2:] - cross_frames[:-2]
            freqs =  2 * np.pi / periods
            observations = min(max_observation, freqs.size)
            if observations < 3:
                warnings.warn(f'observations = {observations} < 3 may give unreliable estimate')
            ddof = 1 if observations > 1 else 0
            freq_std = np.std(freqs[:observations], ddof=ddof)
            freq_delta = max(freq_std * sd_factor, 0.1)
            i_freq_guess[...]       = freqs[0]
            i_freq_lower_bound[...] = freqs[0] - freq_delta
            i_freq_upper_bound[...] = freqs[0] + freq_delta
        return it_params.operands[1], it_params.operands[2], it_params.operands[3]


def guess_C2_tau2(ddm_array: np.ndarray, C1: ArrayLike, freq: ArrayLike, max_observation: int = 5) \
        -> Tuple[Tuple[ArrayLike, ArrayLike, ArrayLike], Tuple[ArrayLike, ArrayLike, ArrayLike]]:
    """
    Guess C2 and tau2 parameters and their bounds based on the decays of the peaks and valleys of ddm_array

    Return
    ======
    A Tuple of tuple of guessed C2, tau2 and their lower and upper bound
    """
    def expodecay_func(x, aC2, tau2):
        return aC2 * np.exp(-x / tau2)

    def expodecay_jac(x, aC2, tau2):
        jac = np.empty((x.size, 2))
        jac[:, 0] = np.exp(-x / tau2)
        jac[:, 1] = aC2 * jac[:, 0] * (x / np.square(tau2))
        return jac

    C1 = np.array(C1)
    freq = np.array(freq)
    expect_shape = list(ddm_array.shape)
    expect_shape.pop()
    expect_shape = tuple(expect_shape)
    # expect C1 and freq with broadcastable shape to ddm_array except the last dimension
    C1 = np.broadcast_to(C1, expect_shape)
    freq = np.broadcast_to(freq, expect_shape)

    it_ddm_array = np.nditer(ddm_array, flags=['external_loop'], order='C', op_flags=[['readwrite']])
    # 1st: guess, 2nd: lower_bound, 3rd: upper_bound
    it_params = np.nditer([C1, freq, None, None, None, None, None, None], order='C')
    with it_ddm_array, it_params:
        it_zip = zip(it_ddm_array, it_params)
        for i_ddm_array, \
                (i_C1, i_freq,
                 i_C2_guess, i_C2_lower_bound, i_C2_upper_bound,
                 i_tau2_guess, i_tau2_lower_bound, i_tau2_upper_bound) \
                in it_zip:
            period = 2 * np.pi / i_freq
            # peak_distance = 0.8 * period
            # peaks, _ = find_peaks(i_ddm_array, distance=peak_distance)
            peak_width = period / 4
            peaks, _ = find_peaks(i_ddm_array, width=peak_width)
            valleys, _ = find_peaks(-i_ddm_array, width=peak_width)
            peaks_height = i_ddm_array[peaks] - i_C1
            valleys_height = -i_ddm_array[valleys] + i_C1
            # detect the first time that the peaks no longer decrease
            peaks_height_drop = np.nonzero(peaks_height[1:] - peaks_height[:-1] > 0)[0]
            valleys_height_drop = np.nonzero(valleys_height[1:] - valleys_height[:-1] > 0)[0]
            peaks_height_last_drop = peaks_height_drop[0] + 1 if peaks_height_drop.size else len(peaks)
            valleys_height_last_drop = valleys_height_drop[0] + 1 if valleys_height_drop.size else len(valleys)

            # extract the first extrema that monotonically decrease
            extrema = np.concatenate((peaks[:peaks_height_last_drop], valleys[:valleys_height_last_drop]))
            extrema_heights = np.concatenate((peaks_height[:peaks_height_last_drop],
                                              valleys_height[:valleys_height_last_drop]))
            assert extrema.size == extrema_heights.size
            observations = min(extrema.size, max_observation)
            assert observations >= 2, 'Require observations >= 2 to extract 2 free parameters: C2i, tau2i'
            extrema_argsort = np.argsort(extrema)
            extrema_limited_argsort = extrema_argsort[:observations]
            extrema_limited = extrema[extrema_limited_argsort]
            extrema_limited_heights = extrema_heights[extrema_limited_argsort]
            gC2, gC2_lower_bound, gC2_upper_bound = guess_C2(i_C1)
            gpC2 = -gC2
            gpC2_lower_bound = -gC2_upper_bound
            gpC2_upper_bound = -gC2_lower_bound
            C2tau2opt, C2tau2cov = curve_fit(expodecay_func, extrema_limited, extrema_limited_heights,
                                             p0=[gpC2, period],
                                             bounds=([gpC2_lower_bound, 0],
                                                     [gpC2_upper_bound, np.inf]))
            i_C2_guess[...] = -C2tau2opt[0]
            i_C2_lower_bound[...] = -1.5 * C2tau2opt[0]
            i_C2_upper_bound[...] = -0.5 * C2tau2opt[0]
            i_tau2_guess[...] = C2tau2opt[1]
            i_tau2_lower_bound[...] = min(period / 4, C2tau2opt[1] / 4)
            i_tau2_upper_bound[...] = C2tau2opt[1] * 2

        return (it_params.operands[2], it_params.operands[3], it_params.operands[4]), \
               (it_params.operands[5], it_params.operands[6], it_params.operands[7]),


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
