"""
Fit an image structure function to a provided model
"""
import numpy as np
from scipy.optimize import curve_fit
from waterwave_ddm.ddm_polar_inspect import polar_space
from waterwave_ddm.models import models, models_jac, models_initial_guess, \
    models_params_lower_bound, models_params_upper_bound
from typing import Tuple


def prepare_xydata(ddm_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare xdata and ydata according to scipy.optimize.curve_fit"""
    r, a, *_ = polar_space(ddm_array.shape)
    tau = np.arange(0, ddm_array.shape[2])

    xdata = np.empty((3, ddm_array.size))
    xdata[0] = np.repeat(r.flatten('C'), tau.size)
    xdata[1] = np.repeat(a.flatten('C'), tau.size)
    xdata[2] = np.tile(tau, r.size)

    ydata = ddm_array.flatten('C')

    return xdata, ydata


def fit_models(ddm_array: np.ndarray):
    xdata, ydata = prepare_xydata(ddm_array)

    for model, jac, initial_guess, lower_bound, upper_bound in \
            zip(models, models_jac, models_initial_guess, models_params_lower_bound, models_params_upper_bound):
        # TODO: Calculate sigmas and include in the curve_fit function call
        optimized_params, pcov = curve_fit(model, xdata, ydata,
                                           jac=jac,
                                           p0=initial_guess,
                                           bounds=(lower_bound, upper_bound))
        yield optimized_params, pcov


def main():
    import argparse
    import pathlib
    parser = argparse.ArgumentParser()
    # parser.add_argument('--cutoffij', type=int, default=1)
    parser.add_argument('--cutofftau', type=int, default=120)
    # parser.add_argument('-o', '--out', type=pathlib.Path, default=False)
    # parser.add_argument('-ao', '--animation_out', type=pathlib.Path, default=False)
    parser.add_argument('ddm_npy_path', type=pathlib.Path)
    params = parser.parse_args()

    ddm_array: np.ndarray = np.load(params.ddm_npy_path)
    print('ddm_array.shape: ', ddm_array.shape)

    for optimized_params, pcov in fit_models(ddm_array[:, :, :params.cutofftau]):
        print(optimized_params)
        perr = np.sqrt(np.diag(pcov))
        print(perr)


if __name__ == '__main__':
    main()
