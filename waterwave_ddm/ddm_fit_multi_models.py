"""
Fit an image structure function to a provided model
"""
import numpy as np
from waterwave_ddm.models import models, ISFModel


def main():
    import argparse
    import pathlib

    parser = argparse.ArgumentParser()
    # parser.add_argument('--cutoffij', type=int, default=1)
    parser.add_argument('--maxtime', type=int, default=None)
    # parser.add_argument('-o', '--out', type=pathlib.Path, default=False)
    # parser.add_argument('-ao', '--animation_out', type=pathlib.Path, default=False)
    parser.add_argument('ddm_npy_path', type=pathlib.Path)
    params = parser.parse_args()

    ddm_array: np.ndarray = np.load(params.ddm_npy_path)
    print('ddm_array.shape: ', ddm_array.shape)
    cut_ddm_array = ddm_array[:, :, :params.maxtime]
    print('cut_ddm_array.shape:', cut_ddm_array.shape)
    xdata, ydata = ISFModel.prepare_xydata(cut_ddm_array)

    for model in models:
        params_optimized, params_covariance = model.fit_model(xdata, ydata)
        pvar = np.diag(params_covariance)
        perr = np.sqrt(pvar)
        perr_frac = perr / params_optimized
        perr_frac_total = np.sqrt(np.sum(pvar / np.square(params_optimized)))
        print()
        print('Model number:', model.number)
        print('Params:')
        print(params_optimized)
        print('Errors:')
        print(perr)
        print('Fractional errors:')
        print(perr_frac)
        print('Total fractional errors:', perr_frac_total)


if __name__ == '__main__':
    main()
