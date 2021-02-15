"""Read FFT data and calculate normalised Image Structure Function or Intermediate Scattering Function and save"""
from cddm.core import acorr, normalize, stats
# from cddm.multitau import log_average
# from cddm.video import crop, asmemmaps  # asarrays  # multiply, normalize_video,
# from cddm.fft import rfft2  # , normalize_fft
# from cddm.core import acorr, normalize, stats
# from cddm.multitau import log_average
import numpy as np


def acorr_save(fft_array, path_out, method='diff', mode='diff'):
    # These codes are copied from the examples in cddm package

    #: now perform auto correlation calculation with default parameters
    data = acorr(fft_array, method=method)
    bg, var = stats(fft_array)

    #: perform normalization and merge data
    data_lin = normalize(data, bg, var, scale=True, mode=mode)

    np.save(path_out / f'auto_correlate_data_lin_{method}_{mode}.npy', data_lin)

    # #: change size, to define time resolution in log space
    # x, y = log_average(data_lin, size=16)

    # #: save the normalized data to numpy files
    # np.save(path_out / 'auto_correlate_t.npy', x)
    # np.save(path_out / 'auto_correlate_data.npy', y)

    return ({
        'bg': bg,
        'var': var,
        'data_lin_shape': data_lin.shape,
        # 't_shape': x.shape,
        # 'data_shape': y.shape,
    }, data_lin)  # , x, y)


def main():
    import argparse
    import pathlib
    # import json

    parser = argparse.ArgumentParser()
    # diff for Image Structure Function
    # fft or corr for Intermediate Scattering Function
    parser.add_argument('--method', default='diff', choices=['diff', 'fft', 'corr'])
    parser.add_argument('--mode', default='diff', choices=['diff', 'corr'])
    parser.add_argument('fft_in', type=pathlib.Path)
    parser.add_argument('ddm_out', type=pathlib.Path)
    params = parser.parse_args()

    # print(params)

    try:
        params.ddm_out.mkdir(mode=0o755, parents=True, exist_ok=True)
    except FileExistsError:
        parser.error('ddm_out must be a directory')

    fft_array = np.load(params.fft_in / 'fft_array_0.npy')
    acorr_info, data_lin = acorr_save(fft_array, params.ddm_out, params.method, params.mode)

    print(acorr_info)
    # json not support np.ndarray and tuple
    # with open(params.ddm_out / 'acorr_info.json', 'w') as j:
    #     json.dump(acorr_info, j, indent=4)


if __name__ == '__main__':
    main()
