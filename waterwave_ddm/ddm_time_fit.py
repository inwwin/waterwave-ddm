import numpy as np
import argparse
from pathlib import Path
from waterwave_ddm.ddm_time_view import DDMDecayCosineViewer


class DDMDecayCosineFitter(DDMDecayCosineViewer):

    def fit_all_wavevectors(self):
        params = np.empty(self.wavevector_shape + (4,))
        errors = np.empty(self.wavevector_shape + (4,))
        errs_frac = np.empty(self.wavevector_shape)
        for _ in self.iterate():
            self.vprint('Considering wavevector index:', self.wavvector_index)
            try:
                popt, _, perr, perr_frac_total = self.fit_model()
            except RuntimeError:
                # If curve_fit could not fit the params
                popt = np.nan
                perr = np.nan
                perr_frac_total = np.nan
            params[self.wavvector_index] = popt
            errors[self.wavvector_index] = perr
            errs_frac[self.wavvector_index] = perr_frac_total
        return params, errors, errs_frac


def main():
    meta_params = ('initial_guess', 'lower_bound', 'upper_bound')
    parser = argparse.ArgumentParser()
    parser.add_argument('-lt', '--lower-time',        dest='fit_lower_time', default=5,      type=int)
    parser.add_argument('-ut', '--upper-time',        dest='fit_upper_time', default=np.inf, type=int)
    parser.add_argument('-tt', '--total-time',        dest='fit_total_time', default=np.inf, type=int)
    parser.add_argument('-glt', '--guess-lower-time', dest='guess_lower_time', default=None, type=int)
    parser.add_argument('-gut', '--guess-upper-time', dest='guess_upper_time', default=None, type=int)
    parser.add_argument('-c1',   nargs=3, default=None, type=float, metavar=meta_params)
    parser.add_argument('-c2',   nargs=3, default=None, type=float, metavar=meta_params)
    parser.add_argument('-freq', nargs=3, default=None, type=float, metavar=meta_params)
    parser.add_argument('-tau2', nargs=3, default=None, type=float, metavar=meta_params)
    parser.add_argument('ddm_npy_path', type=Path)
    coord_subparsers = parser.add_subparsers(title='coord', required=True, dest='coord_system')
    coord_subparsers.add_parser('cartesian', aliases=['cart'])
    polar_parser = coord_subparsers.add_parser('polar')
    polar_parser.add_argument('-a', '--angular_bins', type=int, default=18)
    polar_parser.add_argument('-ao', '--angle_offset', type=float, default=0.)
    polar_parser.add_argument('-p', '--radial_bin_size', type=int, default=2)

    params = parser.parse_args()
    ddm_npy_path = params.ddm_npy_path
    ddm_npy_path: Path
    ddm_array = np.load(ddm_npy_path)
    print('array_shape =', ddm_array.shape)

    def print_null(*args, **kwargs):
        return
        # print(*args, **kwargs)

    fitter = DDMDecayCosineFitter(
        ddm_array=ddm_array,
        fit_lower_time=params.fit_lower_time,
        fit_upper_time=params.fit_upper_time,
        fit_total_time=params.fit_total_time,
        guess_lower_time=params.guess_lower_time,
        guess_upper_time=params.guess_upper_time,
        printer=print_null,
    )
    fitter.c1 = params.c1
    fitter.c2 = params.c2
    fitter.freq = params.freq
    fitter.tau2 = params.tau2

    if params.coord_system.startswith('cart'):
        fitter.set_coord_cartesian(0, 0)
        params.coord_system = 'cartesian'
    elif params.coord_system == 'polar':
        fitter.set_coord_polar(
            angle_index=0,
            radius_index=0,
            angular_bins=params.angular_bins,
            radial_bin_size=params.radial_bin_size,
            angle_offset=params.angle_offset,
        )

    params_fit, params_errors, params_errs_frac = fitter.fit_all_wavevectors()

    params_path = \
        ddm_npy_path.with_stem(ddm_npy_path.stem + '.' + params.coord_system + '.params_fit')
    param_errors_path = \
        ddm_npy_path.with_stem(ddm_npy_path.stem + '.' + params.coord_system + '.param_errors')
    param_error_fracs_total_path = \
        ddm_npy_path.with_stem(ddm_npy_path.stem + '.' + params.coord_system + '.param_error_fracs_total')
    np.save(params_path, params_fit)
    np.save(param_errors_path, params_errors)
    np.save(param_error_fracs_total_path, params_errs_frac)

    if params.coord_system == 'polar':
        median_angle_path = \
            ddm_npy_path.with_stem(ddm_npy_path.stem + '.' + params.coord_system + '.median_angle')
        median_radius_path = \
            ddm_npy_path.with_stem(ddm_npy_path.stem + '.' + params.coord_system + '.median_radius')
        average_angle_path = \
            ddm_npy_path.with_stem(ddm_npy_path.stem + '.' + params.coord_system + '.average_angle')
        average_radius_path = \
            ddm_npy_path.with_stem(ddm_npy_path.stem + '.' + params.coord_system + '.average_radius')
        np.save(median_angle_path, fitter.median_angle)
        np.save(median_radius_path, fitter.median_radius)
        np.save(average_angle_path, fitter.average_angle)
        np.save(average_radius_path, fitter.average_radius)


if __name__ == '__main__':
    main()
