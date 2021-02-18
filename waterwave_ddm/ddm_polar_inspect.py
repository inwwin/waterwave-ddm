"""View the evolution of an image structure function over lag time either radially or azimuthally"""
import numpy as np
# import json
from typing import Tuple


def polar_space(shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Init a meshgrid of cartesian coordinates
    centre = (shape[0] // 2, 0)
    x_i = np.arange(-centre[0], +centre[0] + 1, +1)
    x_j = np.arange(0, shape[1])
    x_i = np.fft.ifftshift(x_i)  # so that our map is compatible with x
    xm_i, xm_j = np.meshgrid(x_i, x_j, indexing='ij')

    # Radius, angle function
    r = np.sqrt(xm_i**2 + xm_j**2)
    a = np.arctan2(xm_i, xm_j)
    # Adjust domain
    a[a > np.pi] -= 2 * np.pi
    # This makes our angle measured anticlockwise with
    # +x_j => 0 degrees
    # +x_i => -pi/2 degrees (bottom)
    # -x_i => +pi/2 degrees (top)
    a *= -1

    return r, a, xm_i, xm_j


def map_to_polar(x: np.ndarray, angular_bins: int, radial_bin_size: float, max_radius: float) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert the domain of an image structure function from cartesian coordinate into polar coordinate
    by averaging over appropriate bins
    """
    r, a, xm_i, xm_j = polar_space(x.shape)
    a = np.rad2deg(a)

    # Results holder array
    # index 0: angle
    # index 1: radius
    # index 2: time
    radial_bins = int(max_radius // radial_bin_size)
    # result = np.empty((angular_bins, radial_bins, x.shape[-1]))
    # average_radius = np.empty((angular_bins, radial_bins))

    lower_angle = np.linspace(-90.0, 90.0, angular_bins, False)
    upper_angle = lower_angle + 180.0 / angular_bins
    median_angle = (lower_angle + upper_angle) / 2

    lower_radius = np.linspace(0.0, max_radius, radial_bins, False)
    upper_radius = lower_radius + radial_bin_size
    median_radius = (lower_radius + upper_radius) / 2

    # Construct bin mapping array
    # index 0: angle
    # index 1: radius
    # index 2,3: i,j
    a_map            =            a[np.newaxis, np.newaxis, :, :]
    r_map            =            r[np.newaxis, np.newaxis, :, :]
    lower_angle_map  =  lower_angle[:, np.newaxis, np.newaxis, np.newaxis]
    upper_angle_map  =  upper_angle[:, np.newaxis, np.newaxis, np.newaxis]
    lower_radius_map = lower_radius[np.newaxis, :, np.newaxis, np.newaxis]
    upper_radius_map = upper_radius[np.newaxis, :, np.newaxis, np.newaxis]
    polar_map = np.logical_and(
        np.logical_and(r_map >= lower_radius_map, r_map < upper_radius_map),
        np.logical_and(a_map >= lower_angle_map, a_map < upper_angle_map)
    )
    # The origin is always included in the first radius
    polar_map[:, 0, 0, 0] = True
    # print(polar_map.shape)
    assert polar_map.shape[0] == angular_bins
    assert polar_map.shape[1] == radial_bins
    assert polar_map.shape[2] == xm_i.shape[0]
    assert polar_map.shape[3] == xm_i.shape[1]

    # elem_exists = np.any(polar_map, axis=(2, 3))
    # Initialise average radius/angle array with median values
    # in order to account for the bins which have no elements
    average_angle, average_radius = \
        np.meshgrid(median_angle, median_radius, indexing='ij')
    assert average_angle.shape == (angular_bins, radial_bins)  # Just a sanity check
    # Similaryly initialise the result array with NaN
    result = np.full((angular_bins, radial_bins, x.shape[-1]), np.nan)

    # average_radius[elem_exists] = \  # This idea won't work because advance indexing always return a copy
    #     np.mean(
    #         r[np.newaxis, np.newaxis, :, :],
    #         axis=(2, 3),
    #         where=polar_map[elem_exists[:, :, np.newaxis, np.newaxis]
    for ai in range(polar_map.shape[0]):
        for ri in range(polar_map.shape[1]):
            mapi = polar_map[ai, ri, ...]
            # For bins which contains elements
            if np.any(mapi):
                average_radius[ai, ri] = np.mean(r, where=mapi)
                average_angle[ai, ri] = np.mean(a, where=mapi)
                result[ai, ri, :] = np.mean(x, axis=(0, 1), where=mapi[..., np.newaxis])
            # No need for else since it is already accounted for in the arrays' initialisation

    return result, median_angle, median_radius, average_angle, average_radius


def ddm_polar_inspection_animation(mode, rai, average_angle, average_radius, ddm_polar, max_ti=100, interval=200):
    """Create lag-time animation of image structure function when viewed over one radius or one angle only"""
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    if mode not in ('r', 'a'):
        raise ValueError('mode must be \'r\' or \'a\'')

    fig, ax = plt.subplots()
    ti_indicator = fig.text(.02, .98, 'ti = 0', verticalalignment='top')
    if mode == 'a':
        ri = rai
        line, = ax.plot(average_angle[:, ri], ddm_polar[:, ri, 0])
        ax.set_ylim(0, np.nanmax(ddm_polar[:, ri, :]))
    else:
        ai = rai
        line, = ax.plot(average_radius[ai, :], ddm_polar[ai, :, 0])
        ax.set_ylim(0, np.nanmax(ddm_polar[ai, :, :]))

    def animation_function(ti, mode):
        if mode == 'a':
            line.set_ydata(ddm_polar[:, rai, ti])
        else:
            line.set_ydata(ddm_polar[rai, :, ti])
        ti_indicator.set_text(f"ti = {ti}")
        return line, ti_indicator,

    animation = FuncAnimation(fig, animation_function, fargs=mode, interval=interval, blit=False, frames=max_ti)

    return animation, fig, ax, line, ti_indicator


def main():
    import argparse
    import pathlib
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--angular_bins', type=int, default=18)
    parser.add_argument('-p', '--radial_bin_size', type=int, default=2)
    parser.add_argument('-i', '--interactive', action='store_true')
    parser.add_argument('-o', '--out', type=pathlib.Path, default=False)
    # parser.add_argument('-ao', '--animation_out', type=pathlib.Path, default=False)
    parser.add_argument('ddm_npy_path', type=pathlib.Path)
    params = parser.parse_args()

    if not (params.interactive or params.out or params.animation_out):
        return

    ddm_array: np.ndarray = np.load(params.ddm_npy_path)
    print('ddm_array.shape: ', ddm_array.shape)
    # ddm_array = np.fft.fftshift(ddm_array, axes=0)
    result, median_angle, median_radius, average_angle, average_radius = \
        map_to_polar(ddm_array, params.angular_bins, params.radial_bin_size, ddm_array.shape[1] - 1)
    print('polar_result.shape: ', result.shape)

    if params.out:
        try:
            params.out.mkdir(mode=0o755, parents=True, exist_ok=True)
        except FileExistsError:
            parser.error('out must be a directory')

        np.save(params.out / 'polar_ddm.npy',      result)
        np.save(params.out / 'median_angle.npy',   median_angle)
        np.save(params.out / 'median_radius.npy',  median_radius)
        np.save(params.out / 'average_angle.npy',  average_angle)
        np.save(params.out / 'average_radius.npy', average_radius)

    # if params.animation_out:
    #     for ri in range(result.shape[1]):
    #         pass

    if params.interactive:
        while True:
            try:
                p = input('a/r radius_index/angle_index max_time interval >> ')
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            animation_params = p.split()
            if len(animation_params) < 2:
                print('Must provide a/r and radius_index/angle_index')
                continue
            mode = animation_params[0]
            if mode not in ('a', 'r'):
                print('a (angular view) r (radial view) or only')
                continue
            rai = int(animation_params[1])  # radius or angle index depending on context
            max_ti = int(animation_params[2]) if len(animation_params) > 2 else 100
            interval = int(animation_params[3]) if len(animation_params) > 3 else 200
            # html_out = animation_params[4] if len(animation_params) > 4 else False

            if mode == 'a':
                print('Inspecting angular view at median radius = ', median_radius[rai])
            else:
                print('Inspecting radial view at median angle = ', median_angle[rai])

            animation, fig, *_ = \
                ddm_polar_inspection_animation(mode, rai, average_angle, average_radius, result, max_ti, interval)

            fig.show()
            # if not html_out:
            #     fig.show()
            # else:
            #     html = animation.to_jshtml(default_mode='loop')
            #     print(html)


if __name__ == '__main__':
    main()
