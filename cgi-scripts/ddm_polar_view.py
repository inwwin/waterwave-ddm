#!/usr/bin/env python3
import numpy as np
from waterwave_ddm.ddm_polar_inspect import map_to_polar, ddm_polar_inspection_animation
import json
import cgi
import cgitb
import os.path as p


def get_cgi_param(params, key, default=None, mapper=None):
    if mapper is None:
        mapper = lambda x: x  # noqa: E731
    try:
        return mapper(params.get(key, (default,))[0])
    except (TypeError, ValueError):
        return default


def main():
    # DIR_OF_THIS_SCRIPT = p.abspath(p.dirname(__file__))
    # config_path = p.join(DIR_OF_THIS_SCRIPT, '.config.json')
    config_path = p.expanduser('~/.config/waterwave_ddm/config.json')
    try:
        with open(config_path, 'r') as config_file:
            config: dict = json.load(config_file)
    except FileNotFoundError:
        print('Status: 500 Server Error')
        print()
        print('No config file')
        return
    cgitb.enable(display=0, logdir=config.get('log_path', p.expanduser('~/.log/waterwave_ddm.log')))
    ddm_data_map: dict = config['data_map']
    default_interval = config.get('default_interval', 200)
    default_max_ti = config.get('default_max_ti', 50)
    default_angular_bins = config.get('default_angular_bins', 18)
    default_radial_bin_size = config.get('default_radial_bin_size', 2)

    params = cgi.parse()
    dataid = get_cgi_param(params, 'dataid')
    data_path: str = ddm_data_map.get(dataid)
    if data_path is None:
        print('Status: 404 Not Found')
        print()
        print('Data not found')
        return
    interval = get_cgi_param(params, 'interval', default_interval, int)
    max_ti = get_cgi_param(params, 'max_ti', default_max_ti, int)
    angular_bins = get_cgi_param(params, 'angular_bins', default_angular_bins, int)
    radial_bin_size = get_cgi_param(params, 'radial_bin_size', default_radial_bin_size, int)
    ai = get_cgi_param(params, 'angle', None, int)
    ri = get_cgi_param(params, 'radius', None, int)
    if ai is None and ri is not None:
        rai = ri
        mode = 'a'
    elif ai is not None and ri is None:
        rai = ai
        mode = 'r'
    else:
        print('Status: 400 Bad Request')
        print()
        print('Please provide either engle or radius')
        return

    ddm_array = np.load(data_path)
    result, median_angle, median_radius, average_angle, average_radius = \
        map_to_polar(ddm_array, angular_bins, radial_bin_size, ddm_array.shape[1] - 1)
    animation, fig, _, _, _ = \
        ddm_polar_inspection_animation(mode, rai, average_angle, average_radius, result, max_ti, interval)
    angular_bins = result.shape[0]
    radial_bins = result.shape[1]

    html_viewer = animation.to_jshtml(default_mode='loop')
    print('Content-Type: text/html')
    print()
    print('<p>Polar ddm shape: {}</p>'.format(result.shape))
    print('<p>After applying the binning setting this data set has {} angle and {} radius values available</p>'
          .format(angular_bins, radial_bins))
    if mode == 'a':
        print('<p>You are viewing angular distribution of this data at median radius = {} varying over lag time</p>'
              .format(median_radius[ri]))
    else:
        print('<p>You are viewing radial distribution of this data at median angle = {} varying over lag time</p>'
              .format(median_angle[ai]))
    print(html_viewer)


if __name__ == '__main__':
    main()
