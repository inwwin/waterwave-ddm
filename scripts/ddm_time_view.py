"""
A helper script for investigation the lag time dependence at a particular wavevector of
an image structure function
"""
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt
import matplotlib.axes
import matplotlib.figure
import matplotlib.animation

parser = argparse.ArgumentParser()
coord_group = parser.add_mutually_exclusive_group(required=True)
coord_group.add_argument('-ij', type=int, nargs=2, metavar=('i', 'j'))
coord_group.add_argument('-xy', type=int, nargs=2, metavar=('x', 'y'))
coord_group.add_argument('-ar', type=float, nargs=2, metavar=('angle', 'radius'))
parser.add_argument('-tm', '--ti_max', type=int, default=None)
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

fig, ax = plt.subplots()
fig: matplotlib.figure.Figure
ax: matplotlib.axes.Axes
ax.plot(ddm_array[i, j, 0:params.ti_max])
plt.show()
