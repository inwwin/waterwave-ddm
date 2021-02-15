"""
A helper script for watching the variation of the entire plane of
an image structure function over different lag time

Important Remark
=====
This script assume a fork version of cddm available [here](https://github.com/inwwin/cddm)
to configure the pause_duration.
"""
import numpy as np
from cddm.viewer import VideoViewer
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('ddm_npy_path', type=pathlib.Path)
params = parser.parse_args()

ddm_array = np.load(params.ddm_npy_path)
print(ddm_array.shape)
ddm_array = np.fft.fftshift(ddm_array, axes=0)
ddm_iter = [ddm_array[:, :, i] for i in range(ddm_array.shape[2])]
viewer = VideoViewer(ddm_iter, vmin=0, vmax=5, count=256)
viewer.pause_duration = 0.25
viewer.show()
