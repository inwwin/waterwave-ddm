"""A helper script for viewing DDM data using cddm's DataViewer"""
import numpy as np
from cddm.viewer import DataViewer
import argparse
import pathlib

parser = argparse.ArgumentParser()
parser.add_argument('ddm_npy_path', type=pathlib.Path)
params = parser.parse_args()

ddm_array = np.load(params.ddm_npy_path)
# print(ddm_array)

#: inspect the data
viewer = DataViewer(semilogx=False)
viewer.set_data(ddm_array)
viewer.set_mask(k=3, angle=0, sector=10)
viewer.plot()
viewer.show()
