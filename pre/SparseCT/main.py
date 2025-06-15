import data
import numpy as np
import visualization
import utils
from PIL import Image

"""
This file contains an example on how to generate the dataset from the functions in this library.
"""

PATH = '../SparseCT/tiff/'
TO_PATH = '../SparseCT/data/'
LOAD = False

if not LOAD:
    ellipsoid_data = data.get_data((2, 256, 256, 50))
    # data.to_tif(ellipsoid_data, PATH, type='xy')
    # data.to_tif(ellipsoid_data, PATH, type='yt')
    data.to_tif(ellipsoid_data, PATH, type='xy')
    #data.to_np(ellipsoid_data, PATH)
else:
    ellipsoid_data = np.load(PATH + 'ellipsoid_dataset.npy')
    print(ellipsoid_data.shape)

#utils.np_to_mat(ellipsoid_data, TO_PATH)


