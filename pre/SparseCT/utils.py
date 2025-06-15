import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from PIL import Image

def np_to_mat(data, TO_PATH):
    """
    Convert the dataset data into a .mat file and puts it in TO_PATH

    :param data: ndarray, an array of dimension (N, x, y, t)
    :param TO_PATH: str, the name of the output PATH
    :return: None
    """
    N = data.shape[0]

    for i in range(N):
        img = data[0]
        name = 'img_' + str(i)
        D = {name: img}

        io.savemat(TO_PATH + 'TomoImg_' + str(i) + '.mat', D)
