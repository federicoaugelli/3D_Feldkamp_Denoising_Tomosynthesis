from create_dataset import get_data, to_tif, to_np
from plot_images import plot_volumes
import numpy as np
import argparse, os

#noise_levels = [0.0, 0.01]
num_projections = 11

def parse_args():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--tiff-path', type=str, default='dataset/tiff/',
                        help='Path to save the dataset')
    parser.add_argument('--np-path', type=str, default='dataset/data/',
                        help='Path to save the dataset in numpy format')
    parser.add_argument('--dim', type=int, nargs=4, default=(30, 256, 256, 50))
    parser.add_argument('--num-vols', type=int, default=25,
                        help='Number of samples to generate')
    parser.add_argument('--noise-level', type=float, default=0.0, 
                        help='Noise level to add to the dataset')
    return parser.parse_args()


'''
LOAD = False

if not LOAD:
    ellipsoid_data = get_data(dim)
    # data.to_tif(ellipsoid_data, PATH, type='xy')
    # data.to_tif(ellipsoid_data, PATH, type='yt')
    data.to_tif(ellipsoid_data, TIFF_PATH, type='xy')
    data.to_np(ellipsoid_data, NP_PATH)
else:
    ellipsoid_data = np.load(PATH + 'ellipsoid_dataset.npy')
    print(ellipsoid_data.shape)
'''
def main():
    args = parse_args()
    TIFF_PATH = args.tiff_path
    NP_PATH = args.np_path
    dim = tuple(args.dim)

    # check if NP_PATH exists contains data
    if not os.path.exists(NP_PATH):
        os.makedirs(NP_PATH)
    if not os.path.exists(TIFF_PATH):
        os.makedirs(TIFF_PATH)
    if not os.listdir(NP_PATH):
        ellipsoid_data = get_data(dim)
        to_tif(ellipsoid_data, TIFF_PATH, type='xy')
        to_np(ellipsoid_data, NP_PATH)

    






if __name__ == "__main__":
    main()
