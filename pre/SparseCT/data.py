from pyellipsoid import drawing

# ellipsoid dataset (2000 grayscale images 512x512, normalized in 0-1)

import numpy as np
import random


def add_ellipse(fig, center_range=(100, 100, 10), axes_range=(30, 30, 3)):
    """
    Add an ellipse to fig. The ellipse will be centered in a random value in the range:
    [center_range, dim - center_range]
    and will have axes of dimension in the range:
    [axes_range, 2 * axes_range]
    The ellipse will be rotated of a random angle between 0 and 90 degrees in each dimension and the opacity will be in
    the range [0, 1].
    If two ellipses overlaps, only the more opaque will be shown in the overlapping.

    :param fig: ndarray, an array that contains the figure
    :param center_range: list or tuple, a list of length 3 that contains the range of the centers in each dimensions (x, y, z)
    :param axes_range: list or tuple, a list of length 3 that contains the range of the axes in each dimensions (x, y, z)
    :return: ndarray, an array that contains the figure
    """
    H, W, D = fig.shape

    ell_center = (random.randrange(center_range[0], H - center_range[0]),
                  random.randrange(center_range[1], W - center_range[1]),
                  random.randrange(center_range[2], D - center_range[2]))

    ell_axes = (random.randrange(axes_range[0], 3*axes_range[0]),
                random.randrange(axes_range[1], 3*axes_range[1]),
                random.randrange(axes_range[2], 3*axes_range[2]))

    ell_angle = np.deg2rad([random.randrange(90), random.randrange(90), random.randrange(90)])

    ell_opacity = random.random() * 0.9

    overlay = drawing.make_ellipsoid_image((D, W, H), ell_center, ell_axes, ell_angle)
    overlay = np.transpose(overlay, (2, 1, 0))
    overlay = overlay.astype('float32')

    fig = unify(fig, overlay, ell_opacity)
    return fig, ell_center


def add_circle(fig, center_range=(100, 100, 10), r=3):
    """
    Add a point to fig. The point will be centered in a random value in the range:
    [center_range, dim - center_range]
    and will have axes of dimension:
    [3, 3, 3]
    The opacity will be 0.9.

    :param fig: ndarray, an array that contains the figure
    :param center_range: list or tuple, a list of length 3 that contains the range of the centers in each dimensions (x, y, z)
    :return: ndarray, an array that contains the figure
    """
    H, W, D = fig.shape

    point_center = (random.randrange(center_range[0], H - center_range[0]),
                    random.randrange(center_range[1], W - center_range[1]),
                    random.randrange(center_range[2], D - center_range[2]))
    point_axes = (r, r, r)
    point_angle = (0, 0, 0)
    point_opacity = 0.9

    overlay = drawing.make_ellipsoid_image((D, W, H), point_center, point_axes, point_angle)
    overlay = np.transpose(overlay, (2, 1, 0))
    overlay = overlay.astype('float32')

    fig = unify(fig, overlay, point_opacity)
    return fig, point_center


def add_line(fig, center_range=(100, 100, 10), length_range=80):
    """
    Add a line segment to fig. The line segment will be centered in a random value in the range:
    [center_range, dim - center_range]
    and will have a length in the range:
    [length_range, 2 * length_range]
    The line segment will be rotated of a random angle between 0 and 90 degrees in each dimension and the opacity will be 0.9.

    :param fig: ndarray, an array that contains the figure
    :param center_range: list or tuple, a list of length 3 that contains the range of the centers in each dimensions (x, y, z)
    :param length_range: int, the range of the length for the line segment
    :return: ndarray, an array that contains the figure
    """
    H, W, D = fig.shape

    line_center = (random.randrange(center_range[0], H - center_range[0]),
                  random.randrange(center_range[1], W - center_range[1]),
                  random.randrange(center_range[2], D - center_range[2]))

    line_axes = (1, random.randrange(length_range, 2*length_range), 1)
    line_angle = np.deg2rad([random.randrange(90), 0, 0])
    line_opacity = 0.9

    overlay = drawing.make_ellipsoid_image((D, W, H), line_center, line_axes, line_angle)
    overlay = np.transpose(overlay, (2, 1, 0))
    overlay = overlay.astype('float32')

    fig = unify(fig, overlay, line_opacity)
    return fig, line_center


def add_concentric_ellipse(fig, center, axes_range=(10, 10, 3)):
    """
    Add an ellipse to fig. The ellipse will be centered in a random value in the range:
    [center_range, dim - center_range]
    and will have axes of dimension in the range:
    [axes_range, 2 * axes_range]
    The ellipse will be rotated of a random angle between 0 and 90 degrees in each dimension and the opacity will be in
    the range [0, 1].
    If two ellipses overlaps, only the more opaque will be shown in the overlapping.

    :param fig: ndarray, an array that contains the figure
    :param center: list or tuple, a list of length 3 that contains the range of the centers in each dimensions (x, y, z)
    :param axes_range: list or tuple, a list of length 3 that contains the range of the axes in each dimensions (x, y, z)
    :return: ndarray, an array that contains the figure
    """
    H, W, D = fig.shape

    ell_center = center

    ell_axes = (random.randrange(axes_range[0], 2*axes_range[0]),
                random.randrange(axes_range[1], 2*axes_range[1]),
                random.randrange(axes_range[2], 2*axes_range[2]))

    ell_angle = np.deg2rad([random.randrange(90), random.randrange(90), random.randrange(90)])

    ell_opacity = (random.random() + 1) * 0.5

    overlay = drawing.make_ellipsoid_image((D, W, H), ell_center, ell_axes, ell_angle)
    overlay = np.transpose(overlay, (2, 1, 0))
    overlay = overlay.astype('float32')

    fig = unify(fig, overlay, ell_opacity)
    return fig


def unify(fig, overlay, opacity):
    """
    A utility function that unifies two images with the condition that if two part of the images overlaps, only the more
    opaque will be shown.

    :param fig: ndarray, an array that contains the first figure
    :param overlay: fig: ndarray, an array that contains the second figure
    :param opacity: int, the opacity of the second image
    :return: ndarray, an array that contains the figure
    """
    H, W, D = fig.shape

    fig[overlay >= fig] = overlay[overlay >= fig] * opacity
    return fig


def get_data(input_shape=(30, 512, 512, 32)):
    """
    Create the dataset. The parameters are defined in the code.

    :return: ndarray, an array of dimension (N, H, W, D) that contains the dataset.
    """
    N, H, W, D = input_shape  # Shape of the dataset.
    center_range = (30, 30, 5)  # Possible centers of the ellipses: (c_range, dim - c_range)
    axes_range = (50, 20, 10)  # Possible radius of the ellipses: (r_range, 100 - r_range)

    ellipsoid_dataset = np.empty((N, H, W, D), dtype=np.float32)
    ell_centers = list()

    for i in range(N):
        N_ell = random.randint(1, 5) + 8  # Number of ellipses in each image.
        N_circle = random.randint(1, 5) + 8  # Number of circles of opacity 0.9.
        N_lines = random.randint(1, 5) + 4  # Number of lines of opacity 0.9
        N_centered_ellipses = random.randint(1, N_ell)  # Number of centered ellipses in each image.

        print('Drawing the ', str(i), '-th Ellipsoid', end='')
        fig = np.zeros((H, W, D))

        for n_ell in range(N_ell):
            fig, ell_center = add_ellipse(fig, center_range, axes_range)
            ell_centers.append(ell_center)
            print('.', end='')
        print('/', end='')

        for n_circle in range(N_circle):
            circle_radius = random.randint(3, 8)
            fig, point_center = add_circle(fig, center_range, circle_radius)
            print('.', end='')
        print('/', end='')

        for n_line in range(N_lines):
            fig, line_center = add_line(fig)
            print('.', end='')
        print('/', end='')

        for n_centered_ellipses in range(N_centered_ellipses):
            fig = add_concentric_ellipse(fig, ell_centers[n_centered_ellipses])
            print('.', end='')

        # fig = fig / fig.max()
        ellipsoid_dataset[i, :, :, :] = fig
        print('')

    return ellipsoid_dataset


def to_npz(data, PATH='./'):
    """
    Create a .npz file that contains data into PATH with the name ellipsoid_dataset.npz

    :param data: ndarray, the file containing the dataset
    :param PATH: str, the PATH in where the file will be saved
    :return: None
    """
    np.savez(PATH + 'ellipsoid_dataset.npz', data)


def to_np(data, PATH='./'):
    """
    Create a .npy file that contains data into PATH with the name ellipsoid_dataset.npz

    :param data: ndarray, the file containing the dataset
    :param PATH: str, the PATH in where the file will be saved
    :return: None
    """
    np.save(PATH + 'ellipsoid_dataset.npy', data)


def to_tif(data, PATH='./', type='xy'):
    """
    Create a .tif file for each element of data in PATH. The name of each .tif file will be ellipsoid_dataset_i.tif where
    i is an increasing counter.

    :param data: ndarray, the file containing the dataset
    :param PATH: str, the PATH in where the file will be saved
    :param type: str, visualization type. Available types:
            - xy
            - yt
            - xt
    :return: None
    """
    import tifffile as tif
    for i in range(data.shape[0]):
        if type == 'xy':
            tif.imwrite(PATH + 'ellipsoid_dataset_'+ str(i) + '_xy.tif', np.transpose(data[i], (2, 0, 1)))
        elif type == 'yt':
            tif.imwrite(PATH + 'ellipsoid_dataset_' + str(i) + '_yt.tif', data[i])
        elif type == 'xt':
            tif.imwrite(PATH + 'ellipsoid_dataset_'+ str(i) + '_xt.tif', np.transpose(data[i], (1, 0, 2)))
