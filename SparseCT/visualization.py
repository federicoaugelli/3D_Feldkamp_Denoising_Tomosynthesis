import numpy as np
import matplotlib.pyplot as plt


def show3d(img):
    """
    Takes as input an image of shape (H, W, D) and shows it slice by slice, in an image matrix of dimension 5 x (D // 5).

    :param img: ndarray, the image
    :return: None
    """
    img = np.transpose(img, (2, 0, 1))
    d = img.shape[0]

    _, axes = plt.subplots(nrows=d//5, ncols=5, figsize=(16, 14))

    val_min = img.min()
    val_max = img.max()

    for ax, image in zip(axes.flatten(), img):
        ax.imshow(image, cmap='gray', vmin=val_min, vmax=val_max)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()