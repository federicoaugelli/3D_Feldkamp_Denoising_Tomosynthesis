from SparseCT.data import get_data
#from SparseCT.visualization import show3d
import numpy as np


dim = (2, 256, 256, 50)
image = get_data(dim)

# plot ndarray image of dimension (N, H, W, D)
def plot_images(image):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, image.shape[0], figsize=(15, 5))
    for i in range(image.shape[0]):
        axes[i].imshow(image[i, :, :, 10], cmap='gray')
        axes[i].axis('off')
    plt.show()

plot_images(image)
