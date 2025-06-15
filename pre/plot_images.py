import matplotlib.pyplot as plt

def plot_image(image, title=None, cmap='gray'):
    """
    Plot a single image with a title.
    """
    plt.figure(figsize=(10, 5))
    plt.imshow(image, cmap=cmap)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# plot multiple images in grid
def plot_n_images(images, title=None, n=3, cmap='gray'):
    """
    Plot n images in a grid.
    """
    n = min(n, len(images))
    fig, axes = plt.subplots(1, n, figsize=(15, 5))
    for i in range(n):
        axes[i].imshow(images[i], cmap=cmap)
        axes[i].axis('off')
    if title:
        plt.suptitle(title)
    plt.show()

# plot ndarray image of dimension (N, H, W, D)
def plot_volumes(image):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, image.shape[0], figsize=(15, 5))
    for i in range(image.shape[0]):
        axes[i].imshow(image[i, :, :, 10], cmap='gray')
        axes[i].axis('off')
    plt.show()
