import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def rel_err(x_true, x_corr):
    # Flatten the images
    x_true = x_true.flatten()
    x_corr = x_corr.flatten()

    # Compute the error
    RE = np.linalg.norm(x_true - x_corr, 2) / np.linalg.norm(x_true, 2)
    return RE


def RMSE(x_true, x_corr):
    # Flatten the images
    x_true = x_true.flatten()
    x_corr = x_corr.flatten()

    # Return the error
    return np.sqrt(np.mean(np.square(x_true - x_corr)))


def PSNR(x_true, x_corr):
    # Ensure data types are float to avoid potential issues with PSNR calculation
    x_true = x_true.astype(np.float64)
    x_corr = x_corr.astype(np.float64)
    # Calculate PSNR slice by slice or volume by volume if needed
    # Assuming x_true and x_corr have the same shape (batch, depth, height, width)
    # We will calculate PSNR for each volume in the batch and average
    psnr_values = []
    for i in range(x_true.shape[0]): # Iterate through the batch dimension
        # Calculate PSNR for the current volume
        psnr_values.append(peak_signal_noise_ratio(x_true[i], x_corr[i], data_range=x_true.max() - x_true.min()))
    return np.mean(psnr_values)


def SSIM(x_true, x_corr):
    # Assuming x_true and x_corr have the shape (batch, depth, height, width)
    # We will calculate SSIM for each 2D slice across all volumes and average
    ssim_values = []
    # Iterate through batch and depth dimensions
    for i in range(x_true.shape[0]):
        for j in range(x_true.shape[1]):
            # Calculate SSIM for the 2D slice (height, width)
            # Set win_size to an appropriate odd value less than or equal to the smaller of height and width
            # Default win_size is 7, which is suitable for 256x256 slices.
            # We also set channel_axis to None as we are processing 2D slices
            ssim_values.append(structural_similarity(x_true[i, j], x_corr[i, j], data_range=x_true.max() - x_true.min(), channel_axis=None))
    return np.mean(ssim_values)

'''
def PSNR(x_true, x_corr):
    # Flatten the images
    x_true = x_true.flatten()
    x_corr = x_corr.flatten()

    mse = np.mean((x_true - x_corr) ** 2)
    if (
        mse == 0
    ):  # MSE is zero means no noise is present in the signal. Therefore PSNR have no importance.
        return 100
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def SSIM(x_true, x_corr):
    return structural_similarity(x_true, x_corr, data_range=1.0)
'''

def TpV(x, p):
    """Compute the p-norm of the gradient of the input image x."""
    # Normalize x
    if np.linalg.norm(x) != 0:
        x = (x - x.min()) / (x.max() - x.min())
    Dh_x = np.diff(x, n=1, axis=1, prepend=0)
    Dv_x = np.diff(x, n=1, axis=0, prepend=0)

    D_x = np.sqrt(np.square(Dh_x) + np.square(Dv_x))
    return np.power(np.sum(np.power(D_x, p)), 1 / p) / np.prod(D_x.shape)


def rTpV(x_true, x_pred, p=1):
    TpV_true = TpV(x_true, p)
    TpV_pred = TpV(x_pred, p)

    return np.abs(TpV_pred - TpV_true) / np.abs(TpV_true)
