import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio


def rel_err(x_true, x_corr):
    x_true = x_true.flatten()
    x_corr = x_corr.flatten()

    RE = np.linalg.norm(x_true - x_corr, 2) / np.linalg.norm(x_true, 2)
    return RE


def RMSE(x_true, x_corr):
    x_true = x_true.flatten()
    x_corr = x_corr.flatten()

    return np.sqrt(np.mean(np.square(x_true - x_corr)))


def PSNR(x_true, x_corr):
    x_true = x_true.astype(np.float64)
    x_corr = x_corr.astype(np.float64)
    psnr_values = []
    for i in range(x_true.shape[0]):
        psnr_values.append(peak_signal_noise_ratio(x_true[i], x_corr[i], data_range=x_true.max() - x_true.min()))
    return np.mean(psnr_values)


def SSIM(x_true, x_corr):
    ssim_values = []
    for i in range(x_true.shape[0]):
        for j in range(x_true.shape[1]):
            ssim_values.append(structural_similarity(x_true[i, j], x_corr[i, j], data_range=x_true.max() - x_true.min(), channel_axis=None))
    return np.mean(ssim_values)