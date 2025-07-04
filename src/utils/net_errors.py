from torch import nn
import torch
from torch.nn import functional as F

def rmse_err(pred, target):
    assert pred.shape == target.shape,
    squared_error = (pred - target)**2
    mse = torch.mean(squared_error)
    rmse = torch.sqrt(mse)
    return rmse

def rel_err(pred, target, epsilon=1e-8):
  assert pred.shape == target.shape,
  return torch.mean(torch.abs((pred - target) / (target + epsilon)))

def psnr_err(pred, target, max_val=1.0):
  assert pred.shape == target.shape,
  mse = torch.mean((pred - target)**2)
  if mse == 0:
    return 100
  return 10 * torch.log10(max_val**2 / mse)

def freq_loss(pred, target):
  """Frequency domain artifact penalty"""
  pred_fft = torch.fft.rfftn(pred, dim=(-3, -2, -1))
  target_fft = torch.fft.rfftn(target, dim=(-3, -2, -1))
  return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

def edge_loss(pred, target):
  """Gradient magnitude preservation"""
  grad_pred = torch.abs(pred[:, :, 1:] - pred[:, :, :-1])
  grad_target = torch.abs(target[:, :, 1:] - target[:, :, :-1])
  return F.l1_loss(grad_pred, grad_target)

class Loss(nn.Module):
  def __init__(self):
    super(Loss, self).__init__()

  def forward(self, pred, target):
    #rmse = rmse_err(pred, target)
    freq = freq_loss(pred, target)
    #ssim = ssim_3d(pred, target)
    psnr = psnr_err(pred, target)
    edge = edge_loss(pred, target)

    return 0.20 * freq + 0.70 * (1 / (psnr + 1e-8)) + 0.10 * edge
