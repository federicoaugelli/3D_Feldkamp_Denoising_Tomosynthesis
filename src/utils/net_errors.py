from torch import nn
import torch

def rmse_err(pred, target):
    assert pred.shape == target.shape, "Input tensors must have the same shape"
    squared_error = (pred - target)**2
    mse = torch.mean(squared_error)
    rmse = torch.sqrt(mse)
    return rmse

def rel_err(pred, target, epsilon=1e-8):
  assert pred.shape == target.shape, "Input tensors must have the same shape"
  return torch.mean(torch.abs((pred - target) / (target + epsilon)))

def psnr_err(pred, target, max_val=1.0):
  assert pred.shape == target.shape, "Input tensors must have the same shape"
  mse = torch.mean((pred - target)**2)
  if mse == 0:
    return 100
  return 10 * torch.log10(max_val**2 / mse)

class Loss(nn.Module):
  def __init__(self):
    super(Loss, self).__init__()

  def forward(self, pred, target):
    rmse = rmse_err(pred, target)
    psnr = psnr_err(pred, target)
    #rel = rel_err(pred, target)
    return rmse + 1 / (psnr +1e-8)
