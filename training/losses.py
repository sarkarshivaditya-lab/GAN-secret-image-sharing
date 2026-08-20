import torch.nn.functional as F

def reconstruction_loss(original, reconstructed):
    return F.mse_loss(reconstructed, original)