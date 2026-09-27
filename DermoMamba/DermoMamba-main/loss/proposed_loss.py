from scipy.ndimage import distance_transform_edt
import torch
from loss.loss import DiceLoss
import torch.nn.functional as F

def compute_attention_mask_boundary(binary_mask):
    dist_foreground = distance_transform_edt(binary_mask == 0)
    dist_background = distance_transform_edt(binary_mask == 1)
    distance_map = dist_foreground + dist_background
    attention_mask = 1 / (1 + distance_map)
    return torch.tensor(attention_mask, dtype=torch.float32)
def Guide_Fusion_Loss(y_pred, y_true):
    dice = DiceLoss()(y_pred, y_true)
    bce = F.binary_cross_entropy_with_logits(torch.sigmoid(y_pred), y_true.float())
    attention_mask = compute_attention_mask_boundary(y_true.cpu()).cuda()
    loss = bce + dice * attention_mask
    return loss.mean()