import torch.nn as nn
import numpy as np
import  torch
import torch.nn.functional as F
from skimage import measure
from torch import einsum
from torch import Tensor
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union

class SoftIoULoss(nn.Module):
    def __init__(self, **kwargs):
        super(SoftIoULoss, self).__init__()

    def forward(self, pred, target):
        # Old One
        pred = torch.sigmoid(pred)
        smooth = 1

        # print("pred.shape: ", pred.shape)
        # print("target.shape: ", target.shape)

        intersection = pred * target
        loss = (intersection.sum() + smooth) / (pred.sum() + target.sum() - intersection.sum() + smooth)

        # loss = (intersection.sum(axis=(1, 2, 3)) + smooth) / \
        #        (pred.sum(axis=(1, 2, 3)) + target.sum(axis=(1, 2, 3))
        #         - intersection.sum(axis=(1, 2, 3)) + smooth)

        loss = 1 - loss.mean()
        # loss = (1 - loss).mean()

        return loss


def Dice( pred, target,warm_epoch=1, epoch=1, layer=0):
        pred = torch.sigmoid(pred)

        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        loss = (2 * intersection_sum + smooth) / \
               (pred_sum + target_sum + intersection_sum + smooth)

        loss = 1 - loss.mean()

        return loss

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        # pred = torch.sigmoid(pred)
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)

        return self.bceloss(pred_, target_)

class temp_DiceLoss(nn.Module):
    def __init__(self):
        super(temp_DiceLoss, self).__init__()

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        smooth = 1

        intersection = pred * target
        intersection_sum = torch.sum(intersection, dim=(1, 2, 3))
        pred_sum = torch.sum(pred, dim=(1, 2, 3))
        target_sum = torch.sum(target, dim=(1, 2, 3))

        # 计算 Dice 系数
        loss = (2 * intersection_sum + smooth) / \
               (pred_sum + target_sum + intersection_sum + smooth)

        # 最终的损失是 1 减去平均的 Dice 系数
        loss = 1 - loss.mean()

        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss


"""
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)

        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth)/(pred_.sum(1) + target_.sum(1) + smooth)
        dice_loss = 1 - dice_score.sum()/size

        return dice_loss
"""

class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = temp_DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)

        loss = self.wd * diceloss + self.wb * bceloss
        return loss


def LLoss(pred, target):
        loss = torch.tensor(0.0, requires_grad=True).to(pred)

        patch_size = pred.shape[0]
        h = pred.shape[2]
        w = pred.shape[3]        
        x_index = torch.arange(0,w,1).view(1, 1, w).repeat((1,h,1)).to(pred) / w
        y_index = torch.arange(0,h,1).view(1, h, 1).repeat((1,1,w)).to(pred) / h
        smooth = 1e-8
        for i in range(patch_size):  

            pred_centerx = (x_index*pred[i]).mean()
            pred_centery = (y_index*pred[i]).mean()

            target_centerx = (x_index*target[i]).mean()
            target_centery = (y_index*target[i]).mean()
           
            angle_loss = (4 / (torch.pi**2) ) * (torch.square(torch.arctan((pred_centery) / (pred_centerx + smooth)) 
                                                            - torch.arctan((target_centery) / (target_centerx + smooth))))

            pred_length = torch.sqrt(pred_centerx*pred_centerx + pred_centery*pred_centery + smooth)
            target_length = torch.sqrt(target_centerx*target_centerx + target_centery*target_centery + smooth)
            
            length_loss = (torch.min(pred_length, target_length)) / (torch.max(pred_length, target_length) + smooth)
        
            loss = loss + (1 - length_loss + angle_loss) / patch_size
        
        return loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count