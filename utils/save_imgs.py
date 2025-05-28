import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt

from scipy.ndimage import zoom
import SimpleITK as sitk
from medpy import metric

# def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
#     msk = msk.squeeze(1).cpu().detach().numpy()
#     msk_pred = msk_pred.squeeze(1).cpu().detach().numpy()
#     img = img.squeeze(0).permute(1,2,0).detach().cpu().numpy()
#     #img = img / 255. if img.max() > 1.1 else img
#     if datasets == 'retinal':
#         msk = np.squeeze(msk, axis=0)
#         msk_pred = np.squeeze(msk_pred, axis=0)
#     else:
#         msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
#         msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

#     plt.figure(figsize=(7,15))

#     plt.subplot(3,1,1)
#     plt.imshow(img)
#     plt.axis('off')

#     plt.subplot(3,1,2)
#     plt.imshow(msk, cmap= 'gray')
#     plt.axis('off')

#     plt.subplot(3,1,3)
#     plt.imshow(msk_pred, cmap = 'gray')
#     plt.axis('off')

#     if test_data_name is not None:
#         save_path = save_path + test_data_name + '_'
#     plt.savefig(save_path + str(i) +'.png')
#     plt.close()
def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    msk_pred = msk_pred.squeeze(1).cpu().detach().numpy()
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    if datasets == 'retinal':
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    plt.figure(figsize=(7, 15))

    plt.imshow(msk_pred, cmap='gray')
    plt.axis('off')

    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()