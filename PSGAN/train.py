import os
import argparse
from torch.backends import cudnn
import functools
import inspect
import logging
from fvcore.common.config import CfgNode
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

import torch
import numpy as np
import PIL
import random
import linecache
from PIL import Image
from torch.utils.data import Dataset


from psgan.solver import Solver
from psgan.preprocess import PreProcess
from tools.data_reader import DataReader

"""
This file defines default options of configurations.
It will be further merged by yaml files and options from
the command-line.
Note that *any* hyper-parameters should be firstly defined
here to enable yaml and command-line configuration.
"""

_C = CfgNode()


# Paths for logging and saving
_C.LOG = CfgNode()
_C.LOG.LOG_PATH = 'log/'
_C.LOG.SNAPSHOT_PATH = 'snapshot/'
_C.LOG.VIS_PATH = 'visulization/'
_C.LOG.SNAPSHOT_STEP = 1024
_C.LOG.LOG_STEP = 8
_C.LOG.VIS_STEP = 2048

# Data settings
_C.DATA = CfgNode()
_C.DATA.PATH = './data'
_C.DATA.NUM_WORKERS = 4
_C.DATA.BATCH_SIZE = 1
_C.DATA.IMG_SIZE = 256

# Training hyper-parameters
_C.TRAINING = CfgNode()
_C.TRAINING.G_LR = 2e-4
_C.TRAINING.D_LR = 2e-4
_C.TRAINING.BETA1 = 0.5
_C.TRAINING.BETA2 = 0.999
_C.TRAINING.C_DIM = 2
_C.TRAINING.G_STEP = 1
_C.TRAINING.NUM_EPOCHS = 50
_C.TRAINING.NUM_EPOCHS_DECAY = 0

# Loss weights
_C.LOSS = CfgNode()
_C.LOSS.LAMBDA_A = 10.0
_C.LOSS.LAMBDA_B = 10.0
_C.LOSS.LAMBDA_IDT = 0.5
_C.LOSS.LAMBDA_CLS = 1
_C.LOSS.LAMBDA_REC = 10
_C.LOSS.LAMBDA_HIS = 1
_C.LOSS.LAMBDA_SKIN = 0.1
_C.LOSS.LAMBDA_EYE = 1
_C.LOSS.LAMBDA_HIS_LIP = _C.LOSS.LAMBDA_HIS
_C.LOSS.LAMBDA_HIS_SKIN = _C.LOSS.LAMBDA_HIS * _C.LOSS.LAMBDA_SKIN
_C.LOSS.LAMBDA_HIS_EYE = _C.LOSS.LAMBDA_HIS * _C.LOSS.LAMBDA_EYE
_C.LOSS.LAMBDA_VGG = 5e-3

# Model structure
_C.MODEL = CfgNode()
_C.MODEL.G_CONV_DIM = 64
_C.MODEL.D_CONV_DIM = 64
_C.MODEL.G_REPEAT_NUM = 6
_C.MODEL.D_REPEAT_NUM = 3
_C.MODEL.NORM = "SN"
_C.MODEL.WEIGHTS = "assets/models"


# Preprocessing
_C.PREPROCESS = CfgNode()
_C.PREPROCESS.UP_RATIO = 0.6 / 0.85  # delta_size / face_size
_C.PREPROCESS.DOWN_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.WIDTH_RATIO = 0.2 / 0.85  # delta_size / face_size
_C.PREPROCESS.LIP_CLASS = [7, 9]
_C.PREPROCESS.FACE_CLASS = [1, 6]
_C.PREPROCESS.LANDMARK_POINTS = 68

# Postprocessing
_C.POSTPROCESS = CfgNode()
_C.POSTPROCESS.WILL_DENOISE = False


def get_config()->CfgNode:
    return _C



def ToTensor(pic):
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img



class MakeupDataloader(Dataset):
    def __init__(self, image_path, preprocess: PreProcess, transform, transform_mask):
        self.image_path = image_path
        self.transform = transform
        self.transform_mask = transform_mask
        self.preprocess = preprocess

        self.reader = DataReader(image_path)

    def __getitem__(self, index):
        (image_s, mask_s, lm_s), (image_r, mask_r, lm_r) = self.reader.pick()
        lm_s = self.preprocess.relative2absolute(lm_s / image_s.size)
        lm_r = self.preprocess.relative2absolute(lm_r / image_r.size)
        image_s = self.transform(image_s)
        mask_s = self.transform_mask(Image.fromarray(mask_s))
        image_r = self.transform(image_r)
        mask_r = self.transform_mask(Image.fromarray(mask_r))

        mask_s, dist_s = self.preprocess.process(
            mask_s.unsqueeze(0), lm_s)
        mask_r, dist_r = self.preprocess.process(
            mask_r.unsqueeze(0), lm_r)
        return [image_s, mask_s, dist_s], [image_r, mask_r, dist_r]

    def __len__(self):
        return len(self.reader)



def get_loader(config, mode="train"):
    # return the DataLoader
    transform = transforms.Compose([
    transforms.Resize(config.DATA.IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_mask = transforms.Compose([
        transforms.Resize(config.DATA.IMG_SIZE, interpolation=PIL.Image.NEAREST),
        ToTensor])

    dataset = MakeupDataloader(
        config.DATA.PATH, transform=transform,
        transform_mask=transform_mask, preprocess=PreProcess(config))
    
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config.DATA.BATCH_SIZE,
                            shuffle=True, num_workers=config.DATA.NUM_WORKERS)
    return dataloader



if __name__ == '__main__':
    config = get_config()
    config.freeze()

    print("Configurations:")
    print(config)

    cudnn.benchmark = True

    data_loader = get_loader(config)
    solver = Solver(config, data_loader=data_loader, device="cuda")
    solver.train()
