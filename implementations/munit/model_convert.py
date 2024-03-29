# 模型转jit

import hydra
import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch.nn.functional as F
import torch

def de_parallel(model):
    if type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel):
        return model.module
    else:
        return model

def get_style_code(dim, dv='cuda', B=1):
    return torch.rand(B, dim).to(dv) * 2 - 1

class ModelWrapper(nn.Module):
    def __init__(self, encoder, decoder, style_dim):
        super().__init__()
        self.style_dim = style_dim
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, s_code):
        B, _, _, _ = x.size()
        # 取 (self.style_dim, self.style_dim) 的随机数做s_code, 在(-1,1)上均匀采样
        # s_code = torch.rand(1, self.style_dim).to(x.device) * 2 - 1
        c_code_1, _ = self.encoder(x)
        x12 = self.decoder(c_code_1, s_code)
        return x12

@torch.inference_mode()
def convert(cfg):
    Enc1 = Encoder(dim=cfg.dim, n_downsample=cfg.n_downsample, n_residual=cfg.n_residual, style_dim=cfg.style_dim)
    Dec2 = Decoder(dim=cfg.dim, n_upsample=cfg.n_downsample, n_residual=cfg.n_residual, style_dim=cfg.style_dim)

    Enc1 = nn.DataParallel(Enc1)
    Dec2 = nn.DataParallel(Dec2)

    Enc1.load_state_dict(torch.load(cfg.encoder))
    Dec2.load_state_dict(torch.load(cfg.decoder))

    model = ModelWrapper(de_parallel(Enc1), de_parallel(Dec2), cfg.style_dim)
    model.to('cuda')
    model.eval()

    transforms_ = [
        transforms.Resize((cfg.img_height, cfg.img_width), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataset = ImageDataset("/data/result/youzhonghui/data/cbct_prompt_png", transforms_=transforms_)
    sample = dataset[0]

    x = sample['A'].unsqueeze(0).to('cuda')
    s_code = get_style_code(cfg.style_dim, 'cuda', 1)
    output = model(x, s_code)

    with torch.no_grad():
        traced_script_module = torch.jit.trace(model, (x, s_code))
        traced_script_module.save(cfg.output)

    jit_model = torch.jit.load(cfg.output)
    jit_model.eval()
    jit_output = jit_model(x, s_code)

    # 将 sample['A'] / output / jit_output / sample['B'] 拼接保存为图片
    img_A = sample['A'].cpu()
    img_B = sample['B'].cpu()
    img_output = output[0].cpu()
    img_jit_output = jit_output[0].cpu()

    imgcat = torch.cat([img_A, img_output, img_jit_output, img_B], dim=2)
    save_image(imgcat, 'output.png', nrow=1, normalize=True)

@hydra.main(version_base='1.3',
            config_path='conf',
            config_name='convert')
def main(cfg):
    convert(cfg)

if __name__ == '__main__':
    main()