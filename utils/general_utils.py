"""
Author: Redal
Date: 2025-11-02
Todo: Build dataset reader for general_utils.py to create gaussian model
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import torch
import sys
from datetime import datetime
import numpy as np
import random
from typing import Tuple
from PIL import Image


def inverse_sigmoid(x:torch.Tensor
                    )->torch.Tensor:
    """将sigmoid函数输出的[0,1]区间值映射回整个实数域,
    常用于机器学习中需要将概率值转换为未归一化的logits时使用
    注意输入的x必须为[0,1]区间内的值"""
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image: Image.Image, 
               resolution: Tuple[int, int]
               )->torch.Tensor:
    """将PIL格式的图像转换为PyTorch张量"""
    resized_pil_image = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_pil_image)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else: # 如果resized_image是灰度图,则增加维度至[C, H, W]
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)
    

def get_expon_lr_func(lr_init:float, 
                      lr_final:float, 
                      lr_delay_steps:int, 
                      lr_delay_mult:int, 
                      max_steps:int):
    """连续学习率衰减函数
    当step=0时,返回的学习率为lr_init;当step=max_steps时,返回的学习率为lr_final;
    如果lr_delay_steps>0,那么学习率将通过lr_delay_mult的某种平滑函数进行缩放,
    使得优化开始时的初始学习率为lr_init*lr_delay_mult,当steps>lr_delay_steps时,学习率将平缓地恢复到正常学习率
    lr_init:初始学习率, lr_final:最终学习率, lr_delay_steps:学习率延迟调整的步数, 
    lr_delay_mult:学习率延迟调整的倍数, max_steps:优化的步数"""
    def helper(step):
        if step < 0 or (lr_init==0.0 and lr_final==0.0): return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1))
        else: delay_rate = 1.0
        # 将step/max_steps计算结果限制在[0,1]之间
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1-t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper


def strip_symmetric(sym):
    """将对称矩阵转换为6维向量"""
    def strip_lowerdiag(L):
        """将Lz矩阵的Lower Diagonal部分转换为6维向量"""
        uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device='cuda')
        uncertainty[:, 0] = L[:, 0, 0]
        uncertainty[:, 1] = L[:, 0, 1]
        uncertainty[:, 2] = L[:, 0, 2]
        uncertainty[:, 3] = L[:, 1, 1]
        uncertainty[:, 4] = L[:, 1, 2]
        uncertainty[:, 5] = L[:, 2, 2]
        return uncertainty
    return strip_lowerdiag(sym)


def build_rotation(r):
    """从四元数到旋转矩阵的转换"""
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + 
                      r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    # 给出旋转矩阵的元素r, x, y, z
    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


def build_scaling_rotation(s, r):
    """构建缩放旋转矩阵"""
    