"""
Author: Redal
Todo: Convert blender data to our own format
Date: 2025/10/28
Homepage: 
"""
import json
import os
import argparse
import numpy as np
from PIL import Image
from typing import List, Dict, Union


# 针对Nerf-Synthetic数据集将Blender生成的渲染数据生成
# 适合多尺度训练的Nerf的数据格式, 提高模型的表现能力
def load_rendering_data(data_dir:str, split:str)->Dict[np.ndarray, Union[np.ndarray, float]]:
    """从数据集中加载图像和metadata, data_dir:数据集的路径目录
    split:数据集中的负责分割train, test, split场景标识"""
    f = f'transforms_{split}.json'
    with open(os.path.join(data_dir, f)) as fp:
        meta = json.load(fp)
    images = []
    cameras = []
    print("正在加载图像数据...")
    for frame in meta['frames']:
        f_name = os.path.join(data_dir, frame['file_path']+'.png')
        with open(f_name, 'rb') as img_file:
            image = np.array(Image.open(img_file), dtype=np.float32)
        cameras.append(frame['transform_matrix'])
        images.append(image)
    # 构建数据字典ret
    ret = {}
    ret['images'] = np.stack(images, axis=0)
    