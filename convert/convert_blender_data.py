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
    # 将图像列表堆叠成形状为(N, H, W, C)的4D数组
    ret['images'] = np.stack(images, axis=0)
    print(f'图像相机数据加载完毕,数据形状{ret['images'].shape}')
    ret['cameras_to_worlds'] = np.stack(cameras, axis=0)
    img_width = ret['images'].shape[2]
    camera_angle_x = float(meta['camera_angle_x'])
    # 计算焦距focal, focal = (0.5*img_width) / (tan(0.5*camera_angle_x))
    ret['focal'] = 0.5 * img_width / np.tan(0.5 * camera_angle_x)
    return ret


def down2(img:np.ndarray)->np.ndarray:
    """降采样式的图像缩放"""
    img_shape = img.shape
    # 类似2×2的均值池化, 将图像分辨率降低一半, 同时保留通道信息
    img_mean = np.mean(np.reshape(img, [img_shape[0]//2, 2, img_shape[1]//2, 2, -1]), (1, 3))
    return img_mean


def convert_to_nerf_data(base_dir:str, new_dir:str, n_down:int):
    """将Blender数据集转换成多尺度的数据格式"""
    if not os.path.exists(new_dir): os.makedirs(new_dir)
    splits = ['train', 'val', 'test']
    big_meta = {}
    # 对于数据集中的每个拆分
    for split in splits:
        print(f'正在处理{split}数据集...')
        data = load_rendering_data(base_dir, split)

        # 保存所有的图像数据到多尺度的存储路径
        img_dir = f'images_{split}'
        os.makedirs(os.path.join(new_dir, img_dir), exist_ok=True)
        file_names = []
        widths = []
        heights = []
        focals = []
        cameras2worlds = []
        loss_mults = []
        labels = []
        nears, fars = [], []
        f = data['focal']
        print(f'正在处理{split}数据集->图像数据...')
        for i, img in enumerate(data['images']):
            for j in range(n_down):
                # 确定图像文件存储路径
                file_name = f'{img_dir}/{i:03d}_d{j}.png'
                file_names.append(file_name)
                file_name = os.path.join(new_dir, file_name)
                # 保存相关图像, 尺寸以及相机参数信息
                with open(file_name, 'wb') as img_file_out:
                    img8 = Image.fromarray(np.uint8(img*255))
                    img8.save(img_file_out)
                widths.append(img.shape[1])
                heights.append(img.shape[0])
                focals.append(f / 2 **j)
                cameras2worlds.append(data['cameras_to_worlds'][i].tolist())
                loss_mults.append(4.0 ** j)
                labels.append(j)
                nears.append(2.0)
                fars.append(6.0)
                img = down2(img)

        # 保存数据集的元数据信息
        meta = {}
        meta['file_path'] = file_names
        meta['cam2world'] = cameras2worlds
        meta['width'] = widths
        meta['height'] = heights
        meta['focal'] = focals
        meta['label'] = labels
        meta['near'] = nears
        meta['far'] = fars
        meta['lossmult'] = loss_mults
        fx = np.array(focals)
        fy = np.array(focals)
        cx = np.array(meta['width'])*0.5
        cy = np.array(meta['height'])*0.5
        arr0 = np.zeros_like(cx)
        arr1 = np.ones_like(cx)
        k_inv = np.array([
            [arr1 / fx,   arr0,     -cx / fx],
            [arr0,      -arr1 / fy, cy / fy],
            [arr0,        arr0,     -arr1]])
        k_inv = np.moveaxis(k_inv, -1, 0)
        meta['pix2cam'] = k_inv.tolist()
        # 按照split分类保存数据
        big_meta[split] = meta
    # 打印相关数据信息+写入json文件的meta数据
    for k in big_meta:
        for j in big_meta[k]:
            print(k, j, type(big_meta[k][j]), np.array(big_meta[k][j]).shape)
    json_file = os.path.join(new_dir, 'metadata.json')
    with open(json_file, 'w') as fp:
        json.dump(big_meta, fp, ensure_ascii=False, indent=4)


def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument("--blender_dir", type=str, help='data root path')
    parser.add_argument('--object_name', type=str, help="While object you want to make multi scale", default=None)
    parser.add_argument('--out_dir', type=str, help='Output Directory')
    parser.add_argument('--n_down', type=int, help="Numbers of scale you want to scale.", default=4)
    args = parser.parse_args()
    # 创建多尺度的数据集
    blender_dir = args.blender_dir
    out_dir = args.out_dir
    n_down = args.n_down
    if not os.path.exists(out_dir): os.makedirs(out_dir)
    scenes = os.listdir(blender_dir)
    if args.object_name is not None:
        scenes = [args.object_name]
    dirs = [os.path.join(blender_dir, f) for f in scenes]
    dirs = [d for d in dirs if os.path.isdir(d)]
    print(dirs)
    for base_dir in dirs:
        new_dir = os.path.join(out_dir, os.path.basename(base_dir))
        convert_to_nerf_data(base_dir, new_dir, n_down)
        print(f'{base_dir}数据集转换完毕!')


if __name__ == '__main__':
    main()
