"""
Author: Redal
Date: 2025-11-02
Todo: 
Homepage: https://github.com/Rtwotwo/3DRepo
"""
import os
from errno import EEXIST
from os import makedirs, path


def mkdir_p(folder_path: str)->None:
    """创建一个目录,相当于在命令行中使用mkdir -p"""
    try: makedirs(folder_path)
    except OSError as exc:
        if exc.errno==EEXIST and path.isdir(folder_path): pass
        else: raise


def searchForMaxIteration(folder:str)->int:
    """查找指定文件夹中文件名所包含的最大迭代次数"""
    saved_iters = [int(fname.split('_')[-1]) for fname in os.listdir(folder)]
    return max(saved_iters)