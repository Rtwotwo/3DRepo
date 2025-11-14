import numpy as np
import torch
import torch.nn as nn
from utils.graphics_utils import getProjectionMatrix
import math
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
import torchvision
from argparse import ArgumentParser
from arguments import PipelineParams

class VirtualCamera(nn.Module):
    def __init__(
        self,
        # 点云的旋转矩阵
        # 绕Z轴的旋转
        angle_z = np.radians(10),

        # 绕Y轴的旋转
        angle_y = np.radians(-35),
        

        # 绕X轴的旋转
        angle_x = np.radians(17),
        

        # 相机的旋转矩阵
        R_camera = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]),

        T=np.array(
            [0, 0, 0],
            dtype=np.float32,
        ),
        FoVy=math.radians(30),#角度化弧度 fov=30°
        image_height=800,
        image_width=800,
    ):
        Rz = np.array([
            [np.cos(angle_z), -np.sin(angle_z), 0],
            [np.sin(angle_z), np.cos(angle_z), 0],
            [0, 0, 1]
            ])
        Ry = np.array([
            [np.cos(angle_y), 0, np.sin(angle_y)],
            [0, 1, 0],
            [-np.sin(angle_y), 0, np.cos(angle_y)]
            ])
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(angle_x), -np.sin(angle_x)],
            [0, np.sin(angle_x), np.cos(angle_x)]
            ])
        
        # 合成最终的点云旋转矩阵
        R_point_cloud = Rz @ Ry @ Rx
        
        # 计算新的相机旋转矩阵
        R_camera_corrected = R_point_cloud @ R_camera,

        print("点云的最终旋转矩阵:")
        print(R_point_cloud)

        print("新的相机旋转矩阵:")
        print(R_camera_corrected)
        
        self.R: np.array = R_camera
        self.T: np.array = T
        self.FoVy: float = FoVy
        self.image_height: int = image_height
        self.image_width: int = image_width
        self.update_attributes()

    def update_attributes(self):

        self.FoVx = compute_fovX(#算fovx
            fovY=self.FoVy,
            height=self.image_height,
            width=self.image_width,
        )

        self.zfar = 100.0
        self.znear = 0.01

        self.world_view_transform = getViewMatrix(self.R, self.T)#view w2c
        self.projection_matrix = getProjectionMatrix(#projection
            znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
        )
        self.full_proj_transform = self.projection_matrix @ self.world_view_transform#？？？？？？？？？？

        self.world_view_transform = self.world_view_transform.transpose(0, 1).cuda()#一维化方便py和c的矩阵统一
        self.projection_matrix = self.projection_matrix.transpose(0, 1).cuda()
        self.full_proj_transform = self.full_proj_transform.transpose(0, 1).cuda()

        self.camera_center = torch.tensor(self.T, dtype=torch.float32).cuda()

        return

    def setLookAt(self, cam_pos, target_pos):
        cam_pos = np.array(cam_pos, dtype=np.float32)
        target_pos = np.array(target_pos, dtype=np.float32)
        y_approx = np.array([0, 1, 0], dtype=np.float32)#预设值
        self.R = LookAt(cam_pos, target_position=target_pos, y_approx=y_approx)
        self.T = cam_pos
        self.update_attributes()
        return


def compute_fovX(fovY, height, width):
    w_h_ratio = float(width) / float(height)

    return math.atan(math.tan(fovY * 0.5) * w_h_ratio) * 2#返回fovx


def getViewMatrix(R, t):#w2c  Rt  -RtT
    Rt = np.zeros((4, 4))#    0    1
    Rt[:3, :3] = R.T
    Rt[:3, 3] = -R.T @ t
    Rt[3, 3] = 1.0

    return torch.tensor(Rt, dtype=torch.float32)


def LookAt(camera_position, target_position, y_approx):

    look_dir = target_position - camera_position
    z_axis = look_dir#相机光轴为z
    z_axis /= np.linalg.norm(z_axis)  # 归一化

    x_axis = np.cross(y_approx, z_axis)#x=y×z(向量)
    x_axis /= np.linalg.norm(x_axis)  # 归一化

    y_axis = np.cross(z_axis, x_axis)#y=z×x(向量)
    y_axis /= np.linalg.norm(y_axis)  # 归一化

    R = np.zeros((3, 3))
    R[:, 0] = x_axis
    R[:, 1] = y_axis
    R[:, 2] = z_axis

    return R#旋转矩阵

def insert_cam(start_pos, end_pos, num_frames):
    start_pos = np.array(start_pos)
    end_pos = np.array(end_pos)
    #插值
    t = np.linspace(0, 1, num_frames)

    #计算插值位置
    interpolated_positions = [start_pos + (end_pos - start_pos)* ti for ti in t]

    return interpolated_positions

@torch.no_grad()
def test_virtual_camera():
    # Set up command line argument parser
    parser = ArgumentParser(description="gaussian splatting")
    pipe = PipelineParams(parser)

    gaussians = GaussianModel(3)
    # gaussians.load_ply("./test.ply")
    gaussians.load_ply("/data1/kongsujie/raoxinyao/exp_results/3d-mip-splatting/Nerf-Synthetic/nerf-synthetic-mtmt/mic/point_cloud/iteration_30000/point_cloud.ply")

    # gaussians._xyz[:,0] = gaussians._xyz[:,0] + 0.2
    # gaussians._xyz[:,1] = gaussians._xyz[:,1] + 0.2
    # gaussians._xyz[:,2] = gaussians._xyz[:,2] + 2

    cam = VirtualCamera()
    # cam.setLookAt(cam_pos=[-3, 0, -3], target_pos=[0,0,0]) # 2.png
    # cam.setLookAt(cam_pos=[-3, 0, 3], target_pos=[0,0,0]) # 3.png
    # cam.setLookAt(cam_pos=[-2, 3, 2], target_pos=[0,0,0]) # 4.png
    # cam.setLookAt(cam_pos=[2, 3, -2], target_pos=[0,0,0]) # 5.png
    # cam.setLookAt(cam_pos=[2, -3, -2], target_pos=[0,0,0]) # 6.png
    # cam.setLookAt(cam_pos=[1, -1, 3], target_pos=[0,0,0]) # 7.png
    cam.setLookAt(cam_pos=[1, -1, 2], target_pos=[0,0,0]) # 8.png
    # cam.setLookAt(cam_pos=[0.5, -0.8, 1.8], target_pos=[0,0,0]) # 9.png
    # cam.setLookAt(cam_pos=[0.2, -0.8, 1.5], target_pos=[0,0,0]) # 10.png
    bg = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    # positions = insert_cam(start_pos, target_pos, num_frames)

    rendering = render(cam, gaussians, pipe, bg)["render"]
    torchvision.utils.save_image(rendering, 
        f"runs/nerf_mtmt/_render/mic_render/8.png")
    # for i, pos in enumerate(positions):
    #     print(f"Frame {i+1}: Position {pos}")
    #     cam.setLookAt(cam_pos=pos, target_pos=[0.4, -0.5, 0.4])
    #     rendering = render(cam, gaussians, pipe, bg)["render"]
    #     torchvision.utils.save_image(rendering, f"./trans_cloud/images/{i+1}.png")

if __name__ == "__main__":
    test_virtual_camera()