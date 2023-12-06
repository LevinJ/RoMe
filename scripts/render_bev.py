"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Thu Nov 30 2023
*  File : render_mesh.py
******************************************* -->

"""



import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from utils.image import render_semantic
from utils.monitorutil import MonitorUtil
from utils.renderer import Renderer
from utils.visualizer import Visualizer


def get_configs():
    args_config = '/home/levin/workspace/nerf/RoMe/configs/nusc_eval.yaml'
    with open(args_config) as file:
        configs = yaml.safe_load(file)
    return configs
class App(object):
    def __init__(self):
        return
    def eval(self, grid_model, pose_model, dataset, renderer, configs, device):
        visualizer = Visualizer(device, configs)
        grid_model.eval()
        
        optim_dict = dict()
        for optim_option in ["vertices_rgb", "vertices_label", "vertices_z", "rotations", "translations"]:
            if configs["lr"].get(optim_option, 0) != 0:
                optim_dict[optim_option] = True
            else:
                optim_dict[optim_option] = False

        
        mesh = grid_model(None, configs["batch_size"])

        bev_features, bev_depth = visualizer(mesh[0])
        if optim_dict["vertices_rgb"]:
            bev_seg = bev_features[0, :, :, 3:-1].detach().cpu().numpy()
        else:
            bev_seg = bev_features[0, :, :, :-1].detach().cpu().numpy()
        bev_rgb = bev_features[0, :, :, :3].detach().cpu().numpy()
        # plt.imshow(bev_rgb)
        # bev_rgb = np.clip(bev_rgb, 0, 1)  # see https://github.com/wandb/client/issues/2722
        # plt.imshow(bev_rgb)
        bev_rgb = (bev_rgb * 255).astype(np.uint8)
        # cv2.imwrite(f"./outputs/bev_rgb.png", bev_rgb[...,::-1])


        bev_seg = np.argmax(bev_seg, axis=-1)
        bev_seg = render_semantic(bev_seg, dataset.filted_color_map)  # RGB fomat
        # plt.imshow(bev_seg)
        # cv2.imwrite(f"./outputs/bev_seg.png", bev_seg[...,::-1])

        # bev_depth = bev_depth[0, :, :, 0].detach().cpu().numpy()

        MonitorUtil(show_img = True, use_bev_height=True).vis_mesh(grid_model, visualizer)
        # plt.switch_backend("tkagg")
        # plt.imshow(img)
        plt.figure()
        plt.imshow(bev_rgb)
        plt.figure()
        plt.imshow(bev_seg)

        plt.show()
        return
    def run(self):
        configs = get_configs()
        device = torch.device("cuda:0")

        if configs["dataset"] == "NuscDataset":
            from datasets.nusc import NuscDataset as Dataset
        elif configs["dataset"] == "KittiDataset":
            from datasets.kitti import KittiDataset as Dataset
        elif configs["dataset"] == "NerfStudio":
            from datasets.nerfstudio import NerfStudio as Dataset
        else:
            raise NotImplementedError("Dataset not implemented")

        renderer = Renderer().to(device)
        dataset = Dataset(configs)

        grid = torch.load(configs["model_path"])
        poses = torch.load(configs["pose_path"])
        grid = grid.to(device)
        poses = poses.to(device)
        self.eval(grid, poses, dataset, renderer, configs, device)
        
        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
