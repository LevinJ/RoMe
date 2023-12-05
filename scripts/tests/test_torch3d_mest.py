"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Fri Dec 01 2023
*  File : test_torch3d_mest.py
******************************************* -->

"""
import torch
from pytorch3d.io import IO


class App(object):
    def __init__(self):
        return
    def run(self):
       

        device=torch.device("cuda:0")
        mesh = IO().load_mesh("/home/levin/workspace/nerf/RoMe/outputs/wandb/run-20231129_153754-s8s24ga1/files/bev_mesh_epoch_7.obj", device=device)
        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
