"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Wed Dec 06 2023
*  File : render_img.py
******************************************* -->

"""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import DataLoader

from models.loss import MESMaskedLoss
from utils.geometry import fps_by_distance
from utils.image import render_semantic
from utils.poseinfo import PoseInfo
from utils.renderer import Renderer


def mse2psnr(mse):
    """
    :param mse: scalar
    :return:    scalar np.float32
    """
    mse = np.maximum(mse, 1e-10)  # avoid -inf or nan when mse is very small.
    psnr = -10.0 * np.log10(mse)
    return psnr.astype(np.float32)


def eval(grid_model, pose_model, dataset, renderer, configs, device):
    grid_model.eval()
    pose_model.eval()
    num_class = dataset.num_class
    pose_xy = np.array(dataset.ref_camera2world_all)[:, :2, 3]
    dataloader = DataLoader(dataset, batch_size=configs["batch_size"],
                            num_workers=configs["num_workers"],
                            shuffle=False,
                            drop_last=False)
    print(f"Get {len(dataset.ref_camera2world_all)} images for mapping")
    # Load grid and optimization toggles
    optim_dict = dict()
    for optim_option in ["vertices_rgb", "vertices_label", "vertices_z", "rotations", "translations"]:
        if configs["lr"].get(optim_option, 0) != 0:
            optim_dict[optim_option] = True
        else:
            optim_dict[optim_option] = False

    radius = configs["waypoint_radius"]
    loss_all = []
    loss_fuction = MESMaskedLoss()
    image_segs = []
    gt_segs = []
    cnt = 0
    with torch.no_grad():
        waypoints = fps_by_distance(pose_xy, min_distance=radius*2, return_idx=False)
        print(f"get {waypoints.shape[0]} waypoints")
        for waypoint in waypoints:
            vertice_waypoint = waypoint + dataset.world2bev[:2, 3]
            activation_idx = grid_model.get_activation_idx(vertice_waypoint, radius)
            dataset.set_waypoint(waypoint, radius)

            sample = dataset[0]
            for key, ipt in sample.items():
                if key == "image_path":
                    continue
                if isinstance(ipt, np.ndarray):
                    sample[key] = torch.tensor(ipt[np.newaxis, ...])
                else:
                    sample[key] = torch.tensor([ipt])
            samples = [sample]
            for sample in samples:
                for key, ipt in sample.items():
                    if key != "image_path":
                        sample[key] = ipt.clone().detach().to(device)
                # image_path = sample["image_path"][0]
                mesh = grid_model(activation_idx, configs["batch_size"])
                pose = pose_model(sample["camera_idx"])
                transform = pose @ sample["Transform_pytorch3d"]
                R_pytorch3d = transform[:, :3, :3]
                T_pytorch3d = transform[:, :3, 3]
                focal_pytorch3d = sample["focal_pytorch3d"]
                p0_pytorch3d = sample["p0_pytorch3d"]
                image_shape = sample["image_shape"]
                cameras = PerspectiveCameras(
                    R=R_pytorch3d,
                    T=T_pytorch3d,
                    focal_length=focal_pytorch3d,
                    principal_point=p0_pytorch3d,
                    image_size=image_shape,
                    device=device
                )

                gt_image = sample["image"]
                gt_seg = sample["static_label"]
                images_feature, depth = renderer({"mesh": mesh, "cameras": cameras})

                silhouette = images_feature[:, :, :, -1]
                silhouette[silhouette > 0] = 1
                silhouette = torch.unsqueeze(silhouette, -1)
                mask = silhouette
                if "static_mask" in sample:
                    static_mask = torch.unsqueeze(sample["static_mask"], -1)
                    mask *= static_mask

                images = images_feature[:, :, :, :3]
                if optim_dict["vertices_rgb"]:
                    images_seg = images_feature[:, :, :, 3:-1]
                else:
                    images_seg = images_feature[:, :, :, :-1]

                mse_loss = loss_fuction(images, gt_image, mask)
                mse_loss_np = mse_loss.cpu().detach().numpy()
                loss_all.append(mse_loss_np)
                images = images.detach().cpu().numpy().squeeze()
                gt_image = gt_image.detach().cpu().numpy().squeeze()
                mask_vis = mask.detach().cpu().numpy().astype(np.uint8)
                if mask_vis.shape[0] == 1:
                    mask_vis = mask_vis.squeeze(0)

                images = (images * 255).astype(np.uint8)[:, :, ::-1]
                gt_image = (gt_image * 255).astype(np.uint8)[:, :, ::-1]
                cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-render.png", images)
                cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-gt.png", gt_image)

                # save seg numpy array
                mask = mask.detach().cpu().numpy().squeeze(3).astype(np.uint8)
                images_seg_np = images_seg.detach().cpu().numpy()
                images_seg_np = np.argmax(images_seg_np, axis=-1)

                vis_seg = render_semantic(images_seg_np[0], dataset.filted_color_map)[:, :, ::-1]
                cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-vis_seg.png", vis_seg)
                blend_image = cv2.addWeighted(gt_image, 0.5, vis_seg, 0.5, 0)
                cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-blend.png", blend_image)

                images_seg_np[images_seg_np == num_class - 1] = 255
                images_seg_np[images_seg_np == 0] = 255
                images_seg_np -= 1
                images_seg_np[images_seg_np == 254] = 255
                image_segs.append(images_seg_np)

                gt_seg_np = gt_seg.detach().cpu().numpy()
                gt_seg_np *= mask
                vis_gt_seg = render_semantic(gt_seg_np[0], dataset.filted_color_map)[:, :, ::-1]
                cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-vis_gt_seg.png", vis_gt_seg)
                gt_seg_np[gt_seg_np == num_class - 1] = 255
                gt_seg_np[gt_seg_np == 0] = 255
                gt_seg_np -= 1
                gt_seg_np[gt_seg_np == 254] = 255
                gt_segs.append(gt_seg_np)
                cnt += 1
                break

    # loss_all = np.array(loss_all)
    # loss_mean = np.mean(loss_all)
    # psnr_mean = mse2psnr(loss_mean)
    # print(f"MSE: {loss_mean:.4f}, PSNR: {psnr_mean:.4f}")
    # if len(image_segs) > 1:
    #     image_segs = np.concatenate(image_segs, axis=0)
    #     gt_segs = np.concatenate(gt_segs, axis=0)
    # else:
    #     image_segs = np.array(image_segs)[None]
    #     gt_segs = np.array(gt_segs)[None]
    # results = eval_metrics(image_segs, gt_segs,
    #                     num_classes=num_class-2,
    #                     ignore_index=255,
    #                     metrics=['mIoU'],
    #                     nan_to_num=None,
    #                     label_map=dict(),
    #                     reduce_zero_label=False)
    # print(results)


def get_configs():
    args_config = '/home/levin/workspace/nerf/RoMe/configs/nerf_road.yaml'
    model_dir = "/home/levin/workspace/nerf/RoMe/outputs/wandb/run-20240104_145112-jybb47oi/files"
    with open(args_config) as file:
        configs = yaml.safe_load(file)
    configs["model_path"] = f"{model_dir}/grid_baseline.pt"
    configs["pose_path"] = f"{model_dir}/pose_baseline.pt"
    configs["batch_size"] = 1
    configs["num_workers"] = 2
    return configs

class App(object):
    def __init__(self):
        return
    def eval(self, grid_model, pose_model, dataset, renderer, configs, device):
        grid_model.eval()
        pose_model.eval()
        num_class = dataset.num_class
        pose_xy = np.array(dataset.ref_camera2world_all)[:, :2, 3]
        # dataloader = DataLoader(dataset, batch_size=configs["batch_size"],
        #                         num_workers=configs["num_workers"],
        #                         shuffle=False,
        #                         drop_last=False)
        print(f"Get {len(dataset.ref_camera2world_all)} images for mapping")
        # Load grid and optimization toggles
        optim_dict = dict()
        for optim_option in ["vertices_rgb", "vertices_label", "vertices_z", "rotations", "translations"]:
            if configs["lr"].get(optim_option, 0) != 0:
                optim_dict[optim_option] = True
            else:
                optim_dict[optim_option] = False

        radius = configs["waypoint_radius"] * 1000
        loss_all = []
        loss_fuction = MESMaskedLoss()
        image_segs = []
        gt_segs = []
        cnt = 0
        with torch.no_grad():
            waypoints = fps_by_distance(pose_xy, min_distance=radius*2, return_idx=False)
            print(f"get {waypoints.shape[0]} waypoints")
            for waypoint in waypoints:
                vertice_waypoint = waypoint + dataset.world2bev[:2, 3]
                activation_idx = grid_model.get_activation_idx(vertice_waypoint, radius)
                dataset.set_waypoint(waypoint, radius)

                sample = dataset[0]
                print(f"sample image {sample['image_path']}")
                #adjust sample values
                # t1, t2, t3 = 1.8998901e+01, 9.8098389e+01, 1.4919682e+00
                # # t1, t2, t3 = 1.95, 9.8098389e+01, 1.4919682e+00
                # sample["camera2world"]  = np.array([[ 3.3148620e-02,  6.8456652e-03,  9.9942696e-01,  t1],
                #                                     [-9.9930471e-01, -1.6848126e-02,  3.3259965e-02,  t2],
                #                                     [ 1.7066160e-02, -9.9983466e-01,  6.2824134e-03,  t3],
                #                                     [ 0.0000000e+00,  0.0000000e+00,  0.0000000e+00,  1.0000000e+00]],
                #                                     dtype=np.float32)
                # sample["world2camera"] = np.linalg.inv(sample["camera2world"] )
                # sample = dataset.opencv_camera2pytorch3d_(sample)
                # Twc = PoseInfo().construct_fromT(sample["camera2world"] )
                # print(f"camera2world= {Twc.get_short_desc()}")

                for key, ipt in sample.items():
                    if key == "image_path":
                        continue
                    if isinstance(ipt, np.ndarray):
                        sample[key] = torch.tensor(ipt[np.newaxis, ...])
                    else:
                        sample[key] = torch.tensor([ipt])

                samples = [sample]
                for sample in samples:
                    for key, ipt in sample.items():
                        if key != "image_path":
                            sample[key] = ipt.clone().detach().to(device)
                    # image_path = sample["image_path"][0]
                    mesh = grid_model(activation_idx, configs["batch_size"])
                    pose = pose_model(sample["camera_idx"])
                    transform = pose @ sample["Transform_pytorch3d"]

                    pose_ext = self.tensor2poseInfo(pose)
                    print(f"extrinsic adjustment {pose_ext.get_short_desc()}")
                    # Twc = self.tensor2poseInfo(transform).I
                    # print(f"Twc= {Twc.get_short_desc()}")
                    R_pytorch3d = transform[:, :3, :3]
                    T_pytorch3d = transform[:, :3, 3]
                    focal_pytorch3d = sample["focal_pytorch3d"]
                    p0_pytorch3d = sample["p0_pytorch3d"]
                    image_shape = sample["image_shape"]
                    
                    cameras = PerspectiveCameras(
                        R=R_pytorch3d,
                        T=T_pytorch3d,
                        focal_length=focal_pytorch3d,
                        principal_point=p0_pytorch3d,
                        image_size=image_shape,
                        device=device
                    )



                    gt_image = sample["image"]
                    gt_seg = sample["static_label"]
                    images_feature, depth = renderer({"mesh": mesh, "cameras": cameras})

                    silhouette = images_feature[:, :, :, -1]
                    silhouette[silhouette > 0] = 1
                    silhouette = torch.unsqueeze(silhouette, -1)
                    mask = silhouette
                    if "static_mask" in sample:
                        static_mask = torch.unsqueeze(sample["static_mask"], -1)
                        mask *= static_mask

                    images = images_feature[:, :, :, :3]
                    if optim_dict["vertices_rgb"]:
                        images_seg = images_feature[:, :, :, 3:-1]
                    else:
                        images_seg = images_feature[:, :, :, :-1]

                    mse_loss = loss_fuction(images, gt_image, mask)
                    mse_loss_np = mse_loss.cpu().detach().numpy()
                    loss_all.append(mse_loss_np)
                    images = images.detach().cpu().numpy().squeeze()
                    gt_image = gt_image.detach().cpu().numpy().squeeze()
                    mask_vis = mask.detach().cpu().numpy().astype(np.uint8)
                    if mask_vis.shape[0] == 1:
                        mask_vis = mask_vis.squeeze(0)

                    images = (images * 255).astype(np.uint8)[:, :, ::-1]
                    gt_image = (gt_image * 255).astype(np.uint8)[:, :, ::-1]
                    cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-render.png", images)
                    cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-gt.png", gt_image)

                    # save seg numpy array
                    mask = mask.detach().cpu().numpy().squeeze(3).astype(np.uint8)
                    images_seg_np = images_seg.detach().cpu().numpy()
                    images_seg_np = np.argmax(images_seg_np, axis=-1)

                    vis_seg = render_semantic(images_seg_np[0], dataset.filted_color_map)[:, :, ::-1]
                    cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-vis_seg.png", vis_seg)
                    blend_image = cv2.addWeighted(gt_image, 0.5, vis_seg, 0.5, 0)
                    cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-blend.png", blend_image)

                    images_seg_np[images_seg_np == num_class - 1] = 255
                    images_seg_np[images_seg_np == 0] = 255
                    images_seg_np -= 1
                    images_seg_np[images_seg_np == 254] = 255
                    image_segs.append(images_seg_np)

                    gt_seg_np = gt_seg.detach().cpu().numpy()
                    gt_seg_np *= mask
                    vis_gt_seg = render_semantic(gt_seg_np[0], dataset.filted_color_map)[:, :, ::-1]
                    cv2.imwrite(f"./outputs/render_img/eval_{cnt:05d}-vis_gt_seg.png", vis_gt_seg)
                    gt_seg_np[gt_seg_np == num_class - 1] = 255
                    gt_seg_np[gt_seg_np == 0] = 255
                    gt_seg_np -= 1
                    gt_seg_np[gt_seg_np == 254] = 255
                    gt_segs.append(gt_seg_np)
                    cnt += 1
                    break
        loss_all = np.array(loss_all)
        loss_mean = np.mean(loss_all)
        psnr_mean = mse2psnr(loss_mean)
        print(f"MSE: {loss_mean:.4f}, PSNR: {psnr_mean:.4f}")
        self.disp_images([images, gt_image])
        # self.disp_images([images, gt_image, blend_image, vis_seg, vis_gt_seg])
    def disp_images(self, imgs):
        for img in imgs:
            plt.figure()
            plt.imshow(img[...,::-1])
        plt.show()
        return
    def tensor2poseInfo(self, pose_tensor):
        pose = pose_tensor.cpu().numpy()[0]
        res = PoseInfo().construct_fromT(pose)
        return res
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
