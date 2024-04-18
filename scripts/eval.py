import argparse

import cv2
import numpy as np
import torch
import yaml
from pytorch3d.renderer import PerspectiveCameras
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.loss import MESMaskedLoss
from utils.geometry import fps_by_distance
from utils.image import render_semantic
from utils.metrics import eval_metrics
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
    video_name = './eval/autolabel_video.avi'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter.fourcc('M','J','P','G'), 15, (configs['image_width'],configs['image_height']))
    with torch.no_grad():
        waypoints = fps_by_distance(pose_xy, min_distance=radius*2, return_idx=False)
        print(f"get {waypoints.shape[0]} waypoints")
        for waypoint in waypoints:
            vertice_waypoint = waypoint + dataset.world2bev[:2, 3]
            activation_idx = grid_model.get_activation_idx(vertice_waypoint, radius)
            dataset.set_waypoint(waypoint, radius)
            for sample in tqdm(dataloader):
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
                cv2.imwrite(f"./eval/eval_{cnt:05d}-render.png", images)
                cv2.imwrite(f"./eval/eval_{cnt:05d}-gt.png", gt_image)

                # save seg numpy array
                mask = mask.detach().cpu().numpy().squeeze(3).astype(np.uint8)
                images_seg_np = images_seg.detach().cpu().numpy()
                images_seg_np = np.argmax(images_seg_np, axis=-1)

                vis_seg = render_semantic(images_seg_np[0], dataset.filted_color_map)[:, :, ::-1]
                cv2.imwrite(f"./eval/eval_{cnt:05d}-vis_seg.png", vis_seg)


                gt_image = sample["image2"]
                gt_image = gt_image.detach().cpu().numpy().squeeze()
                gt_image = (gt_image * 255).astype(np.uint8)[:, :, ::-1]
                vis_seg  = np.concatenate([np.zeros_like(vis_seg), vis_seg], axis=0)
                blend_image = cv2.addWeighted(gt_image, 0.8, vis_seg, 0.2, 0)
                cv2.imwrite(f"./eval/eval_{cnt:05d}-blend.jpg", blend_image)
                video.write(blend_image)


                images_seg_np[images_seg_np == num_class - 1] = 255
                images_seg_np[images_seg_np == 0] = 255
                images_seg_np -= 1
                images_seg_np[images_seg_np == 254] = 255
                image_segs.append(images_seg_np)

                gt_seg_np = gt_seg.detach().cpu().numpy()
                gt_seg_np *= mask
                vis_gt_seg = render_semantic(gt_seg_np[0], dataset.filted_color_map)[:, :, ::-1]
                cv2.imwrite(f"./eval/eval_{cnt:05d}-vis_gt_seg.png", vis_gt_seg)
                gt_seg_np[gt_seg_np == num_class - 1] = 255
                gt_seg_np[gt_seg_np == 0] = 255
                gt_seg_np -= 1
                gt_seg_np[gt_seg_np == 254] = 255
                gt_segs.append(gt_seg_np)
                cnt += 1

    video.release()
    loss_all = np.array(loss_all)
    loss_mean = np.mean(loss_all)
    psnr_mean = mse2psnr(loss_mean)
    print(f"MSE: {loss_mean:.4f}, PSNR: {psnr_mean:.4f}")
    if len(image_segs) > 1:
        image_segs = np.concatenate(image_segs, axis=0)
        gt_segs = np.concatenate(gt_segs, axis=0)
    else:
        image_segs = np.array(image_segs)[None]
        gt_segs = np.array(gt_segs)[None]
    results = eval_metrics(image_segs, gt_segs,
                           num_classes=num_class-2,
                           ignore_index=255,
                           metrics=['mIoU'],
                           nan_to_num=None,
                           label_map=dict(),
                           reduce_zero_label=False)
    print(results)


def get_configs():
    parser = argparse.ArgumentParser(description='G4M config')
    parser.add_argument(
        '--config',
        default="configs/local_carla.yaml",
        help='config yaml path')
    parser.add_argument(
        '--model_dir',
        default="",
        help='model dir, non emtpy for nerf studio format data')
    args = parser.parse_args()
    
    with open(args.config) as file:
        configs = yaml.safe_load(file)

    if args.model_dir != "":
        configs["model_path"] = f"{args.model_dir}/grid_baseline.pt"
        configs["pose_path"] = f"{args.model_dir}/pose_baseline.pt"
        configs["batch_size"] = 1
        configs["num_workers"] = 2
    return configs


if __name__ == "__main__":
    configs = get_configs()
    configs["filter_pose_by_distance"] = True
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
    eval(grid, poses, dataset, renderer, configs, device)
