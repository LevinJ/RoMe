"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Wed Nov 22 2023
*  File : nerfstudio.py
******************************************* -->

"""
import json
import sys
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
from sympy import false

from datasets.base import BaseDataset
from datasets.nusc import NuscDataset
from utils.plane_fit import robust_estimate_flatplane
from utils.pose_util import get_Tgl2cv

sys.path.append('/home/levin/workspace/ros_projects/src/vslam_localization/scripts/nerf/data_convert/nerfstudio')
from gen_mullti_trip_config import MultiTripconfig


def load_from_json(filename: Path):
    """Load a dictionary from a JSON filename.

    Args:
        filename: The filename to load from.
    """
    assert filename.suffix == ".json"
    with open(filename, encoding="UTF-8") as file:
        return json.load(file)
class NerfStudio(NuscDataset):
    def __init__(self, configs):
        BaseDataset.__init__(self)

        self.resized_image_size = (configs["image_width"], configs["image_height"])
        self.trip_config = configs["trip_config"]
        # clip_list = configs["clip_list"]
        # camera_names = configs["camera_names"]
        x_offset = -configs["center_point"]["x"] + configs["bev_x_length"]/2
        y_offset = -configs["center_point"]["y"] + configs["bev_y_length"]/2
        self.world2bev = np.asarray([
            [1, 0, 0, x_offset],
            [0, 1, 0, y_offset],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.min_distance = configs["min_distance"]

        filter_pose_by_distance = True
        if 'filter_pose_by_distance' in configs:
            filter_pose_by_distance = configs['filter_pose_by_distance']
        meta = MultiTripconfig().load_trip_config(
            self.trip_config).get_transform_dict(filter_pose_by_distance)
        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2"]:
            if distort_key in meta:
                distort_fixed = True
                break
        height = []
        width = []
        self.camera_heights = []
        self.camerafront2world = []

        meta["frames"] = sorted(meta["frames"], key=lambda d: d['frame_id']) 
        for idx,frame in enumerate(meta["frames"]):
            
            filepath = frame["file_path"]
            label_filepath = frame["label_path"]

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx = (float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy = (float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx = (float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy = (float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            intrinsic = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            
            glcamera2world = np.array(frame["transform_matrix"]).astype(np.float32)

            camera2world = glcamera2world @ get_Tgl2cv(inv=True).astype(np.float32)

            self.ref_camera2world_all.append(camera2world)
            
            self.cameras_idx_all.append(self.cameraname2id(frame['camera_name']))
            self.cameras_K_all.append(intrinsic.astype(np.float32))

            self.label_filenames_all.append(label_filepath)
            self.image_filenames_all.append(filepath)


            if 'camera_height' in frame:
                self.camera_heights.append(frame['camera_height'])
                self.camerafront2world.append(camera2world)

        # 6. estimate flat plane
        #skip file existence check
        # self.file_check()
        # self.label_valid_check()

        self.camerafront2world = np.array(self.camerafront2world)
        print("before plane estimation, z std = ", self.camerafront2world[:, 2, 3].std())
        #front camera height
        camera_height = np.array(self.camera_heights).mean()
        # camera_height =  1.9

        transform_normal2origin = robust_estimate_flatplane(self.camerafront2world[:, :3, 3]).astype(np.float32)
        transform_normal2origin[2, 3] += camera_height

        self.ref_camera2world_all = transform_normal2origin[None] @ np.array(self.ref_camera2world_all)

        self.camerafront2world = transform_normal2origin[None] @ np.array(self.camerafront2world)
        print("after plane estimation, z std = ", self.camerafront2world[:, 2, 3].std())

        # 7. filter poses in bev range
        all_camera_xy = np.asarray(self.ref_camera2world_all)[:, :2, 3]
        available_mask_x = abs(all_camera_xy[:, 0]) < configs["bev_x_length"] // 2 + 10
        available_mask_y = abs(all_camera_xy[:, 1]) < configs["bev_y_length"] // 2 + 10
        available_mask = available_mask_x & available_mask_y
        available_idx = list(np.where(available_mask)[0])
        print(f"before poses filtering, pose num = {available_mask.shape[0]}")
        self.filter_by_index(available_idx)
        print(f"after poses filtering, pose num = {available_mask.sum()}")
        return
    def cameraname2id(self, cam_name):
        if not hasattr(self, 'camera_dict'):
            self.camera_dict = {}
            self.cur_camera_id = 0
        if not cam_name in self.camera_dict:
            self.camera_dict[cam_name] = self.cur_camera_id
            self.cur_camera_id += 1
        return self.camera_dict[cam_name]
    def __getitem__(self, idx):
        sample = dict()
        sample["idx"] = idx
        sample["camera_idx"] = self.cameras_idx[idx]

        # read image
        image_path = self.image_filenames[idx]
        sample["image_path"] = image_path
        input_image = cv2.imread(image_path)
        camera_name = image_path.split("/")[-2]
        crop_cy = int(self.resized_image_size[1] / 2)
        K = self.cameras_K[idx]
        origin_image_size = input_image.shape
        resized_image = cv2.resize(input_image, dsize=self.resized_image_size, interpolation=cv2.INTER_LINEAR)
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

        #keep original image for evaluation visualization
        sample["image2"] = (np.asarray(resized_image).copy()/255.0).astype(np.float32)

        resized_image = resized_image[crop_cy:, :, :]  # crop the sky
        sample["image"] = (np.asarray(resized_image)/255.0).astype(np.float32)

        # read label
        label_path = self.label_filenames[idx]
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        resized_label = cv2.resize(label, dsize=self.resized_image_size, interpolation=cv2.INTER_NEAREST)
        mask, label = self.label2mask(resized_label)
        if camera_name == "CAM_BACK":
            h = mask.shape[0]
            mask[int(0.83 * h):, :] = 0
        label = self.remap_semantic(label).astype(np.long)

        mask = mask[crop_cy:, :]  # crop the sky
        label = label[crop_cy:, :]
        sample["static_mask"] = mask
        sample["static_label"] = label

        cv_camera2world = self.ref_camera2world[idx]
        camera2world = self.world2bev @ cv_camera2world
        sample["camera2world"]  = camera2world
        sample["world2camera"] = np.linalg.inv(camera2world)
        resized_K = deepcopy(K)
        width_scale = self.resized_image_size[0]/origin_image_size[1]
        height_scale = self.resized_image_size[1]/origin_image_size[0]
        resized_K[0, :] *= width_scale
        resized_K[1, :] *= height_scale
        resized_K[1, 2] -= crop_cy
        sample["camera_K"] = resized_K
        sample["image_shape"] = np.asarray(sample["image"].shape)[:2]
        sample = self.opencv_camera2pytorch3d_(sample)
        return sample
    
    def run(self):
        return 

if __name__ == "__main__":   
    obj= NerfStudio()
    obj.run()
