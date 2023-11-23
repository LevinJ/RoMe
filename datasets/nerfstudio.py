"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Wed Nov 22 2023
*  File : nerfstudio.py
******************************************* -->

"""
import json
from os.path import join
from pathlib import Path

import numpy as np

from datasets.base import BaseDataset
from datasets.nusc import NuscDataset
from utils.plane_fit import robust_estimate_flatplane


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

        # self.nusc = NuScenes(version="v1.0-{}".format(configs["version"]),
        #                      dataroot=configs["base_dir"],
        #                      verbose=True)
        # self.version = configs["version"]
        # self.replace_name = configs["replace_name"]
        # super().__init__()
        self.resized_image_size = (configs["image_width"], configs["image_height"])
        self.base_dir = configs["base_dir"]
        self.image_dir = configs["image_dir"]
        clip_list = configs["clip_list"]
        camera_names = configs["camera_names"]
        x_offset = -configs["center_point"]["x"] + configs["bev_x_length"]/2
        y_offset = -configs["center_point"]["y"] + configs["bev_y_length"]/2
        self.world2bev = np.asarray([
            [1, 0, 0, x_offset],
            [0, 1, 0, y_offset],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        self.min_distance = configs["min_distance"]

        meta = load_from_json(Path(configs["base_dir"])/ "transforms.json")
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
        self.camera_extrinsics = []
        height = []
        width = []
        self.camera_heights = []
        # fx = float(meta["fl_x"])
        # fy = float(meta["fl_y"])
        # cx = float(meta["cx"])
        # cy = float(meta["cy"])
        for idx,frame in enumerate(meta["frames"]):
            if not frame['scene_name'] in clip_list:
                continue
            if not frame['camera_name'] in camera_names:
                continue
            
            filepath = frame["file_path"]
            label_filepath = frame["label_path"]
            #label_fname = Path(self.image_dir/label_filepath)

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
            
            camera2world = np.array(frame["transform_matrix"]).astype(np.float32)
            camera2camerafront = np.array(frame['camera2frontcamera']).astype(np.float32)
            camerafront2world = camera2world @ np.linalg.inv(camera2camerafront)

            self.camera_extrinsics.append(camera2camerafront)
            self.ref_camera2world_all.append(camerafront2world)
            
            self.cameras_idx_all.append(idx)
            self.cameras_K_all.append(intrinsic.astype(np.float32))

            self.label_filenames_all.append(label_filepath.split(self.image_dir)[-1])
            self.image_filenames_all.append(filepath.split(self.base_dir)[-1])


            if 'camera_height' in frame:
                self.camera_heights.append(frame['camera_height'])

        # # start loading all filename and poses
        # samples = [samp for samp in self.nusc.sample]
        # lidar_height = []
        # lidar2world_all = []
        # for scene_name in tqdm(clip_list, desc="Loading data clips"):
        #     records = [samp for samp in samples if
        #                self.nusc.get("scene", samp["scene_token"])["name"] in scene_name]
        #     # sort by timestamp (only to make chronological viz easier)
        #     records.sort(key=lambda x: (x['timestamp']))

        #     # interpolate images from 2HZ to 12 HZ
        #     for index in range(len(records)):
        #         rec = records[index]
        #         # comput lidar key frame poses
        #         rec_token = rec["data"]["LIDAR_TOP"]
        #         samp = self.nusc.get("sample_data", rec_token)
        #         lidar2chassis = self.compute_extrinsic2chassis(samp)
        #         chassis2world = self.compute_chassis2world(samp)
        #         lidar2world = chassis2world @ lidar2chassis
        #         lidar2world_all.append(lidar2world)
        #         lidar_height.append(lidar2chassis[2, 3])
        #         for camera_idx, cam in enumerate(camera_names):
        #             # compute camera key frame poses
        #             rec_token = rec["data"][cam]
        #             samp = self.nusc.get("sample_data", rec_token)
        #             camera2chassis = self.compute_extrinsic2chassis(samp)
        #             if cam == "CAM_FRONT":
        #                 camera_front2_camera_ref = np.eye(4)
        #                 camera_ref2_camera_front = np.eye(4)
        #             else:
        #                 rec_token_front = rec["data"]["CAM_FRONT"]
        #                 samp_front = self.nusc.get("sample_data", rec_token_front)
        #                 camera_front2_camera_ref = self.compute_extrinsic(samp_front, samp)
        #                 camera_ref2_camera_front = np.linalg.inv(camera_front2_camera_ref)
        #             self.camera_extrinsics.append(camera_ref2_camera_front.astype(np.float32))
        #             flag = True
        #             # compute first key frame and framse between first frame and second frame
        #             while flag or not samp["is_key_frame"]:
        #                 flag = False
        #                 rel_camera_path = samp["filename"]
        #                 if True:
        #                     camera2chassis = self.compute_extrinsic2chassis(samp)
        #                     # 1. label path
        #                     rel_label_path = rel_camera_path.replace("/CAM", "/seg_CAM")
        #                     rel_label_path = rel_label_path.replace(".jpg", ".png")
        #                     if self.replace_name:
        #                         rel_label_path = rel_label_path.replace("+", "_")
        #                     self.label_filenames_all.append(rel_label_path)

        #                     # 2. camera path
        #                     self.image_filenames_all.append(rel_camera_path)

        #                     # 3. camera2world
        #                     chassis2world = self.compute_chassis2world(samp)

        #                     ref_camera2world = chassis2world @ camera2chassis @ camera_front2_camera_ref

        #                     self.ref_camera2world_all.append(ref_camera2world.astype(np.float32))

        #                     # 4.camera intrinsic
        #                     calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
        #                     intrinsic = np.array(calibrated_sensor["camera_intrinsic"])
        #                     self.cameras_K_all.append(intrinsic.astype(np.float32))

        #                     # 5. camera index
        #                     self.cameras_idx_all.append(camera_idx)
        #                 # not key frames
        #                 if samp["next"] != "":
        #                     samp = self.nusc.get('sample_data', samp["next"])
        #                 else:
        #                     break

        # 6. estimate flat plane
        self.file_check()
        # self.label_valid_check()

        self.ref_camera2world_all = np.array(self.ref_camera2world_all)
        print("before plane estimation, z std = ", self.ref_camera2world_all[:, 2, 3].std())
        #front camera height
        camera_height = np.array(self.camera_heights).mean()

        transform_normal2origin = robust_estimate_flatplane(self.ref_camera2world_all[:, :3, 3]).astype(np.float32)
        transform_normal2origin[2, 3] += camera_height
        self.ref_camera2world_all = transform_normal2origin[None] @ np.array(self.ref_camera2world_all)
        print("after plane estimation, z std = ", self.ref_camera2world_all[:, 2, 3].std())

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
    
    def run(self):
        return 

if __name__ == "__main__":   
    obj= NerfStudio()
    obj.run()
