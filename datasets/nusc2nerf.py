"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Wed Nov 22 2023
*  File : nusc2nerf.py
******************************************* -->

"""
import json
import os
from copy import deepcopy
from multiprocessing.pool import ThreadPool as Pool
from os.path import join

import cv2
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from pyquaternion import Quaternion
from skspatial.objects import Plane, Points
from tqdm import tqdm

from datasets.base import BaseDataset
from utils.plane_fit import robust_estimate_flatplane
from utils.pose_util import get_Tgl2cv


class App(object):
    def __init__(self):
        return
    def compute_extrinsic(self, samp_a, samp_b):
        """transform from sensor_a to sensor_b"""
        sensor_a2chassis = self.compute_extrinsic2chassis(samp_a)
        sensor_b2chassis = self.compute_extrinsic2chassis(samp_b)
        sensor_a2sensor_b = np.linalg.inv(sensor_b2chassis) @ sensor_a2chassis
        return sensor_a2sensor_b

    def compute_extrinsic2chassis(self, samp):
        calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
        rot = np.array((Quaternion(calibrated_sensor["rotation"]).rotation_matrix))
        tran = np.expand_dims(np.array(calibrated_sensor["translation"]), axis=0)
        sensor2chassis = np.hstack((rot, tran.T))
        sensor2chassis = np.vstack((sensor2chassis, np.array([[0, 0, 0, 1]])))  # [4, 4] camera 3D
        return sensor2chassis
    def compute_chassis2world(self, samp):
        """transform sensor in world coordinate"""
        # comput current frame Homogeneous transformation matrix : from chassis 2 global
        pose_chassis2global = self.nusc.get("ego_pose", samp['ego_pose_token'])
        chassis2global = transform_matrix(pose_chassis2global['translation'],
                                          Quaternion(pose_chassis2global['rotation']),
                                          inverse=False)
        return chassis2global
    def reset_temp_var(self):
        self.camera_height = []
        self.camera2frontcamera = []

        self.image_filenames_all = []
        self.label_filenames_all=[]
        self.cameras_K_all=[]
        self.cameras_idx_all=[]
        self.ref_camera2world_all=[]

        self.cameras_name_all=[]
        self.scene_name_all=[]

        return
    def extract_dataset(self):
        version = "mini" 
        self.base_dir = "/media/levin/DATA/zf/workspace/data/nuscene/v1.0-mini/"
        self.image_dir = "/media/levin/DATA/zf/workspace/data/nuscene/nuScenes_clip/"
        self.nusc = NuScenes(version="v1.0-{}".format(version),
                             dataroot=self.base_dir,
                             verbose=True)
        
        clip_list = ["scene-0655"]
        camera_names = ["CAM_FRONT"]
        self.replace_name = False

        

        # start loading all filename and poses
        samples = [samp for samp in self.nusc.sample]
        # lidar_height = []
        # lidar2world_all = []
        for scene_name in tqdm(clip_list, desc="Loading data clips"):
            self.reset_temp_var()
            records = [samp for samp in samples if
                       self.nusc.get("scene", samp["scene_token"])["name"] in scene_name]
            # sort by timestamp (only to make chronological viz easier)
            records.sort(key=lambda x: (x['timestamp']))

            # interpolate images from 2HZ to 12 HZ
            for _index, rec in enumerate(records):
                for camera_idx, cam in enumerate(camera_names):
                    # compute camera key frame poses
                    rec_token = rec["data"][cam]
                    samp = self.nusc.get("sample_data", rec_token)
                    # camera2chassis = self.compute_extrinsic2chassis(samp)
                    flag = True
                    # compute first key frame and framse between first frame and second frame
                    while flag or not samp["is_key_frame"]:
                        flag = False
                        rel_camera_path = samp["filename"]
                        if True:
                            camera2chassis = self.compute_extrinsic2chassis(samp)
                            if cam == "CAM_FRONT":
                                camera_front2_camera_ref = np.eye(4)
                                camera_ref2_camera_front = np.eye(4)
                                self.camera_height.append(camera2chassis[2, 3])
                            else:
                                rec_token_front = rec["data"]["CAM_FRONT"]
                                samp_front = self.nusc.get("sample_data", rec_token_front)
                                camera_front2_camera_ref = self.compute_extrinsic(samp_front, samp)
                                camera_ref2_camera_front = np.linalg.inv(camera_front2_camera_ref)
                            self.camera2frontcamera.append(camera_ref2_camera_front.astype(np.float32))
                            # 1. label path
                            rel_label_path = rel_camera_path.replace("/CAM", "/seg_CAM")
                            rel_label_path = rel_label_path.replace(".jpg", ".png")
                            if self.replace_name:
                                rel_label_path = rel_label_path.replace("+", "_")
                            self.label_filenames_all.append(join(self.image_dir, rel_label_path) )

                            # 2. camera path
                            self.image_filenames_all.append(join(self.base_dir, rel_camera_path))

                            # 3. camera2world
                            chassis2world = self.compute_chassis2world(samp)

                            ref_camera2world = chassis2world @ camera2chassis @ get_Tgl2cv()

                            self.ref_camera2world_all.append(ref_camera2world.astype(np.float32))

                            # 4.camera intrinsic
                            calibrated_sensor = self.nusc.get("calibrated_sensor", samp["calibrated_sensor_token"])
                            intrinsic = np.array(calibrated_sensor["camera_intrinsic"])
                            self.cameras_K_all.append(intrinsic.astype(np.float32))

                            # 5. camera index
                            self.cameras_idx_all.append(camera_idx)
                            self.cameras_name_all.append(cam)
                            self.scene_name_all.append(scene_name)
                        # not key frames
                        if samp["next"] != "":
                            samp = self.nusc.get('sample_data', samp["next"])
                        else:
                            break
        json_path = f"{self.base_dir}transforms.json"
        self.save_nerf_studio(json_path)
        return
    def save_nerf_studio(self, json_path):
        meta_data = {"k1": 0,
                    "k2": 0,
                    "p1": 0,
                    "p2": 0,
                    'h': 0,
                    'w': 0,
                    "aabb_scale": 16,
                    "frames":[],}
        
        sample_img = self.label_filenames_all[0]
        sample_img = cv2.imread(sample_img)
        meta_data['h'], meta_data['w']= sample_img.shape[:2] 
        os.makedirs(os.path.dirname(json_path), exist_ok=True)


        for idx, img in enumerate(self.image_filenames_all):
            frame = {}
            frame['file_path'] = img
            frame['label_path'] = self.label_filenames_all[idx]
            frame['transform_matrix'] = self.ref_camera2world_all[idx].tolist()
            frame['camera_id'] = self.cameras_idx_all[idx]
            frame['camera_name'] = self.cameras_name_all[idx]
            frame['scene_name'] = self.scene_name_all[idx]
            frame['camera2frontcamera'] =  self.camera2frontcamera [idx].tolist()
            if frame['camera_name'] == "CAM_FRONT":
                frame['camera_height'] = self.camera_height[idx]


            K = self.cameras_K_all[idx]
            frame['fl_x'], frame['cx'] = str(K[0, 0]), str(K[0, 2])
            frame['fl_y'], frame['cy'] = str(K[1, 1]), str(K[1, 2])
            meta_data['frames'].append(frame)


        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(meta_data, f, ensure_ascii=False, indent=4)
            print(f"saved file {json_path}")
        return
    def run(self):
        self.extract_dataset()
        return 

if __name__ == "__main__":   
    obj= App()
    obj.run()
