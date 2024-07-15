""" This file contains the source code to create 2D pose annotations for SLOPER4D dataset. """

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir)

import cv2
import torch
import pickle
import argparse
import numpy as np
from mmpose.apis import init_model
from torch.utils.data import Dataset
from datasets.pose_estimation.vitpose import pose_estimation
from ops.point_cloud_ops import transform_points, get_lidar_projection_mask


class SLOPER4D_pose(Dataset):
    def __init__(self, data_root: str, sequence: str):
        self.bbox_format = "xyxy"
        self.sampled_sequences = [sequence]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        intrinsic_params = [599.628, 599.466, 971.613, 540.258]
        self.camera_intrinsics = np.array([[intrinsic_params[0], 0, intrinsic_params[2]], [0, intrinsic_params[1], intrinsic_params[3]], [0, 0, 1]])
        self.pkl_file = os.path.join(data_root, args.sequence, args.sequence + "_labels.pkl")
        with open(self.pkl_file, "rb") as f:
            self.data = pickle.load(f)
        
        self.image_dir = os.path.join(data_root, sequence, "rgb_data", sequence + "_imgs")
        
        self.img_names = self.data["RGB_frames"]["file_basename"]
        self.bbox = self.data["RGB_frames"]["bbox"]  # 2D bbox of the tracked human (N, [x1, y1, x2, y2])
        self.joints2d = self.data["RGB_frames"]["skel_2d"]  # 2D keypoints (N, [17, 3]), every joint is (x, y, probability)
        self.cam_pose = self.data["RGB_frames"]["cam_pose"]  # extrinsic, world to camera (N, [4, 4])
        
        point_clouds = [[]] * len(self.img_names)
        for i, pf in enumerate(self.data["second_person"]["point_frame"]):
            index = self.data["frame_num"].index(pf)
            if index < len(self.img_names):
                point_clouds[index] = self.data["second_person"]["point_clouds"][i]
        self.point_clouds = point_clouds
        
        self.valid_indices = [i for i in range(len(self.img_names)) if len(self.bbox[i]) > 0 and len(self.point_clouds[i]) > 0]
    
    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        img = cv2.imread(os.path.join(self.image_dir, self.img_names[idx]))
        img_w, img_h = img.shape[1], img.shape[0]
        bboxes = [self.bbox[idx]]
        
        cam_pose = self.cam_pose[idx]
        point_cloud = np.array(self.point_clouds[idx])
        point_cloud = transform_points(point_cloud, cam_pose)
        lidar_mask = get_lidar_projection_mask(point_cloud=point_cloud, cam_resolution=(img_w, img_h), intrinsics=self.camera_intrinsics, kernel_size=(50, 50))
        
        inputs = {"img": img, "bboxes": bboxes, "lidar_mask": lidar_mask}
        meta_info = {"idx": idx}
        
        return inputs, meta_info
    
    def __len__(self):
        return len(self.valid_indices)
    
    def save_prediction(self, inputs, meta_info, predictions):
        idx = meta_info["idx"]
        img = inputs["img"]
        img_w, img_h = img.shape[1], img.shape[0]
        
        keypoints = predictions[0].keypoints.reshape(17, 2)
        keypoints = np.clip(keypoints, 0, [img_w - 1, img_h - 1])
        conf = predictions[0].keypoint_scores.reshape(17, 1)
        
        lidar_visibility = inputs["lidar_mask"][keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)].reshape(-1, 1)
        keypoints = np.hstack((keypoints, conf, lidar_visibility)).reshape(17, 4)
        
        self.data["RGB_frames"]["skel_2d"][idx] = keypoints.tolist()
    
    def save_pkl(self):
        with open(self.pkl_file, "wb") as f:
            pickle.dump(self.data, f)
            
        print("Annotations saved to", self.pkl_file)


def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Running on device:", device)
    
    dataset = SLOPER4D_pose(data_root=args.data, sequence=args.sequence)
    
    config = args.config
    checkpoints = args.checkpoints
    model = init_model(config=config, checkpoint=checkpoints, device=str(device))
    
    pose_estimation(model=model, dataset=dataset)
    dataset.save_pkl()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose estimation SLOPER4D.")
    
    config = os.path.join(project_dir, "third_party/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py")
    checkpoints = os.path.join(project_dir, "models/ViTPose/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth")
    
    parser.add_argument("--data", required=True, type=str, help="HumanM3 data directory.")
    parser.add_argument("--config", default=config, type=str, help="Config file of pose estimation model.")
    parser.add_argument("--checkpoints", default=checkpoints, type=str, help="Checkpoint file of pose estimation model.")
    parser.add_argument("--sequence", default="seq009_running_002", type=str, help="Checkpoint file of pose estimation model.")
    
    args = parser.parse_args()
    
    main(args)
    