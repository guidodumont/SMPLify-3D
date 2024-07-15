""" This file contains the source code to create 2D pose annotations for HumanM3 dataset. """

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_dir)

import cv2
import json
import torch
import argparse
import numpy as np
from mmpose.apis import init_model
from torch.utils.data import Dataset
from ops.point_cloud_ops import points2image
from datasets.pose_estimation.vitpose import pose_estimation



class HumanM3_pose(Dataset):
    def __init__(self, data_root: str):
        self.bbox_format = "xyxy"
        self.data_root = os.path.join(data_root, "test")
        self.sequences = ["basketball1", "basketball2", "intersection", "plaza"]

        self.sequence2cam = {"basketball1": "camera_1", "basketball2": "camera_0", "intersection": "camera_3", "plaza": "camera_2"}
        self.sequence2intrinsics = {}
        self.sequence2extrinsics = {}
        
        self.data = {}
        self.load_data()
    
    def load_data(self):
        self.data["sequence_nr"] = []
        self.data["img_file"] = []
        self.data["camera"] = []
        self.data["frame_id"] = []
        
        for i, sequence in enumerate(self.sequences):
            camera = self.sequence2cam[sequence]
            sequence_path = os.path.join(self.data_root, sequence)
            
            ann_dir = os.path.join(sequence_path, "pose_calib")
            img_dir = os.path.join(sequence_path, "images", camera)
            img_names = sorted(os.listdir(img_dir))
            
            for j, ann in enumerate(sorted(os.listdir(ann_dir))):
                frame_id = int(ann.split(".")[0])
                self.data["sequence_nr"].append(i)
                self.data["img_file"].append(img_names[j])
                self.data["camera"].append(camera)
                self.data["frame_id"].append(frame_id)
            
            calib_dir = os.path.join(sequence_path, "camera_calibration")
            calib_file = os.path.join(calib_dir, f"{camera}.json")
            with open(calib_file, "r") as f:
                calib_data = json.load(f)
            
            intrinsics = np.array(calib_data["intrinsic"])[:, :3]
            extrinsics = np.array(calib_data["extrinsic"])
            
            self.sequence2intrinsics[sequence] = intrinsics
            self.sequence2extrinsics[sequence] = extrinsics
    
    def __len__(self):
        return len(self.data["sequence_nr"])
    
    def __getitem__(self, idx):
        camera = self.data["camera"][idx]
        sequence = self.sequences[self.data["sequence_nr"][idx]]
        frame_id = self.data["frame_id"][idx]
        
        img_file = self.data["img_file"][idx]
        img = cv2.imread(os.path.join(self.data_root, sequence, "images", self.data["camera"][idx], img_file))
        img_w, img_h = img.shape[1], img.shape[0]
        
        sequence_path = os.path.join(self.data_root, sequence)
        ann_file = os.path.join(sequence_path, "pose_calib", str(frame_id) + ".json")
        
        with open(ann_file, "r") as f:
            ann_data = json.load(f)
        
        pose3d = np.array(list(ann_data.values())).reshape(-1, 15, 3)
        track_ids = np.array(list(ann_data.keys())).ravel()
        
        bboxes = []
        for pose, track_id in zip(pose3d, track_ids):
            intrinsics = self.sequence2intrinsics[sequence]
            extrinsics = self.sequence2extrinsics[sequence]
            pose_img = points2image(points=pose, intrinsics=intrinsics, extrinsics=extrinsics)
            
            factor_x, factor_y = 0.35, 0.15
            min_x, max_x = np.min(pose_img[:, 0]) - 0.2, np.max(pose_img[:, 0]) + 0.2
            min_y, max_y = np.min(pose_img[:, 1]) - 0.2, np.max(pose_img[:, 1]) + 0.2
            w, h = max_x - min_x, max_y - min_y
            bboxes.append([int(min_x - factor_x * w), int(min_y - factor_y * h), int(max_x + factor_x * w), int(max_y + factor_y * h)])
            
        inputs = {"img": img, "bboxes": bboxes, "lidar_mask": np.ones((img_h, img_w), dtype=np.uint8)}
        meta_info = {"idx": idx, "camera": camera, "sequence": sequence, "frame_id": frame_id, "img_w": img_w, "img_h": img_h, "track_ids": track_ids}
        
        return inputs, meta_info
    
    def save_prediction(self, inputs, meta_info, predictions):
        sequence = meta_info["sequence"]
        frame_id = meta_info["frame_id"]
        camera = meta_info["camera"]
        img_w, img_h = meta_info["img_w"], meta_info["img_h"]
        
        data = {}
        data["camera"] = camera
        for i, track_id in enumerate(meta_info["track_ids"]):
            keypoints = predictions[i].keypoints.reshape(17, 2)
            keypoints = np.clip(keypoints, 0, [img_w - 1, img_h - 1])
            conf = predictions[i].keypoint_scores.reshape(17, 1)
            
            lidar_visibility = inputs["lidar_mask"][keypoints[:, 1].astype(int), keypoints[:, 0].astype(int)].reshape(-1, 1)
            keypoints = np.hstack((keypoints, conf, lidar_visibility)).reshape(17, 4)
        
            data[track_id] = keypoints.tolist()
        
        save_dir = os.path.join(self.data_root, sequence, "pose2d")
        os.makedirs(save_dir, exist_ok=True)
        save_file = os.path.join(save_dir, f"{frame_id}.json")
        
        with open(save_file, "w") as f:
            json.dump(data, f, indent=4)
        

def main(args):
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Running on device:", device)
    
    dataset = HumanM3_pose(data_root=args.data)
    
    config = args.config
    checkpoints = args.checkpoints
    model = init_model(config=config, checkpoint=checkpoints, device=str(device))
    
    pose_estimation(model=model, dataset=dataset)
    print("Annotations saved to all sequences in:", dataset.data_root)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pose estimation HumanM3.")
    
    config = os.path.join(project_dir, "third_party/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_ViTPose-large_8xb64-210e_coco-256x192.py")
    checkpoints = os.path.join(project_dir, "models/ViTPose/td-hm_ViTPose-large_8xb64-210e_coco-256x192-53609f55_20230314.pth")
    
    parser.add_argument("--data", required=True, type=str, help="HumanM3 data directory.")
    parser.add_argument("--config", default=config, type=str, help="Config file of pose estimation model.")
    parser.add_argument("--checkpoints", default=checkpoints, type=str, help="Checkpoint file of pose estimation model.")
    
    args = parser.parse_args()
    
    main(args)
    