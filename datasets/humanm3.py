""" This file contains the dataset class for the HumanM3 dataset """

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, "third_party/cliff"))
sys.path.append(project_dir)

import pickle
import glob
import cv2
import json
import smplx
import torch
import argparse
import open3d as o3d
from tqdm import tqdm
from utils.keypoint_utils import *
from smplx.lbs import vertices2joints
from datasets.dataset import BaseDataset
from pytorch3d.structures import Pointclouds
from third_party.cliff.common.imutils import process_image
from ops.point_cloud_ops import points2image, transform_points
from third_party.cliff.common.utils import estimate_focal_length
from utils.visualization_utils import draw_2d_bbox, draw_2d_skeleton
from utils.keypoint_utils import get_coco17_skeleton, get_coco17_skeleton_colors


class HumanM3(BaseDataset):
    def __init__(self, data_root: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = os.path.join(data_root, "test")
        sequences = ["basketball1", "basketball2", "intersection", "plaza"]
        
        self.metrics = ["mpjpe", "mpjpe_pa"]
        self.has_labels = True
        
        self.smpl_dir = os.path.join(project_dir, "models/parametric_body_models")
        self.smpl_model_neutral = smplx.create(self.smpl_dir, gender="neutral").to(self.device)
        self.J_regressor = torch.tensor(np.load(os.path.join(self.smpl_dir, "smpl/J_regressor_humanm3.npy"), allow_pickle=True)).float().to(self.device)
        
        self.sequence2cam = {"basketball1": "camera_1", "basketball2": "camera_0", "intersection": "camera_3", "plaza": "camera_2"}
        self.sequence2intrinsics = {}
        self.sequence2extrinsics = {}
        self.img_dirs = {}
        for sequence in sequences:
            calib_file = os.path.join(self.data_root, sequence, "camera_calibration", f"{self.sequence2cam[sequence]}.json")
            calib_data = json.load(open(calib_file, "r"))
            self.sequence2intrinsics[sequence] = np.array(calib_data["intrinsic"])[:, :3]
            self.sequence2extrinsics[sequence] = np.array(calib_data["extrinsic"])
            self.img_dirs[sequence] = os.path.join(self.data_root, sequence, "images", self.sequence2cam[sequence])
        
        self.num_joints = 15
        
        if not os.path.exists(os.path.join(self.data_root, "test.pkl")):
            self.process_data()
            sys.exit()
        else:
            self.data = pickle.load(open(os.path.join(self.data_root, "test.pkl"), "rb"))
            
        super(HumanM3, self).__init__(name="HumanM3", sequences=sequences, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    def process_data(self):
        print("Processing data...")
        
        pose_files = {}
        for sequence in self.sequences:
            pose_files[sequence] = sorted(os.listdir(os.path.join(self.data_root, sequence, 'pose_calib')))
        
        data = []
        json_files = glob.glob(os.path.join(self.data_root, '*', 'smpl_estimated', '*.json'))
        for file in tqdm(json_files):
            smpl_data = json.load(open(file, 'r'))
            
            names = file.split('/')
            sequence, time = names[-3], names[-1].replace('.json', '')
            
            intrinsics = self.sequence2intrinsics[sequence]
            extrinsics = self.sequence2extrinsics[sequence]
            
            pcd_file = file.replace('json', 'pcd').replace('smpl_estimated', 'pointcloud').replace(time, str(int(time)).zfill(6))
            point_cloud = np.array(o3d.io.read_point_cloud(pcd_file).points)
            
            file_idx = np.where(np.array(pose_files[sequence]) == time + '.json')[0][0]
            images = sorted(os.listdir(os.path.join(self.data_root, sequence, 'images', self.sequence2cam[sequence])))
            img_file = images[file_idx]
            
            pose_file = os.path.join(self.data_root, sequence, 'pose_calib', time + '.json')
            pose_data = json.load(open(pose_file, 'r'))
            for tracking_id in smpl_data.keys():
                pose3d = np.array(pose_data[tracking_id]).reshape(-1, 3)
                min_x, max_x = np.min(pose3d[:, 0]) - 0.2, np.max(pose3d[:, 0]) + 0.2
                min_y, max_y = np.min(pose3d[:, 1]) - 0.2, np.max(pose3d[:, 1]) + 0.2
                min_z, max_z = np.min(pose3d[:, 2]), np.max(pose3d[:, 2]) + 0.2
                
                valid_indices = np.where((point_cloud[:, 0] >= min_x) & (point_cloud[:, 0] <= max_x) & (point_cloud[:, 1] >= min_y) & (point_cloud[:, 1] <= max_y) & (point_cloud[:, 2] >= min_z) & (point_cloud[:, 2] <= max_z))[0]
                human_points = point_cloud[valid_indices]
                human_points = transform_points(human_points, extrinsics)
                
                pose3d = transform_points(point_cloud=pose3d, transformation=extrinsics)
                pose_img = points2image(points=pose3d, intrinsics=intrinsics)
                
                factor_x, factor_y = 0.35, 0.15
                min_x, max_x = np.min(pose_img[:, 0]) - 0.2, np.max(pose_img[:, 0]) + 0.2
                min_y, max_y = np.min(pose_img[:, 1]) - 0.2, np.max(pose_img[:, 1]) + 0.2
                w, h = max_x - min_x, max_y - min_y
                bbox2d_xyxy = [int(min_x - factor_x * w), int(min_y - factor_y * h), int(max_x + factor_x * w), int(max_y + factor_y * h)]
                bbox2d_xywh = [bbox2d_xyxy[0], bbox2d_xyxy[1], bbox2d_xyxy[2] - bbox2d_xyxy[0], bbox2d_xyxy[3] - bbox2d_xyxy[1]]
                
                data.append({'sequence': sequence, 'time': time, 'tracking_id': tracking_id, 'pose3d': pose3d, 'bbox2d_xywh': bbox2d_xywh, 'human_points': human_points, 'img_file': img_file})
        
        pkl_file = os.path.join(self.data_root, 'test.pkl')
        pickle.dump(data, open(pkl_file, 'wb'))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        
        time = data["time"]
        pose3d = data["pose3d"]
        img_name = data["img_file"]
        sequence = data["sequence"]
        tracking_id = data["tracking_id"]
        bbox2d_xywh = data["bbox2d_xywh"]
        human_points = data["human_points"]
        bbox2d_xyxy = [bbox2d_xywh[0], bbox2d_xywh[1], bbox2d_xywh[0] + bbox2d_xywh[2], bbox2d_xywh[1] + bbox2d_xywh[3]]
        
        img_bgr = cv2.imread(os.path.join(self.data_root, sequence, "images", self.sequence2cam[sequence], img_name))
        img_rgb = img_bgr[:, :, ::-1]
        img_w = img_bgr.shape[1]
        img_h = img_bgr.shape[0]
        
        intrinsics = self.sequence2intrinsics[sequence]
        extrinsics = self.sequence2extrinsics[sequence]
        
        est_focal_length = estimate_focal_length(img_h=img_h, img_w=img_w)
        est_camera_intrinsics = torch.Tensor([[est_focal_length, 0, img_w / 2], [0, est_focal_length, img_h / 2], [0, 0, 1]]).float()
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(orig_img_rgb=img_rgb, bbox=bbox2d_xyxy)
        
        keypoints_file = os.path.join(self.data_root, sequence, "pose2d", time + ".json")
        with open(keypoints_file, "r") as f:
            keypoints = json.load(f)
        
        keypoints = np.array(keypoints[tracking_id])
        joints = keypoints[:, :2]
        joints_conf = keypoints[:, 2]
        img_visibility = joints_conf
        lidar_visibility = keypoints[:, 3]
        joints2d = np.hstack((joints, joints_conf.reshape(-1, 1), img_visibility.reshape(-1, 1), lidar_visibility.reshape(-1, 1)))
        
        inputs = {"norm_img": norm_img, "scale": scale, "center": center, "bbox": bbox2d_xywh, "joints2d": joints2d, "point_cloud": human_points, "img_w": img_w, "img_h": img_h, "mask_area": 1, "extrinsics": extrinsics,
                  "camera_intrinsics": intrinsics, "est_focal_length": est_focal_length, "est_camera_intrinsics": est_camera_intrinsics}
        
        meta_info = {"sequence": sequence, "img_name": img_name, "tracking_id": tracking_id, "img_id": time}
        
        targets = {"joints3d": pose3d}
        
        return inputs, targets, meta_info
    
    def collate_fn(self, data):
        raw_inputs, raw_targets, raw_meta_info = zip(*data)
        
        inputs = {}
        inputs["norm_img"] = torch.tensor(np.array([item["norm_img"] for item in raw_inputs])).to(self.device).float()
        inputs["scale"] = torch.tensor(np.array([item["scale"] for item in raw_inputs])).to(self.device).float()
        inputs["center"] = torch.stack([item["center"] for item in raw_inputs]).to(self.device).float()
        inputs["bbox"] = torch.tensor(np.array([item["bbox"] for item in raw_inputs])).to(self.device).float()
        inputs["img_w"] = torch.tensor(np.array([item["img_w"] for item in raw_inputs])).to(self.device).float()
        inputs["img_h"] = torch.tensor(np.array([item["img_h"] for item in raw_inputs])).to(self.device).float()
        inputs["mask_area"] = torch.tensor(np.array([item["mask_area"] for item in raw_inputs])).to(self.device).float()
        inputs["est_focal_length"] = torch.tensor(np.array([item["est_focal_length"] for item in raw_inputs])).to(self.device).float()
        inputs["est_camera_intrinsics"] = torch.stack([item["est_camera_intrinsics"] for item in raw_inputs]).to(self.device).float()
        inputs["joints2d"] = torch.tensor(np.array([item["joints2d"] for item in raw_inputs])).to(self.device).float()
        inputs["extrinsics"] = torch.tensor(np.array([item["extrinsics"] for item in raw_inputs])).to(self.device).float()
        inputs["camera_intrinsics"] = torch.tensor(np.array([item["camera_intrinsics"] for item in raw_inputs])).to(self.device).float()
        
        meta_info = {}
        meta_info["sequence"] = np.array([item["sequence"] for item in raw_meta_info])
        meta_info["img_name"] = np.array([item["img_name"] for item in raw_meta_info])
        meta_info["tracking_id"] = np.array([item["tracking_id"] for item in raw_meta_info])
        meta_info["img_id"] = np.array([item["img_id"] for item in raw_meta_info])
        
        targets = {}
        targets["joints3d"] = torch.tensor(np.array([item["joints3d"] for item in raw_targets])).to(self.device).float()
        
        point_cloud_list = [torch.Tensor(item["point_cloud"]) for item in raw_inputs]
        inputs["point_cloud"] = Pointclouds(point_cloud_list).to(self.device)
        
        del raw_inputs, raw_targets, raw_meta_info
        
        return inputs, targets, meta_info
    
    def evaluation_prep(self, outputs: torch.Tensor, targets: torch.Tensor, align_root: bool):
        targets["joints3d"] = targets["joints3d"] * 1000
        target_root_joint = targets["joints3d"][:, 0]
        
        outputs["vertices"] = outputs["vertices"] * 1000
        outputs["joints3d"] = vertices2joints(J_regressor=self.J_regressor, vertices=outputs["vertices"]).to(self.device).float()
        output_root_joint = outputs["joints3d"][:, 0]
        
        if align_root:
            outputs["vertices"] = outputs["vertices"] - output_root_joint.unsqueeze(1)
            outputs["joints3d"] = outputs["joints3d"] - output_root_joint.unsqueeze(1)
            targets["joints3d"] = targets["joints3d"] - target_root_joint.unsqueeze(1)
        
        return outputs, targets
    
    def visualize_sample(self, inputs, meta_info, targets, idx, vis_dimension=2):
        sequence = meta_info["sequence"][idx]
        img_name = meta_info["img_name"][idx]
        tracking_id = meta_info["tracking_id"][idx]
        
        if vis_dimension == 2:
            print("Sequence:", sequence, "Image:", img_name)
            
            img = cv2.imread(os.path.join(self.data_root, sequence, "images", self.sequence2cam[sequence], img_name))
            bbox = inputs["bbox"][idx].cpu().numpy()
            bbox_xywh = bbox.astype(int)
            bbox_xyxy = np.array([bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]])
            img = draw_2d_bbox(img=img, bboxes_xyxy=[bbox_xyxy], detection_id=[tracking_id])
            
            joints2d = inputs["joints2d"][idx].cpu().numpy()[:, :2]
            img = draw_2d_skeleton(img=img, joints2d=joints2d, skeleton=get_coco17_skeleton(), colors=get_coco17_skeleton_colors())
            
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow("Image", img)
            cv2.waitKey(0)
        
        else:
            super().visualize_sample(inputs=inputs, meta_info=meta_info, targets=targets, idx=idx, vis_dimension=vis_dimension)
            
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HumanM3 dataset.")
    parser.add_argument("--data", required=True, type=str, help="HumanM3 test data directory.")
    parser.add_argument("--vis_dim", default=3, choices=[2, 3], type=int, help="Visualization dimension.")
    
    args = parser.parse_args()
    
    dataset = HumanM3(data_root=args.data)

    inputs, targets, meta_info = dataset.get_single_sample(idx=0)
    dataset.visualize_sample(inputs=inputs, meta_info=meta_info, targets=targets, idx=0, vis_dimension=args.vis_dim)
