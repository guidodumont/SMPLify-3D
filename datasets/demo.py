""" This file contains the dataset class for the provided demo data (sample of the HumanM3 dataset) """

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, "third_party/cliff"))
sys.path.append(project_dir)

import cv2
import json
import smplx
import torch
import pickle
import argparse
from utils.keypoint_utils import *
from smplx.lbs import vertices2joints
from datasets.dataset import BaseDataset
from pytorch3d.structures import Pointclouds
from third_party.cliff.common.imutils import process_image
from third_party.cliff.common.utils import estimate_focal_length
from utils.visualization_utils import draw_2d_bbox, draw_2d_skeleton
from utils.keypoint_utils import get_coco17_skeleton, get_coco17_skeleton_colors


class DemoDataset(BaseDataset):
    def __init__(self, data_root: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = data_root
        sequences = ["basketball1"]
        
        self.metrics = ["mpjpe", "mpjpe_pa"]
        self.has_labels = True
        
        self.smpl_dir = os.path.join(project_dir, "models/parametric_body_models")
        self.smpl_model_neutral = smplx.create(self.smpl_dir, gender="neutral").to(self.device)
        self.J_regressor = torch.tensor(np.load(os.path.join(self.smpl_dir, "smpl/J_regressor_humanm3.npy"), allow_pickle=True)).float().to(self.device)
        
        self.sequence2cam = {"basketball1": "camera_1"}
        self.sequence2intrinsics = {"basketball1": np.array([[1208.95178, 0.0, 1012.43334],
                                                             [0.0, 1209.42656, 762.148531],
                                                             [0.0, 0.0, 1.0]])}
        self.sequence2extrinsics = {"basketball1": np.array([[-0.4357130825519562, -0.8990759253501892, -0.04262133315205574, 13.762267112731934],
                                                             [0.10995721817016602, -0.006170473527163267, -0.9939171671867371, 1.9691928625106812],
                                                             [0.8933440446853638, -0.4377492368221283, 0.1015484482049942, 9.830440521240234],
                                                             [0.0, 0.0, 0.0, 1.0]])}
        self.img_dirs = {"basketball1": self.data_root}
        self.num_joints = 15
        
        self.data = pickle.load(open(os.path.join(self.data_root, "data.pkl"), "rb"))
        # data_root = os.path.join(project_dir, "../data/humanm3/test")
        # self.data = pickle.load(open(os.path.join(data_root, "test.pkl"), "rb"))
        # test = self.data[805:814]
        # pickle.dump(test, open("/home/guido/Documents/thesis_research/SMPLify-3D/demo_data/data.pkl", "wb"))
        super(DemoDataset, self).__init__(name="demo", sequences=sequences, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
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
        
        img_bgr = cv2.imread(os.path.join(self.data_root, img_name))
        img_rgb = img_bgr[:, :, ::-1]
        img_w = img_bgr.shape[1]
        img_h = img_bgr.shape[0]
        
        intrinsics = self.sequence2intrinsics[sequence]
        extrinsics = self.sequence2extrinsics[sequence]
        
        est_focal_length = estimate_focal_length(img_h=img_h, img_w=img_w)
        est_camera_intrinsics = torch.Tensor([[est_focal_length, 0, img_w / 2], [0, est_focal_length, img_h / 2], [0, 0, 1]]).float()
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(orig_img_rgb=img_rgb, bbox=bbox2d_xyxy)
        
        keypoints_file = os.path.join(self.data_root, "2d_keypoints.json")
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
        
        meta_info = {"sequence": sequence, "img_name": img_name, "tracking_id": tracking_id, "img_id": time, "idx": idx}
        
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
        meta_info["idx"] = np.array([item["idx"] for item in raw_meta_info])
        
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
            
            img = cv2.imread(os.path.join(self.data_root, img_name))
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
    
    data = os.path.join(project_dir, "data/demo")
    parser.add_argument("--data", default=data, type=str, help="Demo data directory.")
    parser.add_argument("--vis_dim", default=3, choices=[2, 3], type=int, help="Visualization dimension.")
    
    args = parser.parse_args()
    
    dataset = DemoDataset(data_root=args.data)
    
    inputs, targets, meta_info = dataset.get_single_sample(idx=0)
    dataset.visualize_sample(inputs=inputs, meta_info=meta_info, targets=targets, idx=0, vis_dimension=args.vis_dim)
