""" This file contains the dataset class for the SLOPER4D dataset """

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, "third_party/cliff"))
sys.path.append(project_dir)

import cv2
import torch
import smplx
import pickle
import argparse
from utils.keypoint_utils import *
from smplx.lbs import vertices2joints
from datasets.dataset import BaseDataset
from pytorch3d.structures import Pointclouds
from ops.point_cloud_ops import transform_points
from scipy.spatial.transform import Rotation as R
from third_party.cliff.common.imutils import process_image
from third_party.cliff.common.utils import estimate_focal_length


class SLOPER4D(BaseDataset):
    def __init__(self, data_root: str, sequence: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_root = data_root
        
        self.metrics = ["pve", "mpere", "mpjpe", "mpjpe_pa"]
        self.num_joints = 24
        self.has_labels = True
        
        self.pkl_file = os.path.join(data_root, sequence, sequence + "_labels.pkl")
        with open(self.pkl_file, "rb") as f:
            self.data = pickle.load(f)
        
        self.img_dirs = {sequence: os.path.join(data_root, sequence, "rgb_data", sequence + "_imgs")}
        self.smpl_dir = os.path.join(project_dir, "models/parametric_body_models")
        self.smpl_model_neutral = smplx.create(self.smpl_dir, gender="neutral").to(self.device)
        self.smpl_model_male = smplx.create(self.smpl_dir, gender="male").to(self.device)
        self.smpl_model_female = smplx.create(self.smpl_dir, gender="female").to(self.device)
        
        intrinsic_params = [599.628, 599.466, 971.613, 540.258]
        self.camera_intrinsics = np.array([[intrinsic_params[0], 0, intrinsic_params[2]], [0, intrinsic_params[1], intrinsic_params[3]], [0, 0, 1]])
        self.length = self.data["total_frames"] if "total_frames" in self.data else len(self.data["frame_num"])
        
        self.world2lidar, self.lidar_tstamps = self.get_lidar_data()
        self.load_3d_data(self.data)
        self.load_rgb_data(self.data)
        
        super(SLOPER4D, self).__init__(name="SLOPER4D", sequences=[sequence], device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        self.check_length()
    
    def get_lidar_data(self):
        lidar_traj = self.data["first_person"]["lidar_traj"].copy()
        lidar_tstamps = lidar_traj[:self.length, -1]
        world2lidar = np.array([np.eye(4)] * self.length)
        world2lidar[:, :3, :3] = R.from_quat(lidar_traj[:self.length, 4: 8]).inv().as_matrix()
        world2lidar[:, :3, 3:] = -world2lidar[:, :3, :3] @ lidar_traj[:self.length, 1:4].reshape(-1, 3, 1)
        
        return world2lidar, lidar_tstamps
    
    def load_rgb_data(self, data):
        self.cam = data["RGB_info"]
        if "RGB_frames" not in data:
            data["RGB_frames"] = {}
            world2lidar, lidar_tstamps = self.get_lidar_data()
            data["RGB_frames"]["file_basename"] = [""] * self.length
            data["RGB_frames"]["lidar_tstamps"] = lidar_tstamps[:self.length]
            data["RGB_frames"]["bbox"] = [[]] * self.length
            data["RGB_frames"]["skel_2d"] = [[]] * self.length
            data["RGB_frames"]["cam_pose"] = self.cam["lidar2cam"] @ world2lidar
            self.save_pkl(overwrite=True)
        
        self.img_names = data["RGB_frames"]["file_basename"]  # synchronized img file names
        self.lidar_tstamps = data["RGB_frames"]["lidar_tstamps"]  # synchronized ldiar timestamps
        self.bbox = data["RGB_frames"]["bbox"]  # 2D bbox of the tracked human (N, [x1, y1, x2, y2])
        self.joints2d = data["RGB_frames"]["skel_2d"]  # 2D keypoints (N, [17, 3]), every joint is (x, y, probability)
        self.cam_pose = data["RGB_frames"]["cam_pose"]  # extrinsic, world to camera (N, [4, 4])
    
    def load_3d_data(self, data, person="second_person", points_num=1024):
        assert self.length <= len(data["frame_num"]), f"RGB length must be less than point cloud length"
        point_clouds = [[]] * self.length
        if "point_clouds" in data[person]:
            for i, pf in enumerate(data[person]["point_frame"]):
                index = data["frame_num"].index(pf)
                if index < self.length:
                    point_clouds[index] = data[person]["point_clouds"][i]
        
        sp = data["second_person"]
        self.smpl_pose = sp["opt_pose"][:self.length].astype(np.float32)  # n x 72 array of scalars
        self.global_trans = sp["opt_trans"][:self.length].astype(np.float32)  # n x 3 array of scalars
        self.betas = sp["beta"]  # n x 10 array of scalars
        self.smpl_gender = sp["gender"]  # male/female/neutral
        self.point_clouds = point_clouds  # list of n arrays, each of shape (x_i, 3)

    def updata_pkl(self, img_name,
                   bbox=None,
                   cam_pose=None,
                   keypoints=None):
        if img_name in self.img_names:
            index = self.img_names.index(img_name)
            if bbox is not None:
                self.data["RGB_frames"]["bbox"][index] = bbox
            if keypoints is not None:
                self.data["RGB_frames"]["skel_2d"][index] = keypoints
            if cam_pose is not None:
                self.data["RGB_frames"]["cam_pose"][index] = cam_pose
        else:
            print(f"{img_name} is not in the synchronized labels file")
    
    def get_rgb_frames(self, ):
        return self.data["RGB_frames"]
    
    def save_pkl(self, overwrite=False):
        
        save_path = self.pkl_file if overwrite else self.pkl_file[:-4] + "_updated.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"{save_path} saved")
    
    def check_length(self):
        # Check if all the lists inside rgb_frames have the same length
        assert all(len(lst) == self.length for lst in [self.bbox, self.joints2d,
                                                       self.lidar_tstamps,
                                                       self.smpl_pose, self.global_trans,
                                                       self.point_clouds])
        
        self.valid_indices = [i for i in range(self.length) if len(self.bbox[i]) > 0 and len(self.joints2d[i]) > 0 and len(self.point_clouds[i]) > 0 and np.sum(np.array(self.joints2d[i])[:, -1]) >= 10]
    
    def __getitem__(self, idx):
        idx = self.valid_indices[idx]
        
        img_bgr = cv2.imread(os.path.join(self.img_dirs[self.sequences[0]], self.img_names[idx]))
        img_rgb = img_bgr[:, :, ::-1]
        img_w = img_bgr.shape[1]
        img_h = img_bgr.shape[0]
        
        bbox2d_xyxy = self.bbox[idx]
        bbox2d_xywh = [bbox2d_xyxy[0], bbox2d_xyxy[1], bbox2d_xyxy[2] - bbox2d_xyxy[0], bbox2d_xyxy[3] - bbox2d_xyxy[1]]
        
        est_focal_length = estimate_focal_length(img_h=img_h, img_w=img_w)
        est_camera_intrinsics = torch.Tensor([[est_focal_length, 0, img_w / 2], [0, est_focal_length, img_h / 2], [0, 0, 1]]).float()
        norm_img, center, scale, crop_ul, crop_br, _ = process_image(orig_img_rgb=img_rgb, bbox=bbox2d_xyxy)
        
        cam_pose = self.cam_pose[idx]
        point_cloud = np.array(self.point_clouds[idx])
        point_cloud = transform_points(point_cloud, cam_pose)

        keypoints = np.array(self.joints2d[idx])
        joints = keypoints[:, :2]
        joints_conf = keypoints[:, 2]
        img_visibility = joints_conf
        lidar_visibility = keypoints[:, 3]
        
        joints2d = np.hstack((joints, joints_conf.reshape(-1, 1), img_visibility.reshape(-1, 1), lidar_visibility.reshape(-1, 1)))
        
        inputs = {"norm_img": norm_img, "scale": scale, "center": center, "bbox": bbox2d_xywh, "joints2d": joints2d, "point_cloud": point_cloud, "img_w": img_w, "img_h": img_h, "mask_area": 1, "extrinsics": cam_pose,
                  "camera_intrinsics": self.camera_intrinsics, "est_focal_length": est_focal_length, "est_camera_intrinsics": est_camera_intrinsics}
        
        img_name = self.img_names[idx]
        img_id = int(img_name[:-4].replace(".", ""))
        meta_info = {"sequence": self.sequences[0], "img_name": img_name, "img_id": img_id, "tracking_id": -1}
        
        gt_poses = self.smpl_pose[idx]
        gt_global_t = self.global_trans[idx]
        gt_betas = self.betas
        gender = 0 if self.smpl_gender == "male" else 1
        targets = {"poses": gt_poses, "betas": gt_betas, "global_t": gt_global_t, "gender": gender}
        
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
        meta_info["img_id"] = np.array([item["img_id"] for item in raw_meta_info])
        meta_info["tracking_id"] = np.array([item["tracking_id"] for item in raw_meta_info])
        
        targets = {}
        targets["poses"] = torch.tensor(np.array([item["poses"] for item in raw_targets])).to(self.device).float()
        targets["betas"] = torch.tensor(np.array([item["betas"] for item in raw_targets])).to(self.device).float()
        targets["global_t"] = torch.tensor(np.array([item["global_t"] for item in raw_targets])).to(self.device).float()
        targets["gender"] = torch.tensor(np.array([item["gender"] for item in raw_targets])).to(self.device).float()
        
        male_idx = np.where(targets["gender"].cpu().numpy() == 1)[0]
        female_idx = np.where(targets["gender"].cpu().numpy() == 0)[0]
        
        targets["vertices"] = torch.zeros((inputs["norm_img"].shape[0], 6890, 3)).to(self.device).float()
        targets["joints3d"] = torch.zeros((inputs["norm_img"].shape[0], 24, 3)).to(self.device).float()
        if len(male_idx) > 0:
            smpl_output = self.smpl_model_neutral(betas=targets["betas"][male_idx], body_pose=targets["poses"][male_idx, 3:], global_orient=targets["poses"][male_idx, :3], transl=targets["global_t"][male_idx], pose2rot=True)
            targets["vertices"][male_idx] = smpl_output.vertices.to(self.device).float()
            targets["joints3d"][male_idx] = vertices2joints(J_regressor=self.smpl_model_neutral.J_regressor, vertices=smpl_output.vertices)
        
        if len(female_idx) > 0:
            smpl_output = self.smpl_model_neutral(betas=targets["betas"][female_idx], body_pose=targets["poses"][female_idx, 3:], global_orient=targets["poses"][female_idx, :3], transl=targets["global_t"][female_idx], pose2rot=True)
            targets["vertices"][female_idx] = smpl_output.vertices.to(self.device).float()
            targets["joints3d"][female_idx] = vertices2joints(J_regressor=self.smpl_model_neutral.J_regressor, vertices=smpl_output.vertices).to(self.device).float()
        
        # Transform 3D keypoints and vertices to camera space
        for key in ["joints3d", "vertices"]:
            points = targets[key]
            points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
            points = torch.bmm(points, inputs["extrinsics"].transpose(1, 2))
            targets[key] = points[..., :3]
        
        point_cloud_list = [torch.Tensor(item["point_cloud"]) for item in raw_inputs]
        inputs["point_cloud"] = Pointclouds(point_cloud_list).to(self.device)
        
        del raw_inputs, raw_targets, raw_meta_info, points
        
        return inputs, targets, meta_info
    
    def __len__(self):
        return len(self.valid_indices)
    
    def evaluation_prep(self, outputs: torch.Tensor, targets: torch.Tensor, align_root: bool):
        targets["vertices"] = targets["vertices"] * 1000
        targets["joints3d"] = vertices2joints(J_regressor=self.smpl_model_neutral.J_regressor, vertices=targets["vertices"]).to(self.device).float()
        target_root_joint = targets["joints3d"][:, 0].clone()
        
        outputs["vertices"] = outputs["vertices"] * 1000
        outputs["joints3d"] = vertices2joints(J_regressor=self.smpl_model_neutral.J_regressor, vertices=outputs["vertices"]).to(self.device).float()
        output_root_joint = outputs["joints3d"][:, 0].clone()
        
        if align_root:
            outputs["vertices"] = outputs["vertices"] - output_root_joint.unsqueeze(1)
            targets["vertices"] = targets["vertices"] - target_root_joint.unsqueeze(1)
            outputs["joints3d"] = outputs["joints3d"] - output_root_joint.unsqueeze(1)
            targets["joints3d"] = targets["joints3d"] - target_root_joint.unsqueeze(1)
        
        return outputs, targets
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SLOPER4D dataset.")
    parser.add_argument("--data", required=True, type=str, help="SLOPER4D data directory.")
    parser.add_argument("--sequence", default="seq009_running_002", type=str, help="Data sequence.")
    parser.add_argument("--vis_dim", default=3, choices=[2, 3], type=int, help="Visualization dimension.")
    
    args = parser.parse_args()
    
    dataset = SLOPER4D(data_root=args.data, sequence=args.sequence)

    inputs, targets, meta_info = dataset.get_single_sample(idx=0)
    dataset.visualize_sample(inputs=inputs, meta_info=meta_info, targets=targets, idx=0, vis_dimension=args.vis_dim)
