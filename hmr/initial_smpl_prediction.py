""" This files inferences the CLIFF model for human mesh recovery. """

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, "third_party/cliff"))

import torch
from pytorch3d import transforms
import third_party.cliff.common.constants as constants
from models.parametric_body_models.smpl.smpl import SMPL
from third_party.cliff.models.cliff_hr48.cliff import CLIFF as cliff_hr48
from third_party.cliff.models.cliff_res50.cliff import CLIFF as cliff_res50
from third_party.cliff.common.utils import strip_prefix_if_present, cam_crop2full

constants.SMPL_MODEL_DIR = os.path.join(project_dir, "models/parametric_body_models/smpl")


class CLIFF:
    def __init__(self, device: torch.device, backbone: str, checkpoints: str):
        self.device = device
        self.backbone = backbone
        self.checkpoints = checkpoints
        
        # Create CLIFF model
        cliff = eval("cliff_" + self.backbone)
        self.model = cliff(constants.SMPL_MEAN_PARAMS).to(self.device)
        
        # Load the pretrained model
        print("Load the CLIFF checkpoint from path:", self.checkpoints)
        state_dict = torch.load(self.checkpoints)["model"]
        state_dict = strip_prefix_if_present(state_dict, prefix="module.")
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()
        
        # Setup the SMPL model
        self.smpl_model = SMPL(constants.SMPL_MODEL_DIR, gender="neutral").to(device)
        
    def inference(self, inputs):
        norm_img = inputs["norm_img"]
        center = inputs["center"]
        scale = inputs["scale"]
        img_h = inputs["img_h"]
        img_w = inputs["img_w"]
        est_focal_length = inputs["est_focal_length"]
        
        cx, cy, b = center[:, 0], center[:, 1], scale * 200
        bbox_info = torch.stack([cx - img_w / 2., cy - img_h / 2., b], dim=-1)
        bbox_info[:, :2] = bbox_info[:, :2] / est_focal_length.unsqueeze(-1) * 2.8  # [-1, 1]
        bbox_info[:, 2] = (bbox_info[:, 2] - 0.24 * est_focal_length) / (0.06 * est_focal_length)  # [-1, 1]
        
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_cam_crop = self.model(norm_img, bbox_info)
        
        # convert the camera parameters from the crop camera to the full camera
        full_img_shape = torch.stack((img_h, img_w), dim=-1)
        pred_cam_full = cam_crop2full(pred_cam_crop, center, scale, full_img_shape, est_focal_length)
        
        pred_output = self.smpl_model(betas=pred_betas, body_pose=pred_rotmat[:, 1:], global_orient=pred_rotmat[:, [0]], pose2rot=False, transl=pred_cam_full)
        pred_poses = transforms.matrix_to_axis_angle(pred_rotmat).contiguous().view(-1, 72)  # N*72
        vertices = pred_output.vertices
        
        return pred_poses, pred_betas, pred_cam_full, pred_output.joints, vertices
