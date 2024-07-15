""" This file contains the loss functions used within the mesh optimization. """

import torch
from pytorch3d import transforms
from pytorch3d.loss import chamfer_distance
from ops.mesh_attributes import get_visible_points
from utils.keypoint_utils import get_body_part_id2smpl_pose_idx

def perspective_projection(points: torch.Tensor, camera_intrinsics: torch.Tensor):
    projected_points = points / points[:, :, -1].unsqueeze(-1)
    projected_points = torch.einsum('bij,bkj->bki', camera_intrinsics, projected_points)
    
    return projected_points[:, :, :-1]


def gmof(x, sigma):
    """
    Geman-McClure error function
    """
    x_squared = x ** 2
    sigma_squared = sigma ** 2
    return (sigma_squared * x_squared) / (sigma_squared + x_squared)


def get_joint_reprojection_loss(opt_joints2d, goal_joints2d, joints_conf, loss_sigma, align_joints=False):
    if align_joints:
        opt_center_hips = (opt_joints2d[:, 11] + opt_joints2d[:, 12]) / 2
        goal_center_hips = (goal_joints2d[:, 11] + goal_joints2d[:, 12]) / 2
        translation = goal_center_hips - opt_center_hips
        opt_joints2d = opt_joints2d + translation[:, None, :]
    
    reprojection_error = gmof(opt_joints2d[:, :16] - goal_joints2d[:, :16], loss_sigma)
    reprojection_loss = reprojection_error.sum(dim=-1) * (joints_conf[:, :16] ** 2)
    
    return reprojection_loss.sum(dim=-1)


def get_pose_prior_loss(pose_prior, pose, betas, weight):
    return pose_prior(pose, betas) * weight ** 2


def get_unnatural_joint_bending_loss(pose, weight):
    # We subtract 3 because pose does not include the global rotation of the model
    loss = torch.exp(pose[:, [55-3, 58-3, 12-3, 15-3]] * torch.tensor([1., -1., -1, -1.], device=pose.device)) ** 2
    return loss.sum(dim=-1) * weight ** 2


def get_shape_prior_loss(betas, weight):
    return (betas ** 2).sum(dim=-1) * weight ** 2


def get_3d_alignment_loss(vertices, faces, camera_t, body_part_visibility, point_cloud, weight, dot_product_threshold=0.0):
    visible_points, _ = get_visible_points(vertices=vertices, faces=faces, mesh_locations=camera_t, body_part_visibility=body_part_visibility, dot_product_threshold=dot_product_threshold)
    
    return chamfer_distance(x=visible_points, y=point_cloud, batch_reduction=None, point_reduction="mean", single_directional=False)[0] * weight ** 2


def get_occluded_body_part_loss(pose, initial_pose_quat, body_part_visibility, weight):
    opt_body_pose_qaut = transforms.axis_angle_to_quaternion(pose.view(-1, 23, 3))
    opt_body_pose_qaut = opt_body_pose_qaut / torch.norm(opt_body_pose_qaut, dim=-1, keepdim=True)
    initial_pose_quat = initial_pose_quat / torch.norm(initial_pose_quat, dim=-1, keepdim=True)
    dot_product = torch.sum(opt_body_pose_qaut * initial_pose_quat, dim=-1)
    
    dot_product_weight = torch.ones_like(dot_product)
    for body_part_idx, smpl_pose_idx in get_body_part_id2smpl_pose_idx().items():
        dot_product_weight[:, smpl_pose_idx] = body_part_visibility[:, body_part_idx]
    dot_product = dot_product * dot_product_weight
    
    return torch.abs(torch.sum(1 - dot_product ** 2, dim=-1)) * weight ** 2
