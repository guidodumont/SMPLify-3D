""" This file contains the optimzation class for SMPLify-3D. """

import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from optimizer.losses import *
from utils.keypoint_utils import *
from smplx.lbs import vertices2joints
from ops.mesh_attributes import get_visible_points
from third_party.cliff.prior import MaxMixturePrior
from ops.mesh_alignment import initial_mesh_alignment
import third_party.cliff.common.constants as constants
from models.parametric_body_models.smpl.smpl import SMPL

constants.SMPL_MODEL_DIR = os.path.join(project_dir, "../pretrained_models/parametric_models/smpl")


class Optimization:
    def __init__(self, camera_intrinsics, config, device=torch.device("cuda")):
        self.device = device
        self.config = config
        self.camera_intrinsics = camera_intrinsics
        self.cam_resolution = torch.tensor([int(camera_intrinsics[0, 0, 2] * 2), int(camera_intrinsics[0, 1, 2] * 2)]).to(self.device)
        
        self.joint_confidence = self.config.SMPLify3D.joint_confidence
        self.num_gaussians = self.config.SMPLify3D.num_gaussians
        self.lr = self.config.SMPLify3D.lr
        self.adam_betas = self.config.SMPLify3D.adam_betas
        
        self.pose_prior = MaxMixturePrior(prior_folder=constants.SMPL_MODEL_DIR,
                                          num_gaussians=self.num_gaussians,
                                          dtype=torch.float32).to(self.device)
        
        self.smpl_model_neutral = SMPL(constants.SMPL_MODEL_DIR, gender="neutral").to(self.device)
        J_regressor = np.load(os.path.join(constants.SMPL_MODEL_DIR, "J_regressor_coco.npy"), allow_pickle=True)
        self.J_regressor = torch.tensor(J_regressor).to(self.device).float()
        self.skeleton2body_part = get_coco17_skeleton2body_parts()
    
    def __call__(self, init_pose, init_betas, init_cam_t, goal_joints2d, point_clouds, mask_areas):
        batch_size = init_pose.shape[0]
        
        pred_camera_translation = init_cam_t.clone()
        pred_body_pose = init_pose[:, 3:].detach().clone()
        pred_global_orient = init_pose[:, :3].detach().clone()
        pred_betas = init_betas.detach().clone()
        
        '''
        ########################################################################################################
        ######################################### Body part visibility #########################################
        ########################################################################################################
        '''
        visibility_conf_img = goal_joints2d[:, :, 3]
        lidar_visibility = goal_joints2d[:, :, 4]
        
        thr = np.min([self.joint_confidence, np.max(visibility_conf_img.cpu().numpy())])
        joint_visibility_img = torch.where((visibility_conf_img >= thr), 1, 0)
        joint_visibility_lidar = torch.where((lidar_visibility >= thr), 1, 0)
        
        body_part_visibility_img = torch.zeros((batch_size, len(self.skeleton2body_part.keys()))).to(self.device)
        body_part_visibility_lidar = torch.zeros((batch_size, len(self.skeleton2body_part.keys()))).to(self.device)
        for i, (body_part, joint_idxs) in enumerate(self.skeleton2body_part.items()):
            sampled_joint_visibility = joint_visibility_img[:, joint_idxs]
            sum_visibility = torch.sum(sampled_joint_visibility, dim=-1)
            body_part_visibility_img[:, i] = torch.where(sum_visibility == len(joint_idxs), 1, 0)
            
            sampled_joint_visibility = joint_visibility_lidar[:, joint_idxs]
            sum_visibility = torch.sum(sampled_joint_visibility, dim=-1)
            body_part_visibility_lidar[:, i] = torch.where(sum_visibility == len(joint_idxs), 1, 0)

        '''
        ########################################################################################################
        ######################################## Initial mesh alignment ########################################
        ########################################################################################################
        '''
        smpl_output = self.smpl_model_neutral(betas=pred_betas, body_pose=pred_body_pose, global_orient=pred_global_orient, transl=pred_camera_translation, pose2rot=True)
        pred_vertices = smpl_output.vertices
        
        dot_product_threshold = self.config.SMPLify3D.dot_product_threshold
        visible_points, _ = get_visible_points(vertices=pred_vertices, faces=self.smpl_model_neutral.faces, mesh_locations=pred_camera_translation, body_part_visibility=body_part_visibility_lidar, dot_product_threshold=dot_product_threshold)
        
        convered, RTs, Xt = initial_mesh_alignment(visible_points=visible_points, point_clouds=point_clouds, max_iterations=100)
        pred_camera_translation = (RTs.s[:, None, None] * torch.bmm(pred_camera_translation.unsqueeze(1), RTs.R) + RTs.T[:, None, :]).view(batch_size, 3)
        
        goal_joints2d = goal_joints2d[:, :, :2]
        
        '''
        ########################################################################################################
        ########################################## Mesh optimization ###########################################
        ########################################################################################################
        '''
        
        ########################################################################################################
        ######################################### Scaling optimization #########################################
        ########################################################################################################
        
        pred_betas.requires_grad = True
        pred_body_pose.requires_grad = False
        pred_global_orient.requires_grad = False
        pred_camera_translation.requires_grad = False
        body_opt_params = [pred_betas]

        loss_sigma = self.config.SMPLify3D.scaling_optimization.loss_sigma
        shape_prior_weight = self.config.SMPLify3D.scaling_optimization.shape_prior_weight
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.lr, betas=self.adam_betas)
        for i in range(self.config.SMPLify3D.scaling_optimization.iterations):
            smpl_output = self.smpl_model_neutral(betas=pred_betas, body_pose=pred_body_pose, global_orient=pred_global_orient, pose2rot=True, transl=pred_camera_translation)
            pred_vertices = smpl_output.vertices

            pred_joints3d = vertices2joints(J_regressor=self.J_regressor, vertices=pred_vertices)
            pred_joints2d = perspective_projection(points=pred_joints3d, camera_intrinsics=self.camera_intrinsics)

            reprojection_loss = get_joint_reprojection_loss(opt_joints2d=pred_joints2d, goal_joints2d=goal_joints2d, joints_conf=visibility_conf_img, loss_sigma=loss_sigma, align_joints=True)
            shape_prior_loss = get_shape_prior_loss(betas=pred_betas, weight=shape_prior_weight)
            total_loss = reprojection_loss + shape_prior_loss
            loss = total_loss.sum()

            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        ########################################################################################################
        ######################################### 3D mesh optimization #########################################
        ########################################################################################################
        
        initial_pose = pred_body_pose.clone().detach().view(batch_size, 23, 3)
        initial_pose_quat = transforms.axis_angle_to_quaternion(initial_pose)

        pred_body_pose.requires_grad = True
        pred_betas.requires_grad = True
        pred_global_orient.requires_grad = True
        pred_camera_translation.requires_grad = False
        body_opt_params = [pred_body_pose, pred_betas, pred_global_orient]

        loss_sigma = self.config.SMPLify3D.mesh_optimization.loss_sigma
        pose_prior_weight = self.config.SMPLify3D.mesh_optimization.pose_prior_weight
        angle_prior_weight = self.config.SMPLify3D.mesh_optimization.angle_prior_weight
        shape_prior_weight = self.config.SMPLify3D.mesh_optimization.shape_prior_weight
        point_cloud_weight = self.config.SMPLify3D.mesh_optimization.point_cloud_weight
        angle_deviation_weight = self.config.SMPLify3D.mesh_optimization.angle_deviation_weight
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.lr, betas=self.adam_betas)
        for i in range(self.config.SMPLify3D.mesh_optimization.iterations):
            smpl_output = self.smpl_model_neutral(betas=pred_betas, body_pose=pred_body_pose, global_orient=pred_global_orient, pose2rot=True, transl=pred_camera_translation)
            pred_vertices = smpl_output.vertices

            pred_joints3d = vertices2joints(J_regressor=self.J_regressor, vertices=pred_vertices)
            pred_joints2d = perspective_projection(points=pred_joints3d, camera_intrinsics=self.camera_intrinsics)

            reprojection_loss = get_joint_reprojection_loss(opt_joints2d=pred_joints2d, goal_joints2d=goal_joints2d, joints_conf=visibility_conf_img, loss_sigma=loss_sigma, align_joints=True)
            pose_prior_loss = get_pose_prior_loss(pose_prior=self.pose_prior, pose=pred_body_pose, betas=pred_betas, weight=pose_prior_weight)
            angle_prior_loss = get_unnatural_joint_bending_loss(pose=pred_body_pose, weight=angle_prior_weight)
            shape_prior_loss = get_shape_prior_loss(betas=pred_betas, weight=shape_prior_weight)
            angle_deviation_loss = get_occluded_body_part_loss(pose=pred_body_pose, initial_pose_quat=initial_pose_quat, body_part_visibility=body_part_visibility_img, weight=angle_deviation_weight)
            data3d_loss = get_3d_alignment_loss(vertices=pred_vertices, faces=self.smpl_model_neutral.faces, camera_t=pred_camera_translation, body_part_visibility=body_part_visibility_lidar, point_cloud=point_clouds, weight=point_cloud_weight, dot_product_threshold=dot_product_threshold)
            total_loss = reprojection_loss + pose_prior_loss + angle_prior_loss + shape_prior_loss + data3d_loss + angle_deviation_loss
            loss = total_loss.sum()

            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()

        ########################################################################################################
        ########################################### Image alignment ############################################
        ########################################################################################################
        
        pred_body_pose.requires_grad = False
        pred_global_orient.requires_grad = False
        pred_betas.requires_grad = False
        pred_camera_translation.requires_grad = True
        body_opt_params = [pred_betas, pred_camera_translation]

        loss_sigma = self.config.SMPLify3D.image_alignment.loss_sigma
        point_cloud_weight = self.config.SMPLify3D.image_alignment.point_cloud_weight
        body_optimizer = torch.optim.Adam(body_opt_params, lr=self.lr, betas=self.adam_betas)
        for i in range(self.config.SMPLify3D.image_alignment.iterations):
            smpl_output = self.smpl_model_neutral(betas=pred_betas, body_pose=pred_body_pose, global_orient=pred_global_orient, pose2rot=True, transl=pred_camera_translation)
            pred_vertices = smpl_output.vertices

            pred_joints3d = vertices2joints(J_regressor=self.J_regressor, vertices=pred_vertices)
            pred_joints2d = perspective_projection(points=pred_joints3d, camera_intrinsics=self.camera_intrinsics)

            reprojection_loss = get_joint_reprojection_loss(opt_joints2d=pred_joints2d, goal_joints2d=goal_joints2d, joints_conf=visibility_conf_img, loss_sigma=loss_sigma, align_joints=False)
            data3d_loss = get_3d_alignment_loss(vertices=pred_vertices, faces=self.smpl_model_neutral.faces, camera_t=pred_camera_translation, body_part_visibility=body_part_visibility_lidar, point_cloud=point_clouds, weight=point_cloud_weight, dot_product_threshold=dot_product_threshold)
            total_loss = reprojection_loss + data3d_loss
            loss = total_loss.sum()

            body_optimizer.zero_grad()
            loss.backward()
            body_optimizer.step()
        
        ########################################################################################################
        ########################################## Reprojection loss ###########################################
        ########################################################################################################
        
        with torch.no_grad():
            smpl_output = self.smpl_model_neutral(betas=pred_betas, body_pose=pred_body_pose, global_orient=pred_global_orient, pose2rot=True, transl=pred_camera_translation)
            pred_vertices = smpl_output.vertices

            pred_joints3d = vertices2joints(J_regressor=self.J_regressor, vertices=pred_vertices)
            pred_joints2d = perspective_projection(points=pred_joints3d, camera_intrinsics=self.camera_intrinsics)

            keypoints_distances = torch.norm(pred_joints2d - goal_joints2d, dim=-1)
            reprojection_losses = torch.mean(keypoints_distances, dim=-1) / (torch.abs(mask_areas) + 1e-9)
            chamfer_distances = get_3d_alignment_loss(vertices=pred_vertices, faces=self.smpl_model_neutral.faces, camera_t=pred_camera_translation, body_part_visibility=body_part_visibility_lidar, point_cloud=point_clouds, weight=1, dot_product_threshold=dot_product_threshold)
        
        final_vertices = smpl_output.vertices.detach()
        final_joints3d = vertices2joints(J_regressor=self.smpl_model_neutral.J_regressor, vertices=final_vertices)
        final_pose = torch.cat([pred_global_orient, pred_body_pose], dim=-1).detach()
        final_betas = pred_betas.detach()
        
        return final_vertices, final_joints3d, final_pose, final_betas, pred_camera_translation, reprojection_losses, chamfer_distances
    