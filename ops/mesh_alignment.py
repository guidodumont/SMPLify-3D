""" This file contains the source code to perform initial ICP alignment between the visible vertices of the mesh and observed LiDAR point cloud."""

import torch
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.ops.points_alignment import SimilarityTransform


def initial_mesh_alignment(visible_points: Pointclouds, point_clouds: Pointclouds, max_iterations: int = 100):
        init_RTs = initial_guess(visible_points=visible_points, point_clouds=point_clouds)
        
        # Perform ICP and update the camera translation
        result = pytorch3d.ops.iterative_closest_point(X=visible_points, Y=point_clouds, init_transform=init_RTs, max_iterations=max_iterations)
        Xt = result.Xt
        RTs = result.RTs
        converged = result.converged
        
        return converged, RTs, Xt
    
    
def initial_guess(visible_points: Pointclouds, point_clouds: Pointclouds):
    device = visible_points.device
    batch_size = len(visible_points)
    
    visible_bounds = visible_points.get_bounding_boxes()
    visible_centers = visible_bounds[:, :, 0] + ((visible_bounds[:, :, 1] - visible_bounds[:, :, 0]) / 2)
    
    point_cloud_bounds = point_clouds.get_bounding_boxes()
    point_cloud_centers = point_cloud_bounds[:, :, 0] + ((point_cloud_bounds[:, :, 1] - point_cloud_bounds[:, :, 0]) / 2)
    
    # Get initial transformation matrix
    T = point_cloud_centers - visible_centers
    R = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    s = torch.ones(batch_size, device=device)
    RTs = SimilarityTransform(R=R, T=T, s=s)
    
    return RTs
