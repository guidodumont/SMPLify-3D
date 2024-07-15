""" This file contains the source code for the body part visibility filter. """

import os

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
smpl_dir = os.path.join(project_dir, "models/parametric_body_models/smpl")

import json
import torch
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from pytorch3d.structures import Pointclouds

segmentation_map = json.load(open(os.path.join(smpl_dir, "vert_segmentation.json"), "r"))
segmentation_map = list(segmentation_map.values())
segmentation_map = [torch.tensor(segmentation_map[i]) for i in range(len(segmentation_map))]
segmentation_map = pad_sequence(segmentation_map, batch_first=True, padding_value=0).float()
uniform_vertices = torch.tensor(json.load(open(os.path.join(smpl_dir, "uniform_vertices.json"), "r")))


def calculate_normals(vertices: torch.Tensor, faces: np.ndarray):
    batch_size = vertices.shape[0]
    faces = torch.from_numpy(faces.astype(np.int32)).long().detach().to(vertices.device)
    
    faces = faces.unsqueeze(0).repeat(batch_size, 1, 1)
    offsets = torch.arange(batch_size, device=vertices.device) * vertices.shape[1]
    faces += offsets.view(-1, 1, 1)
    
    faces = faces.reshape(faces.shape[1] * batch_size, 3)
    vertices = vertices.reshape(vertices.shape[1] * batch_size, 3)
    
    # Compute face normals
    v0 = vertices[faces[:, 0], :]
    v1 = vertices[faces[:, 1], :] - v0
    v2 = vertices[faces[:, 2], :] - v0
    
    face_normals = torch.cross(v1, v2, dim=1)
    
    # Normalize face normals
    face_normals = torch.nn.functional.normalize(face_normals, p=2, dim=1)
    
    # Accumulate normals at each vertex
    normals = torch.zeros_like(vertices)
    normals.index_add_(0, faces[:, 0], face_normals)
    normals.index_add_(0, faces[:, 1], face_normals)
    normals.index_add_(0, faces[:, 2], face_normals)
    
    # Normalize vertex normals
    normals = torch.nn.functional.normalize(normals, p=2, dim=1)
    normals = normals.reshape(batch_size, vertices.shape[0] // batch_size, 3)
    
    return normals


def get_visible_points(vertices: torch.Tensor, faces: np.ndarray, mesh_locations: torch.Tensor, body_part_visibility: torch.Tensor, dot_product_threshold=0.0):
    centered_vertices = vertices - mesh_locations.unsqueeze(1)
    
    batch_size = centered_vertices.shape[0]
    normals = calculate_normals(vertices=centered_vertices, faces=faces)
    viewing_directions = mesh_locations / torch.norm(mesh_locations, dim=1, keepdim=True)  # Normalize the viewing direction
    
    # Calculate the view vector and the dot product
    view_vector = (viewing_directions.unsqueeze(1) - centered_vertices) / torch.norm(viewing_directions.unsqueeze(1) - centered_vertices, dim=1, keepdim=True)
    view_vector = view_vector / torch.norm(view_vector, dim=2, keepdim=True)
    dot_products = torch.sum(normals * -view_vector, dim=2)
    
    visible_indices = [torch.arange(dot_products.size(1), device=dot_products.device)[dot_products[i] > dot_product_threshold] for i in range(dot_products.size(0))]
    visible_indices_pad = pad_sequence(visible_indices, batch_first=True, padding_value=0)
    uniform_mask = torch.isin(visible_indices_pad, uniform_vertices.to(dot_products.device))

    segmentation_map.to(body_part_visibility.device)
    segmentation_map_batch = segmentation_map.unsqueeze(0).expand(batch_size, -1, -1).to(body_part_visibility.device)
    body_part_visibility_batch = body_part_visibility.unsqueeze(-1).expand(-1, -1, segmentation_map_batch.shape[-1])
    visible_part_indices = body_part_visibility_batch * segmentation_map_batch
    visible_part_indices = visible_part_indices.view(batch_size, -1)
    visibility_mask = torch.isin(visible_indices_pad, visible_part_indices)

    visible_points = []
    for i in range(batch_size):
        mask = uniform_mask[i] & visibility_mask[i]
        visible_points.append(vertices[i, visible_indices_pad[i][mask]])

    visible_points = Pointclouds(points=visible_points)

    return visible_points, visible_indices
