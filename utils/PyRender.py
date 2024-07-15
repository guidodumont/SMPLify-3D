""" This file contains the PyRender class which is used to render the 3D mesh using PyRender library."""

# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import os
import trimesh
import pyrender
import numpy as np
import colorsys
import cv2


class PyRender(object):
    def __init__(self, camera_intrinsics, img_w, img_h, faces=None):
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        self.renderer = pyrender.OffscreenRenderer(viewport_width=img_w, viewport_height=img_h, point_size=1.0)
        self.fx, self.fy, self.cx, self.cy = camera_intrinsics[0][0], camera_intrinsics[1][1], camera_intrinsics[0][2], camera_intrinsics[1][2]
        self.faces = faces
    
    def render_front_view(self, verts, bg_img_rgb=None, bg_color=(0, 0, 0, 0), mesh_colors_rgb: np.ndarray=None):
        # Create a scene for each image and render all meshes
        scene = pyrender.Scene(bg_color=bg_color, ambient_light=np.ones(3) * 0)
        # Create camera. Camera will always be at [0,0,0]
        camera = pyrender.camera.IntrinsicsCamera(fx=self.fx, fy=self.fy, cx=self.cx, cy=self.cy)
        scene.add(camera, pose=np.eye(4))
        
        # Create light source
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        # for DirectionalLight, only rotation matters
        light_pose = trimesh.transformations.rotation_matrix(np.radians(-45), [1, 0, 0])
        scene.add(light, pose=light_pose)
        light_pose = trimesh.transformations.rotation_matrix(np.radians(45), [0, 1, 0])
        scene.add(light, pose=light_pose)
        
        # Need to flip x-axis
        rot = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        # multiple person
        num_people = len(verts)
        
        # for every person in the scene
        for n in range(num_people):
            mesh = trimesh.Trimesh(verts[n], self.faces)
            mesh.apply_transform(rot)
            if mesh_colors_rgb is not None:
                mesh_color = mesh_colors_rgb[n][::-1]
            else:
                mesh_color = colorsys.hsv_to_rgb(float(n) / num_people, 0.5, 1.0)
            
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.2,
                alphaMode="BLEND",
                baseColorFactor=mesh_color)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material, wireframe=False)
            scene.add(mesh, "mesh")
        
        # Alpha channel was not working previously, need to check again
        # Until this is fixed use hack with depth image to get the opacity
        color_rgba, depth_map = self.renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color_rgb = np.copy(color_rgba[:, :, :3])
        
        # return cv2.addWeighted(bg_img_rgb, 0, color_rgb, 1, 0)
        if bg_img_rgb is None:
            return color_rgb
        else:
            mask = depth_map <= 0
            color_rgb[mask] = bg_img_rgb[mask]
            return cv2.addWeighted(bg_img_rgb, 0, color_rgb, 1, 0)
    
    def render_side_view(self, verts):
        centroid = verts.mean(axis=(0, 1))  # n*6890*3 -> 3
        # make the centroid at the image center (the X and Y coordinates are zeros)
        centroid[:2] = 0
        aroundy = cv2.Rodrigues(np.array([0, np.radians(90.), 0]))[0][np.newaxis, ...]  # 1*3*3
        pred_vert_arr_side = np.matmul((verts - centroid), aroundy) + centroid
        side_view = self.render_front_view(pred_vert_arr_side)
        return side_view
    
    def delete(self):
        """
        Need to delete before creating the renderer next time
        """
        self.renderer.delete()