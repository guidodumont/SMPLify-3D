""" This file contain utility functions for visualization of 2D and 3D data. """

import cv2
import open3d as o3d
import open3d.visualization as vis
from utils.keypoint_utils import *
from utils.line_mesh import LineMesh

color_pallet = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255), (199, 100, 0),
                (72, 0, 118), (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
                (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174), (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255), (151, 0, 95),
                (9, 80, 61), (84, 105, 51), (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65), (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
                (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161), (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120), (183, 130, 88), (95, 32, 0),
                (130, 114, 135), (110, 129, 133), (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62), (65, 70, 15), (127, 167, 115), (59, 105, 106),
                (142, 108, 45), (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1), (246, 0, 122), (191, 162, 208)]


def draw_2d_bbox(img: np.ndarray, bboxes_xyxy: np.ndarray, detection_id: np.ndarray = None, show=False, additional_text="", show_id=True):
    for i in range(len(bboxes_xyxy)):
        bbox = bboxes_xyxy[i]
        bbox = bbox.astype(int)
        x1, y1, x2, y2 = bbox
        
        color = (0, 0, 255)
        if detection_id is not None:
            color = color_pallet[int(detection_id[i]) % len(color_pallet)][::-1]
            
            if show_id or additional_text != "":
                text = str(int(detection_id[i])) + additional_text
                img = cv2.putText(img, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    if not show:
        return img
    else:
        cv2.imshow("2D Bounding Box Visualization", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_2d_skeleton(img: np.ndarray, joints2d: np.ndarray, skeleton: np.ndarray, colors: list = None, show=False):
    joints2d = joints2d.astype(int)
    
    for i, connection in enumerate(skeleton):
        start_idx, end_idx = connection
        start_point = tuple(joints2d[start_idx, :2])
        end_point = tuple(joints2d[end_idx, :2])
        
        if start_point == (0, 0) or end_point == (0, 0):
            continue
        
        color = colors[i][::-1] if colors is not None else (0, 0, 255)
        img = cv2.line(img, start_point, end_point, color=color, thickness=2)
    
    if not show:
        return img
    else:
        cv2.imshow("Stick Figure Visualization", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def draw_3d_kinematic_tree(joints3d: np.ndarray, skeleton: list, colors: list, id: str = ""):
    geoms = []
    
    for connection, color in zip(skeleton, colors):
        start_idx, end_idx = connection
        points = np.array([joints3d[start_idx], joints3d[end_idx]])
        line_mesh = LineMesh(points, None, color, radius=0.01).cylinder_segments[0]
        geoms.append({"name": id + str(connection), "geometry": line_mesh, "material": vis.rendering.MaterialRecord()})
    
    return geoms


def draw_point_cloud(point_cloud: o3d.geometry.PointCloud, show: bool = False, id: str = "", color: np.ndarray = None):
    if color is not None:
        point_cloud.paint_uniform_color(color)
    
    geoms = [{"name": id + "_point_cloud", "geometry": point_cloud}]
    
    if show:
        vis.draw(geoms)
    else:
        return geoms


def draw_mesh(mesh: o3d.geometry.TriangleMesh, show: bool = False, id: str = "", color: np.ndarray = None):
    material = vis.rendering.MaterialRecord()
    material.shader = "defaultLit"
    
    if color is not None:
        material.base_color = color
        
        if len(color) == 4:
            material.shader = "defaultLitTransparency"
    else:
        material.base_color = [191/255, 162/255, 208/255, 1]
    
    geoms = [{"name": id + "mesh", "geometry": mesh, "material": material}]
    
    if show:
        vis.draw(geoms)
    else:
        return geoms
