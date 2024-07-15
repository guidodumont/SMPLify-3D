""" Base class for all datasets """

import os
import cv2
import time
import torch
import open3d as o3d
import open3d.visualization as vis
from utils.keypoint_utils import *
from utils.PyRender import PyRender
from torch.utils.data import Dataset
from ops.point_cloud_ops import points2image
from utils.smpl_annotations import AnnotationsSMPL
from utils.visualization_utils import draw_2d_skeleton, draw_2d_bbox, draw_mesh, draw_point_cloud, draw_3d_kinematic_tree

project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class BaseDataset(Dataset):
    def __init__(self, name: str, sequences: list, device: torch.device):
        self.name = name
        self.sequences = sequences
        self.device = device
        self.image_crop_factor = 1
        
        self.temp_results = {"inputs": {}, "outputs": {}, "meta_info": {}}
        self.ann_smpl = AnnotationsSMPL(smpl_dir=self.smpl_dir)
    
    def get_single_sample(self, idx):
        inputs, targets, meta_info = self.__getitem__(idx=idx)
        inputs, targets, meta_info = self.collate_fn([tuple([inputs, targets, meta_info])])
        
        return inputs, targets, meta_info
    
    def visualize_output_sample(self, inputs, meta_info, outputs, targets, idx, vis_dimension=3, show=True):
        pred_vertices = outputs["vertices"][idx].cpu().numpy().reshape(-1, 3)
        if vis_dimension == 2:
            sequence = meta_info["sequence"][idx]
            img_name = meta_info["img_name"][idx]
            camera_intrinsics = inputs["camera_intrinsics"][idx].cpu().numpy()
            
            print("Visualizing: sequence:", sequence, ", image:", img_name)
            
            img = cv2.imread(os.path.join(self.img_dirs[sequence], img_name))
            img = cv2.resize(img, (int(img.shape[1] * self.image_crop_factor), int(img.shape[0] * self.image_crop_factor)))
            
            bbox_xywh = inputs["bbox"][idx].cpu().numpy()
            bbox_xyxy = np.array([[bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]])
            img = draw_2d_bbox(img=img, bboxes_xyxy=np.array(bbox_xyxy), show=False)
            
            color = np.array([[0, 166, 214]])
            renderer = PyRender(camera_intrinsics=camera_intrinsics, img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl_model_neutral.faces)
            img = renderer.render_front_view(torch.from_numpy(pred_vertices).unsqueeze(0), bg_img_rgb=img.copy(), mesh_colors_rgb=color)
            renderer.delete()
            
            if show:
                cv2.namedWindow("Predicted output", cv2.WINDOW_NORMAL)
                cv2.imshow("Predicted output", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                return img
        else:
            skeleton = get_humanm3_skeleton()
            J_regressor = np.load(os.path.join(self.smpl_dir, "smpl/J_regressor_humanm3.npy"), allow_pickle=True)
            
            geoms = []
            point_cloud = inputs["point_cloud"].points_list()[idx]
            point_cloud_geo = o3d.geometry.PointCloud()
            point_cloud_geo.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy().reshape(-1, 3))
            geoms.extend(draw_point_cloud(point_cloud_geo, id="Point cloud", show=False, color=np.array([0, 0, 0])))
            
            pred_mesh_geo = o3d.geometry.TriangleMesh()
            pred_mesh_geo.vertices = o3d.utility.Vector3dVector(pred_vertices)
            pred_mesh_geo.triangles = o3d.utility.Vector3iVector(self.smpl_model_neutral.faces)
            geoms.extend(draw_mesh(mesh=pred_mesh_geo, color=np.array([0, 166/255, 214/255, 0.7]), id="Predicted", show=False))
            
            joints = np.dot(J_regressor, pred_vertices)
            geoms.extend(draw_3d_kinematic_tree(joints3d=joints, skeleton=skeleton, colors=[[1, 0, 0]] * len(skeleton), id="Predicted"))
            
            if "vertices" in targets:
                gt_vertices = targets["vertices"][idx].cpu().numpy().reshape(-1, 3)
                gt_mesh_geo = o3d.geometry.TriangleMesh()
                gt_mesh_geo.vertices = o3d.utility.Vector3dVector(gt_vertices)
                gt_mesh_geo.triangles = o3d.utility.Vector3iVector(self.smpl_model_neutral.faces)
                geoms.extend(draw_mesh(mesh=gt_mesh_geo, color=np.array([0, 1, 0, 0.7]), id="GT", show=False))
                
                joints = np.dot(J_regressor, gt_vertices)
                geoms.extend(draw_3d_kinematic_tree(joints3d=joints, skeleton=skeleton, colors=[[0, 1, 0]] * len(skeleton), id="GT"))
                
            elif "joints3d" in targets:
                joints = targets["joints3d"][idx].cpu().numpy()
                geoms.extend(draw_3d_kinematic_tree(joints3d=joints, skeleton=skeleton, colors=[[0, 1, 0]] * len(skeleton), id="GT"))
            
            if show:
                vis.draw(geoms, show_skybox=False)
            else:
                return geoms
    
    def visualize_sample(self, inputs, meta_info, targets, idx, vis_dimension=2):
        sequence = meta_info["sequence"][idx]
        img_name = meta_info["img_name"][idx]
        
        print("Sequence:", sequence, "Image:", img_name)
        
        if vis_dimension == 2:
            camera_intrinsics = inputs["camera_intrinsics"][idx].cpu().numpy()
            bbox_xywh = inputs["bbox"][idx].cpu().numpy()
            bbox_xyxy = np.array([bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]])
            
            img = cv2.imread(os.path.join(self.img_dirs[sequence], img_name))
            img = cv2.resize(img, (int(img.shape[1] * self.image_crop_factor), int(img.shape[0] * self.image_crop_factor)))
            img = draw_2d_bbox(img=img, bboxes_xyxy=[bbox_xyxy], show=False)
            
            if len(targets) > 0:
                vertices = targets["vertices"][idx].cpu().numpy().reshape(-1, 3)
                renderer = PyRender(camera_intrinsics=camera_intrinsics, img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl_model_neutral.faces)
                img = renderer.render_front_view(torch.from_numpy(vertices).unsqueeze(0), bg_img_rgb=img.copy())
                renderer.delete()
            
            joints2d = inputs["joints2d"][idx, :, :2].cpu().numpy()
            img = draw_2d_skeleton(img=img, joints2d=joints2d, skeleton=get_coco17_skeleton(), colors=get_coco17_skeleton_colors(), show=False)
            
            cv2.namedWindow("Data sample", cv2.WINDOW_NORMAL)
            cv2.imshow("Data sample", img)
            
            # Draw point cloud
            point_cloud_img = cv2.imread(os.path.join(self.img_dirs[sequence], img_name))
            point_cloud_img = cv2.resize(point_cloud_img, (int(point_cloud_img.shape[1] * self.image_crop_factor), int(point_cloud_img.shape[0] * self.image_crop_factor)))
            
            point_cloud = inputs["point_cloud"].points_list()[idx].cpu().numpy().reshape(-1, 3)
            point_cloud_uv = points2image(points=point_cloud, intrinsics=camera_intrinsics)
            for uv in point_cloud_uv:
                point_cloud_img = cv2.circle(point_cloud_img, (int(uv[0]), int(uv[1])), 2, (0, 255, 0), -1)
            
            cv2.namedWindow("Point cloud", cv2.WINDOW_NORMAL)
            cv2.imshow("Point cloud", point_cloud_img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            geoms = []
            point_cloud = inputs["point_cloud"].points_list()[idx]
            point_cloud_geo = o3d.geometry.PointCloud()
            point_cloud_geo.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy().reshape(-1, 3))
            geoms.extend(draw_point_cloud(point_cloud_geo, id="Point cloud", show=False, color=np.array([0, 0, 0])))
            
            if "vertices" in targets:
                vertices = targets["vertices"][idx].cpu().numpy().reshape(-1, 3)
                mesh_geo = o3d.geometry.TriangleMesh()
                mesh_geo.vertices = o3d.utility.Vector3dVector(vertices)
                mesh_geo.triangles = o3d.utility.Vector3iVector(self.smpl_model_neutral.faces)
                geoms.extend(draw_mesh(mesh=mesh_geo, color=np.array([0, 1, 0, 0.7]), id="GT", show=False))
                
                J_regressor = np.load(os.path.join(self.smpl_dir, "smpl/J_regressor_humanm3.npy"), allow_pickle=True)
                
                joints = np.dot(J_regressor, vertices)
                skeleton = get_humanm3_skeleton()
                geoms.extend(draw_3d_kinematic_tree(joints3d=joints, skeleton=skeleton, colors=[[0, 1, 0]] * len(skeleton), id="GT"))
            elif "joints3d" in targets:
                joints = targets["joints3d"][idx].cpu().numpy()
                skeleton = get_humanm3_skeleton()
                geoms.extend(draw_3d_kinematic_tree(joints3d=joints, skeleton=skeleton, colors=[[0, 1, 0]] * len(skeleton), id="GT"))
            
            vis.draw(geoms, show_skybox=False)
    
    def save_batch_HMR_results(self, inputs, outputs, meta_info, final_batch=False):
        if self.temp_results["inputs"] != {}:
            for key in inputs.keys():
                inputs[key] = torch.cat((self.temp_results["inputs"][key], inputs[key]), dim=0)
            for key in outputs.keys():
                outputs[key] = torch.cat((self.temp_results["outputs"][key], outputs[key]), dim=0)
            for key in meta_info.keys():
                meta_info[key] = np.concatenate((self.temp_results["meta_info"][key], meta_info[key]), axis=0)
        
        img_ids = [meta_info["img_id"][i] for i in range(meta_info["img_id"].shape[0])]
        unique_img_ids, unique_idxs, counts = np.unique(img_ids, return_index=True, return_counts=True)
        last_frame_length = counts[-1]
        
        if not final_batch:
            unique_img_ids, unique_idxs, counts = unique_img_ids[:-1], unique_idxs[:-1], counts[:-1]
        
        for img_id, unique_idx, count in zip(unique_img_ids, unique_idxs, counts):
            sequence = meta_info["sequence"][unique_idx]
            camera_intrinsics = inputs["camera_intrinsics"][unique_idx].cpu().numpy().astype(float).tolist()
            date = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            info_dict = self.ann_smpl.create_info_dict(sequence=sequence,
                                                       date_created=str(date),
                                                       skeleton2d=get_coco17_skeleton().tolist(),
                                                       camera_intrinsics=camera_intrinsics)
            
            img_w = int(inputs["img_w"][unique_idx].cpu().numpy())
            img_h = int(inputs["img_h"][unique_idx].cpu().numpy())
            img_name = str(meta_info["img_name"][unique_idx])
            img_dict = self.ann_smpl.create_image_dict(img_id=img_id,
                                                       file_name=img_name,
                                                       width=img_w,
                                                       height=img_h)
            
            tracking_ids, body_poses, betas, global_orients, global_ts, vertices, joints3d, joints2d, bboxes, projection_errors = [], [], [], [], [], [], [], [], [], []
            for i in range(count):
                tracking_ids.append(int(meta_info["tracking_id"][unique_idx + i]))
                
                betas.append(outputs["betas"][unique_idx + i].cpu().numpy().astype(float).tolist())
                body_poses.append(outputs["poses"][unique_idx + i, 3:].cpu().numpy().astype(float).tolist())
                global_orients.append(outputs["poses"][unique_idx + i, :3].cpu().numpy().astype(float).tolist())
                global_ts.append(outputs["global_t"][unique_idx + i].cpu().numpy().astype(float).tolist())
                projection_errors.append(outputs["projection_errors"][unique_idx + i].cpu().numpy().astype(float).tolist())
                
                joints3d.append(outputs["joints3d"][unique_idx + i].cpu().numpy().astype(float).tolist())
                joints2d.append(inputs["joints2d"][unique_idx + i].cpu().numpy().astype(float).tolist())
                bboxes.append(inputs["bbox"][unique_idx + i].cpu().numpy().astype(float).tolist())
            
            annotations_dict = self.ann_smpl.create_annotation_dict(idxs=np.arange(unique_idx, unique_idx + count).tolist(),
                                                                    keypoints=joints2d,
                                                                    body_pose=body_poses,
                                                                    betas=betas,
                                                                    global_orient=global_orients,
                                                                    global_t=global_ts,
                                                                    joints3d=joints3d,
                                                                    tracking_ids=tracking_ids,
                                                                    bboxes=bboxes,
                                                                    projection_errors=projection_errors)
            
            if self.name == "demo":
                save_dir = self.data_root
            else:
                save_dir = os.path.join(self.data_root, sequence, "smpl")
                
            self.ann_smpl.save(anns=[{"info": info_dict, "image": img_dict, "annotations": annotations_dict}], save_dir=save_dir)
        
        if counts[-1] > 0 and not final_batch:
            self.temp_results = {"inputs": {}, "outputs": {}, "meta_info": {}}
            
            for key in inputs.keys():
                self.temp_results["inputs"][key] = inputs[key][-last_frame_length:]
            for key in outputs.keys():
                self.temp_results["outputs"][key] = outputs[key][-last_frame_length:]
            for key in meta_info.keys():
                self.temp_results["meta_info"][key] = meta_info[key][-last_frame_length:]
            