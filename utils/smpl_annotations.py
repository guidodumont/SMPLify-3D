""" This script is used to create, load, and visualize SMPL annotations. """

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_dir)

import cv2
import json
import copy
import torch
import open3d as o3d
from utils.keypoint_utils import *
import open3d.visualization as vis
from utils.PyRender import PyRender
from argparse import ArgumentParser
from config.config import create_config_object
from models.parametric_body_models.smpl.smpl import SMPL
from utils.visualization_utils import color_pallet, draw_2d_skeleton, draw_mesh, draw_point_cloud, draw_3d_kinematic_tree, draw_2d_bbox


class AnnotationsSMPL:
    def __init__(self, smpl_dir: str, ann_dir: str = None, img_dir: str = None):
        self.info_template = {
            "sequence": "",
            "date_created": "",
            "skeleton2d": [],
            "camera_intrinsics": [],
        }
        
        self.image_template = {
            "image_id": int(-1),
            "file_name": "",
            "width": int(-1),
            "height": int(-1),
        }
        
        self.annotation_template = {
            "idx": -1,
            "bbox": [],
            "keypoints": [],
            "hmr": {
                "body_pose": [],
                "betas": [],
                "global_orient": [],
                "global_t": [],
                "joints3d": [],
            },
            "reprojection_error": float(-1),
            "chamfer_distance": float(-1),
        }
        
        if not smpl_dir.endswith("smpl"):
            smpl_dir = os.path.join(smpl_dir, "smpl")
        self.smpl_dir = smpl_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.smpl_model = SMPL(self.smpl_dir, gender="neutral").to(self.device)
        self.J_regressor_humanM3 = np.load(os.path.join(project_dir, "models/parametric_body_models/smpl/J_regressor_humanm3.npy"))
        
        if ann_dir == img_dir is not None:
            if not os.path.exists(ann_dir):
                raise FileNotFoundError(f"Annotation directory {ann_dir} does not exist.")
            if not os.path.exists(img_dir):
                raise FileNotFoundError(f"Image directory {img_dir} does not exist.")
            
            self.ann_dir = ann_dir
            self.img_dir = img_dir
            
            self.img_ids = []
            self.img_names = []
            for file in os.listdir(ann_dir):
                if file.endswith(".json") and file.startswith("smpl"):
                    ann = json.load(open(os.path.join(ann_dir, file), "r"))
                    self.img_ids.append(ann["image"]["image_id"])
                    self.img_names.append(ann["image"]["file_name"])
            self.img_ids = np.array(self.img_ids)
            sorted_idx = np.argsort(self.img_ids)
            self.img_ids = self.img_ids[sorted_idx]
            self.img_names = np.array(self.img_names)[sorted_idx]
    
    def create_info_dict(self, sequence: str = None, date_created: str = None, skeleton2d: list = None, camera_intrinsics: list = None) -> dict:
        """
        Create a dictionary for the information object within of a SMPL annotation.
        
        args:
            sequence (int, optional): Sequence number of the annotation.
            date_created (str, optional): Date when the annotation was created.
            skeleton2d (list, optional): 2D skeleton save in the annotation.
            camera_intrinsics (list, optional): Camera intrinsics of the annotation.
            
        returns:
            dict: Information dictionary of the annotation.
        """
        info_dict = self.info_template.copy()
        info_dict["sequence"] = str(sequence) if sequence is not None else self.info_template["sequence"]
        info_dict["date_created"] = str(date_created) if date_created is not None else self.info_template["date_created"]
        info_dict["skeleton2d"] = list(skeleton2d) if skeleton2d is not None else self.info_template["skeleton2d"]
        info_dict["camera_intrinsics"] = list(camera_intrinsics) if camera_intrinsics is not None else self.info_template["camera_intrinsics"]
        
        return info_dict
    
    def create_image_dict(self, img_id: int, file_name, width: int, height: int) -> dict:
        """
        Create a dictionary for the image object within of a SMPL annotation.
        
        args:
            img_id (int): Image ID of the annotation.
            file_name (str): Image file name.
            width (int): Image width.
            height (int): Image height.
            
        returns:
            dict: Image dictionary of the annotation.
        """
        img_dict = self.image_template.copy()
        
        img_dict["image_id"] = int(img_id)
        img_dict["file_name"] = str(file_name)
        img_dict["width"] = int(width)
        img_dict["height"] = int(height)
        
        return img_dict
    
    def create_annotation_dict(self, idxs: list, keypoints: list, joints3d: list, tracking_ids: list, bboxes: list, body_pose: list = [], betas: list = [], global_orient: list = [], global_t: list = [], reprojection_errors: list = [], chamfer_distances: list = []) -> dict:
        """
        Create a dictionary for the annotation object within of a SMPL annotation.
        
        args:
            idxs (list): Index of the annotation within the dataloader (n).
            keypoints (list): 2D keypoints (..., 17, 3)
            joints3d (list): 3D joints (..., 23, 3)
            tracking_ids (list): Tracking IDs of the annotations.
            bboxes (list): Bounding boxes of the annotations (..., x, y, w, h).
            body_pose (list, optional): Body pose parameters (..., 69)
            betas (list, optional): Shape parameters (..., 10)
            global_orient (list, optional): Global orientation parameters (..., 3)
            global_t (list, optional): Global translation parameters (..., 3)
            vertices (list, optional): SMPL vertices (..., 6890, 3)
            
        returns:
            dict: Annotation dictionary of the annotation.
        """
        anns = {}
        for i in range(len(keypoints)):
            anns = copy.deepcopy(anns)
            ann_dict = self.annotation_template.copy()
            ann_dict["idx"] = int(idxs[i])
            ann_dict["bbox"] = list(bboxes[i])
            ann_dict["keypoints"] = list(keypoints[i])
            ann_dict["hmr"]["joints3d"] = list(joints3d[i])
            ann_dict["hmr"]["body_pose"] = list(body_pose[i]) if body_pose != [] else self.annotation_template["hmr"]["body_pose"]
            ann_dict["hmr"]["betas"] = list(betas[i]) if betas != [] else self.annotation_template["hmr"]["betas"]
            ann_dict["hmr"]["global_orient"] = list(global_orient[i]) if global_orient != [] else self.annotation_template["hmr"]["global_orient"]
            ann_dict["hmr"]["global_t"] = list(global_t[i]) if global_t != [] else self.annotation_template["hmr"]["global_t"]
            ann_dict["reprojection_error"] = float(reprojection_errors[i]) if reprojection_errors != [] else self.annotation_template["reprojection_error"]
            ann_dict["chamfer_distance"] = float(chamfer_distances[i]) if chamfer_distances != [] else self.annotation_template["chamfer_distance"]
            
            tracking_id = int(tracking_ids[i])
            anns[tracking_id] = ann_dict
        
        return anns
    
    def save(self, anns: list, save_dir: str) -> None:
        """
        Save the SMPL annotations to a JSON file.
        
        args:
            anns (list): List of SMPL annotations.
            save_dir (str): Directory to save the annotations.
        """
        os.makedirs(save_dir, exist_ok=True)
        
        for ann in anns:
            id = ann["image"]["image_id"]
            file_name = "smpl_" + str(id) + ".json"
            
            with open(os.path.join(save_dir, file_name), "w") as f:
                json.dump(ann, f, indent=4)
    
    def load(self, ann_file: int) -> dict:
        """
        Load the SMPL annotations from a JSON file.
        
        args:
            ann_file (str): Annotation file path.
            
        returns:
            dict: SMPL annotations.
        """
        
        return json.load(open(ann_file, "r"))
            
    
    def get_annotation(self, ann: dict, tracking_id: int = None) -> None or dict:
        """
        Get the annotation of a specific tracking ID.
        
        args:
            ann (dict): SMPL annotations.
            tracking_id (int): Tracking ID of the annotation.
            
        returns:
            dict: Annotation of the tracking ID.
        """
        for id, annotation in ann["annotations"].items():
            if int(id) == tracking_id:
                return annotation
        return None
    
    def get_smpl_vertices(self, anns: dict):
        betas = torch.tensor([ann["hmr"]["betas"] for ann in anns["annotations"].values()]).float().to(self.device)
        body_pose = torch.tensor([ann["hmr"]["body_pose"] for ann in anns["annotations"].values()]).float().to(self.device)
        global_orient = torch.tensor([ann["hmr"]["global_orient"] for ann in anns["annotations"].values()]).float().to(self.device)
        global_t = torch.tensor([ann["hmr"]["global_t"] for ann in anns["annotations"].values()]).float().to(self.device)
        
        smpl_output = self.smpl_model(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=global_t)
        vertices = smpl_output.vertices.cpu().numpy()
        
        return vertices
    
    def show_annotations(self, anns: dict, dataset, show_joints: bool = True, vis_dim: int = 2, show: bool = True) -> None or np.ndarray:
        """
        Show the SMPL annotations.
        
        args:
            anns (dict): SMPL annotations.
            show_joints (bool, optional): Show the joints in the visualization.
            vis_dimension (int, optional): Dimension of the visualization (2 or 3).
            show (bool, optional): Show visualization.
            
        returns:
            None or np.ndarray: Image with annotations.
        """
        
        if vis_dim == 3 and not show:
            raise ValueError("Cannot save 3D visualization.")
        
        sequence = anns["info"]["sequence"]
        img_dir = dataset.img_dirs[sequence]
        img = self.load_image(ann=anns, img_dir=img_dir)
        
        global_t = torch.tensor([ann["hmr"]["global_t"] for ann in anns["annotations"].values()]).float().to(self.device)
        distance = torch.norm(global_t, dim=1).view(1, -1)
        sorted_idx = torch.argsort(distance, descending=False).cpu().numpy()[0]
        sample_idx = np.array([ann["idx"] for ann in anns["annotations"].values()])[sorted_idx]
        
        pred_vertices = self.get_smpl_vertices(anns=anns)
        pred_vertices = pred_vertices[sorted_idx]
        
        camera_intrinsics = anns["info"]["camera_intrinsics"]
        tracking_ids = np.array(list(anns["annotations"].keys())).astype(int)[sorted_idx]
        mesh_colors = np.array([color_pallet[int(tracking_id) % len(color_pallet)] for tracking_id in list(anns["annotations"].keys())]).astype(int)[sorted_idx]
        if vis_dim == 2:
            annotations = np.array([ann for ann in anns["annotations"].values()]).ravel()[sorted_idx]

            for tracking_id, ann, color in zip(tracking_ids, annotations, mesh_colors):
                bbox_xywh = ann["bbox"]
                bbox_xyxy = np.array([[bbox_xywh[0], bbox_xywh[1], bbox_xywh[0] + bbox_xywh[2], bbox_xywh[1] + bbox_xywh[3]]])

                img = draw_2d_bbox(img=img, bboxes_xyxy=bbox_xyxy, detection_id=[tracking_id], show_id=False, show=False)
                if show_joints:
                    keypoints = np.array(ann["keypoints"])[:, :2]
                    skeleton = anns["info"]["skeleton2d"]
                    img = draw_2d_skeleton(img=img, joints2d=keypoints, skeleton=skeleton, colors=get_coco17_skeleton_colors(), show=False)

            renderer = PyRender(camera_intrinsics=camera_intrinsics, img_w=img.shape[1], img_h=img.shape[0], faces=self.smpl_model.faces)
            img = renderer.render_front_view(torch.from_numpy(pred_vertices), bg_img_rgb=img.copy(), mesh_colors_rgb=mesh_colors)
            renderer.delete()
        
        else:
            geoms = []
            for i, idx in enumerate(sample_idx):
                inputs, targets, meta_info = dataset.get_single_sample(idx=idx)
                point_cloud = inputs["point_cloud"].points_list()[0]
                
                point_cloud_geo = o3d.geometry.PointCloud()
                point_cloud_geo.points = o3d.utility.Vector3dVector(point_cloud.cpu().numpy().reshape(-1, 3))
                geoms.extend(draw_point_cloud(point_cloud_geo, id=str(i) + "_pc", show=False, color=np.array([0, 0, 0])))
                
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(pred_vertices[i])
                mesh.triangles = o3d.utility.Vector3iVector(self.smpl_model.faces)
                color = np.hstack((mesh_colors[i] / 255, 0.8))
                geoms.extend(draw_mesh(mesh=mesh, show=False, id=str(i) + "_pred", color=color))
                
                if show_joints:
                    joints3d = np.dot(self.J_regressor_humanM3, pred_vertices[i]).reshape(-1, 3)
                    joints3d = joints3d.reshape(-1, 3)
                    geoms.extend(draw_3d_kinematic_tree(joints3d=joints3d, skeleton=get_humanm3_skeleton(), colors=[[1, 0, 0]] * len(get_humanm3_skeleton()), id=str(i) + "_pred"))
                    
                    if "vertices" in targets:
                        vertices = targets["vertices"][0].cpu().numpy().reshape(-1, 3)
                        J_regressor = np.load(os.path.join(self.smpl_dir, "J_regressor_humanm3.npy"), allow_pickle=True)
                        
                        joints = np.dot(J_regressor, vertices)
                        skeleton = get_humanm3_skeleton()
                        geoms.extend(draw_3d_kinematic_tree(joints3d=joints, skeleton=skeleton, colors=[[0, 1, 0]] * len(skeleton), id=str(i) + "_GT"))
                    elif "joints3d" in targets:
                        joints = targets["joints3d"][0].cpu().numpy()
                        skeleton = get_humanm3_skeleton()
                        geoms.extend(draw_3d_kinematic_tree(joints3d=joints, skeleton=skeleton, colors=[[0, 1, 0]] * len(skeleton), id=str(i) + "_GT"))
                    
        if show:
            if vis_dim == 2:
                cv2.namedWindow("Predicted output", cv2.WINDOW_NORMAL)
                cv2.imshow("Predicted output", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                vis.draw(geoms, show_skybox=False)
        
        else:
            return img
    
    def load_image(self, ann: dict, img_dir: str) -> np.ndarray:
        """
        Load the image of the annotation.
        
        args:
            ann (dict): SMPL annotations.
            
        returns:
            np.ndarray: Image of the annotation.
        """
        img_name = ann["image"]["file_name"]
        img = cv2.imread(os.path.join(img_dir, img_name))
        
        return img


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--ann_file", type=str, required=True, help="Path to SMPL annotation file")
    parser.add_argument("--vis_dim", default=3, choices=[2, 3], type=int, help="Visualization dimension.")
    
    args = parser.parse_args()
    config = create_config_object(config_path=args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    from datasets.humanm3 import HumanM3
    from datasets.demo import DemoDataset
    from datasets.sloper4d import SLOPER4D
    
    if config.dataset.name == "demo":
        dataset = DemoDataset(data_root=args.data)
    elif config.dataset.name == "SLOPER4D":
        dataset = SLOPER4D(data_root=args.data, sequence=config.dataset.sequence)
    elif config.dataset.name == "HumanM3":
        dataset = HumanM3(data_root=args.data)
    else:
        raise ValueError("Invalid dataset name")
    
    smpl_ann = AnnotationsSMPL(smpl_dir=os.path.join(project_dir, "models/parametric_body_models/smpl"))
    anns = smpl_ann.load(ann_file=args.ann_file)
    smpl_ann.show_annotations(anns=anns, dataset=dataset, show_joints=True, show=True, vis_dim=args.vis_dim)
