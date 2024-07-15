""" This file contains the source code to run SMPLify-3D on a given dataset. Native support for demo, SLOPER4D and HumanM3 datasets. """

import os
import sys

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(project_dir, "third_party/cliff"))
sys.path.append(project_dir)

import copy
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from datasets.humanm3 import HumanM3
from datasets.sloper4d import SLOPER4D
from datasets.demo import DemoDataset
from evaluation.metrics import Metrics
from torch.utils.data import DataLoader
from hmr.initial_smpl_prediction import CLIFF
from config.config import create_config_object
from optimizer.optimization import Optimization
import third_party.cliff.common.constants as constants
from models.parametric_body_models.smpl.smpl import SMPL

constants.SMPL_MODEL_DIR = os.path.join(project_dir, "models/parametric_body_models/smpl")


class SMPLify3D:
    def __init__(self, args, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.cliff_backbone = "hr48"
        self.cliff_checkpoints = args.cliff_checkpoints
        self.batch_size = self.config.dataset.batch_size
        
        self.smpl_model = SMPL(constants.SMPL_MODEL_DIR, gender="neutral").to(self.device)
        self.cliff = CLIFF(device=self.device, backbone=self.cliff_backbone, checkpoints=self.cliff_checkpoints)
        
    def inference(self, dataset):
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=dataset.collate_fn, num_workers=self.config.dataset.num_workers)
        
        if dataset.has_labels:
            metrics = Metrics(metrics_to_track=dataset.metrics, dataset=dataset, save_path=os.path.join(project_dir, "outputs", self.config.dataset.name), align_root=self.config.metrics.align_root)
            
        itter = 0
        final_batch = False
        for inputs, targets, meta_info in tqdm(data_loader):
            # inputs, targets, meta_info = dataset.get_single_sample(idx=809)
            cliff_poses, cliff_betas, cliff_global_t, cliff_joints, cliff_vertices = self.cliff.inference(inputs)
            
            batch_size = inputs["img_h"].shape[0]
            point_clouds = inputs["point_cloud"].to(self.device)
            joints2d = inputs["joints2d"].to(self.device).float()
            camera_intrinsics = inputs["camera_intrinsics"].to(self.device).float()
            mask_areas = inputs["mask_area"].to(self.device).float()
    
            optimization = Optimization(camera_intrinsics=camera_intrinsics, config=self.config)
            results = optimization(init_pose=cliff_poses.detach(), init_betas=cliff_betas.detach(), init_cam_t=cliff_global_t.detach(), goal_joints2d=joints2d, point_clouds=point_clouds, mask_areas=mask_areas)
    
            new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t, projection_errors = results
            
            with torch.no_grad():
                outputs = {}
                outputs["poses"] = new_opt_pose
                outputs["betas"] = new_opt_betas
                outputs["global_t"] = new_opt_cam_t
                outputs["joints3d"] = new_opt_joints
                outputs["projection_errors"] = projection_errors
                outputs["vertices"] = new_opt_vertices
    
                # dataset.visualize_output_sample(inputs=inputs, meta_info=meta_info, outputs=outputs, targets=targets, idx=0, vis_dimensions=2)
                # dataset.visualize_output_sample(inputs=inputs, meta_info=meta_info, outputs=outputs, targets=targets, idx=0, vis_dimensions=3)
                
                if dataset.has_labels:
                    metrics.update_per_batch(outputs=copy.deepcopy(outputs), targets=copy.deepcopy(targets), meta_info=copy.deepcopy(meta_info))
        
                if itter % self.config.save.every_n_batches == 0:
                    batched_inputs = inputs
                    batched_meta_info = meta_info
                    batched_outputs = outputs
    
                    del batched_inputs["point_cloud"]
                else:
                    for key in batched_inputs.keys():
                        batched_inputs[key] = torch.cat((batched_inputs[key], inputs[key]), dim=0)
                    for key in batched_meta_info.keys():
                        batched_meta_info[key] = np.concatenate((batched_meta_info[key], meta_info[key]), axis=0)
                    for key in batched_outputs.keys():
                        batched_outputs[key] = torch.cat((batched_outputs[key], outputs[key]), dim=0)
    
                if len(data_loader) == itter + 1:
                    final_batch = True
    
                if batch_size == self.config.save.every_n_batches * data_loader.batch_size and itter > 0 or itter == np.ceil(len(dataset) / data_loader.batch_size) - 1:
                    dataset.save_batch_HMR_results(inputs=copy.deepcopy(batched_inputs), outputs=copy.deepcopy(batched_outputs), meta_info=copy.deepcopy(batched_meta_info), final_batch=final_batch)
                torch.cuda.empty_cache()
                itter += 1
                
        if dataset.has_labels:
            metrics.compute_metrics()
            metrics.save_metrics()

            print("Final metrics:")
            for metric, value in metrics.combined_metrics.items():
                print(metric, value)
            print()


if __name__ == "__main__":
    cliff_checkpoints = os.path.join(project_dir, "models/cliff/hr48-PA43.0_MJE69.0_MVE81.2_3dpw.pt")
    
    parser = ArgumentParser()
    
    parser.add_argument("--data", type=str, required=True, help="Data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--cliff_checkpoints", type=str, default=cliff_checkpoints, help="CLIFF checkpoint file")

    args = parser.parse_args()
    config = create_config_object(config_path=args.config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.dataset.name == "demo":
        dataset = DemoDataset(data_root=args.data)
    elif config.dataset.name == "SLOPER4D":
        dataset = SLOPER4D(data_root=args.data, sequence=config.dataset.sequence)
    elif config.dataset.name == "HumanM3":
        dataset = HumanM3(data_root=args.data)
    else:
        raise ValueError("Invalid dataset name")
    
    smplify3d = SMPLify3D(args=args, config=config)
    smplify3d.inference(dataset=dataset)
    