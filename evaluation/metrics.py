""" This file contains the metrics used for evaluation.
Parts of the code are adapted from https://github.com/akanazawa/hmr and  https://github.com/akashsengupta1997/STRAPS-3DHumanShapePose
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

class Metrics:
    """
    Tracks metrics during evaluation.
    
    Options:                                                                                                    Requirements:
    - pve: Sum of per vertex errors.                                                                           | vertices
    - mpjpe: Sum of mean per joint position errors.                                                            | joints3d
    - mpjpe_pa: Sum of mean per joint position errors after Procrustes analysis.                               | joints3d
    - pck: Percentage of Correct Keypoints.                                                                    | joints3d
    - auc: Area under the curve of the PCK curve.                                                              | joints3d
    - mpere: Mean per edge position error.                                                                     | vertices
    """
    
    def __init__(self, metrics_to_track, dataset, save_path=None, align_root=False):
        self.metrics_to_track = metrics_to_track
        self.dataset = dataset
        self.sequences = [dataset.sequences] if isinstance(dataset.sequences, str) else dataset.sequences
        self.save_path = save_path
        
        num_joints = dataset.num_joints
        self.smpl_faces = dataset.smpl_model_neutral.faces
        
        self.possible_metrics = ["pve", "mpjpe", "mpjpe_pa", "pck", "auc", "mpere"]
        self.required_attributes = {"pve": "vertices", "mpjpe": "joints3d", "mpjpe_pa": "joints3d", "pck": "joints3d", "auc": "joints3d", "mpere": "vertices"}
        self.num_per_sample = {"pve": 6890, "mpjpe": num_joints, "mpjpe_pa": num_joints, "pck": num_joints, "auc": num_joints, "mpere": 13776}
        
        for metric in self.metrics_to_track:
            assert metric in self.possible_metrics, f"Invalid metric: {metric}"
        
        self.mpje_errors, self.per_frame_metrics, self.metric_sums, self.total_samples, self.sequence_metrics = {}, {}, {}, {}, {}
        self.align_root = align_root
        
        self.initialize()
    
    def initialize(self):
        for sequence in self.sequences:
            self.mpje_errors[sequence] = []
            self.total_samples[sequence] = 0
            
            metric_dict_zeros, metric_dict_lists = {}, {}
            for metric_type in self.metrics_to_track:
                metric_dict_zeros[metric_type] = 0
                metric_dict_lists[metric_type] = []
            
            self.metric_sums[sequence] = metric_dict_zeros
            self.sequence_metrics[sequence] = metric_dict_zeros
            self.per_frame_metrics[sequence] = metric_dict_lists
    
    
    def update_per_batch(self, outputs: dict, targets: dict, meta_info: dict):
        for metric in self.metrics_to_track:
            assert self.required_attributes[metric] in outputs.keys(), f"Attribute {self.required_attributes[metric]} not found in pred_dict."
            assert self.required_attributes[metric] in targets.keys(), f"Attribute {self.required_attributes[metric]} not found in target_dict."

        outputs, targets = self.dataset.evaluation_prep(outputs=outputs, targets=targets, align_root=self.align_root)
        
        # Convert tensors to numpy arrays
        for key in targets.keys():
            targets[key] = targets[key].cpu().numpy()
        for key in outputs.keys():
            outputs[key] = outputs[key].cpu().numpy()
        
        sequences = meta_info["sequence"]
        unique_sequences, counts = np.unique(sequences, return_counts=True)
        sequence_mapping = {i: np.where(sequences == i)[0] for i in unique_sequences}
        
        for i, sequence in enumerate(unique_sequences):
            self.total_samples[sequence] += counts[i]
        
        # -------- Update metrics sums --------
        if "pve" in self.metrics_to_track:
            pve_batch = np.linalg.norm(outputs["vertices"] - targets["vertices"], axis=-1)  # (bsize, 6890)
            
            for sequence in unique_sequences:
                indices = sequence_mapping[sequence]
                self.metric_sums[sequence]["pve"] += np.sum(pve_batch[indices])  # scalar
                self.per_frame_metrics[sequence]["pve"].extend(np.mean(pve_batch[indices], axis=-1))  # (bs,)
        
        # Mean per joint position error
        if "mpjpe" in self.metrics_to_track or "auc" in self.metrics_to_track or "pck" in self.metrics_to_track:
            mpjpe_batch = np.linalg.norm(outputs["joints3d"] - targets["joints3d"], axis=-1)  # (bsize, 14)

            for sequence in unique_sequences:
                indices = sequence_mapping[sequence]
                self.metric_sums[sequence]["mpjpe"] += np.sum(mpjpe_batch[indices])  # scalar
                self.mpje_errors[sequence].extend(mpjpe_batch[indices].ravel())
                self.per_frame_metrics[sequence]["mpjpe"].extend(np.mean(mpjpe_batch[indices], axis=-1))  # (bs,)
        
        # Mean per joint position error, Procrustes analysis
        if "mpjpe_pa" in self.metrics_to_track:
            pred_joints3d_pa = procrustes_analysis_batch(outputs["joints3d"], targets["joints3d"])
            mpjpe_pa_batch = np.linalg.norm(pred_joints3d_pa - targets["joints3d"], axis=-1)  # (bsize, 14)
            
            for sequence in unique_sequences:
                indices = sequence_mapping[sequence]
                self.metric_sums[sequence]["mpjpe_pa"] += np.sum(mpjpe_pa_batch[indices])  # scalar
                self.per_frame_metrics[sequence]["mpjpe_pa"].extend(np.mean(mpjpe_pa_batch[indices], axis=-1))  # (bs,)
        
        if "mpere" in self.metrics_to_track:
            v1_idx, v2_idx, v3_idx = self.smpl_faces[:, 0], self.smpl_faces[:, 1], self.smpl_faces[:, 2]
            pred_v1, pred_v2, pred_v3 = outputs["vertices"][:, v1_idx], outputs["vertices"][:, v2_idx], outputs["vertices"][:, v3_idx]
            target_v1, target_v2, target_v3 = targets["vertices"][:, v1_idx], targets["vertices"][:, v2_idx], targets["vertices"][:, v3_idx]
            
            pred_l1, pred_l2, pred_l3 = np.linalg.norm(pred_v1 - pred_v2, axis=-1), np.linalg.norm(pred_v3 - pred_v2, axis=-1), np.linalg.norm(pred_v3 - pred_v1, axis=-1)
            gt_l1, gt_l2, gt_l3 = np.linalg.norm(target_v1 - target_v2, axis=-1), np.linalg.norm(target_v3 - target_v2, axis=-1), np.linalg.norm(target_v3 - target_v1, axis=-1)
            diff1 = np.abs(pred_l1 - gt_l1)
            diff2 = np.abs(pred_l2 - gt_l2)
            diff3 = np.abs(pred_l3 - gt_l3)
            mpere = (diff1 / gt_l1 + diff2 / gt_l2 + diff3 / gt_l3) / 3
            
            for sequence in unique_sequences:
                indices = sequence_mapping[sequence]
                self.metric_sums[sequence]["mpere"] += np.sum(mpere[indices])
                self.per_frame_metrics[sequence]["mpere"].extend(np.mean(mpere[indices], axis=-1))
    
    def compute_metrics_per_sequence(self) -> dict:
        for sequence in self.sequences:
            metric_sums = self.metric_sums[sequence]
            total_samples = self.total_samples[sequence]
            mpje_errors = np.array(self.mpje_errors[sequence])
        
            if total_samples == 0:
                continue
                
            for metric_type in self.metrics_to_track:
                num_per_sample = self.num_per_sample[metric_type]
                
                if metric_type == "auc":
                    pck_aucs = []
                    threshold_range = np.arange(0, 200)
                    for pck_thr in threshold_range:
                        pck = self.compute_pck(errors=mpje_errors, threshold=pck_thr)
                        pck_aucs.append(pck)
                    
                    self.sequence_metrics[sequence][metric_type] = self.compute_auc(xpts=threshold_range / threshold_range.max(), ypts=pck_aucs)
                elif metric_type == "pck":
                    pck = self.compute_pck(errors=mpje_errors, threshold=50)
                    self.sequence_metrics[sequence][metric_type] = pck / (total_samples * num_per_sample)
                else:
                    self.sequence_metrics[sequence][metric_type] = metric_sums[metric_type] / (total_samples * num_per_sample)
    
    def compute_metrics(self):
        self.compute_metrics_per_sequence()
        
        combined_metrics = {}
        combined_metrics["sequences"] = self.sequences
        samples_per_sequence = np.array([self.total_samples[sequence] for sequence in self.sequences])
        for i, metric_type in enumerate(self.metrics_to_track):
            combined_metrics[metric_type] = np.average([self.sequence_metrics[sequence][metric_type] for sequence in self.sequences], weights=samples_per_sequence)
        
        self.combined_metrics = combined_metrics
    
    def save_metrics(self):
        for sequence in self.sequences:
            metrics = self.sequence_metrics[sequence]
            
            if self.per_frame_metrics is not None:
                for metric_type in self.metrics_to_track:
                    if len(self.per_frame_metrics[sequence][metric_type]) > 0:
                        metrics[metric_type + "_per_frame"] = np.asarray(self.per_frame_metrics[sequence][metric_type]).astype(float).tolist()
                        
            json_object = json.dumps(metrics, indent=4)
            save_dir = os.path.join(self.save_path, sequence)
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "metrics.json"), "w") as outfile:
                outfile.write(json_object)
        
        json_object = json.dumps(self.combined_metrics, indent=4)
        save_dir = os.path.join(self.save_path)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "metrics.json"), "w") as outfile:
            outfile.write(json_object)
        
        self.plot_metrics(sequences=self.sequences, save=True)
        
    
    def plot_metrics(self, sequences: list, save: bool=False):
        for sequence in sequences:
            plt.figure(figsize=(12, 8))
            
            for metric_type in self.metrics_to_track:
                if len(self.per_frame_metrics[sequence][metric_type]) > 0:
                    plt.plot(self.per_frame_metrics[sequence][metric_type], label=metric_type.replace("_", "-").upper())
            
            plt.title("Metrics per frame for sequence: {}".format(sequence))
            plt.xlabel("Instance")
            plt.ylabel("Error [mm]")
            plt.legend()
            plt.grid()
            
            if save:
                path_to_save = os.path.join(self.save_path, sequence, "metrics.png")
                plt.savefig(path_to_save)
                plt.close()
                
        if not save:
            plt.show()
    
    
    def compute_pck(self, errors, threshold):
        """
        Computes Percentage-Correct Keypoints
        :param errors: N x 12 x 1
        :param THRESHOLD: Threshold value used for PCK
        :return: the final PCK value
        """
        errors_pck = errors <= threshold
        
        return  np.mean(errors_pck)
    
    def compute_auc(self, xpts, ypts):
        """
        Calculates the AUC.
        :param xpts: Points on the X axis - the threshold values
        :param ypts: Points on the Y axis - the pck value for that threshold
        :return: The AUC value computed by integrating over pck values for all thresholds
        """
        a = np.min(xpts)
        b = np.max(xpts)
        from scipy import integrate
        myfun = lambda x: np.interp(x, xpts, ypts)
        auc = integrate.quad(myfun, a, b)[0]
        return auc
                

def compute_similarity_transform(S1, S2):
    """
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    """
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert(S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R"K) is R=U*V", where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale*(R.dot(mu1))

    # 7. Error:
    S1_hat = scale*R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


def procrustes_analysis_batch(S1, S2):
    """Batched version of compute_similarity_transform."""
    S1_hat = np.zeros_like(S1)
    for i in range(S1.shape[0]):
        S1_hat[i] = compute_similarity_transform(S1[i], S2[i])
    return S1_hat
