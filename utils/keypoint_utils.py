""" This file contains keypoints and skeleton definitions for different formats. """

import numpy as np

def get_coco17_joint_names():
    return [
        'nose',  # 0
        'leye',  # 1
        'reye',  # 2
        'lear',  # 3
        'rear',  # 4
        'lshoulder',  # 5
        'rshoulder',  # 6
        'lelbow',  # 7
        'relbow',  # 8
        'lwrist',  # 9
        'rwrist',  # 10
        'lhip',  # 11
        'rhip',  # 12
        'lknee',  # 13
        'rknee',  # 14
        'lankle',  # 15
        'rankle',  # 16
    ]


def get_coco17_skeleton():
    return np.array(
        [
            [0, 1], # nose - leye
            [1, 3], # leye - lear
            [0, 2], # nose - reye
            [2, 4], # reye - rear
            [5, 6], # lshoulder - rshoulder
            [5, 11], # lshoulder - lhip
            [11, 12], # lhip - rhip
            [12, 6], # rhip - rshoulder
            [5, 7], # lshoulder - lelbow
            [7, 9], # lelbow - lwrist
            [6, 8], # rshoulder - relbow
            [8, 10], # relbow - rwrist
            [11, 13], # lhip - lknee
            [13, 15], # lknee - lankle
            [12, 14], # rhip - rknee
            [14, 16], # rknee - rankle
        ]
    )


def get_coco17_skeleton_colors():
    return [
        [50, 205, 50], # nose - leye
        [50, 205, 50], # leye - lear
        [255, 0, 0], # nose - reye
        [255, 0, 0], # reye - rear
        [0, 0, 255], # lshoulder - rshoulder
        [50, 205, 50], # lshoulder - lhip
        [0, 0, 255], # lhip - rhip
        [255, 0, 0], # rhip - rshoulder
        [50, 205, 50], # lshoulder - lelbow
        [50, 205, 50], # lelbow - lwrist
        [255, 0, 0], # rshoulder - relbow
        [255, 0, 0], # relbow - rwrist
        [50, 205, 50], # lhip - lknee
        [50, 205, 50], # lknee - lankle
        [255, 0, 0], # rhip - rknee
        [255, 0, 0], # rknee - rankle
    ]


def get_coco17_skeleton2body_parts():
    return {
        "head": [1, 2], # 0
        "chest": [5, 6], # 1
        "r_shoulder": [6], # 2
        "r_upper_arm": [6, 8], # 3
        "r_lower_arm": [8, 10], # 4
        "r_hand": [10], # 5
        "l_shoulder": [5], # 6
        "l_upper_arm": [5, 7], # 7
        "l_lower_arm": [7, 9], # 8
        "l_hand": [9], # 9
        "belly": [11, 12], # 10
        "hips": [11, 12], # 11
        "r_upper_leg": [12, 14], # 12
        "r_lower_leg": [14, 16], # 13
        "r_foot": [16], # 14
        "l_upper_leg": [11, 13], # 15
        "l_lower_leg": [13, 15], # 16
        "l_foot": [15], # 17
    }


def get_humanm3_joint_names():
    return [
        'pelvis', # 0
        'left_hip', # 1
        'right_hip', # 2
        'left_knee', # 3
        'right_knee', # 4
        'left_ankle', # 5
        'right_ankle', # 6
        'neck', # 7
        'head', # 8
        'left_shoulder', # 9
        'right_shoulder', # 10
        'left_elbow', # 11
        'right_elbow', # 12
        'left_wrist', # 13
        'right_wrist', # 14
    ]


def get_humanm3_skeleton():
    return np.array([
        [0, 1],
        [0, 2],
        [1, 3],
        [2, 4],
        [3, 5],
        [4, 6],
        [0, 7],
        [7, 8],
        [7, 9],
        [7, 10],
        [9, 11],
        [10, 12],
        [11, 13],
        [12, 14]
    ])


def get_smpl_joint_names():
    return [
        'hips',  # 0
        'leftUpLeg',  # 1
        'rightUpLeg',  # 2
        'spine',  # 3
        'leftLeg',  # 4
        'rightLeg',  # 5
        'spine1',  # 6
        'leftFoot',  # 7
        'rightFoot',  # 8
        'spine2',  # 9
        'leftToeBase',  # 10
        'rightToeBase',  # 11
        'neck',  # 12
        'leftShoulder',  # 13
        'rightShoulder',  # 14
        'head',  # 15
        'leftArm',  # 16
        'rightArm',  # 17
        'leftForeArm',  # 18
        'rightForeArm',  # 19
        'leftHand',  # 20
        'rightHand',  # 21
        'leftHandIndex1',  # 22
        'rightHandIndex1',  # 23
    ]


def get_smpl_skeleton():
    return np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 4],
            [2, 5],
            [3, 6],
            [4, 7],
            [5, 8],
            [6, 9],
            [7, 10],
            [8, 11],
            [9, 12],
            [9, 13],
            [9, 14],
            [12, 15],
            [13, 16],
            [14, 17],
            [16, 18],
            [17, 19],
            [18, 20],
            [19, 21],
            [20, 22],
            [21, 23],
        ]
    )


def get_smpl_parts2joints():
    return {
        "head": [],
        "chest": [],
        "r_shoulder": [],
        "r_upper_arm": [13],
        "r_lower_arm": [16, 18],
        "r_hand": [20, 22],
        "l_shoulder": [],
        "l_upper_arm": [12],
        "l_lower_arm": [15, 17],
        "l_hand": [19, 21],
        "belly": [],
        "hips": [],
        "r_upper_leg": [1],
        "r_lower_leg": [3],
        "r_foot": [7, 10],
        "l_upper_leg": [0],
        "l_lower_leg": [2],
        "l_foot": [6, 9],
    }


def get_body_part_id2smpl_pose_idx():
    return {
        0: 12,
        2: 14,
        3: 17,
        4: 19,
        5: 21,
        6: 13,
        7: 16,
        8: 18,
        9: 20,
        12: 2,
        13: 5,
        14: 8,
        15: 1,
        16: 4,
        17: 7,
    }
