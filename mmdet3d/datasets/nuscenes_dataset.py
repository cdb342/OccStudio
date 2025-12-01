# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Dubing Chen
import tempfile
from os import path as osp
import os
import mmcv
import numpy as np
import pyquaternion
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box as NuScenesBox
from .utils import nuscenes_get_rt_matrix
from ..core import show_result
from ..core.bbox import Box3DMode, Coord3DMode, LiDARInstance3DBoxes
from .builder import DATASETS
from .custom_3d import Custom3DDataset
from .pipelines import Compose
from tqdm import tqdm
import csv
import math
import torch
from .ego_pose_extractor import EgoPoseDataset
from torch.utils.data import DataLoader
from .ray_metrics import main as ray_based_miou
from .ray_metrics import process_one_sample, generate_lidar_rays
import pickle, gzip
from mmcv.parallel.data_container import DataContainer
import copy
from .rayiou_metrics import main_rayiou, main_raypq
from .ego_pose_dataset import EgoPoseDatasetOcc3D
occ3d_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]


def trans_matrix(T, R):
    tm = np.eye(4)
    tm[:3, :3] = R.rotation_matrix
    tm[:3, 3] = T
    return tm
@DATASETS.register_module()
class NuScenesDataset(Custom3DDataset):
    r"""NuScenes Dataset.

    This class serves as the API for experiments on the NuScenes Dataset.

    Please refer to `NuScenes Dataset <https://www.nuscenes.org/download>`_
    for data downloading.

    Args:
        ann_file (str): Path of annotation file.
        pipeline (list[dict], optional): Pipeline used for data processing.
            Defaults to None.
        data_root (str): Path of dataset root.
        classes (tuple[str], optional): Classes used in the dataset.
            Defaults to None.
        load_interval (int, optional): Interval of loading the dataset. It is
            used to uniformly sample the dataset. Defaults to 1.
        with_velocity (bool, optional): Whether include velocity prediction
            into the experiments. Defaults to True.
        modality (dict, optional): Modality to specify the sensor data used
            as input. Defaults to None.
        box_type_3d (str, optional): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes.
            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool, optional): Whether to filter empty GT.
            Defaults to True.
        test_mode (bool, optional): Whether the dataset is in test mode.
            Defaults to False.
        eval_version (bool, optional): Configuration version of evaluation.
            Defaults to  'detection_cvpr_2019'.
        use_valid_flag (bool, optional): Whether to use `use_valid_flag` key
            in the info file as mask to filter gt_boxes and gt_names.
            Defaults to False.
        img_info_prototype (str, optional): Type of img information.
            Based on 'img_info_prototype', the dataset will prepare the image
            data info in the type of 'mmcv' for official image infos,
            'bevdet' for BEVDet, and 'bevdet4d' for BEVDet4D.
            Defaults to 'mmcv'.
        multi_adj_frame_id_cfg (tuple[int]): Define the selected index of
            reference adjcacent frames.
        ego_cam (str): Specify the ego coordinate relative to a specified
            camera by its name defined in NuScenes.
            Defaults to None, which use the mean of all cameras.
    """
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    DefaultAttribute = {
        'car': 'vehicle.parked',
        'pedestrian': 'pedestrian.moving',
        'trailer': 'vehicle.parked',
        'truck': 'vehicle.parked',
        'bus': 'vehicle.moving',
        'motorcycle': 'cycle.without_rider',
        'construction_vehicle': 'vehicle.parked',
        'bicycle': 'cycle.without_rider',
        'barrier': '',
        'traffic_cone': '',
    }
    AttrMapping = {
        'cycle.with_rider': 0,
        'cycle.without_rider': 1,
        'pedestrian.moving': 2,
        'pedestrian.standing': 3,
        'pedestrian.sitting_lying_down': 4,
        'vehicle.moving': 5,
        'vehicle.parked': 6,
        'vehicle.stopped': 7,
    }
    AttrMapping_rev = [
        'cycle.with_rider',
        'cycle.without_rider',
        'pedestrian.moving',
        'pedestrian.standing',
        'pedestrian.sitting_lying_down',
        'vehicle.moving',
        'vehicle.parked',
        'vehicle.stopped',
    ]
    # https://github.com/nutonomy/nuscenes-devkit/blob/57889ff20678577025326cfc24e57424a829be0a/python-sdk/nuscenes/eval/detection/evaluate.py#L222 # noqa
    ErrNameMapping = {
        'trans_err': 'mATE',
        'scale_err': 'mASE',
        'orient_err': 'mAOE',
        'vel_err': 'mAVE',
        'attr_err': 'mAAE'
    }
    CLASSES = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
               'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
               'barrier')

    def __init__(self,
                 ann_file=None,
                 pipeline=None,
                 data_root=None,
                 classes=None,
                 load_interval=1,
                 with_velocity=True,
                 modality=None,
                 box_type_3d='LiDAR',
                 filter_empty_gt=True,
                 test_mode=False,
                 eval_version='detection_cvpr_2019',
                 use_valid_flag=False,
                 img_info_prototype='mmcv',
                 multi_adj_frame_id_cfg=None,
                 occupancy_path='/mount/dnn_data/occupancy_2023/gts',
                 ego_cam='CAM_FRONT',
                 # SOLLOFusion
                 use_sequence_group_flag=False,
                 sequences_split_num=1,
                 stereo=False,
                 openocc=False,
                 use_corner_case_data=None,
                 corner_case_degree=False,
                 eval_mask=False,
                 occupancy_path2='data/nuscenes/gts',
                 gts_surroundocc=False,
                 occ_size=[200,200,16],
                 cal_metric_in_model=False,
                 class_names=['others','barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
                            'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
                            'driveable_surface', 'other_flat', 'sidewalk',
                            'terrain', 'manmade', 'vegetation','free'],
                 dynamic_object_idx=[2,3,4,5,6,7,9,10],
                ):
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            classes=classes,
            modality=modality,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode)
        self.occupancy_path = occupancy_path
        self.occupancy_path2=occupancy_path2
        
        self.with_velocity = with_velocity
        self.eval_version = eval_version
        from nuscenes.eval.detection.config import config_factory

        self.eval_detection_configs = config_factory(self.eval_version)
        if self.modality is None:
            self.modality = dict(
                use_camera=False,
                use_lidar=True,
                use_radar=False,
                use_map=False,
                use_external=False,
            )


        self.img_info_prototype = img_info_prototype
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.ego_cam = ego_cam
        self.stereo = stereo

        # SOLOFusion
        self.use_sequence_group_flag = use_sequence_group_flag
        self.sequences_split_num = sequences_split_num
        # sequences_split_num splits eacgh sequence into sequences_split_num parts.
        
        if self.use_sequence_group_flag:
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.
            
        #########
        self.openocc=openocc
        if openocc:
            self.evaluate=self.evaluate_miou
            self.scene_frames = {}
            for info in self.data_infos:
                scene_token = self.get_scene_token(info)
                if scene_token not in self.scene_frames:
                    self.scene_frames[scene_token] = []
                self.scene_frames[scene_token].append(info)
                ############################
        self.use_corner_case_data=use_corner_case_data
        self.corner_case_degree=corner_case_degree
        if use_corner_case_data:

            self.update_data_path()
        self.eval_mask=eval_mask
        self.gts_surroundocc=gts_surroundocc
        self.occ_size=occ_size
        self.cal_metric_in_model=cal_metric_in_model
        self.class_names=class_names
        self.dynamic_object_idx=np.array(dynamic_object_idx)
    def update_data_path(self,):
        for info in  self.data_infos:
            for cam in info['cams']:
                data_path=info['cams'][cam]['data_path']
                revised_data_path=data_path.replace('samples',osp.join(self.use_corner_case_data,self.corner_case_degree))
                info['cams'][cam]['data_path']=revised_data_path
    def get_scene_token(self, info):
        # if self.dataset_type == 'openocc_v2':
            # meta info of openocc_v2 don't have scene_token
            # extract scene name from 'occ_path' instead
            # if the custom data info contains scene_token, we just use it.
        if 'scene_token' in info:
            scene_name = info['scene_token']
        else:
            scene_name = info['occ_path'].split('openocc_v2/')[-1].split('/')[0]
        return scene_name

    def get_ego_from_lidar(self, info):
        ego_from_lidar = trans_matrix(
            np.array(info['lidar2ego_translation']), 
            Quaternion(info['lidar2ego_rotation']))

        return ego_from_lidar

    def get_global_pose(self, info, inverse=False):

        global_from_ego = trans_matrix(
            np.array(info['ego2global_translation']), 
            Quaternion(info['ego2global_rotation']))
        ego_from_lidar = trans_matrix(
            np.array(info['lidar2ego_translation']), 
            Quaternion(info['lidar2ego_rotation']))
        pose = global_from_ego.dot(ego_from_lidar)
        if inverse:
            pose = np.linalg.inv(pose)
        return pose
    def get_origin(self, idx):
        info = self.data_infos[idx]
        ref_sample_token = info['token']
        ref_lidar_from_global = self.get_global_pose(info, inverse=True)
        ref_ego_from_lidar = self.get_ego_from_lidar(info)

        scene_token = self.get_scene_token(info)
        scene_frame = self.scene_frames[scene_token]
        ref_index = scene_frame.index(info)

        # NOTE: getting output frames
        output_origin_list = []
        for curr_index in range(len(scene_frame)):
            # if this exists a valid target
            if scene_frame[curr_index]['token'] == info['token'] :
                origin_tf = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            else:
                # transform from the current lidar frame to global and then to the reference lidar frame
                global_from_curr = self.get_global_pose(scene_frame[curr_index], inverse=False)
                ref_from_curr = ref_lidar_from_global.dot(global_from_curr)
                origin_tf = np.array(ref_from_curr[:3, 3], dtype=np.float32)

            origin_tf_pad = np.ones([4])
            origin_tf_pad[:3] = origin_tf  # pad to [4]
            origin_tf = np.dot(ref_ego_from_lidar[:3], origin_tf_pad.T).T  # [3]

            # origin
            if np.abs(origin_tf[0]) < 39 and np.abs(origin_tf[1]) < 39:
                output_origin_list.append(origin_tf)
        # select 8 origins
        if len(output_origin_list) > 8:
            select_idx = np.round(np.linspace(0, len(output_origin_list) - 1, 8)).astype(np.int64)
            output_origin_list = [output_origin_list[i] for i in select_idx]

        output_origin_tensor = torch.from_numpy(np.stack(output_origin_list))  # [T, 3]
        return DataContainer(output_origin_tensor)

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info['valid_flag']
            gt_names = set(info['gt_names'][mask])
        else:
            gt_names = set(info['gt_names'])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        
        data = mmcv.load(ann_file, file_format='pkl')
        data_infos = list(sorted(data['infos'], key=lambda e: e['timestamp']))
        data_infos = data_infos[::self.load_interval]#[13400:]#[::-1]
        
        self.metadata = data['metadata']
        self.version = self.metadata['version']
        return data_infos

    def process_video_sequences(self,data_info):
        processed_data = []
        current_sequence = []
        current_scene = None

        for frame in data_info:
            if frame['scene_name'] != current_scene:
                if current_sequence:
                    # Process the previous sequence
                    if len(current_sequence) < 40:
                        # Pad with the last frame
                        current_sequence.extend([current_sequence[-1]] * (40 - len(current_sequence)))
                    elif len(current_sequence) > 40:
                        # Truncate to 40 frames
                        current_sequence = current_sequence[:40]
                    processed_data.extend(current_sequence)

                # Start a new sequence
                current_sequence = [frame]
                current_scene = frame['scene_name']
            else:
                current_sequence.append(frame)

        # Process the last sequence
        if current_sequence:
            if len(current_sequence) < 40:
                current_sequence.extend([current_sequence[-1]] * (40 - len(current_sequence)))
            elif len(current_sequence) > 40:
                current_sequence = current_sequence[:40]
            processed_data.extend(current_sequence)

        return processed_data

    def verify_processed_data(self,original_data, processed_data):
        def group_by_scene(data):
            scenes = {}
            for frame in data:
                scenes.setdefault(frame['scene_name'], []).append(frame)
            return scenes

        original_scenes = group_by_scene(original_data)
        processed_scenes = group_by_scene(processed_data)

        for scene_name, processed_frames in processed_scenes.items():
            original_frames = original_scenes[scene_name]

            # Check 1: Ensure each sequence has exactly 40 frames
            if len(processed_frames) != 40:
                print(f"Error: Sequence {scene_name} has {len(processed_frames)} frames instead of 40.")
                return False

            # Check 2: Ensure frames with the same scene_name are consecutive
            if any(frame['scene_name'] != scene_name for frame in processed_frames):
                print(f"Error: Sequence {scene_name} contains frames from different scenes.")
                return False

            # Check 3 & 4: Verify correct handling of short and long sequences
            if len(original_frames) < 40:
                # Check if the last frame is correctly duplicated
                if processed_frames[-1] != original_frames[-1]:
                    print(f"Error: Last frame of short sequence {scene_name} is not correctly duplicated.")
                    return False
            else:
                # Check if only the first 40 frames are kept
                if processed_frames != original_frames[:40]:
                    print(f"Error: Long sequence {scene_name} is not correctly truncated.")
                    return False

        print("All checks passed. The processed data is correct.")
        return True



    def _set_sequence_group_flag(self):
        """
        Set each sequence to be a different group
        """
            
        res = []
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            if idx != 0 and self.data_infos[idx]['scene_name'] !=self.data_infos[idx-1]['scene_name']:
                # Not first frame and # of sweeps is 0 -> new sequence
                curr_sequence += 1
            res.append(curr_sequence)

        self.flag = np.array(res, dtype=np.int64)

        if self.sequences_split_num != 1:
            if self.sequences_split_num == 'all':
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0
                for curr_flag in range(len(bin_counts)):
                    
                    curr_sequence_length = np.array(
                        list(range(0, bin_counts[curr_flag],math.ceil(bin_counts[curr_flag] / self.sequences_split_num)))
                        + [bin_counts[curr_flag]])
                    for sub_seq_idx in (curr_sequence_length[1:] - curr_sequence_length[:-1]):
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        curr_new_flag += 1

                assert len(new_flags) == len(self.flag)
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocol modified from SECOND.Pytorch
        input_dict = dict(
            index=index,
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            scene_name=info['scene_name'],
             scene_token=info['scene_token'],
            timestamp=info['timestamp'] / 1e6,
            lidarseg_filename=info.get('lidarseg_filename', 'None') 
        )
        if 'ann_infos' in info:
            input_dict['ann_infos'] = info['ann_infos']
            
        if self.modality['use_camera']:
            if self.img_info_prototype == 'mmcv':
                image_paths = []
                lidar2img_rts = []

                for cam_type, cam_info in info['cams'].items():
                    image_paths.append(cam_info['data_path'])
                    
                    # obtain lidar to image transformation matrix
                    lidar2cam_r = np.linalg.inv(
                        cam_info['sensor2lidar_rotation'])
                    lidar2cam_t = cam_info[
                        'sensor2lidar_translation'] @ lidar2cam_r.T
                    lidar2cam_rt = np.eye(4)
                    lidar2cam_rt[:3, :3] = lidar2cam_r.T
                    lidar2cam_rt[3, :3] = -lidar2cam_t
                    intrinsic = cam_info['cam_intrinsic']
                    viewpad = np.eye(4)
                    viewpad[:intrinsic.shape[0], :intrinsic.
                            shape[1]] = intrinsic
                    lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                    lidar2img_rts.append(lidar2img_rt)
                    cam_position = np.linalg.inv(lidar2cam_rt.T) @ np.array([0., 0., 0., 1.]).reshape([4, 1])
                    cam_positions.append(cam_position.flatten()[:3])
                   

                input_dict.update(
                    dict(
                        
                        img_filename=image_paths,
                        lidar2img=lidar2img_rts,
                    ))

                if not self.test_mode:
                    annos = self.get_ann_info(index)
                    input_dict['ann_info'] = annos
            else:   
                assert 'bevdet' in self.img_info_prototype
                input_dict.update(dict(curr=info))
                if '4d' in self.img_info_prototype:
                    info_adj_list = self.get_adj_info(info, index)
                    input_dict.update(dict(adjacent=info_adj_list))
            if self.use_sequence_group_flag:
                input_dict['sample_index'] = index
                input_dict['sequence_group_idx'] = self.flag[index]
                input_dict['start_of_sequence'] = index == 0 or self.flag[index - 1] != self.flag[index]
                # Get a transformation matrix from current keyframe lidar to previous keyframe lidar
                # if they belong to same sequence.
                input_dict['end_of_sequence'] = index == len(self.data_infos)-1 or  self.flag[index ] != self.flag[index+1]
                # print(len(self.data_infos)-1,111111111111111)
                input_dict['nuscenes_get_rt_matrix'] = dict(
                    lidar2ego_rotation = self.data_infos[index]['lidar2ego_rotation'],
                    lidar2ego_translation = self.data_infos[index]['lidar2ego_translation'],
                    ego2global_rotation = self.data_infos[index]['ego2global_rotation'],
                    ego2global_translation = self.data_infos[index]['ego2global_translation'],
                )
                if not input_dict['start_of_sequence']:
                    input_dict['curr_to_prev_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index - 1],
                        "lidar", "lidar"))
                    input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index - 1], self.data_infos[index],
                        "lidar", "global")) # TODO: Note that global is same for all.
                    input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index - 1],
                        "ego", "ego"))
                    input_dict['prev_occ_gt_path']=self.data_infos[index-1]['occ_path']
                else:
                    input_dict['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                    input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix( 
                        self.data_infos[index], self.data_infos[index], "lidar", "global")
                        )
                    input_dict['curr_to_prev_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index],
                        "ego", "ego"))
                    input_dict['prev_occ_gt_path']=None
                input_dict['global_to_curr_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                    self.data_infos[index], self.data_infos[index],
                    "global", "lidar"))
                if not input_dict['end_of_sequence']:
                    input_dict['curr_to_next_lidar_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index + 1],
                        "lidar", "lidar"))
                    input_dict['next_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index + 1], self.data_infos[index],
                        "lidar", "global"))
                    input_dict['curr_to_next_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index + 1],
                        "ego", "ego"))
                    input_dict['next_occ_gt_path']=self.data_infos[index+1]['occ_path']
                else:
                    input_dict['curr_to_next_lidar_rt'] = torch.eye(4).float()
                    input_dict['next_lidar_to_global_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index], "lidar", "global"))
                    input_dict['curr_to_next_ego_rt'] = torch.FloatTensor(nuscenes_get_rt_matrix(
                        self.data_infos[index], self.data_infos[index],
                        "ego", "ego"))
                    input_dict['next_occ_gt_path']=None
                
        input_dict['occ_gt_path'] = self.data_infos[index]['occ_path']   
        if self.openocc:
            input_dict['ray_origin']=self.get_origin(index)     
            scene_frame = self.scene_frames[info['scene_token']]
            input_dict['scene_frame'] = scene_frame
        return input_dict

    def get_adj_info(self, info, index):
        info_adj_list = []
        adj_id_list = list(range(*self.multi_adj_frame_id_cfg))
        if self.stereo:
            assert self.multi_adj_frame_id_cfg[0] == 1
            assert self.multi_adj_frame_id_cfg[2] == 1
            adj_id_list.append(self.multi_adj_frame_id_cfg[1])
        for select_id in adj_id_list:
            select_id = max(index - select_id, 0)
            if not self.data_infos[select_id]['scene_token'] == info[
                    'scene_token']:
                info_adj_list.append(info)
            else:
                info_adj_list.append(self.data_infos[select_id])
        return info_adj_list
    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info['valid_flag']
        else:
            mask = info['num_lidar_pts'] > 0
        gt_bboxes_3d = info['gt_boxes'][mask]
        gt_names_3d = info['gt_names'][mask]
        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info['gt_velocity'][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
        # the same as KITTI (0.5, 0.5, 0)
        gt_bboxes_3d = LiDARInstance3DBoxes(
            gt_bboxes_3d,
            box_dim=gt_bboxes_3d.shape[-1],
            origin=(0.5, 0.5, 0.5)).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d)
        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES
       
        print('Start to convert detection format...')
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            boxes = det['boxes_3d'].tensor.numpy()
            scores = det['scores_3d'].numpy()
            labels = det['labels_3d'].numpy()
            sample_id = det.get('index', sample_id)
            # from IPython import embed
            # embed()
            # exit()

            sample_token = self.data_infos[sample_id]['token']
            
            trans = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_translation']
            rot = self.data_infos[sample_id]['cams'][
                self.ego_cam]['ego2global_rotation']
            rot = pyquaternion.Quaternion(rot)
            annos = list()
            for i, box in enumerate(boxes):
                name = mapped_class_names[labels[i]]
                center = box[:3]
                wlh = box[[4, 3, 5]]
                box_yaw = box[6]
                box_vel = box[7:].tolist()
                box_vel.append(0)
                quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw)
                nusc_box = NuScenesBox(center, wlh, quat, velocity=box_vel)
                nusc_box.rotate(rot)
                nusc_box.translate(trans)
                if np.sqrt(nusc_box.velocity[0]**2 +
                           nusc_box.velocity[1]**2) > 0.2:
                    if name in [
                            'car',
                            'construction_vehicle',
                            'bus',
                            'truck',
                            'trailer',
                    ]:
                        attr = 'vehicle.moving'
                    elif name in ['bicycle', 'motorcycle']:
                        attr = 'cycle.with_rider'
                    else:
                        attr = self.DefaultAttribute[name]
                else:
                    if name in ['pedestrian']:
                        attr = 'pedestrian.standing'
                    elif name in ['bus']:
                        attr = 'vehicle.stopped'
                    else:
                        attr = self.DefaultAttribute[name]
                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=nusc_box.center.tolist(),
                    size=nusc_box.wlh.tolist(),
                    rotation=nusc_box.orientation.elements.tolist(),
                    velocity=nusc_box.velocity[:2],
                    detection_name=name,
                    detection_score=float(scores[i]),
                    attribute_name=attr,
                )
                annos.append(nusc_anno)
            # other views results of the same frame should be concatenated
            if sample_token in nusc_annos:
                pass
                # nusc_annos[sample_token].extend(annos)
            else:
                nusc_annos[sample_token] = annos
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(self,
                         result_path,
                         logger=None,
                         metric='bbox',
                         result_name='pts_bbox'):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str, optional): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str, optional): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        from nuscenes import NuScenes
        from nuscenes.eval.detection.evaluate import NuScenesEval

        output_dir = osp.join(*osp.split(result_path)[:-1])

        nusc = NuScenes(
            version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        # import pdb;pdb.set_trace()
        nusc_eval = NuScenesEval(
            nusc,
            config=self.eval_detection_configs,
            result_path=result_path,
            eval_set=eval_set_map[self.version],
            output_dir=output_dir,
            verbose=False)
        nusc_eval.main(render_curves=False)

        # record metrics
        metrics = mmcv.load(osp.join(output_dir, 'metrics_summary.json'))
        detail = dict()
        metric_prefix = f'{result_name}_NuScenes'
        for name in self.CLASSES:
            for k, v in metrics['label_aps'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['label_tp_errors'][name].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
            for k, v in metrics['tp_errors'].items():
                val = float('{:.4f}'.format(v))
                detail['{}/{}'.format(metric_prefix,
                                      self.ErrNameMapping[k])] = val

        detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
        detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']
        return detail

    def evaluate(self, results,
                       logger=None,
                        metric='bbox',
                        jsonfile_prefix='test',
                        result_names=['pts_bbox'],
                        show=False,
                        out_dir=None,
                        pipeline=None,
                        save=False,
                        use_image_mask=True,
                        **kwargs,
                        ):
            results_dict = {}
            
            if results[0].get('pred_occupancy', None) is not None:
                results_dict.update(self.evaluate_occupancy(results,use_image_mask=use_image_mask, show_dir=jsonfile_prefix, save=save, **kwargs))
                
            if results[0].get('iou', None) is not None:
                results_dict.update(self.evaluate_mask(results))
            
            if results[0].get('pts_bbox', None) is not None:
                results_dict.update(self.evaluate_bbox(results, logger=logger,
                        metric=metric,
                        jsonfile_prefix=jsonfile_prefix,
                        result_names=result_names,
                        show=show,
                        out_dir=out_dir,
                        pipeline=pipeline))
            mmcv.mkdir_or_exist(jsonfile_prefix)
            with open(osp.join(jsonfile_prefix, 'results.csv'), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(results_dict.keys()))
                writer.writerow(list(results_dict.values()))

            return results_dict
    
    def evaluate_rayiou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts, occ_preds, inst_gts, inst_preds, lidar_origins = [], [], [], [], []
        print('\nStarting Evaluation...')

        sample_tokens = [info['token'] for info in self.data_infos]
       
        data_loader = DataLoader(
            EgoPoseDatasetOcc3D(self.data_infos), num_workers=8)
        
        print('\nStarting Evaluation...')
        processed_set = set()
        for occ_pred_w_index in tqdm(occ_results):

            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)
            
            output_origin=data_loader.dataset[index][1].unsqueeze(0)
            
            info = self.data_infos[index]

            occ_path = os.path.join(self.occupancy_path, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']

            sem_pred = occ_pred_w_index['pred_occupancy']  # [B, N]
            occ_class_names = occ3d_class_names

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(sem_pred)
        
        if len(inst_preds) > 0:
            results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins, occ_class_names=occ_class_names)
            results.update(main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names))
            return results
        else:
            return main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names)  
            
    def evaluate_occupancy(self, occ_results, runner=None, show_dir=None, save=False,use_image_mask=True, **eval_kwargs):
        from .occ_metrics import Metric_mIoU, Metric_FScore

        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)

        self.eval_fscore = False
        if  self.eval_fscore:
            self.fscore_eval_metrics = Metric_FScore(
                leaf_size=10,
                threshold_acc=0.4,
                threshold_complete=0.4,
                voxel_size=[0.4, 0.4, 0.4],
                range=[-40, -40, -1, 40, 40, 5.4],
                void=[17, 255],
                use_lidar_mask=False,
                use_image_mask=True,
            )
                
        count = 0
        print('\nStarting Evaluation...')
        processed_set = set()
        semantic_iou=[]
        occ_iou=[]

        for occ_pred_w_index in tqdm(occ_results):
            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)

            if self.cal_metric_in_model:
                semantic_iou.append(occ_pred_w_index['semantic_iou'])
                occ_iou.append(occ_pred_w_index['occ_iou'])
            else:
                occ_pred = occ_pred_w_index['pred_occupancy']
                info = self.data_infos[index]
                scene_name = info['scene_name']
                sample_token = info['token']
                
                if self.gts_surroundocc:
                    lidar_path=info['lidar_path'] 
                    occupancy_file_path=lidar_path.replace('samples/LIDAR_TOP','gts_surroundocc/samples')
                    occupancy_file_path=occupancy_file_path+'.npy'
                    data = np.load(occupancy_file_path)

                    occ = np.zeros(self.occ_size)
                    occ[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]
                    occ = np.where(occ== 0, 17, occ)

                    gt_semantics=occ
                    mask_camera=np.ones_like(gt_semantics).astype(bool)
                    mask_lidar=np.ones_like(gt_semantics).astype(bool)
                else:
                    occupancy_file_path = osp.join(self.occupancy_path, scene_name, sample_token, 'labels.npz')
                    occ_gt = np.load(occupancy_file_path)
        
                    gt_semantics = occ_gt['semantics']
                    if self.eval_mask:
                        occupancy_file_path2 = osp.join(self.occupancy_path2, scene_name, sample_token, 'labels.npz')
                        occ_gt2 = np.load(occupancy_file_path2)
                        gt_semantics2=occ_gt2['semantics']
                        diff_mask=gt_semantics!=gt_semantics2
                        mask_camera=mask_camera&(~diff_mask)
                    else:
                        mask_lidar = occ_gt['mask_lidar'].astype(bool)
                        
                        mask_camera = occ_gt['mask_camera'].astype(bool)       
                    if use_image_mask:
                        mask_camera = mask_camera
                    else:
                        mask_camera = np.ones_like(mask_camera).astype(bool)

                self.occ_eval_metrics.add_batch(occ_pred[mask_camera], gt_semantics, mask_lidar, mask_camera)
                if self.eval_fscore:
                    self.fscore_eval_metrics.add_batch(occ_pred[mask_camera], gt_semantics, mask_lidar, mask_camera)


        if self.cal_metric_in_model:
            res = {}
            num_samples=len(semantic_iou)
            semantic_iou=np.stack(semantic_iou)
            semantic_iou=semantic_iou.sum(0)
            semantic_iou=semantic_iou[:,0]/(semantic_iou[:,1]+semantic_iou[:,2]-semantic_iou[:,0]+1e-8)
            if self.gts_surroundocc:
                semantic_iou=semantic_iou[1:]
                class_names=self.class_names[1:]
                dynamic_object_idx=self.dynamic_object_idx-1
            else:
                class_names=self.class_names
                dynamic_object_idx=self.dynamic_object_idx
            print(f'===> per class IoU of {num_samples} samples:')
            #miou
            for ind_class in range(len(semantic_iou)):
                print(f'===> {class_names[ind_class]} - IoU = ' + str(round(semantic_iou[ind_class] * 100, 4)))
                res[class_names[ind_class]] = round(semantic_iou[ind_class] * 100, 2)

            print(f'===> mIoU of {num_samples} samples: ' + str(round(np.nanmean(semantic_iou[:len(semantic_iou)-1]) * 100, 2)))
            res['Overall mIoU'] =  round(np.nanmean(semantic_iou[:len(semantic_iou)-1]) * 100, 2)
            
            #miou_D
            print(f'===> mIoU_D = ' + str(round(np.nanmean(semantic_iou[dynamic_object_idx]) * 100, 4)))
            res['mIoU_D'] =  round(np.nanmean(semantic_iou[dynamic_object_idx]) * 100, 2)

            occ_iou=np.stack(occ_iou)
            occ_iou=occ_iou.sum(0)
            occ_iou=occ_iou[:,0]/(occ_iou[:,1]+occ_iou[:,2]-occ_iou[:,0]+1e-8)

            #occupied
            
            print(f'===> occupied - IoU = ' + str(round(occ_iou[0] * 100, 4)))
            res['occupied iou']=round(occ_iou[0] * 100, 2)
            return res
        res = self.occ_eval_metrics.count_miou()

        if self.eval_fscore:
            res.update(self.fscore_eval_metrics.count_fscore())
        return res 
        
    def evaluate_mask(self, results):
        results_dict = {}
        iou = 0
        # ret_f1=[0,0,0,0,0]
        for i in range(len(results)):
            iou+=results[i]['iou']
        n=len(results)
        iou = iou/n
        results_dict['iou'] = iou
        return results_dict
        

    def evaluate_bbox(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix='test',
                 result_names=['pts_bbox'],
                 show=False,
                 out_dir=None,
                 pipeline=None):
        """Evaluation in nuScenes protocol.

        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str], optional): Metrics to be evaluated.
                Default: 'bbox'.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str, optional): The prefix of json files including
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool, optional): Whether to visualize.
                Default: False.
            out_dir (str, optional): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.

        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        processed_set = set()
        results_=[]
        
        if 'index' in results[0]:
            for result in tqdm(results):
                index = result['index']
                if index in processed_set: continue
                processed_set.add(index)
                results_.append(result)
        else:
            results_=results
        result_files, tmp_dir = self.format_results_det(results_, jsonfile_prefix)
        


        if isinstance(result_files, dict):
            results_dict = dict()
            for name in result_names:
                print('Evaluating bboxes of {}'.format(name))
                ret_dict = self._evaluate_single(result_files[name])
            results_dict.update(ret_dict)
        elif isinstance(result_files, str):
            results_dict = self._evaluate_single(result_files)

        if tmp_dir is not None:
            tmp_dir.cleanup()

        if show or out_dir:
            self.show(results, out_dir, show=show, pipeline=pipeline)

        return results_dict

    def _build_default_pipeline(self):
        """Build the default pipeline for this dataset."""
        pipeline = [
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=dict(backend='disk')),
            dict(
                type='LoadPointsFromMultiSweeps',
                sweeps_num=10,
                file_client_args=dict(backend='disk')),
            dict(
                type='DefaultFormatBundle3D',
                class_names=self.CLASSES,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ]
        return Compose(pipeline)

    def show(self, results, out_dir, show=False, pipeline=None):
        """Results visualization.

        Args:
            results (list[dict]): List of bounding boxes results.
            out_dir (str): Output directory of visualization result.
            show (bool): Whether to visualize the results online.
                Default: False.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        """
        assert out_dir is not None, 'Expect out_dir, got none.'
        pipeline = self._get_pipeline(pipeline)
        for i, result in enumerate(results):
            if 'pts_bbox' in result.keys():
                result = result['pts_bbox']
            data_info = self.data_infos[i]
            pts_path = data_info['lidar_path']
            file_name = osp.split(pts_path)[-1].split('.')[0]
            points = self._extract_data(i, pipeline, 'points').numpy()
            # for now we convert points into depth mode
            points = Coord3DMode.convert_point(points, Coord3DMode.LIDAR,
                                               Coord3DMode.DEPTH)
            inds = result['scores_3d'] > 0.1
            gt_bboxes = self.get_ann_info(i)['gt_bboxes_3d'].tensor.numpy()
            show_gt_bboxes = Box3DMode.convert(gt_bboxes, Box3DMode.LIDAR,
                                               Box3DMode.DEPTH)
            pred_bboxes = result['boxes_3d'][inds].tensor.numpy()
            show_pred_bboxes = Box3DMode.convert(pred_bboxes, Box3DMode.LIDAR,
                                                 Box3DMode.DEPTH)
            show_result(points, show_gt_bboxes, show_pred_bboxes, out_dir,
                        file_name, show)
    def evaluate_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        flow_gts = []
        occ_preds = []
        flow_preds = []
        lidar_origins = []
        density_preds = []

        print('\nStarting Evaluation...')

        if 'LightwheelOcc' in self.version:
            # lightwheelocc is 10Hz, downsample to 1/5
            if self.load_interval == 5:
                data_infos = self.data_infos
            elif self.load_interval == 1:
                print('[WARNING] Please set `load_interval` to 5 in for LightwheelOcc val/test!')
                print('[WARNING] Current format_results will continue!')
                data_infos = self.data_infos[::5]
            else:
                raise ValueError('Please set `load_interval` to 5 in for LightwheelOcc val/test!')

            ego_pose_dataset = EgoPoseDataset(data_infos, dataset_type='lightwheelocc')
        else:
            ego_pose_dataset = EgoPoseDataset(self.data_infos, dataset_type='openocc_v2')

        data_loader_kwargs={
            "pin_memory": False,
            "shuffle": False,
            "batch_size": 1,
            "num_workers": 8,
        }

        data_loader = DataLoader(
            ego_pose_dataset,
            **data_loader_kwargs,
        )

        print('\nStarting Evaluation...')
        processed_set = set()
        for occ_pred_w_index in tqdm(occ_results):
            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)

            info = self.data_infos[index]
            scene_name = info['scene_name']
            sample_token = info['token']
            occupancy_file_path = osp.join(self.occupancy_path, scene_name, sample_token, 'labels.npz')
            occ_gt = np.load(occupancy_file_path)
 
            occ_gt = np.load(occupancy_file_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']
            gt_flow = occ_gt['flow']

            output_origin=data_loader.dataset[index][1].unsqueeze(0)
            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            flow_gts.append(gt_flow)

            if occ_pred_w_index['pred_occupancy'] is not None:
                occ_preds.append(occ_pred_w_index['pred_occupancy'])
            else:
                occ_preds.append(gt_semantics)

            if occ_pred_w_index['pred_flow'] is not None:
                flow_pred=occ_pred_w_index['pred_flow']
            else:
                flow_pred=np.zeros_like(gt_flow)

            flow_preds.append(flow_pred)   
            density_preds.append(occ_pred_w_index['pred_density'])

        if 'log_dirs' not in eval_kwargs:
            eval_kwargs['log_dirs'] = 'openocc_training_eval_results'
        ray_based_miou(occ_preds, occ_gts, flow_preds, flow_gts, lidar_origins,density_preds,eval_kwargs['log_dirs'])
        return None
    def format_results_det(self, results, jsonfile_prefix=None):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        # """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) >= len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        # currently the output prediction results could be in two formats
        # 1. list of dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...)
        # 2. list of dict('pts_bbox' or 'img_bbox':
        #     dict('boxes_3d': ..., 'scores_3d': ..., 'labels_3d': ...))
        # this is a workaround to enable evaluation of both formats on nuScenes
        # refer to https://github.com/open-mmlab/mmdetection3d/issues/449
        if not ('pts_bbox' in results[0] or 'img_bbox' in results[0]):
            result_files = self._format_bbox(results, jsonfile_prefix)
        else:
            # should take the inner dict out of 'pts_bbox' or 'img_bbox' dict
            result_files = dict()
            for name in ['pts_bbox']:
                print(f'\nFormating bboxes of {name}')
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_)})
        return result_files, tmp_dir

    def format_results(self, occ_results, submission_prefix, **kwargs):
        from torch.utils.cpp_extension import load
        dvr = load("dvr", sources=["tools/ray_iou/lib/dvr/dvr.cpp", "tools/ray_iou/lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])
        if submission_prefix is not None:
            try:
                mmcv.mkdir_or_exist(submission_prefix)
            except:
                submission_prefix=os.path.join('wll',submission_prefix)

        result_dict = {}
        result_occ_dict = {}
        result_flow_dict = {}
        result_coord_dict={}

        if 'LightwheelOcc' in self.version:
            # lightwheelocc is 10Hz, downsample to 1/5
            if self.load_interval == 5:
                data_infos = self.data_infos
            elif self.load_interval == 1:
                print('[WARNING] Please set `load_interval` to 5 in for LightwheelOcc test submission!')
                print('[WARNING] Current format_results will continue!')
                data_infos = self.data_infos[::5]
            else:
                raise ValueError('Please set `load_interval` to 5 in for LightwheelOcc test submission!')

            ego_pose_dataset = EgoPoseDataset(data_infos, dataset_type='lightwheelocc')
        else:
            ego_pose_dataset = EgoPoseDataset(self.data_infos, dataset_type='openocc_v2')

        data_loader_kwargs={
            "pin_memory": False,
            "shuffle": False,
            "batch_size": 1,
            "num_workers": 8,
        }

        data_loader = DataLoader(
            ego_pose_dataset,
            **data_loader_kwargs,
        )

        sample_tokens = [info['token'] for info in self.data_infos]

        lidar_rays = generate_lidar_rays()
        lidar_rays = torch.from_numpy(lidar_rays)

        processed_set = set()
        for occ_pred_w_index in tqdm(occ_results):
            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)

            info = self.data_infos[index]
            scene_name = info['scene_name']
            sample_token = info['token']
            output_origin=data_loader.dataset[index][1].unsqueeze(0)
            token=data_loader.dataset[index][0]
            sem_pred=occ_pred_w_index['pred_occupancy']

            flow_pred=occ_pred_w_index['pred_flow']


            pcd_pred,coords = process_one_sample(sem_pred, lidar_rays, output_origin, flow_pred,dvr,return_coords=True)



            pcd_cls = pcd_pred[:, 0].astype(np.int8)
            pcd_dist = pcd_pred[:, 1].astype(np.float16)
            pcd_flow = pcd_pred[:, 2:4].astype(np.float16)

            sample_dict = {
                'pcd_cls': pcd_cls,
                'pcd_dist': pcd_dist,
                'pcd_flow': pcd_flow
            }
            result_dict.update({token: sample_dict})
            result_occ_dict.update({token: sem_pred.astype(np.int8)})
            result_coord_dict.update({token: coords})
            result_flow_dict.update({token: flow_pred.astype(np.float16)})


        save_path_occ = os.path.join('openocc_results',kwargs['save_path'],str(kwargs['tta']) ,'occ','occ.gz')
        if not os.path.exists(os.path.dirname(save_path_occ)):
            try:
                os.makedirs(os.path.dirname(save_path_occ))
            except:
                save_path_occ=os.path.join('wll',save_path_occ)
                os.makedirs(os.path.dirname(save_path_occ))
        with gzip.open(save_path_occ, 'wb', compresslevel=9) as f:
            pickle.dump(result_occ_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\nCompress and saving results to {save_path_occ}.')

        save_path_flow = os.path.join('openocc_results',kwargs['save_path'],str(kwargs['tta']) ,'flow','flow.gz')
        if not os.path.exists(os.path.dirname(save_path_flow)):
            try:
                os.makedirs(os.path.dirname(save_path_flow))
            except:
                save_path_flow=os.path.join('wll',save_path_flow)
                os.makedirs(os.path.dirname(save_path_flow))
        with gzip.open(save_path_flow, 'wb', compresslevel=9) as f:
            pickle.dump(result_flow_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\nCompress and saving results to {save_path_flow}.')

        final_submission_dict = {
            'method': 'zz',
            'team': 'zzz',
            'authors': ['z'],
            'e-mail': '',
            'institution / company': '',
            'country / region': '',
            'results': result_dict
        }

        save_path = os.path.join('openocc_results',kwargs['save_path'],str(kwargs['tta']) ,'sub','submission.gz')

        if not os.path.exists(os.path.dirname(save_path)):
            try:
                os.makedirs(os.path.dirname(save_path))
            except:
                save_path=os.path.join('wll',save_path)
                os.makedirs(os.path.dirname(save_path))

        with gzip.open(save_path, 'wb', compresslevel=9) as f:
            pickle.dump(final_submission_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f'\nCompress and saving results to {save_path}.')
        print('Finished.')

def output_to_nusc_box(detection, with_velocity=True):
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection['boxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = box3d.gravity_center.numpy()
    box_dims = box3d.dims.numpy()
    box_yaw = box3d.yaw.numpy()

    # our LiDAR coordinate system -> nuScenes box coordinate system
    nus_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(box3d)):
        quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if with_velocity:
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (0, 0, 0)

        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity)
        box_list.append(box)
    return box_list
    

@DATASETS.register_module()
class NuscenesOccupancy(NuScenesDataset):

    CLASSES = [
        "empty",
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction",
        "motorcycle",
        "pedestrian",
        "trafficcone",
        "trailer",
        "truck",
        "driveable_surface",
        "other",
        "sidewalk",
        "terrain",
        "mannade",
        "vegetation",
    ]

    def __init__(self, occupancy_info='data/nuscenes/occupancy_category.json', **kwargs):

        super().__init__(**kwargs)
        self.CLASSES = [
            "empty",
            "barrier",
            "bicycle",
            "bus",
            "car",
            "construction",
            "motorcycle",
            "pedestrian",
            "trafficcone",
            "trailer",
            "truck",
            "driveable_surface",
            "other",
            "sidewalk",
            "terrain",
            "mannade",
            "vegetation",
        ]

        self.occupancy_info = mmcv.load(occupancy_info)

    def get_cat_ids(self, idx):
        """Get category distribution of single scene.

        Args:
            idx (int): Index of the data_info.

        Returns:
            dict[list]: for each category, if the current scene
                contains such boxes, store a list containing idx,
                otherwise, store empty list.
        """
        info = self.data_infos[idx]

        token = info['token']
        category = self.occupancy_info[token]
        cat_ids = []
        for k, v in category.items():
            k = int(k)
            if k == 17: continue
            logv = max((np.log(v)/np.log(100)).round(),1)
            cat_ids.extend([k] * int(logv))
        return cat_ids
    

def lidar_nusc_box_to_global(info,
                             boxes,
                             classes,
                             eval_configs,
                             eval_version='detection_cvpr_2019'):
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: 'detection_cvpr_2019'

    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
