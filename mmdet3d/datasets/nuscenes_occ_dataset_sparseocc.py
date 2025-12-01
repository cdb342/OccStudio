import os
import mmcv
import glob
import torch
import numpy as np
from tqdm import tqdm
from .builder import DATASETS
from mmdet3d.datasets import NuScenesDataset
from nuscenes.eval.common.utils import Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from torch.utils.data import DataLoader
from mmdet3d.models.sparseocc.utils import sparse2dense
from .rayiou_metrics import main_rayiou, main_raypq
from .ego_pose_dataset import EgoPoseDatasetOcc3D
import copy
occ3d_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
openocc_class_names= [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
@DATASETS.register_module()
class NuSceneOcc_SparseOcc(NuScenesDataset):    
    def __init__(self, occ_gt_root, *args, **kwargs):

        super().__init__(filter_empty_gt=False, *args, **kwargs)
        self.occ_gt_root = occ_gt_root
        self.data_infos = self.load_annotations(self.ann_file)

        self.token2scene = {}
        for gt_path in glob.glob(os.path.join(self.occ_gt_root, '*/*/*.npz')):
            token = gt_path.split('/')[-2]
            scene_name = gt_path.split('/')[-3]
            self.token2scene[token] = scene_name

        for i in range(len(self.data_infos)):
           
            scene_name = self.token2scene[self.data_infos[i]['token']]
            self.data_infos[i]['scene_name'] = scene_name

    def collect_sweeps(self, index, into_past=150, into_future=0):
        all_sweeps_prev = []
        curr_index = index
        while len(all_sweeps_prev) < into_past:
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            if len(curr_sweeps) == 0:
                break
            all_sweeps_prev.extend(curr_sweeps)
            all_sweeps_prev.append(self.data_infos[curr_index - 1]['cams'])
            curr_index = curr_index - 1
        
        all_sweeps_next = []
        curr_index = index + 1
        while len(all_sweeps_next) < into_future:
            if curr_index >= len(self.data_infos):
                break
            curr_sweeps = self.data_infos[curr_index]['sweeps']
            all_sweeps_next.extend(curr_sweeps[::-1])
            all_sweeps_next.append(self.data_infos[curr_index]['cams'])
            curr_index = curr_index + 1

        return all_sweeps_prev, all_sweeps_next

    def get_data_info(self, index):
        info = self.data_infos[index]
        sweeps_prev, sweeps_next = self.collect_sweeps(index)

        ego2global_translation = info['ego2global_translation']
        ego2global_rotation = info['ego2global_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        lidar2ego_rotation = info['lidar2ego_rotation']
        ego2global_rotation_mat = Quaternion(ego2global_rotation).rotation_matrix
        lidar2ego_rotation_mat = Quaternion(lidar2ego_rotation).rotation_matrix

        input_dict = dict(
            sample_idx=info['token'],
            sweeps={'prev': sweeps_prev, 'next': sweeps_next},
            timestamp=info['timestamp'] / 1e6,
            ego2global_translation=ego2global_translation,
            ego2global_rotation=ego2global_rotation_mat,
            lidar2ego_translation=lidar2ego_translation,
            lidar2ego_rotation=lidar2ego_rotation_mat,
        )

        ego2lidar = transform_matrix(lidar2ego_translation, Quaternion(lidar2ego_rotation), inverse=True)
        input_dict['ego2lidar'] = [ego2lidar for _ in range(6)]
        input_dict['occ_path'] = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')

        if self.modality['use_camera']:
            img_paths = []
            img_timestamps = []
            lidar2img_rts = []

            for _, cam_info in info['cams'].items():
                img_paths.append(os.path.relpath(cam_info['data_path']))
                img_timestamps.append(cam_info['timestamp'] / 1e6)

                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T

                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                
                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

            input_dict.update(dict(
                img_filename=img_paths,
                img_timestamp=img_timestamps,
                lidar2img=lidar2img_rts,
            ))

        if not self.test_mode:
            annos = self.get_ann_info(index)
            input_dict['ann_info'] = annos

        return input_dict

    def evaluate(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        from .occ_metrics import Metric_mIoU, Metric_FScore
        self.occ_eval_metrics = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)
        self.occ_eval_metrics_non_mask = Metric_mIoU(
            num_classes=18,
            use_lidar_mask=False,
            use_image_mask=True)
        
        occ_gts, occ_preds, inst_gts, inst_preds, lidar_origins = [], [], [], [], []
        print('\nStarting Evaluation...')

        sample_tokens = [info['token'] for info in self.data_infos]

        data_loader = DataLoader(
            EgoPoseDatasetOcc3D(self.data_infos), num_workers=8)
        
        print('\nStarting Evaluation...')
        processed_set = set()
        
        for occ_pred_w_index in tqdm(occ_results):
            token = occ_pred_w_index['index']
            index = sample_tokens.index(token)
            if index in processed_set: continue
            processed_set.add(index)
            
            output_origin=data_loader.dataset[index][1].unsqueeze(0)
            
            info = self.data_infos[index]

            occ_path = os.path.join(self.occ_gt_root, info['scene_name'], info['token'], 'labels.npz')
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['semantics']

            sem_pred = torch.from_numpy(occ_pred_w_index['sem_pred'])  # [B, N]
            occ_loc = torch.from_numpy(occ_pred_w_index['occ_loc'].astype(np.int64))  # [B, N, 3]
            
            data_type = self.occ_gt_root.split('/')[-1]
            if data_type == 'gts' or data_type == 'occ3d_panoptic':
                occ_class_names = occ3d_class_names
            elif data_type == 'openocc_v2':
                occ_class_names = openocc_class_names
            else:
                raise ValueError
            free_id = len(occ_class_names) - 1
            
            occ_size = list(gt_semantics.shape)
            sem_pred, _ = sparse2dense(occ_loc, sem_pred, dense_shape=occ_size, empty_value=free_id)
            sem_pred = sem_pred.squeeze(0).numpy()

            if 'pano_inst' in occ_pred_w_index.keys():
                pano_inst = torch.from_numpy(occ_pred_w_index['pano_inst'])
                pano_sem = torch.from_numpy(occ_pred_w_index['pano_sem'])

                pano_inst, _ = sparse2dense(occ_loc, pano_inst, dense_shape=occ_size, empty_value=0)
                pano_sem, _ = sparse2dense(occ_loc, pano_sem, dense_shape=occ_size, empty_value=free_id)
                pano_inst = pano_inst.squeeze(0).numpy()
                pano_sem = pano_sem.squeeze(0).numpy()
                sem_pred = pano_sem

                gt_instances = occ_gt['instances']
                inst_gts.append(gt_instances)
                inst_preds.append(pano_inst)
            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(sem_pred)
            
            
            mask_camera=occ_gt['mask_camera'].astype(np.bool)
            mask_camera_=copy.deepcopy(mask_camera)
            gt_semantics_=copy.deepcopy(gt_semantics)

            self.occ_eval_metrics.add_batch(sem_pred[mask_camera], gt_semantics, None, mask_camera)

            mask_camera_non_mask = np.ones_like(mask_camera).astype(bool)
            self.occ_eval_metrics_non_mask.add_batch(sem_pred[mask_camera_non_mask], gt_semantics, None, mask_camera_non_mask)

            
        res = self.occ_eval_metrics.count_miou()
        print('\n=================results_non_mask====================')
        res_none_mask = self.occ_eval_metrics_non_mask.count_miou()

        if len(inst_preds) > 0:
            results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins, occ_class_names=occ_class_names)
            results.update(main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names))
            return results
        else:
            return main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names)  

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        for index, occ_pred in enumerate(tqdm(occ_results)):
            info = self.data_infos[index]
            sample_token = info['token']
            save_path = os.path.join(submission_prefix, '{}.npz'.format(sample_token))
            np.savez_compressed(save_path, occ_pred.astype(np.uint8))
        
        print('\nFinished.')
