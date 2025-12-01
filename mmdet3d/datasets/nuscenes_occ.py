import os
import cv2
import gzip
import pickle
import mmcv
import torch
import numpy as np
from tqdm import tqdm
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from .builder import DATASETS
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion
from nuscenes.utils.geometry_utils import transform_matrix
from .ray_metrics import main as ray_based_miou
from .ray_metrics import process_one_sample, generate_lidar_rays
from torch.utils.data import DataLoader
from nuscenes.nuscenes import NuScenes
from .ego_pose_extractor import EgoPoseDataset


@DATASETS.register_module()
class NuSceneOcc(NuScenesDataset):
    r"""NuScenes Dataset.

    This datset only add camera intrinsics and extrinsics to the results.
    """

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file)

        data_infos = data['infos'][::self.load_interval]
        self.metadata = data['metadata']
        self.version = self.metadata['version']

        if 'LightwheelOcc' in self.version:
            # LightwheelOcc use relative path, join with data_root
            for info in data_infos:
                info['occ_path'] = os.path.join(self.data_root, info['occ_path'])
                for cam_info in info['cams'].values():
                    cam_info['data_path'] = os.path.join(self.data_root, cam_info.pop('cam_path'))
                    cam_info['sensor2lidar_rotation'] = Quaternion(cam_info['sensor2lidar_rotation']).rotation_matrix

        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.

        Returns:
            dict: Data information that will be passed to the data \
                preprocessing pipelines. It includes the following keys:

                - sample_idx (str): Sample index.
                - pts_filename (str): Filename of point clouds.
                - sweeps (list[dict]): Infos of sweeps.
                - timestamp (float): Sample timestamp.
                - img_filename (str, optional): Image filename.
                - lidar2img (list[np.ndarray], optional): Transformations \
                    from lidar to different cameras.
                - ann_info (dict): Annotation info.
        """
        info = self.data_infos[index]
        # standard protocal modified from SECOND.Pytorch
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'] if 'lidar_path' in info else '',
            sweeps=info['sweeps'] if 'sweeps' in info else [],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            timestamp=info['timestamp'] / 1e6,
        )

        lidar2ego_rotation = info['lidar2ego_rotation']
        lidar2ego_translation = info['lidar2ego_translation']
        ego2lidar = transform_matrix(translation=lidar2ego_translation, rotation=Quaternion(lidar2ego_rotation),
                                     inverse=True)
        input_dict['ego2lidar'] = ego2lidar
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])
                # obtain lidar to image transformation matrix
                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info[
                                  'sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t
                intrinsic = np.array(cam_info['cam_intrinsic'], dtype=np.float32)
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)
                lidar2img_rts.append(lidar2img_rt)

                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt.T)
            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                ))

        if not self.test_mode:
            input_dict['occ_path'] = info.get('occ_path', None)

        return input_dict

    def __getitem__(self, idx):
        """Get item from infos according to the given index.
        Returns:
            dict: Data dictionary of the corresponding index.
        """

        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def evaluate_miou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts = []
        flow_gts = []
        occ_preds = []
        flow_preds = []
        lidar_origins = []

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

        sample_tokens = [info['token'] for info in self.data_infos]

        for i, batch in tqdm(enumerate(data_loader), ncols=50):
            token = batch[0][0]
            output_origin = batch[1]
            
            data_id = sample_tokens.index(token)
            info = self.data_infos[data_id]

            occ_gt = np.load(info['occ_path'], allow_pickle=True)
            gt_semantics = occ_gt['semantics']
            gt_flow = occ_gt['flow']

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            flow_gts.append(gt_flow)
            occ_preds.append(occ_results[data_id]['occ_results'].cpu().numpy())
            flow_preds.append(occ_results[data_id]['flow_results'].cpu().numpy())

        ray_based_miou(occ_preds, occ_gts, flow_preds, flow_gts, lidar_origins)

    def format_results(self, occ_results, submission_prefix, **kwargs):
        if submission_prefix is not None:
            mmcv.mkdir_or_exist(submission_prefix)

        result_dict = {}

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

        for batch in tqdm(data_loader, ncols=80, desc='Formatting results'):
            token = batch[0][0]
            output_origin = batch[1]

            data_id = sample_tokens.index(token)

            occ_pred = occ_results[data_id]
            sem_pred = occ_pred['occ_results'].cpu().numpy()
            sem_pred = np.reshape(sem_pred, [200, 200, 16])

            flow_pred = occ_pred['flow_results'].cpu().numpy()
            flow_pred = np.reshape(flow_pred, [200, 200, 16, 2])

            pcd_pred = process_one_sample(sem_pred, lidar_rays, output_origin, flow_pred)

            pcd_cls = pcd_pred[:, 0].astype(np.int8)
            pcd_dist = pcd_pred[:, 1].astype(np.float16)
            pcd_flow = pcd_pred[:, 2:4].astype(np.float16)

            sample_dict = {
                'pcd_cls': pcd_cls,
                'pcd_dist': pcd_dist,
                'pcd_flow': pcd_flow
            }
            result_dict.update({token: sample_dict})

        final_submission_dict = {
            'method': '',
            'team': '',
            'authors': [''],
            'e-mail': '',
            'institution / company': '',
            'country / region': '',
            'results': result_dict
        }

        save_path = os.path.join(submission_prefix, 'submission.gz')
        print(f'\nCompress and saving results to {save_path}.')

        with gzip.open(save_path, 'wb', compresslevel=9) as f:
            pickle.dump(final_submission_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        print('Finished.')
