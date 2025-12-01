import os
import copy
import random
import pickle
from functools import reduce

from tqdm import tqdm
import numpy as np
import torch
import mmcv
from mmcv.parallel import DataContainer as DC
from mmcv.utils import print_log
from .builder import DATASETS
from mmdet3d.core.bbox import Box3DMode, points_cam2img
from mmdet3d.datasets.kitti_dataset import KittiDataset
from mmdet3d.core.bbox import get_box_type
from mmdet3d.core.bbox import (Box3DMode, CameraInstance3DBoxes, Coord3DMode, LiDARInstance3DBoxes, points_cam2img)
from nuscenes.eval.common.utils import quaternion_yaw, Quaternion

from .zltwaymo import CustomWaymoDataset
from .occ_metrics import Metric_FScore, Metric_mIoU
import math
import os.path as osp
import csv
from torch.utils.data import DataLoader
from .ego_pose_dataset import EgoPoseDatasetOcc3D
from .rayiou_metrics import main_rayiou

@DATASETS.register_module()
class WaymoDatasetOcc(CustomWaymoDataset):

    CLASSES = ('Car', 'Pedestrian', 'Sign', 'Cyclist')

    def __init__(self,
                 *args,
                 load_interval=1,
                 history_len=1, 
                 input_sample_policy=None,
                 skip_len=0,
                 withimage=True,
                 pose_file=None,
                 offset=0,
                 use_streaming=False,
                 multi_adj_frame_id_cfg=None,
                 stereo=False,
                 use_sequence_group_flag=False,
                 sequences_split_num=1,
                 cam_names=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_SIDE_LEFT','CAM_SIDE_RIGHT'],
                 occupancy_path='',
                 class_names=['TYPE_GENERALOBJECT','TYPE_VEHICLE','TYPE_PEDESTRIAN','TYPE_SIGN','TYPE_BICYCLIST',
                                'TYPE_TRAFFIC_LIGHT','TYPE_POLE','TYPE_CONSTRUCTION_CONE','TYPE_BICYCLE','TYPE_MOTORCYCLE',
                            'TYPE_BUILDING','TYPE_VEGETATION','TYPE_TREE_TRUNK','TYPE_ROAD','TYPE_WALKABLE','TYPE_FREE',],
                FREE_LABEL=23,
                 **kwargs):
        with open(pose_file, 'rb') as f:
            pose_all = pickle.load(f)
            self.pose_all = pose_all
            # import pdb;pdb.set_trace()
        self.length_waymo = sum([len(scene) for k, scene in pose_all.items()])
        self.history_len = history_len
        self.input_sample_policy = input_sample_policy
        self.skip_len = skip_len
        self.withimage = withimage
        self.load_interval_waymo = load_interval
        self.length = self.length_waymo
        self.offset = offset
        self.evaluation_kwargs = kwargs
        self.use_streaming = use_streaming
        
        self.multi_adj_frame_id_cfg = multi_adj_frame_id_cfg
        self.stereo = stereo
        super().__init__(*args, **kwargs)
        self.prepare_scene_token()
        self.cam_names=cam_names
        
        
        # SOLOFusion
        self.use_sequence_group_flag = use_sequence_group_flag
        self.sequences_split_num = sequences_split_num
        
        if self.use_sequence_group_flag:
  
            ######################
            self._set_sequence_group_flag() # Must be called after load_annotations b/c load_annotations does sorting.
        
        if self.test_mode == False:
            self.data_infos=self.data_infos
        self.prepare_img_info()
        self.occupancy_path=occupancy_path
        self.class_names=class_names
        self.FREE_LABEL=FREE_LABEL
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
        
    def __len__(self):
        return self.length_waymo // self.load_interval_waymo

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        if self.use_streaming:
            return self.prepare_streaming_train_data(idx)
        
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data
    def prepare_scene_token(self):
     
        for i in range(len(self.data_infos)):
            self.data_infos[i]['scene_token']=self.data_infos[i]['image']['image_idx']% 1000000 // 1000
            self.data_infos[i]['scene_name']=self.data_infos[i]['image']['image_idx']% 1000000 // 1000
    def prepare_img_info(self):
        for i in range(len(self.data_infos)):
            info = self.data_infos[i]
        
        
            sample_idx = info['image']['image_idx']
            scene_idx = sample_idx % 1000000 // 1000
            frame_idx = sample_idx % 1000000 % 1000
            img_filename = os.path.join(self.data_root, info['image']['image_path'])


            if self.modality['use_camera']:

                info['cams']={}
                for idx_img in range(self.num_views):
                    pose = self.pose_all[scene_idx][frame_idx][idx_img]

                    intrinsics = pose['intrinsics'] # sensor2img
                    sensor2ego = pose['sensor2ego']
                    lidar2img = intrinsics @ np.linalg.inv(sensor2ego)
                    ego2global = pose['ego2global']
                    
                    # Attention! (this code means the pose info dismatch the image data file)
                    if idx_img == 2: 
                        image_path=img_filename.replace('image_0', f'image_3')
                       
                    elif idx_img == 3: 
                        image_path=img_filename.replace('image_0', f'image_2')
                        
                    else:
                        image_path=img_filename.replace('image_0', f'image_{idx_img}')
                       

                    
                    cam_data={}
                    cam_data['cam_intrinsic']=intrinsics[:3,:3]
                    cam_data['data_path']=image_path
                
                    cam_data['cam2camego']=sensor2ego
                    cam_data['camego2global']=ego2global
                    
                    info['cams'][self.cam_names[idx_img]]=cam_data


            info['lidar2lidarego']=torch.eye(4)
            info['lidarego2global']=self.pose_all[scene_idx][frame_idx][0]['ego2global']
            self.data_infos[i]=info
        
    def prepare_streaming_train_data(self, index):
        index = int(index * self.load_interval_waymo)
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example

    def prepare_test_data(self, index):
        """
        Prepare data for testing.
        Args:
            index (int): Index for accessing the target data.

        Returns:
            dict: Testing data dict of the corresponding index.
        """

        index += self.offset
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return example
    
    def get_input_idx(self, idx_list):
        '''
        sample the input index list
        Args:
            idx_list (List[int]): the index list from `index - self.history_len` to `index`. 
                                  It contains current frame index, but it dropped another random frame index to add randomness. 
                                  So the length is `self.history_len`.
        Returns:
            sampled_idx_list (List[int]): the index list after sampling
        '''

        if self.input_sample_policy['type'] == 'normal':
            return idx_list
        
        elif self.input_sample_policy['type'] == 'large interval':
            sampled_idx_list = []
            for i in range(0, self.input_sample_policy['number']):
                sampled_idx = max(0, self.history_len - 1 - i * self.input_sample_policy['interval'])
                sampled_idx_list.append(idx_list[sampled_idx])
            return sorted(sampled_idx_list)
        
        elif self.input_sample_policy['type'] == 'random interval':
            fix_interval = self.input_sample_policy['fix interval']
            slow_interval = random.randint(0, fix_interval-1)
            random_interval = random.choice([fix_interval, slow_interval])

            sampled_idx_list = []
            for i in range(0, self.input_sample_policy['number']):
                sampled_idx = max(self.history_len - 1 - i * random_interval, 0)
                sampled_idx_list.append(idx_list[sampled_idx])
                
            return sorted(sampled_idx_list)
        
        else:
            raise NotImplementedError('not implemented input_sample_policy type')


    def union2one(self, queue):
        """
        convert sample queue into one single sample.
        Args: 
            queue (List[Dict]): the sample queue
        Returns:
            queue (Dict): the single sample
        """
        
        # Step 1: 1. union the `img` tensor into a single tensor. 
        # 2. union the `img_metas` dict into a dict[dict]
        # 3. add prev_bev_exists and scene_token
        prev_scene_token=None
        imgs_list = [each['img'].data for each in queue]
        metas_map = {}
        for i, each in enumerate(queue):
            metas_map[i] = each['img_metas'].data
            if metas_map[i]['sample_idx']//1000 != prev_scene_token:
                metas_map[i]['prev_bev_exists'] = False
                prev_scene_token = metas_map[i]['sample_idx'] // 1000
                metas_map[i]['scene_token']= prev_scene_token

            else:
                metas_map[i]['scene_token'] = prev_scene_token
                metas_map[i]['prev_bev_exists'] = True

        # Step 2: pack them together
        queue[-1]['img'] = DC(torch.stack(imgs_list), cpu_only=False, stack=True)
        queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
        queue = queue[-1]

        return queue


    def get_data_info(self, index):
        '''
        get the data info according to the index. Most of them are image meta data. 
        Args: 
            index (Int): the index of the data.
        Returns:
            input dict (Dict): the data info dict.
        '''

        # Step 1: get the data info
        info = self.data_infos[index]
        
        
        # Step 2: get the image file name and idx
        sample_idx = info['image']['image_idx']
        scene_idx = sample_idx % 1000000 // 1000
        frame_idx = sample_idx % 1000000 % 1000
        img_filename = os.path.join(self.data_root, info['image']['image_path'])
       
        # # Step 3: get the `lidar2img` (why here it get the lidar2img and in the following code it get another lidar2img)
        # rect = info['calib']['R0_rect'].astype(np.float32)
        # Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)
        # P0 = info['calib']['P0'].astype(np.float32)
        # lidar2img = P0 @ rect @ Trv2c

        # the Tr_velo_to_cam is computed for all images but not saved in .info for img1-4
        # the size of img0-2: 1280x1920; img3-4: 886x1920. Attention

        # Step 4: get the image paths, lidar2img, intrinsics, sensor2ego for each image
        if self.modality['use_camera']:
            image_paths = []
            lidar2img_rts = []
            intrinsics_rts = []
            sensor2ego_rts = []

            for idx_img in range(self.num_views):
                pose = self.pose_all[scene_idx][frame_idx][idx_img]

                intrinsics = pose['intrinsics'] # sensor2img
                sensor2ego = pose['sensor2ego']
                lidar2img = intrinsics @ np.linalg.inv(sensor2ego)
                ego2global = pose['ego2global']
                
                # Attention! (this code means the pose info dismatch the image data file)
                if idx_img == 2: 
                    image_path=img_filename.replace('image_0', f'image_3')
                    image_paths.append(image_path)
                elif idx_img == 3: 
                    image_path=img_filename.replace('image_0', f'image_2')
                    image_paths.append(image_path)
                else:
                    image_path=img_filename.replace('image_0', f'image_{idx_img}')
                    image_paths.append(image_path)

                lidar2img_rts.append(lidar2img)
                intrinsics_rts.append(intrinsics)
                sensor2ego_rts.append(sensor2ego)
                
           

        # Step 5: get the pts filename by function `_get_pts_filename` in class `CustomWaymoDataset`
        pts_filename = self._get_pts_filename(sample_idx)

        # info['lidar2lidarego']=torch.eye(4)
        # info['lidarego2global']=self.pose_all[scene_idx][frame_idx][0]['ego2global']
        # Step 6: pack the data info into a dict
        input_dict = dict(
            index=index,
            sample_idx=sample_idx,
            pts_filename=pts_filename,
            scene_idx=scene_idx,
            img_prefix=None,
        )

        if self.modality['use_camera']:
            input_dict['img_filename'] = image_paths
            input_dict['lidar2img'] = lidar2img_rts
            input_dict['cam_intrinsic'] = intrinsics_rts
            input_dict['sensor2ego'] = sensor2ego_rts
            ego2global = self.pose_all[scene_idx][frame_idx][0]['ego2global']
            input_dict['ego2global'] = ego2global
            input_dict['global_to_curr_lidar_rt'] = np.linalg.inv(pose['ego2global'])

            input_dict.update(dict(curr=info))
            
            info_adj_list = self.get_adj_info(info, index)
            input_dict.update(dict(adjacent=info_adj_list))


        # Step 7: get the annos info
        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        # Step 8: get the can_bus info (In `waymo` dataset, we do not have can_bus info)
        can_bus = np.zeros(9)
        input_dict['can_bus'] = can_bus
        if self.use_sequence_group_flag:
            input_dict['sample_index'] = index
            input_dict['sequence_group_idx'] = self.flag[index]
            input_dict['start_of_sequence'] = index == 0 or self.flag[index - 1] != self.flag[index]
            # Get a transformation matrix from current keyframe lidar to previous keyframe lidar
            # if they belong to same sequence.
            input_dict['end_of_sequence'] = index == len(self.data_infos)-1 or  self.flag[index ] != self.flag[index+1]

            if not input_dict['start_of_sequence']:

                input_dict['curr_to_prev_lidar_rt'] =torch.matmul(torch.FloatTensor( self.pose_all[scene_idx][frame_idx-1][idx_img]['ego2global']).inverse(),\
                    torch.FloatTensor(ego2global))
                input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(self.pose_all[scene_idx][frame_idx-1][idx_img]['ego2global']) # TODO: Note that global is same for all.
                input_dict['curr_to_prev_ego_rt'] = input_dict['curr_to_prev_lidar_rt']
                
                
            else:

                input_dict['curr_to_prev_lidar_rt'] = torch.eye(4).float()
                input_dict['prev_lidar_to_global_rt'] = torch.FloatTensor(ego2global)
                input_dict['curr_to_prev_ego_rt'] = input_dict['curr_to_prev_lidar_rt']
                    
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
        '''
        get the annotation info according to the index.
        Args:
            index (Int): the index of the data.
        Returns:
            annos (Dict): the annotation info dict.
        '''

        if self.test_mode == True:
            info = self.data_infos[index]
        else: info = self.data_infos[index]
        
        rect = info['calib']['R0_rect'].astype(np.float32)
        Trv2c = info['calib']['Tr_velo_to_cam'].astype(np.float32)

        annos = info['annos']
        # we need other objects to avoid collision when sample
        annos = self.remove_dontcare(annos)

        loc = annos['location']
        dims = annos['dimensions']
        rots = annos['rotation_y']
        gt_names = annos['name']
        gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                      axis=1).astype(np.float32)

        gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(
            self.box_mode_3d, np.linalg.inv(rect @ Trv2c))


        gt_bboxes = annos['bbox']

        selected = self.drop_arrays_by_name(gt_names, ['DontCare'])
        gt_bboxes = gt_bboxes[selected].astype('float32')
        gt_names = gt_names[selected]
        gt_labels = []
        for cat in gt_names:
            if cat in self.CLASSES:
                gt_labels.append(self.CLASSES.index(cat))
            else:
                gt_labels.append(-1)
        gt_labels = np.array(gt_labels).astype(np.int64)
        gt_labels_3d = copy.deepcopy(gt_labels)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            bboxes=gt_bboxes,
            labels=gt_labels,
            gt_names=gt_names)
        return anns_results


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
        
    def evaluate_occupancy(self, occ_results, runner=None, show_dir=None, save=False,use_image_mask=True, **eval_kwargs):
        eval_half_height=eval_kwargs.get('eval_half_height', False)
        from .occ_metrics import Metric_mIoU, Metric_FScore

        self.occ_eval_metrics = Metric_mIoU(
            num_classes=16,
            use_lidar_mask=False,
            use_image_mask=True,
            class_names=self.class_names,
            dynamic_object_idx=[1,2,4,8,9])
        self.occ_eval_metrics_non_mask = Metric_mIoU(
            num_classes=16,
            use_lidar_mask=False,
            use_image_mask=True,
            class_names=self.class_names,
            dynamic_object_idx=[1,2,4,8,9])
        if eval_half_height:
            self.occ_eval_metrics_half_height_top = Metric_mIoU(
                num_classes=16,
                use_lidar_mask=False,
                use_image_mask=True,
            class_names=self.class_names,
            dynamic_object_idx=[1,2,4,8,9])
            self.occ_eval_metrics_half_height_bottom = Metric_mIoU(
                num_classes=16,
                use_lidar_mask=False,
                use_image_mask=True,
            class_names=self.class_names,
            dynamic_object_idx=[1,2,4,8,9])

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
        for occ_pred_w_index in tqdm(occ_results):
            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)

            occ_pred = occ_pred_w_index['pred_occupancy']
            info = self.data_infos[index]
            
            sample_idx = info['image']['image_idx']
            # scene_idx = sample_idx % 1000000 // 1000
            # frame_idx = sample_idx % 1000000 % 1000
        
            pts_filename = self._get_pts_filename(sample_idx)
        
            basename = os.path.basename(pts_filename)
            seq_name = basename[1:4]
            frame_name = basename[4:7]

            occupancy_file_path = os.path.join(self.occupancy_path, seq_name,  '{}_04.npz'.format(frame_name))

            occ_gt = np.load(occupancy_file_path)
 
        
            gt_semantics = occ_gt['voxel_label']
            
            gt_semantics[gt_semantics == self.FREE_LABEL] = len(self.class_names) - 1
            
            
            mask_infov = occ_gt['infov'].astype(bool)
            mask_lidar = occ_gt['origin_voxel_state'].astype(bool)
            mask_camera = occ_gt['final_voxel_state'].astype(bool)
            
            mask = np.ones_like(gt_semantics).astype(bool) # 200, 200, 16

            mask = mask & mask_infov

            mask = mask & mask_camera
            mask_camera = mask.astype(bool)
            

          
            if use_image_mask:
                mask_camera = mask_camera
            else:
                mask_camera = np.ones_like(mask_camera).astype(bool)

            mask_camera_=copy.deepcopy(mask_camera)
            gt_semantics_=copy.deepcopy(gt_semantics)
            if eval_kwargs.get('return_half', False):
                mask_camera=mask_camera[..., :mask_camera.shape[-1]//2]
                gt_semantics=gt_semantics[..., :gt_semantics.shape[-1]//2]

            self.occ_eval_metrics.add_batch(occ_pred[mask_camera], gt_semantics, mask_lidar, mask_camera)
            if self.eval_fscore:
                self.fscore_eval_metrics.add_batch(occ_pred[mask_camera], gt_semantics, mask_lidar, mask_camera)

            mask_camera_non_mask = np.ones_like(mask_camera).astype(bool)
            self.occ_eval_metrics_non_mask.add_batch(occ_pred[mask_camera_non_mask], gt_semantics, mask_lidar, mask_camera_non_mask)

            if eval_half_height:

                mask_camera_half_height_top=mask_camera_[..., -mask_camera_.shape[-1]//2:]
                mask_camera_half_height_bottom=mask_camera_[..., :mask_camera_.shape[-1]//2]
                occ_pred_top=occ_pred[..., -mask_camera_.shape[-1]//2:]
                occ_pred_bottom=occ_pred[..., :mask_camera_.shape[-1]//2]
                gt_semantics_top=gt_semantics_[..., -mask_camera_.shape[-1]//2:]
                gt_semantics_bottom=gt_semantics_[..., :mask_camera_.shape[-1]//2]
                self.occ_eval_metrics_half_height_top.add_batch(occ_pred_top[mask_camera_half_height_top], gt_semantics_top, mask_lidar, mask_camera_half_height_top)
                self.occ_eval_metrics_half_height_bottom.add_batch(occ_pred_bottom[mask_camera_half_height_bottom], gt_semantics_bottom, mask_lidar, mask_camera_half_height_bottom)
   
        res = self.occ_eval_metrics.count_miou()
        print('\n=================results_non_mask====================')
        res_none_mask = self.occ_eval_metrics_non_mask.count_miou()
        
        if eval_half_height:
            print('\n=================results_half_height_top================')
            res_half_height_top = self.occ_eval_metrics_half_height_top.count_miou()
            print('\n=================results_half_height_bottom================')
            res_half_height_bottom = self.occ_eval_metrics_half_height_bottom.count_miou()
            res_half_height_top= {k+'_half_height_top':v for k,v in res_half_height_top.items()}
            res_half_height_bottom= {k+'_half_height_bottom':v for k,v in res_half_height_bottom.items()}
            res.update(res_half_height_top)
            res.update(res_half_height_bottom)

        res_none_mask= {k+'_non_mask':v for k,v in res_none_mask.items()}
        res.update(res_none_mask)
        if self.eval_fscore:
            res.update(self.fscore_eval_metrics.count_fscore())
        

        return res 
    
    def evaluate_rayiou(self, occ_results, runner=None, show_dir=None, **eval_kwargs):
        occ_gts, occ_preds, inst_gts, inst_preds, lidar_origins = [], [], [], [], []
        print('\nStarting Evaluation...')


                
        data_loader = DataLoader(
            EgoPoseDatasetOcc3D(self.data_infos, dataset='waymo'),num_workers=8)
        
        print('\nStarting Evaluation...')
        processed_set = set()
        for occ_pred_w_index in tqdm(occ_results):

            index = occ_pred_w_index['index']
            if index in processed_set: continue
            processed_set.add(index)
            
            output_origin=data_loader.dataset[index][1].unsqueeze(0)
            
            info = self.data_infos[index]

            sample_idx = info['image']['image_idx']
            # scene_idx = sample_idx % 1000000 // 1000
            # frame_idx = sample_idx % 1000000 % 1000
        
            pts_filename = self._get_pts_filename(sample_idx)
        
            basename = os.path.basename(pts_filename)
            seq_name = basename[1:4]
            frame_name = basename[4:7]

            occ_path = os.path.join(self.occupancy_path, seq_name,  '{}_04.npz'.format(frame_name))
            
            occ_gt = np.load(occ_path, allow_pickle=True)
            gt_semantics = occ_gt['voxel_label']

            sem_pred = occ_pred_w_index['pred_occupancy']  # [B, N]

            occ_class_names = self.class_names

            lidar_origins.append(output_origin)
            occ_gts.append(gt_semantics)
            occ_preds.append(sem_pred)
        
        if len(inst_preds) > 0:
            results = main_raypq(occ_preds, occ_gts, inst_preds, inst_gts, lidar_origins, occ_class_names=occ_class_names)
            results.update(main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names))
            return results
        else:
            return main_rayiou(occ_preds, occ_gts, lidar_origins, occ_class_names=occ_class_names)