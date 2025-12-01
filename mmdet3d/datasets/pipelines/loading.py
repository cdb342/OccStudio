# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Dubing Chen
import mmcv
import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
import os.path as osp
from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile
from ...core.bbox import LiDARInstance3DBoxes
from ..builder import PIPELINES
from copy import deepcopy
import cv2
import os
from torchvision.transforms.functional import rotate
from mmcv.parallel.data_container import DataContainer
import torch.nn.functional as F
import torchvision.transforms.functional as TF
occ_class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]
def trans_matrix(T, R):
    tm = np.eye(4)
    tm[:3, :3] = R.rotation_matrix
    tm[:3, 3] = T
    return tm
def get_scene_token( info):
    if 'scene_token' in info:
        scene_name = info['scene_token']
    else:
        scene_name = info['occ_path'].split('openocc_v2/')[-1].split('/')[0]
    return scene_name
def get_ego_from_lidar(info):
    ego_from_lidar = trans_matrix(
        np.array(info['lidar2ego_translation']), 
        Quaternion(info['lidar2ego_rotation']))

    return ego_from_lidar

def get_global_pose( info, inverse=False):

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
def get_origin( info,scene_frame):

    ref_lidar_from_global =get_global_pose(info, inverse=True)
    ref_ego_from_lidar = get_ego_from_lidar(info)


    # NOTE: getting output frames
    output_origin_list = []
    for curr_index in range(len(scene_frame)):

        global_from_curr = get_global_pose(scene_frame[curr_index], inverse=False)
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




@PIPELINES.register_module()
class LoadOccFlowGTFromFile(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """


    def __init__(self, occupancy_path='/mount/dnn_data/occupancy_2023/gts',
                num_classes=17,
                ignore_nonvisible=False,
                mask='mask_camera',
                ignore_classes=[],
                fix_void=True,
                flow_mask=False,
                occ_ray_mask=False,
                occ_ray_mask2flow_only=False,
                load_next_occ=False,
                expert_classes=None,
                ) :
        self.occupancy_path = occupancy_path
        self.num_classes = num_classes
        self.ignore_nonvisible = ignore_nonvisible
        self.mask = mask

        self.ignore_classes=ignore_classes

        self.fix_void = fix_void
        self.flow_mask = flow_mask
        self.occ_ray_mask=occ_ray_mask
        self.occ_ray_mask2flow_only=occ_ray_mask2flow_only
        self.load_next_occ=load_next_occ
        self.expert_classes=expert_classes

    def trans_occ(self,occupancy,results):
        # to BEVDet format
        occupancy = occupancy.permute(2, 0, 1)
        occupancy = torch.rot90(occupancy, 1, [1, 2])
        occupancy = torch.flip(occupancy, [1])
        occupancy = occupancy.permute(1, 2, 0)

        
        if self.fix_void:
            occupancy[occupancy<255] = occupancy[occupancy<255] + 1


        for class_ in self.ignore_classes:
            occupancy[occupancy==class_] = 255
        if results['rotate_bda'] != 0:
            occupancy = occupancy.permute(2, 0, 1)
            occupancy = rotate(occupancy, -results['rotate_bda'], fill=255).permute(1, 2, 0)

        if results['flip_dx']:
            occupancy = torch.flip(occupancy, [1])


        if results['flip_dy']:
            occupancy = torch.flip(occupancy, [0])
                ###############
        return occupancy



    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        scene_name = results['curr']['scene_name']
        sample_token = results['curr']['token']


        occupancy_file_path = osp.join(self.occupancy_path, scene_name, sample_token, 'labels.npz')
        data = np.load(occupancy_file_path)
        
        occupancy = torch.tensor(data['semantics'])
        occ_flow= torch.tensor(data['flow'])
        if self.occ_ray_mask:
            occupancy_original = occupancy.clone()
            ray_mask_save_path=occupancy_file_path.replace('openocc_v2','openocc_v2_ray_mask')
            metas=np.load(ray_mask_save_path)
            ray_mask=metas['ray_mask2']
            ray_mask=torch.tensor(ray_mask)
            if not self.occ_ray_mask2flow_only:
                occupancy[~ray_mask]=255


        # to FBOcc format
        occupancy = occupancy.permute(2, 0, 1)
        occupancy = torch.rot90(occupancy, 1, [1, 2])

        occupancy = torch.flip(occupancy, [1])
        occupancy = occupancy.permute(1, 2, 0)
        #########
        occ_flow = occ_flow.permute(2, 0, 1,3)
        occ_flow = torch.rot90(occ_flow, 1, [1, 2])
        occ_flow=occ_flow[...,[1,0]]
        # occ_flow[...,0]=-occ_flow[...,0]

        occ_flow = torch.flip(occ_flow, [1])
        # occ_flow[...,0]=-occ_flow[...,0]
        occ_flow = occ_flow.permute(1, 2, 0,3)

        #########
        if self.occ_ray_mask:
            occupancy_original = occupancy_original.permute(2, 0, 1)
            occupancy_original = torch.rot90(occupancy_original, 1, [1, 2])
            occupancy_original = torch.flip(occupancy_original, [1])
            occupancy_original = occupancy_original.permute(1, 2, 0)
        
        #########
        #########
        if self.expert_classes is not None:
            for class_ in occ_class_names:
                if class_ not in self.expert_classes:
                    occupancy[occupancy==occ_class_names.index]=0   
        #############

        if self.flow_mask:
            
            mask=occupancy<8
            if self.occ_ray_mask2flow_only:
                mask=ray_mask&mask
            occ_flow[~mask]=float('inf')

        if self.fix_void:
            occupancy[occupancy<255] = occupancy[occupancy<255] + 1
            ########
            if self.occ_ray_mask:
                occupancy_original[occupancy_original<255] = occupancy_original[occupancy_original<255] + 1
            ########

        for class_ in self.ignore_classes:
            occupancy[occupancy==class_] = 255

        if results['rotate_bda'] != 0:
            # import pdb; pdb.set_trace()
            occupancy = occupancy.permute(2, 0, 1)
            occupancy = rotate(occupancy, -results['rotate_bda'], fill=255).permute(1, 2, 0)
            
            ######################
            occ_flow = occ_flow.permute(3,2, 0, 1)
            occ_flow = rotate(occ_flow, -results['rotate_bda'], fill=float('inf')).permute(2, 3, 1,0)
            ######################
            ######################
            if self.occ_ray_mask:
                occupancy_original = occupancy_original.permute(2, 0, 1)
                occupancy_original = rotate(occupancy_original, -results['rotate_bda'], fill=255).permute(1, 2, 0)
            ######################

        if results['flip_dx']:
            occupancy = torch.flip(occupancy, [1])
            ############
            occ_flow = torch.flip(occ_flow, [1])
            occ_flow[...,1]=-occ_flow[...,1]
            ############
             ############
            if self.occ_ray_mask:
                occupancy_original = torch.flip(occupancy_original, [1])
            ############

        if results['flip_dy']:
            occupancy = torch.flip(occupancy, [0])
            ###############
            occ_flow = torch.flip(occ_flow, [0])
            occ_flow[...,0]=-occ_flow[...,0]
            ###############
            ###############
            if self.occ_ray_mask:
                occupancy_original = torch.flip(occupancy_original, [0])
            ###############

        occ_flow[occ_flow==-float('inf')]=float('inf')
        occ_flow[occ_flow[...,1]==float('inf')]=float('inf')
        occ_flow[occ_flow[...,0]==float('inf')]=float('inf')

        results['gt_occupancy'] = occupancy
        results['gt_occupancy_ori'] = occupancy
        results['gt_occ_flow'] = occ_flow

        if self.load_next_occ:
            next_occ_gt_path=results['next_occ_gt_path']
            if next_occ_gt_path is not None:
                next_occ_gt_path=next_occ_gt_path.replace('gts','openocc_v2')
                next_occ_gt_path=os.path.join(next_occ_gt_path,'labels.npz')
                if os.path.exists(next_occ_gt_path):
                    next_occ_gt=np.load(next_occ_gt_path)['semantics']
                    next_occ_gt=torch.tensor(next_occ_gt)
                    # to BEVDet format
                    next_occ_gt = next_occ_gt.permute(2, 0, 1)
                    next_occ_gt = torch.rot90(next_occ_gt, 1, [1, 2])
                    next_occ_gt = torch.flip(next_occ_gt, [1])
                    next_occ_gt = next_occ_gt.permute(1, 2, 0)
            else:
                next_occ_gt=torch.tensor(np.nan)
            results['future_gt_occ']=DataContainer(next_occ_gt)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)
            
@PIPELINES.register_module()
class LoadSemanticImageMask(object):
    def __init__(self, mask_file_path='./data/nus_sem'):
        self.mask_file_path = mask_file_path
    
    def __call__(self, results):

        masks = []
        for cam in results['cam_names']:
            data_token = results['curr']['cams'][cam]['sample_data_token']
            filename = osp.join(self.mask_file_path, data_token+'.png')
            img = Image.open(filename)
            img_augs = results['img_augs'][cam]
            resize, resize_dims, crop, flip, rotate = img_augs        
            img = self.img_transform_core(img, resize_dims, crop, flip, rotate)
            img = np.array(img)
            masks.append(img)
        masks = np.stack(masks, 0)
        results['gt_img_sem_masks'] = torch.from_numpy(masks)
        return results
        
    
    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims, resample=0)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate, resample=0, expand=0)
        return img


@PIPELINES.register_module()
class LoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool, optional): Whether to convert the img to float32.
            Defaults to False.
        color_type (str, optional): Color type of the file.
            Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        
        try:
            mmcv.use_backend('turbojpeg')
        except ImportError:
            mmcv.use_backend('cv2')
        

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data.
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formatting.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@PIPELINES.register_module()
class LoadImageFromFileMono3D(object):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """


    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        super().__call__(results)
        results['cam2img'] = results['img_info']['cam_intrinsic']
        return results


import torch.nn as nn
@PIPELINES.register_module()
class LoadVqGT(object):

    def __init__(self, vqgt_path='data/vq_res18_gt',) :
        self.vqgt_path = vqgt_path


    def __call__(self, results):
        scene_name = results['curr']['scene_name']
        sample_token = results['curr']['token']
        vqgt= np.load(osp.join(self.vqgt_path, scene_name, sample_token,'labels.npz'))
        vqgt=vqgt['arr_0']
        vqgt = torch.tensor(vqgt).long()
        results['vqgt'] = vqgt.reshape(200,200)
        return results

@PIPELINES.register_module()
class LoadOccupancy(object):
    """Load an image from file in monocular 3D object detection. Compared to 2D
    detection, additional camera parameters need to be loaded.

    Args:
        kwargs (dict): Arguments are the same as those in
            :class:`LoadImageFromFile`.
    """

    def __init__(self, occupancy_path='/mount/dnn_data/occupancy_2023/gts',
                    num_classes=17,
                    ignore_nonvisible=False,
                    mask='mask_camera',
                    ignore_classes=[],
                    fix_void=True,
                    flow_path=None,
                    load_flow=False,
                    label2_path='visual/aug_gts',
                    gts_surroundocc=False,
                    gts_openoccupancy=False,
                    occ_size=[200,200,16],
                    ) :
        self.occupancy_path = occupancy_path
        self.num_classes = num_classes
        self.ignore_nonvisible = ignore_nonvisible
        self.mask = mask

        self.ignore_classes=ignore_classes

        self.fix_void = fix_void
        self.flow_path=flow_path
        self.load_flow=load_flow
        self.label2_path=label2_path
        self.gts_surroundocc=gts_surroundocc
        self.gts_openoccupancy=gts_openoccupancy
        self.occ_size=occ_size


    def __call__(self, results):
        """Call functions to load image and get image meta information.

        Args:
            results (dict): Result dict from :obj:`mmdet.CustomDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """
        
        if self.gts_surroundocc:
            lidar_path=results['pts_filename'] 
            occupancy_file_path=lidar_path.replace('samples/LIDAR_TOP','gts_surroundocc/samples')
            occupancy_file_path=occupancy_file_path+'.npy'
            data = np.load(occupancy_file_path)
            occ = np.zeros(self.occ_size)
            occ[data[:, 0], data[:, 1], data[:, 2]] = data[:, 3]
            occ = np.where(occ== 0, 17, occ)
            occupancy=torch.tensor(occ.copy()).long()
            visible_mask=torch.ones_like(occupancy).bool()
        elif self.gts_openoccupancy:
            scene_token = 'scene_'+results['curr']['scene_token']
            lidar_token = results['curr']['lidar_token']
            occupancy_file_path = osp.join(self.occupancy_path.replace('gts','nuScenes-Occupancy-v0.1'), scene_token, 'occupancy',lidar_token)+'.npy'
            data = np.load(occupancy_file_path)
            occ = np.zeros(self.occ_size)
            occ[data[:, 2], data[:, 1], data[:, 0]] = data[:, 3]
            occ = np.where(occ== 0, 17, occ)
            occupancy=torch.tensor(occ.copy()).long()
            visible_mask=torch.ones_like(occupancy).bool()
        else:
            scene_name = results['curr']['scene_name']
            sample_token = results['curr']['token']


            occupancy_file_path = osp.join(self.occupancy_path, scene_name, sample_token, 'labels.npz')
            data = np.load(occupancy_file_path)
            occupancy = torch.tensor(data['semantics'])
            visible_mask = torch.tensor(data[self.mask])

        ######
        occupancy_original = occupancy.clone()
        #####
        if self.ignore_nonvisible:
            occupancy[~visible_mask.to(torch.bool)] = 255
        if self.load_flow:
            flow_file_path = osp.join(self.flow_path, scene_name, sample_token, 'labels.npz')
            data = np.load(flow_file_path)
            occ_flow = torch.tensor(data['flow'])


        # to FBOcc format
        occupancy = occupancy.permute(2, 0, 1)
        occupancy = torch.rot90(occupancy, 1, [1, 2])
        occupancy = torch.flip(occupancy, [1])
        occupancy = occupancy.permute(1, 2, 0)
        #########
        
        occupancy_original = occupancy_original.permute(2, 0, 1)
        occupancy_original = torch.rot90(occupancy_original, 1, [1, 2])
        occupancy_original = torch.flip(occupancy_original, [1])
        occupancy_original = occupancy_original.permute(1, 2, 0)

        if self.load_flow:
            occ_flow = occ_flow.permute(2, 0, 1,3)
            occ_flow = torch.rot90(occ_flow, 1, [1, 2])
            occ_flow=occ_flow[...,[1,0]]
            # occ_flow[...,0]=-occ_flow[...,0]

            occ_flow = torch.flip(occ_flow, [1])
            # occ_flow[...,0]=-occ_flow[...,0]
            occ_flow = occ_flow.permute(1, 2, 0,3)
        
        if self.fix_void:
            occupancy[occupancy<255] = occupancy[occupancy<255] + 1
            ########
            occupancy_original[occupancy_original<255] = occupancy_original[occupancy_original<255] + 1
            ########

        for class_ in self.ignore_classes:
            occupancy[occupancy==class_] = 255

        if results['rotate_bda'] != 0:
            occupancy = occupancy.permute(2, 0, 1)
            occupancy = rotate(occupancy, -results['rotate_bda'], fill=255).permute(1, 2, 0)
            
            ######################
            occupancy_original = occupancy_original.permute(2, 0, 1)
            occupancy_original = rotate(occupancy_original, -results['rotate_bda'], fill=255).permute(1, 2, 0)
            ######################
            if self.load_flow:
                ######################
                occ_flow = occ_flow.permute(3,2, 0, 1)
                occ_flow = rotate(occ_flow, -results['rotate_bda'], fill=float('inf')).permute(2, 3, 1,0)
                ######################

        if results['flip_dx']:
            occupancy = torch.flip(occupancy, [1])
            ############
            occupancy_original = torch.flip(occupancy_original, [1])
            ############
            if self.load_flow:
                                ############
                occ_flow = torch.flip(occ_flow, [1])
                occ_flow[...,1]=-occ_flow[...,1]
                ############
        if results['flip_dy']:
            occupancy = torch.flip(occupancy, [0])
            ###############
            occupancy_original = torch.flip(occupancy_original, [0])
            ###############
            if self.load_flow:
                    ###############
                occ_flow = torch.flip(occ_flow, [0])
                occ_flow[...,0]=-occ_flow[...,0]

        results['gt_occupancy'] = occupancy
        results['visible_mask'] = visible_mask
        results['visible_mask_bev'] = (occupancy==255).sum(-1)
        results['gt_occupancy_ori'] = occupancy_original

        if self.load_flow:
            results['gt_occ_flow'] = occ_flow
        return results


@PIPELINES.register_module()
class LoadPointsFromMultiSweeps(object):
    """Load points from multiple sweeps.

    This is usually used for nuScenes dataset to utilize previous sweeps.

    Args:
        sweeps_num (int, optional): Number of sweeps. Defaults to 10.
        load_dim (int, optional): Dimension number of the loaded points.
            Defaults to 5.
        use_dim (list[int], optional): Which dimension to use.
            Defaults to [0, 1, 2, 4].
        time_dim (int, optional): Which dimension to represent the timestamps
            of each points. Defaults to 4.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
        pad_empty_sweeps (bool, optional): Whether to repeat keyframe when
            sweeps is empty. Defaults to False.
        remove_close (bool, optional): Whether to remove close points.
            Defaults to False.
        test_mode (bool, optional): If `test_mode=True`, it will not
            randomly sample sweeps but select the nearest N frames.
            Defaults to False.
    """

    def __init__(self,
                 sweeps_num=10,
                 load_dim=5,
                 use_dim=[0, 1, 2, 4],
                 time_dim=4,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 remove_close=False,
                 translate2ego=False,
                 test_mode=False):
        self.load_dim = load_dim
        self.sweeps_num = sweeps_num
        self.use_dim = use_dim
        self.time_dim = time_dim
        assert time_dim < load_dim, \
            f'Expect the timestamp dimension < {load_dim}, got {time_dim}'
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.remove_close = remove_close
        self.test_mode = test_mode
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        self.translate2ego = translate2ego
        
    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)
        return points

    def _remove_close(self, points, radius=1.0):
        """Removes point too close within a certain radius from origin.

        Args:
            points (np.ndarray | :obj:`BasePoints`): Sweep points.
            radius (float, optional): Radius below which points are removed.
                Defaults to 1.0.

        Returns:
            np.ndarray: Points after removing.
        """
        if isinstance(points, np.ndarray):
            points_numpy = points
        elif isinstance(points, BasePoints):
            points_numpy = points.tensor.numpy()
        else:
            raise NotImplementedError
        x_filt = np.abs(points_numpy[:, 0]) < radius
        y_filt = np.abs(points_numpy[:, 1]) < radius
        not_close = np.logical_not(np.logical_and(x_filt, y_filt))
        return points[not_close]

    def __call__(self, results):
        """Call function to load multi-sweep point clouds from files.

        Args:
            results (dict): Result dict containing multi-sweep point cloud
                filenames.

        Returns:
            dict: The result dict containing the multi-sweep points data.
                Added key and value are described below.

                - points (np.ndarray | :obj:`BasePoints`): Multi-sweep point
                    cloud arrays.
        """
        points = results['points']
        points.tensor[:, self.time_dim] = 0
        sweep_points_list = [points]
        ts = results['timestamp']
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                if self.remove_close:
                    sweep_points_list.append(self._remove_close(points))
                else:
                    sweep_points_list.append(points)
        else:
            if len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = np.arange(self.sweeps_num)
            else:
                choices = np.random.choice(
                    len(results['sweeps']), self.sweeps_num, replace=False)
            for idx in choices:
                sweep = results['sweeps'][idx]
                points_sweep = self._load_points(sweep['data_path'])
                points_sweep = np.copy(points_sweep).reshape(-1, self.load_dim)
                if self.remove_close:
                    points_sweep = self._remove_close(points_sweep)
                sweep_ts = sweep['timestamp'] / 1e6
                points_sweep[:, :3] = points_sweep[:, :3] @ sweep[
                    'sensor2lidar_rotation'].T
                points_sweep[:, :3] += sweep['sensor2lidar_translation']
                points_sweep[:, self.time_dim] = ts - sweep_ts
                points_sweep = points.new_point(points_sweep)
                sweep_points_list.append(points_sweep)

        points = points.cat(sweep_points_list)
        points = points[:, self.use_dim]
        results['points'] = points
        if self.translate2ego:
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
            results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)
            results['points'].tensor[:, :3]  = results['points'].tensor[:, :3].matmul(lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3]
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return f'{self.__class__.__name__}(sweeps_num={self.sweeps_num})'


@PIPELINES.register_module()
class PointsFromLidartoEgo(object):
    
    def __init__(self, translate2ego=True):
        self.translate2ego = translate2ego

    def __call__(self, results):
        if self.translate2ego:
            lidar2lidarego = np.eye(4, dtype=np.float32)
            lidar2lidarego[:3, :3] = Quaternion(
            results['curr']['lidar2ego_rotation']).rotation_matrix
            lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            lidar2lidarego = torch.from_numpy(lidar2lidarego)
            results['points'].tensor[:, :3]  = results['points'].tensor[:, :3].matmul(lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3]
        return results


@PIPELINES.register_module()
class PointSegClassMapping(object):
    """Map original semantic class to valid category ids.

    Map valid classes as 0~len(valid_cat_ids)-1 and
    others as len(valid_cat_ids).

    Args:
        valid_cat_ids (tuple[int]): A tuple of valid category.
        max_cat_id (int, optional): The max possible cat_id in input
            segmentation mask. Defaults to 40.
    """

    def __init__(self, valid_cat_ids, max_cat_id=40):
        assert max_cat_id >= np.max(valid_cat_ids), \
            'max_cat_id should be greater than maximum id in valid_cat_ids'

        self.valid_cat_ids = valid_cat_ids
        self.max_cat_id = int(max_cat_id)

        # build cat_id to class index mapping
        neg_cls = len(valid_cat_ids)
        self.cat_id2class = np.ones(
            self.max_cat_id + 1, dtype=np.int) * neg_cls
        for cls_idx, cat_id in enumerate(valid_cat_ids):
            self.cat_id2class[cat_id] = cls_idx

    def __call__(self, results):
        """Call function to map original semantic class to valid category ids.

        Args:
            results (dict): Result dict containing point semantic masks.

        Returns:
            dict: The result dict containing the mapped category ids.
                Updated key and value are described below.

                - pts_semantic_mask (np.ndarray): Mapped semantic masks.
        """
        assert 'pts_semantic_mask' in results
        pts_semantic_mask = results['pts_semantic_mask']

        converted_pts_sem_mask = self.cat_id2class[pts_semantic_mask]

        results['pts_semantic_mask'] = converted_pts_sem_mask
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(valid_cat_ids={self.valid_cat_ids}, '
        repr_str += f'max_cat_id={self.max_cat_id})'
        return repr_str


@PIPELINES.register_module()
class NormalizePointsColor(object):
    """Normalize color of points.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
    """

    def __init__(self, color_mean):
        self.color_mean = color_mean

    def __call__(self, results):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
                Updated key and value are described below.

                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = results['points']
        assert points.attribute_dims is not None and \
            'color' in points.attribute_dims.keys(), \
            'Expect points have color attribute'
        if self.color_mean is not None:
            points.color = points.color - \
                points.color.new_tensor(self.color_mean)
        points.color = points.color / 255.0
        results['points'] = points
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(color_mean={self.color_mean})'
        return repr_str


@PIPELINES.register_module()
class LoadPointsFromFile(object):
    """Load Points From File.

    Load points from file.

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:
            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int, optional): The dimension of the loaded points.
            Defaults to 6.
        use_dim (list[int], optional): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool, optional): Whether to use shifted height.
            Defaults to False.
        use_color (bool, optional): Whether to use color features.
            Defaults to False.
        file_client_args (dict, optional): Config dict of file clients,
            refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details. Defaults to dict(backend='disk').
    """

    def __init__(self,
                 coord_type,
                 load_dim=6,
                 use_dim=[0, 1, 2],
                 shift_height=False,
                 use_color=False,
                 dtype='float32',
                 file_client_args=dict(backend='disk'),
                 translate2ego=True,
                 point_with_semantic=False,
                 ):
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        if dtype=='float32':
            self.dtype = np.float32
        elif dtype== 'float16':
            self.dtype = np.float16
        else:
            assert False
        self.translate2ego = translate2ego
        self.point_with_semantic=point_with_semantic

    def _load_points(self, pts_filename):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            pts_bytes = self.file_client.get(pts_filename)
            points = np.frombuffer(pts_bytes, dtype=self.dtype)
        except ConnectionError:
            mmcv.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=self.dtype)

        return points


    def __call__(self, results):
        """Call function to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
                Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_filename = results['pts_filename']
        points = self._load_points(pts_filename)
        points = points.reshape(-1, self.load_dim)
        points = points[:, self.use_dim]



        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = dict()
            attribute_dims.update(
                dict(color=[
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
                ]))

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)

        results['points'] = points
        if self.translate2ego:
            if 'lidar2lidarego' not in results['curr']:
                lidar2lidarego = np.eye(4, dtype=np.float32)
                lidar2lidarego[:3, :3] = Quaternion(
                results['curr']['lidar2ego_rotation']).rotation_matrix
                lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
                lidar2lidarego = torch.from_numpy(lidar2lidarego)
            else:
                lidar2lidarego=results['curr']['lidar2lidarego']
            results['points'].tensor[:, :3]  = results['points'].tensor[:, :3].matmul(lidar2lidarego[:3, :3].T) + lidar2lidarego[:3, 3]

        ##################################3
        if self.point_with_semantic:
            # lidar_sd_token = token_map[results['sample_idx']]
            # lidarseg_labels_filename = os.path.join(nusc.dataroot,
            #                                         nusc.get('lidarseg', lidar_sd_token)['filename'])

            # points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
            # points_label = np.vectorize(learning_map.__getitem__)(points_label)
            save_dir='./data/lidar_seg'
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            point_name=results['pts_filename'].split('/')[-1].split('.')[0]
            save_path=os.path.join(save_dir,point_name+'.npz')
            results['points_semantics']=np.load(save_path)['lidar_seg'].flatten()
            
            # np.savez_compressed(save_path, lidar_seg=points_label)
        #############################################
        
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'file_client_args={self.file_client_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        return repr_str




@PIPELINES.register_module()
class LoadPointsFromDict(LoadPointsFromFile):
    """Load Points From Dict."""

    def __call__(self, results):
        assert 'points' in results
        return results


@PIPELINES.register_module()
class LoadAnnotations3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype=np.int64,
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_infos'][0]
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_infos'][1]
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_infos']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_infos']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int64)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int64)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_infos']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int64)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@PIPELINES.register_module()
class PointToMultiViewDepth(object):

    def __init__(self, grid_config, downsample=1,load_semantic_map=False,fix_void=False):
        self.downsample = downsample
        self.grid_config = grid_config
        self.load_semantic_map=load_semantic_map
        self.fix_void=fix_void

    def points2depthmap(self, points, height, width):
        height, width = height // self.downsample, width // self.downsample
        depth_map = torch.zeros((height, width), dtype=torch.float32)
        coor = torch.round(points[:, :2] / self.downsample)
        depth = points[:, 2]
        if self.load_semantic_map:
            semantics=points[:,3]
        kept1 = (coor[:, 0] >= 0) & (coor[:, 0] < width) & (
            coor[:, 1] >= 0) & (coor[:, 1] < height) & (
                depth < self.grid_config['depth'][1]) & (
                    depth >= self.grid_config['depth'][0])
        coor, depth = coor[kept1], depth[kept1]
        if self.load_semantic_map:
            semantics=semantics[kept1]
        ranks = coor[:, 0] + coor[:, 1] * width
        sort = (ranks + depth / 100.).argsort()
        coor, depth, ranks = coor[sort], depth[sort], ranks[sort]
        if self.load_semantic_map:
            semantics=semantics[sort]

        kept2 = torch.ones(coor.shape[0], device=coor.device, dtype=torch.bool)
        kept2[1:] = (ranks[1:] != ranks[:-1])
        coor, depth = coor[kept2], depth[kept2]
        if self.load_semantic_map:
            semantics=semantics[kept2]
        coor = coor.to(torch.long)
        depth_map[coor[:, 1], coor[:, 0]] = depth
        if self.load_semantic_map:
            semantic_map=torch.ones((height, width), dtype=torch.float32)*255
            if self.fix_void:
                semantics=semantics+1
            semantic_map[coor[:, 1], coor[:, 0]] = semantics
            return depth_map,semantic_map
        return depth_map

    def __call__(self, results):
        points_lidar = results['points']
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans, bda = results['img_inputs'][4:]
        depth_map_list = []
        semantic_map_list=[]
        for cid in range(len(results['cam_names'])):
            cam_name = results['cam_names'][cid]
            # print(cam_name,11111111111111)
            # lidar2lidarego = np.eye(4, dtype=np.float32)
            # lidar2lidarego[:3, :3] = Quaternion(
            #     results['curr']['lidar2ego_rotation']).rotation_matrix
            # lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
            # lidar2lidarego = torch.from_numpy(lidar2lidarego)

            if 'lidarego2global' not in  results['curr']:
                lidarego2global = np.eye(4, dtype=np.float32)
                lidarego2global[:3, :3] = Quaternion(
                    results['curr']['ego2global_rotation']).rotation_matrix
                lidarego2global[:3, 3] = results['curr']['ego2global_translation']
            else:
                lidarego2global=results['curr']['lidarego2global']
            lidarego2global = torch.from_numpy(lidarego2global)

            if 'cam2camego' not in  results['curr']['cams'][cam_name]:
                cam2camego = np.eye(4, dtype=np.float32)
                cam2camego[:3, :3] = Quaternion(
                    results['curr']['cams'][cam_name]
                    ['sensor2ego_rotation']).rotation_matrix
                cam2camego[:3, 3] = results['curr']['cams'][cam_name][
                    'sensor2ego_translation']
            else:
                cam2camego=results['curr']['cams'][cam_name]['cam2camego']
            cam2camego = torch.from_numpy(cam2camego)

            if 'camego2global' not in results['curr']['cams'][cam_name]:
                camego2global = np.eye(4, dtype=np.float32)
                camego2global[:3, :3] = Quaternion(
                    results['curr']['cams'][cam_name]
                    ['ego2global_rotation']).rotation_matrix
                camego2global[:3, 3] = results['curr']['cams'][cam_name][
                    'ego2global_translation']
            else:
                camego2global=results['curr']['cams'][cam_name]['camego2global']
            camego2global = torch.from_numpy(camego2global)

            cam2img = np.eye(4, dtype=np.float32)
            cam2img = torch.from_numpy(cam2img)
            cam2img[:3, :3] = intrins[cid]

            lidar2cam = torch.inverse(camego2global.matmul(cam2camego)).matmul(lidarego2global)
            # lidarego2global.matmul(lidar2lidarego))

            lidar2img = cam2img.matmul(lidar2cam.float())
            points_img = points_lidar.tensor[:, :3].matmul(
                lidar2img[:3, :3].T) + lidar2img[:3, 3].unsqueeze(0)
            points_img = torch.cat(
                [points_img[:, :2] / points_img[:, 2:3], points_img[:, 2:3]],
                1)
            points_img = points_img.matmul(
                post_rots[cid].T) + post_trans[cid:cid + 1, :]
            if self.load_semantic_map:
                points_semantics=results['points_semantics']
                points_img=torch.cat([points_img,torch.from_numpy(points_semantics).unsqueeze(1)],1)
            depth_map = self.points2depthmap(points_img, imgs.shape[2],imgs.shape[3])  
            
            if self.load_semantic_map:
                depth_map,semantic_map=depth_map
                semantic_map_list.append(semantic_map)
            depth_map_list.append(depth_map)
          
        depth_map = torch.stack(depth_map_list)

        results['gt_depth'] = depth_map
        
        if self.load_semantic_map:
            semantic_map=torch.stack(semantic_map_list)
            results['gt_semantic_map'] = semantic_map
        return results





def mmlabNormalize(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, debug=False):
    from mmcv.image.photometric import imnormalize
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    to_rgb = to_rgb
    if debug:
        print('warning, debug in mmlabNormalize')
        img = np.asarray(img) # not normalize for visualization
    else:
        img = imnormalize(np.array(img), mean, std, to_rgb)
    img = torch.tensor(img).float().permute(2, 0, 1).contiguous()
    return img



@PIPELINES.register_module()
class PrepareImageInputs(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
        self,
        data_config,
        is_train=False,
        sequential=False,
        ego_cam='CAM_FRONT',

        normalize_cfg=dict(
             mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True, debug=False,
        ),
        save_stereo=False,
        depth_his_fusion_pre=False,
        geometry_his_fusion=False,
        img_scale=None,
        additional_sem_gt_path=None,
        fix_void=False,
        gts_surroundocc=False,
        depth_stereo=True,
    ):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = mmlabNormalize
        self.sequential = sequential
        self.ego_cam = ego_cam
        self.normalize_cfg = normalize_cfg
        
        self.save_stereo=save_stereo
        self.depth_his_fusion_pre=depth_his_fusion_pre
        self.geometry_his_fusion=geometry_his_fusion
        self.img_scale = img_scale
        self.additional_sem_gt_path=additional_sem_gt_path
        self.fix_void=fix_void
        self.gts_surroundocc=gts_surroundocc
        self.depth_stereo=depth_stereo
    def pad(self, img):
        # to pad the 5 input images into a same size (for Waymo)
        if img.shape[0] != self.img_scale[0]:
            # print(self.img_scale,2222222222222)
            padded = np.zeros((self.img_scale[0],self.img_scale[1],3)).astype(np.uint8)
            # print(padded.shape,33333333333333333)
            padded[0:img.shape[0], 0:img.shape[1], :] = img
            img = padded
        return img
    def get_rot(self, h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran, resize, resize_dims,
                      crop, flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(
                self.data_config['cams']):
            cam_names = np.random.choice(
                self.data_config['cams'],
                self.data_config['Ncams'],
                replace=False)
        else:
            cam_names = self.data_config['cams']
        return cam_names

    def sample_augmentation(self, H, W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW) / float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) *
                         newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW) / float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_sensor2ego_transformation(self,
                                      cam_info,
                                      key_info,
                                      cam_name,
                                      ego_cam=None):
        if ego_cam is None:
            ego_cam = cam_name
        # import pdb;pdb.set_trace()
        if 'cam2camego' not in cam_info['cams'][cam_name]:
            w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
            # sweep sensor to sweep ego
            sweepsensor2sweepego_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            sweepsensor2sweepego_tran = torch.Tensor(
                cam_info['cams'][cam_name]['sensor2ego_translation'])
            sweepsensor2sweepego = sweepsensor2sweepego_rot.new_zeros((4, 4))
            sweepsensor2sweepego[3, 3] = 1
            sweepsensor2sweepego[:3, :3] = sweepsensor2sweepego_rot
            sweepsensor2sweepego[:3, -1] = sweepsensor2sweepego_tran
        else:
            sweepsensor2sweepego=torch.from_numpy(cam_info['cams'][cam_name]['cam2camego'])
        if 'camego2global' not in cam_info['cams'][cam_name]:
            # sweep ego to global
            w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
            sweepego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            sweepego2global_tran = torch.Tensor(
                cam_info['cams'][cam_name]['ego2global_translation'])
            sweepego2global = sweepego2global_rot.new_zeros((4, 4))
            sweepego2global[3, 3] = 1
            sweepego2global[:3, :3] = sweepego2global_rot
            sweepego2global[:3, -1] = sweepego2global_tran
        else:
            sweepego2global=torch.from_numpy(cam_info['cams'][cam_name]['camego2global'])
        if 'camego2global' not in key_info['cams'][ego_cam]:
            # global sensor to cur ego
            w, x, y, z = key_info['cams'][ego_cam]['ego2global_rotation']
            keyego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            keyego2global_tran = torch.Tensor(
                key_info['cams'][ego_cam]['ego2global_translation'])
            keyego2global = keyego2global_rot.new_zeros((4, 4))
            keyego2global[3, 3] = 1
            keyego2global[:3, :3] = keyego2global_rot
            keyego2global[:3, -1] = keyego2global_tran
        else:
            keyego2global=torch.from_numpy(key_info['cams'][ego_cam]['camego2global'])
        global2keyego = keyego2global.inverse()

        sweepsensor2keyego = \
            global2keyego @ sweepego2global @ sweepsensor2sweepego

        if 'camego2global' not in key_info['cams'][cam_name]:
            # global sensor to cur ego
            w, x, y, z = key_info['cams'][cam_name]['ego2global_rotation']
            keyego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            keyego2global_tran = torch.Tensor(
                key_info['cams'][cam_name]['ego2global_translation'])
            keyego2global = keyego2global_rot.new_zeros((4, 4))
            keyego2global[3, 3] = 1
            keyego2global[:3, :3] = keyego2global_rot
            keyego2global[:3, -1] = keyego2global_tran
        else:
            keyego2global=torch.from_numpy(key_info['cams'][cam_name]['camego2global'])
        global2keyego = keyego2global.inverse()

        if 'cam2camego' not in key_info['cams'][cam_name]:
            # cur ego to sensor
            w, x, y, z = key_info['cams'][cam_name]['sensor2ego_rotation']
            keysensor2keyego_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            keysensor2keyego_tran = torch.Tensor(
                key_info['cams'][cam_name]['sensor2ego_translation'])
            keysensor2keyego = keysensor2keyego_rot.new_zeros((4, 4))
            keysensor2keyego[3, 3] = 1
            keysensor2keyego[:3, :3] = keysensor2keyego_rot
            keysensor2keyego[:3, -1] = keysensor2keyego_tran
        else:
            keysensor2keyego=torch.from_numpy(key_info['cams'][cam_name]['cam2camego'])
        keyego2keysensor = keysensor2keyego.inverse()
        keysensor2sweepsensor = (
            keyego2keysensor @ global2keyego @ sweepego2global
            @ sweepsensor2sweepego).inverse()
        # if cam_name==cam_names[0]:
        #     print(rot,tran,1111111111)
        return sweepsensor2keyego, keysensor2sweepsensor


    def get_sensor_transforms(self, cam_info, cam_name):
        if 'cam2camego' not in cam_info['cams'][cam_name]:
            w, x, y, z = cam_info['cams'][cam_name]['sensor2ego_rotation']
            # sweep sensor to sweep ego
            sensor2ego_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            sensor2ego_tran = torch.Tensor(
                cam_info['cams'][cam_name]['sensor2ego_translation'])
            sensor2ego = sensor2ego_rot.new_zeros((4, 4))
            sensor2ego[3, 3] = 1
            sensor2ego[:3, :3] = sensor2ego_rot
            sensor2ego[:3, -1] = sensor2ego_tran
        else:
            sensor2ego=torch.from_numpy(cam_info['cams'][cam_name]['cam2camego'])
            
        if 'camego2global' not in cam_info['cams'][cam_name]:
            # sweep ego to global
            w, x, y, z = cam_info['cams'][cam_name]['ego2global_rotation']
            ego2global_rot = torch.Tensor(
                Quaternion(w, x, y, z).rotation_matrix)
            ego2global_tran = torch.Tensor(
                cam_info['cams'][cam_name]['ego2global_translation'])
            ego2global = ego2global_rot.new_zeros((4, 4))
            ego2global[3, 3] = 1
            ego2global[:3, :3] = ego2global_rot
            ego2global[:3, -1] = ego2global_tran
        else:
            ego2global=torch.from_numpy(cam_info['cams'][cam_name]['camego2global'])
        return sensor2ego, ego2global

    def get_inputs(self, results, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        sensor2egos = []
        ego2globals = []
        cam_names = self.choose_cams()
        results['cam_names'] = cam_names
        canvas = []
        sensor2sensors = []
        results['img_augs'] = {}
        depth_anything_maps=[]
        semantic_anything_maps=[]
        semantic_maps=[]
        for cam_name in cam_names:
            cam_data = results['curr']['cams'][cam_name]
            filename = cam_data['data_path']

            try:
                img = Image.open(filename)
            except:
                img = Image.open(filename.replace('png','jpg'))
            if self.img_scale is not None:

                img=np.asarray(img)
                img = self.pad(img)
                
        
                img=Image.fromarray(img)

            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])

            sensor2keyego, sensor2sensor = \
                self.get_sensor2ego_transformation(results['curr'],
                                                   results['curr'],
                                                   cam_name,
                                                   self.ego_cam)
            if self.gts_surroundocc:
                ##############
                lidar2lidarego = np.eye(4, dtype=np.float32)
                lidar2lidarego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                lidar2lidarego[:3, 3] = results['curr']['lidar2ego_translation']
                lidar2lidarego = torch.from_numpy(lidar2lidarego)

                if 'lidarego2global' not in  results['curr']:
                    lidarego2global = np.eye(4, dtype=np.float32)
                    lidarego2global[:3, :3] = Quaternion(
                        results['curr']['ego2global_rotation']).rotation_matrix
                    lidarego2global[:3, 3] = results['curr']['ego2global_translation']
                else:
                    lidarego2global=results['curr']['lidarego2global']
                lidarego2global = torch.from_numpy(lidarego2global)

                if 'camego2global' not in results['curr']['cams'][self.ego_cam]:
                    camego2global = np.eye(4, dtype=np.float32)
                    camego2global[:3, :3] = Quaternion(
                        results['curr']['cams'][self.ego_cam]
                        ['ego2global_rotation']).rotation_matrix
                    camego2global[:3, 3] = results['curr']['cams'][self.ego_cam][
                        'ego2global_translation']
                else:
                    camego2global=results['curr']['cams'][self.ego_cam]['camego2global']
                camego2global = torch.from_numpy(camego2global)

                sensor2lidar=torch.inverse(lidarego2global.matmul(lidar2lidarego)).matmul(camego2global.matmul(sensor2keyego))
                sensor2keyego=sensor2lidar
                ##############
            rot = sensor2keyego[:3, :3]
            tran = sensor2keyego[:3, 3]
            
            sensor2ego, ego2global = \
                self.get_sensor_transforms(results['curr'], cam_name)
            # image view augmentation (resize, crop, horizontal flip, rotate)
            if results.get('tta_config', None) is not None:
                flip = results['tta_config']['tta_flip']
            else: flip = None
            img_augs = self.sample_augmentation(
                H=img.height, W=img.width, flip=flip, scale=scale)
            resize, resize_dims, crop, flip, rotate = img_augs
            results['img_augs'][cam_name] = img_augs

            if self.additional_sem_gt_path is not None:
                semantic_filename = filename.replace('./data/nuscenes/samples', self.additional_sem_gt_path)
                semantic_filename = semantic_filename+'.npz'
                
                semantic_map = np.load(semantic_filename)['sem'].reshape(900, 1600)
                semantic_map[semantic_map==17]=255
                semantic_map = torch.from_numpy(semantic_map)
                semantic_map=F.interpolate(
                    semantic_map.float().unsqueeze(0).unsqueeze(0),
                    size=resize_dims[::-1],
                    mode='nearest').squeeze(0).squeeze(0)
                semantic_map=semantic_map.numpy()

                semantic_map = Image.fromarray(semantic_map)

                semantic_map = semantic_map.crop(crop)
                if flip:
                    semantic_map = semantic_map.transpose(method=Image.FLIP_LEFT_RIGHT)
                semantic_map = semantic_map.rotate(rotate)
                # return img
                semantic_map = np.array(semantic_map)
                
                semantic_map = torch.from_numpy(semantic_map)
                semantic_maps.append(semantic_map)

            img, post_rot2, post_tran2 = \
                self.img_transform(img, post_rot,
                                   post_tran,
                                   resize=resize,
                                   resize_dims=resize_dims,
                                   crop=crop,
                                   flip=flip,
                                   rotate=rotate)
            # print(TF.to_tensor(img).shape,444444444444)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            canvas.append(np.array(img))

            imgs.append(self.normalize_img(img, **self.normalize_cfg))

            if self.sequential and not self.save_stereo and self.depth_stereo:
                assert 'adjacent' in results
                for adj_info in results['adjacent']:
                    filename_adj = adj_info['cams'][cam_name]['data_path']
                    try:
                        img_adjacent = Image.open(filename_adj)
                    except:
                        img_adjacent = Image.open(filename_adj.replace('png','jpg'))
                    if self.img_scale is not None:
                        img_adjacent=np.asarray(img_adjacent)
                        img_adjacent = self.pad(img_adjacent)
                        img_adjacent=Image.fromarray(img_adjacent)
                    img_adjacent = self.img_transform_core(
                        img_adjacent,
                        resize_dims=resize_dims,
                        crop=crop,
                        flip=flip,
                        rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent, **self.normalize_cfg))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)
            sensor2sensors.append(sensor2sensor)
            sensor2egos.append(sensor2ego)
            ego2globals.append(ego2global)

        if self.sequential  or (self.depth_his_fusion_pre or self.geometry_his_fusion):
            adj_sensor2egos = []
            adj_ego2globals = []

            for adj_info in results['adjacent']:

                for cam_name in cam_names:
                    sensor2ego, ego2global = \
                        self.get_sensor_transforms(adj_info, cam_name)
                    adj_sensor2egos.append(sensor2ego)
                    adj_ego2globals.append(ego2global)
            adj_sensor2egos=torch.stack(adj_sensor2egos)
            adj_ego2globals=torch.stack(adj_ego2globals)
    
        imgs = torch.stack(imgs)
        
        sensor2egos = torch.stack(sensor2egos).float()
        ego2globals = torch.stack(ego2globals).float()

        rots = torch.stack(rots).float()
        trans = torch.stack(trans).float()
        intrins = torch.stack(intrins).float()
        post_rots = torch.stack(post_rots).float()
        post_trans = torch.stack(post_trans).float()
        sensor2sensors = torch.stack(sensor2sensors).float()

        results['canvas'] = canvas
        results['sensor2sensors'] = sensor2sensors
        
        if self.additional_sem_gt_path is not None:
            semantic_maps=torch.stack(semantic_maps)
            if self.fix_void:
                semantic_maps[semantic_maps!=255]+=1
            results['gt_semantic_map'] = semantic_maps
        if self.sequential  or (self.depth_his_fusion_pre or self.geometry_his_fusion):
            return (imgs, rots, trans, intrins, post_rots, post_trans), (sensor2egos, ego2globals), (adj_sensor2egos.float(), adj_ego2globals.float())
        else:
            return (imgs, rots, trans, intrins, post_rots, post_trans), (sensor2egos, ego2globals)

    def __call__(self, results):
        if self.sequential  or (self.depth_his_fusion_pre or self.geometry_his_fusion):
            results['img_inputs'], results['aux_cam_params'], results['adj_aux_cam_params'] = self.get_inputs(results)
        else:
            results['img_inputs'], results['aux_cam_params'] = self.get_inputs(results)
        
        return results


@PIPELINES.register_module()
class LoadAnnotationsBEVDepth(object):

    def __init__(self, bda_aug_conf, classes,with_pts_bbox=False, is_train=True):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.classes = classes
        self.with_pts_bbox=with_pts_bbox

    def sample_bda_augmentation(self, tta_config=None):
        """Generate bda augmentation values based on bda_config."""
        if self.is_train:
            rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
            scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
            flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
            flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        else:
            rotate_bda = 0
            scale_bda = 1.0
            if tta_config is not None:
                flip_dx = tta_config['flip_dx']
                flip_dy = tta_config['flip_dy']
            else:
                flip_dx = False
                flip_dy = False

        return rotate_bda, scale_bda, flip_dx, flip_dy

    def bev_transform(self,  rotate_angle, scale_ratio, flip_dx,
                      flip_dy,gt_boxes=None):
        rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
        rot_sin = torch.sin(rotate_angle)
        rot_cos = torch.cos(rotate_angle)
        rot_mat = torch.Tensor([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0],
                                [0, 0, 1]])
        scale_mat = torch.Tensor([[scale_ratio, 0, 0], [0, scale_ratio, 0],
                                  [0, 0, scale_ratio]])
        flip_mat = torch.Tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        if flip_dx:
            flip_mat = flip_mat @ torch.Tensor([[-1, 0, 0], [0, 1, 0],
                                                [0, 0, 1]])
        if flip_dy:
            flip_mat = flip_mat @ torch.Tensor([[1, 0, 0], [0, -1, 0],
                                                [0, 0, 1]])
        rot_mat = flip_mat @ (scale_mat @ rot_mat)
        if self.with_pts_bbox:
            if gt_boxes.shape[0] > 0:
                gt_boxes[:, :3] = (
                    rot_mat @ gt_boxes[:, :3].unsqueeze(-1)).squeeze(-1)
                gt_boxes[:, 3:6] *= scale_ratio
                gt_boxes[:, 6] += rotate_angle
                if flip_dx:
                    gt_boxes[:,
                            6] = 2 * torch.asin(torch.tensor(1.0)) - gt_boxes[:,
                                                                            6]
                if flip_dy:
                    gt_boxes[:, 6] = -gt_boxes[:, 6]
                gt_boxes[:, 7:] = (
                    rot_mat[:2, :2] @ gt_boxes[:, 7:].unsqueeze(-1)).squeeze(-1)
        if self.with_pts_bbox:
            return gt_boxes, rot_mat
        else:
            return  rot_mat

    def __call__(self, results):
        if self.with_pts_bbox:
            gt_boxes, gt_labels = results['ann_infos']
            gt_boxes, gt_labels = torch.Tensor(np.array(gt_boxes)), torch.tensor(np.array(gt_labels))
        else:
            gt_boxes=None
        tta_confg = results.get('tta_config', None)
        rotate_bda, scale_bda, flip_dx, flip_dy = self.sample_bda_augmentation(tta_confg
        )
        bda_mat = torch.zeros(4, 4)
        bda_mat[3, 3] = 1
        
        bda_rot = self.bev_transform( rotate_bda, scale_bda,
                                                flip_dx, flip_dy,gt_boxes)
        if self.with_pts_bbox:
            gt_boxes, bda_rot=bda_rot
        bda_mat[:3, :3] = bda_rot
        if self.with_pts_bbox:
            if len(gt_boxes) == 0:
                gt_boxes = torch.zeros(0, 9)
            results['gt_bboxes_3d'] = \
                LiDARInstance3DBoxes(gt_boxes, box_dim=gt_boxes.shape[-1],
                                    origin=(0.5, 0.5, 0.5))
                        
            results['gt_labels_3d'] = gt_labels
        imgs, rots, trans, intrins = results['img_inputs'][:4]
        post_rots, post_trans = results['img_inputs'][4:]
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots,
                                 post_trans, bda_rot)
        
        results['flip_dx'] = flip_dx
        results['flip_dy'] = flip_dy
        results['rotate_bda'] = rotate_bda
        results['scale_bda'] = scale_bda
        
        return results

