import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
import os
import pickle
@PIPELINES.register_module()
class LoadOccGTFromFileBEVFormer(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    note that we read image in BGR style to align with opencv.imread
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(
            self,
            data_root,
            occ3d_aug_mask=False,
        ):
        self.data_root = data_root
        self.occ3d_aug_mask=occ3d_aug_mask
        scene_map_path='./data/nuscenes/scene_map.pkl'
        
        with open(scene_map_path, 'rb') as f:
            scene_map=pickle.load(f)
        self.scene_map=scene_map

    def __call__(self, results):

        scene_token = results['scene_token']
        scene_name=self.scene_map[scene_token]
        sample_token = results['sample_idx']
        occupancy_file_path = os.path.join(self.data_root, scene_name, sample_token, 'labels.npz')
        data = np.load(occupancy_file_path)
       
        semantics = data['semantics']
        if self.occ3d_aug_mask:
            
            sem_mask=data['sem_mask']
            try:
                visible_mask=data['visible_mask']
            except:
                visible_mask=data['occ_mask']
            mask_camera=sem_mask + visible_mask
            mask_lidar = np.zeros((200,200,16),dtype=np.uint8)
        else:

            mask_lidar = data['mask_lidar']
            mask_camera = data['mask_camera']

        results['voxel_semantics'] = semantics
        results['mask_lidar'] = mask_lidar
        results['mask_camera'] = mask_camera


        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        return "{} (data_root={}')".format(
            self.__class__.__name__, self.data_root)