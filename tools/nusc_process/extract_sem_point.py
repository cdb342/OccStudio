from nuscenes.nuscenes import NuScenes
import yaml
import pickle
import numpy as np
import os
import mmcv

nusc = NuScenes(version='v1.0-trainval',
                        dataroot='data/nuscenes/',
                        verbose=True)
label_mapping_file='mmdet3d/datasets/pipelines/nuscenes.yaml'         
with open(label_mapping_file, 'r') as stream:
    nuscenesyaml = yaml.safe_load(stream)                        
learning_map = nuscenesyaml['learning_map']


for sample in mmcv.track_iter_progress(nusc.sample):
    lidar_token = sample['data']['LIDAR_TOP']
    lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

    lidar_sd_token=nusc.get('lidarseg', lidar_token)['token']

    lidarseg_labels_filename = os.path.join(nusc.dataroot,nusc.get('lidarseg', lidar_sd_token)['filename'])

    points_label = np.fromfile(lidarseg_labels_filename, dtype=np.uint8).reshape([-1, 1])
    points_label = np.vectorize(learning_map.__getitem__)(points_label)

    save_dir='./data/lidar_seg2'
    point_name=lidar_path.split('/')[-1].split('.')[0]
    save_path=os.path.join(save_dir,point_name+'.npz')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.savez_compressed(save_path, lidar_seg=points_label)
