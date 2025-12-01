
import pickle
import numpy as np
import torch
import tqdm
import os.path as osp

with open(f'./data/nuscenes/bevdetv2-nuscenes_infos_train.pkl', 'rb') as f:
    nusc_data = pickle.load(f)['infos']

freq=[0]*18

occ_size=[512,512,40]
for i in tqdm.tqdm(range(len(nusc_data))):
    
    occ_path='./data/nuscenes/gts'
    
    scene_token = 'scene_'+nusc_data[i]['scene_token']
    lidar_token = nusc_data[i]['lidar_token']

    occupancy_file_path = osp.join(occ_path.replace('gts','nuScenes-Occupancy-v0.1'), scene_token, 'occupancy',lidar_token)+'.npy'
    data = np.load(occupancy_file_path)
    occ = np.zeros(occ_size)
    occ[data[:, 2], data[:, 1], data[:, 0]] = data[:, 3]
    occ = np.where(occ== 0, 17, occ)

    occupancy=torch.tensor(occ.copy()).long()

    for j in range(len(freq)):
        mask_j=occupancy==j
        
        freq[j]+=mask_j.sum().item()
        
    if i%100==0:
        print(freq)
print(freq)