import pickle

with open(f'./data/nuscenes/bevdetv2-nuscenes_infos_val.pkl', 'rb') as f:
    nusc_data = pickle.load(f)
info=nusc_data['infos']

info = list(sorted(info, key=lambda e: e['timestamp']))
for info_i in (info):

    if info_i['occ_path']=='./data/nuscenes/gts/scene-0161/4a7fecabf16742068d3e02c053f78ce7':
        
        print(info_i['lidar_path'])
        print(info_i['scene_token'])
        print(info_i['lidar_token'])
