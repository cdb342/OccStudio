import pickle
with open(f'./data/nuscenes/bevdetv2-nuscenes_infos_train_val.pkl', 'rb') as f:
    nusc_data = pickle.load(f)

info=nusc_data['infos']

info = list(sorted(info, key=lambda e: e['timestamp']))

map_=dict()
for info_i in info:
    map_[info_i['scene_token']]=info_i['scene_name']
    
save_path='./data/nuscenes/scene_map.pkl'
with open(save_path, 'wb') as f:
    pickle.dump(map_, f)

print('done')







