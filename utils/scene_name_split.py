import pickle

with open(f'./data/nuscenes/bevdetv2-nuscenes_infos_val.pkl', 'rb') as f:
    nusc_data = pickle.load(f)
info=nusc_data['infos']

info = list(sorted(info, key=lambda e: e['timestamp']))

scene_names=set()
for i in range(len(info)):
    scene_names.add(info[i]['scene_name'])

info_scene_split=dict()
scene_names=list(scene_names)
scene_names=sorted(scene_names)


for scene_name in scene_names:
    info_scene_split[scene_name]=[]
    
for i in range(len(info)):
    info_scene_split[info[i]['scene_name']].append(info[i])
    
    
  

results={'infos':info_scene_split,'metadata':'val_scene_split'}

out_path='./data/nuscenes/bevdetv2-nuscenes_infos_val_scene_split.pkl'

with open(out_path, 'wb') as f:
    pickle.dump(results, f)

print(len(results['infos']))    
print('saved to: ',out_path)