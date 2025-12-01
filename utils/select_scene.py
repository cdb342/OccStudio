import pickle

with open(f'./data/nuscenes/bevdetv2-nuscenes_infos_val.pkl', 'rb') as f:
    nusc_data = pickle.load(f)
info=nusc_data['infos']

info = list(sorted(info, key=lambda e: e['timestamp']))

scene_names=set()
for i in range(len(info)):
    scene_names.add(info[i]['scene_name'])

mini_info=[]
scene_names=list(scene_names)
scene_names=sorted(scene_names)
selected_scene='scene-1060'

print(scene_names)
print(selected_scene)
for i in range(len(info)):
    if info[i]['scene_name']==selected_scene:
        mini_info.append(info[i])

results={'infos':mini_info,'metadata':selected_scene}
out_path='./data/nuscenes/bevdetv2-nuscenes_infos_val_{}.pkl'.format(selected_scene)

with open(out_path, 'wb') as f:
    pickle.dump(results, f)

print(len(results['infos']))    
print(selected_scene ,'saved to: ',out_path)