import pickle
with open(f'./data/nuscenes/bevdetv2-nuscenes_infos_train.pkl', 'rb') as f:
    nusc_data_train = pickle.load(f)
with open(f'./data/nuscenes/bevdetv2-nuscenes_infos_val.pkl', 'rb') as f:
    nusc_data_val = pickle.load(f)
info_train=nusc_data_train['infos']
info_val=nusc_data_val['infos']
import pdb;pdb.set_trace()

info_train=info_train+info_val
out_path='./data/nuscenes/bevdetv2-nuscenes_infos_train_val.pkl'
results={'infos':info_train,'metadata':nusc_data_train['metadata']}
with open(out_path, 'wb') as f:
    pickle.dump(results, f)
import pdb;pdb.set_trace()
print(len(results['infos']))    
print('saved to: ',out_path)


