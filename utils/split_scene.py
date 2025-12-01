import pickle
with open(f'./data/nuscenes/bevdetv2-nuscenes_infos_train.pkl', 'rb') as f:
    nusc_data = pickle.load(f)

info=nusc_data['infos']

info = list(sorted(info, key=lambda e: e['timestamp']))


split_num=4

result_info=[]
len_info=len(info)
for i in range(split_num):
    info_i=info[(len_info//split_num+1)*i:(len_info//split_num+1)*(i+1)]
    results_i={'infos':info_i,'metadata':'train_{}'.format(i)}

    out_path='./data/nuscenes/bevdetv2-nuscenes_infos_train_{}.pkl'.format(i)
    
    with open(out_path, 'wb') as f:
        pickle.dump(results_i, f)
    











