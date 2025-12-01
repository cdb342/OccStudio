# +----------------------+-------+-------+-------+--------+------------+-------+
# |     Class Names      | IoU@1 | IoU@2 | IoU@4 | AVE@TP | AVE&GT_RAY |  AVE  |
# +----------------------+-------+-------+-------+--------+------------+-------+
# |         car          | 0.531 | 0.605 | 0.630 | 0.380  |   0.971    | 0.409 |
# |        truck         | 0.436 | 0.541 | 0.591 | 0.305  |   0.690    | 0.472 |
# |       trailer        | 0.217 | 0.275 | 0.368 | 0.595  |   0.987    | 0.333 |
# |         bus          | 0.501 | 0.608 | 0.664 | 0.809  |   1.209    | 0.917 |
# | construction_vehicle | 0.262 | 0.335 | 0.355 | 0.154  |   0.289    | 0.221 |
# |       bicycle        | 0.234 | 0.259 | 0.268 | 0.272  |   1.613    | 0.530 |
# |      motorcycle      | 0.255 | 0.327 | 0.339 | 0.593  |   2.019    | 0.843 |
# |      pedestrian      | 0.333 | 0.380 | 0.398 | 0.342  |   0.672    | 0.517 |
# |     traffic_cone     | 0.320 | 0.340 | 0.350 |  nan   |    nan     |  nan  |
# |       barrier        | 0.423 | 0.461 | 0.476 |  nan   |    nan     |  nan  |
# |  driveable_surface   | 0.460 | 0.562 | 0.683 |  nan   |    nan     |  nan  |
# |      other_flat      | 0.273 | 0.320 | 0.359 |  nan   |    nan     |  nan  |
# |       sidewalk       | 0.254 | 0.303 | 0.354 |  nan   |    nan     |  nan  |
# |       terrain        | 0.242 | 0.308 | 0.368 |  nan   |    nan     |  nan  |
# |       manmade        | 0.434 | 0.511 | 0.562 |  nan   |    nan     |  nan  |
# |      vegetation      | 0.309 | 0.429 | 0.511 |  nan   |    nan     |  nan  |
# +----------------------+-------+-------+-------+--------+------------+-------+
# |         MEAN         | 0.343 | 0.410 | 0.455 | 0.431  |   1.056    | 0.530 |
# +----------------------+-------+-------+-------+--------+------------+-------+
# --- Occ score:0.41924476684554235
# we follow the online training settings  from solofusion
num_gpus = 8
samples_per_gpu = 2
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu) * 4.554)
num_epochs = 18
checkpoint_epoch_interval = 1
use_custom_eval_hook=True

# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order 
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# By default, 3D detection datasets randomly choose another sample if there is
# no GT object in the current sample. This does not make sense when doing
# sequential sampling of frames, so we disable it.
filter_empty_gt = False

history_cat_num = 16


_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
# Global
# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

data_config = {
    'cams': [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT',
        'CAM_BACK', 'CAM_BACK_RIGHT'
    ],
    'Ncams':
    6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5)

use_checkpoint = True
sync_bn = True


# Model
grid_config = {
    'x': [-40, 40, 0.4],
    'y': [-40, 40, 0.4],
    'z': [-1, 5.4, 6.4],
    'depth': [1.0, 45.0, 0.5],
}

depth_channels = int((grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2])

numC_Trans=80
_dim_ = 256

empty_idx = 17  
num_cls = 18  
fix_void = num_cls == 18

voxel_out_channel = 48
occ_encoder_channels = [64, 64*2, 64*4,64*8]

not_use_history=False
use_sequence_group_flag=not not_use_history

depth_stereo=True
multi_adj_frame_id_cfg = (1, 1, 1)
load_point_semantic=True
img_seg_weight=0.1
sem_sup_prototype=True
depth2occ_intra=True
length=22
depth2occ_inter=True
soft_filling=True
use_depth_supervision=True
geometry_denoise=True
depth_loss_ce=True
wo_assign=True
use_mask_net2=True

occ_backbone_2d=True
occ_2d_out_channels=256

pred_flow=True
open_occ=True
flow_l2_loss=True
flow_cosine_loss=True
flow_bin_fixed=True
sup_bin=True
flow_mask=True
occ_ray_mask=True
flow_with_his=True
fow_history_cat_num=1
use_flow_bin_decoder=True
flow_gt_denoise=True

torch_sparse_coor=True
model = dict(
    type='ALOCC',
    use_depth_supervision=use_depth_supervision,
    fix_void=fix_void,
    history_cat_num=history_cat_num,
    single_bev_num_channels=numC_Trans,
    readd=True,
    not_use_history=not_use_history,
    depth_stereo=depth_stereo,
    grid_config=grid_config,
    downsample=16,
    img_seg_weight=img_seg_weight,
    depth_loss_ce=depth_loss_ce,
    depth2occ_intra=depth2occ_intra,
    sem_sup_prototype=sem_sup_prototype,
    use_mask_net2=use_mask_net2,
    
    occ_backbone_2d=occ_backbone_2d,
    occ_2d_out_channels=occ_2d_out_channels,
    occ_2d=occ_backbone_2d,
    
    pred_flow=pred_flow,
    voxel_out_channel=voxel_out_channel,

    img_backbone=dict(
        # pretrained='torchvision://resnet50',
        # pretrained='ckpts/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0,2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=use_checkpoint,
        style='pytorch'),
    img_neck=dict(
        type='CustomFPN',
        in_channels=[1024, 2048],
        out_channels=_dim_,
        num_outs=1,
        start_level=0,
        with_cp=use_checkpoint,
        out_ids=[0]),
    depth_net=dict(
        type='CM_DepthNet', 
        in_channels=_dim_,
        context_channels=numC_Trans,
        downsample=16,
        grid_config=grid_config,
        depth_channels=depth_channels,
        with_cp=use_checkpoint,
        loss_depth_weight=0.05,
        use_dcn=False,
        stereo=depth_stereo,
        input_size=data_config['input_size'],
        bias=5.,
        mid_channels=_dim_,
        aspp_mid_channels=96,
        depth2occ_intra=depth2occ_intra,
        length=length,
        depth2occ_inter=depth2occ_inter,
        geometry_denoise=geometry_denoise,
    ),
    view_transformer=dict(
        type='LSSViewTransformerFunction',
        grid_config=grid_config,
        input_size=data_config['input_size'],
        soft_filling=soft_filling,
        depth2occ_inter=depth2occ_inter,
        occ_2d=occ_backbone_2d,
        torch_sparse_coor=torch_sparse_coor,
        downsample=16),
    pre_process=dict(
        type='CustomResNet',
        numC_input=numC_Trans,
        num_layer=[1, ],
        num_channels=[numC_Trans, ],
        stride=[1, ],
        backbone_output_ids=[0, ]),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * 1,
        num_channels=occ_encoder_channels,
        stride=[1,2,2,2],
        num_layer=[1, 2, 4,4],
        backbone_output_ids=[0,1,2,3],
        dim_z=16,),
    img_bev_encoder_neck=dict(
        type='FPN_LSS2',
        in_channels=sum(occ_encoder_channels),
        out_channels=occ_2d_out_channels,
        with_cp=use_checkpoint,
        extra_upsample=None,
        input_feature_index=(0,1,2,3)),
    alocc_head=dict(
        type='ALOccHead',
        feat_channels=voxel_out_channel,
        out_channels=voxel_out_channel,
        num_queries=num_cls,
        num_occupancy_classes=num_cls,
        sample_weight_gamma=0.25,
        num_transformer_feat_level=0,
        # using the original transformer decoder
        #########
        wo_assign=wo_assign,
        mask_embed2=use_mask_net2,
        out_channels_embed2=numC_Trans,
        num_points_img=1056,
        pred_flow=pred_flow,
        open_occ=open_occ,
        flow_l2_loss=flow_l2_loss,
        flow_cosine_loss=flow_cosine_loss,
        flow_with_his=flow_with_his,
        single_bev_num_channels=voxel_out_channel,
        history_cat_num=fow_history_cat_num,
        sup_bin=sup_bin,
        use_flow_bin_decoder=use_flow_bin_decoder,
        flow_bin_fixed=flow_bin_fixed,
        flow_gt_denoise=flow_gt_denoise,
        ############
        transformer_decoder=dict(
            type='DetrTransformerDecoder',
            return_intermediate=True,
            num_layers=0,
            transformerlayers=None,
            init_cfg=None),
            # loss settings
            loss_cls=dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=2.0,
                reduction='mean',
                class_weight=[1.0] * (num_cls )+ [0.1]),
            loss_mask=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                reduction='mean',
                loss_weight=5.0),
            loss_dice=dict(
                type='DiceLoss',
                use_sigmoid=True,
                activate=True,
                reduction='mean',
                naive_dice=True,
                eps=1.0,
                loss_weight=5.0),
        
            train_cfg=dict(
                num_points=12544 * 2,
                oversample_ratio=3.0,
                importance_sample_ratio=0.75,
                assigner=None,
                sampler=None,
            ),
    ),
    pts_bbox_head=None)

# Data
dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')
occupancy_path = 'data/nuscenes/openocc_v2'


train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=True),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        point_with_semantic=load_point_semantic,
        file_client_args=file_client_args),
    dict(type='PointToMultiViewDepth', downsample=1, grid_config=grid_config,load_semantic_map=load_point_semantic,fix_void=fix_void),
    dict(type='LoadOccFlowGTFromFile', ignore_nonvisible=True, fix_void=fix_void, occupancy_path=occupancy_path,flow_mask=flow_mask,occ_ray_mask=occ_ray_mask),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs',  'gt_occupancy','gt_occupancy_ori', 'gt_occ_flow', 'gt_depth','aux_cam_params','adj_aux_cam_params','gt_semantic_map'])
]

test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs', data_config=data_config, sequential=True),
            dict(
                type='LoadAnnotationsBEVDepth',
                bda_aug_conf=bda_aug_conf,
                classes=class_names,
                is_train=False),
            dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=5,
                file_client_args=file_client_args),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img_inputs','aux_cam_params','adj_aux_cam_params'])
            ]
        )
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

share_data_config = dict(
    type=dataset_type,
    classes=class_names,
    modality=input_modality,
    img_info_prototype='bevdet4d',
    occupancy_path=occupancy_path,
    use_sequence_group_flag=use_sequence_group_flag,
    stereo=depth_stereo,
    multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
    openocc=True,
)

test_data_config = dict(
    pipeline=test_pipeline,
    sequences_split_num=test_sequences_split_num,
    ann_file=data_root + 'bevdetv2-nuscenes_infos_val.pkl')

data = dict(
    samples_per_gpu=samples_per_gpu,
    workers_per_gpu=4,
    test_dataloader=dict(runner_type='IterBasedRunnerEval'),
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        img_info_prototype='bevdet4d',
        sequences_split_num=train_sequences_split_num,
        use_sequence_group_flag=use_sequence_group_flag,
        filter_empty_gt=filter_empty_gt,
        stereo=depth_stereo,
        multi_adj_frame_id_cfg=multi_adj_frame_id_cfg,
        box_type_3d='LiDAR'),
    val=test_data_config,
    test=test_data_config)

for key in ['val', 'test']:
    data[key].update(share_data_config)

# Optimizer
lr = 2e-4
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-2)
optimizer_config = dict(grad_clip=dict(max_norm=5, norm_type=2))
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=200,
    warmup_ratio=0.001,
    step=[num_iters_per_epoch*num_epochs,])
runner = dict(type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(
    interval=checkpoint_epoch_interval * num_iters_per_epoch)
evaluation = dict(
    interval=num_epochs * num_iters_per_epoch, pipeline=test_pipeline)


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
custom_hooks = [
    dict(
        type='MEGVIIEMAHook',
        init_updates=10560,
        priority='NORMAL',
        interval=1*num_iters_per_epoch,
    ),
    dict(
        type='SequentialControlHook',
        temporal_start_iter=num_iters_per_epoch *2,
    ),
    dict(
        type='FusionRateControlHook',
        temporal_start_iter=num_iters_per_epoch *2,
        temporal_end_iter=num_iters_per_epoch *4,
    ),

    dict(
        type='FusionRateControlDepthHook',
        temporal_start_iter=0,
        temporal_end_iter=num_iters_per_epoch *6,
    ),
]


load_from='ckpts/pretrain/bevdet-r50-4d-stereo-cbgs.pth'