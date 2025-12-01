# ===> per class IoU of 6019 samples:
# ===> others - IoU = 14.2496
# ===> barrier - IoU = 48.557
# ===> bicycle - IoU = 28.7098
# ===> bus - IoU = 44.8032
# ===> car - IoU = 52.8442
# ===> construction_vehicle - IoU = 24.7022
# ===> motorcycle - IoU = 29.1905
# ===> pedestrian - IoU = 29.0308
# ===> traffic_cone - IoU = 32.0381
# ===> trailer - IoU = 34.6078
# ===> truck - IoU = 39.6132
# ===> driveable_surface - IoU = 82.3772
# ===> other_flat - IoU = 46.9503
# ===> sidewalk - IoU = 54.8474
# ===> terrain - IoU = 57.7063
# ===> manmade - IoU = 44.7206
# ===> vegetation - IoU = 39.3186
# ===> free - IoU = 89.799
# ===> mIoU of 6019 samples: 41.43
# ===> occupied - IoU = 70.0431
# ===> mIoU_D = 35.4377
# we follow the online training settings  from solofusion
num_gpus = 8
samples_per_gpu = 2
num_iters_per_epoch = int(28130 // (num_gpus * samples_per_gpu) * 4.554)
num_epochs = 12
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
    'depth': [1.0, 45.0, 1.],
}

depth_channels = int((grid_config['depth'][1]-grid_config['depth'][0])//grid_config['depth'][2])

numC_Trans=40
_dim_ = 256

empty_idx = 18  # noise 0-->255
num_cls = 19  # 0 others, 1-16 obj, 17 free
fix_void = num_cls == 19

voxel_out_channel = 32
occ_encoder_channels = [32, 32*2, 32*4]

not_use_history=False
use_sequence_group_flag=not not_use_history

depth_stereo=False
multi_adj_frame_id_cfg = (1, 1, 1)
load_point_semantic=True
img_seg_weight=0.1
sem_sup_prototype=True
depth2occ_intra=True
length=22
depth2occ_inter=False
soft_filling=True
use_depth_supervision=True
geometry_denoise=True
depth_loss_ce=True
wo_assign=True
use_mask_net2=True

occ_backbone_2d=True
occ_2d_out_channels=128
pre_gen_his_grid=True

model = dict(
    type='ALOCC',
    use_depth_supervision=use_depth_supervision,
    fix_void=fix_void,
    history_cat_num=history_cat_num,
    single_bev_num_channels=numC_Trans,
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
    voxel_out_channel=voxel_out_channel,
    occ_2d=occ_backbone_2d,
    pre_gen_his_grid=pre_gen_his_grid,

    img_backbone=dict(
        # pretrained='torchvision://resnet50',
        # pretrained='ckpts/resnet50-0676ba61.pth',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
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
        downsample=16),
    img_bev_encoder_backbone=dict(
        type='CustomResNet',
        numC_input=numC_Trans * 1,
        num_channels=occ_encoder_channels,
        stride=[1,2,2],
        num_layer=[1, 2, 4],
        backbone_output_ids=[0,1,2],
        dim_z=16,),
    img_bev_encoder_neck=dict(
        type='FPN_LSS3',
        in_channels=sum(occ_encoder_channels),
        out_channels=occ_2d_out_channels,
        with_cp=use_checkpoint,
        extra_upsample=None,
        input_feature_index=(0,1,2)),
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
occupancy_path = 'data/nuscenes/gts'


train_pipeline = [
    dict(
        type='PrepareImageInputs',
        is_train=True,
        data_config=data_config,
        sequential=depth_stereo),
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
    dict(type='LoadOccupancy', ignore_nonvisible=True, fix_void=fix_void, occupancy_path=occupancy_path),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D', keys=['img_inputs',  'gt_occupancy', 'gt_depth','aux_cam_params','gt_semantic_map'])
]

test_pipeline = [
    dict(
        type='CustomDistMultiScaleFlipAug3D',
        tta=False,
        transforms=[
            dict(type='PrepareImageInputs', data_config=data_config, sequential=depth_stereo),
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
            dict(type='Collect3D', keys=['points', 'img_inputs','aux_cam_params'])
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
        type='FusionRateControlDepthHook',
        temporal_start_iter=0,
        temporal_end_iter=num_iters_per_epoch *6,
    ),
]


load_from='ckpts/pretrain/bevdet-r50-4d-stereo-cbgs.pth'