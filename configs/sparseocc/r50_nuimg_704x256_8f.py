
# +----------------------+----------+----------+----------+
# |     Class Names      | RayIoU@1 | RayIoU@2 | RayIoU@4 |
# +----------------------+----------+----------+----------+
# |        others        |  0.077   |  0.088   |  0.092   |
# |       barrier        |  0.327   |  0.392   |  0.420   |
# |       bicycle        |  0.190   |  0.234   |  0.240   |
# |         bus          |  0.465   |  0.589   |  0.660   |
# |         car          |  0.454   |  0.555   |  0.591   |
# | construction_vehicle |  0.113   |  0.199   |  0.262   |
# |      motorcycle      |  0.216   |  0.279   |  0.295   |
# |      pedestrian      |  0.267   |  0.322   |  0.345   |
# |     traffic_cone     |  0.249   |  0.282   |  0.292   |
# |       trailer        |  0.148   |  0.209   |  0.302   |
# |        truck         |  0.346   |  0.455   |  0.527   |
# |  driveable_surface   |  0.538   |  0.618   |  0.702   |
# |      other_flat      |  0.260   |  0.297   |  0.329   |
# |       sidewalk       |  0.217   |  0.266   |  0.317   |
# |       terrain        |  0.213   |  0.288   |  0.358   |
# |       manmade        |  0.213   |  0.316   |  0.394   |
# |      vegetation      |  0.153   |  0.251   |  0.355   |
# +----------------------+----------+----------+----------+
# |         MEAN         |  0.261   |  0.332   |  0.381   |
# +----------------------+----------+----------+----------+
# {'RayIoU': 0.3247803386586359, 'RayIoU@1': 0.26148665870168625, 'RayIoU@2': 0.3316777932312336, 'RayIoU@4': 0.38117656404298794}
_base_ = ['../_base_/datasets/nus-3d.py', '../_base_/default_runtime.py']
dataset_type = 'NuSceneOcc_SparseOcc'
dataset_root = 'data/nuscenes/'
occ_gt_root = 'data/nuscenes/gts'
# file_client_args = dict(backend='disk')

total_epochs = 24
batch_size = 2

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-40, -40, -1.0, 40, 40, 5.4]
occ_size = [200, 200, 16]

img_norm_cfg = dict(
    mean=[123.675, 116.280, 103.530],
    std=[58.395, 57.120, 57.375],
    to_rgb=True
)

# For nuScenes we usually do 10-class detection
det_class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

occ_class_names = [
    'others', 'barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
    'driveable_surface', 'other_flat', 'sidewalk',
    'terrain', 'manmade', 'vegetation', 'free'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

_dim_ = 256
_num_points_ = 4
_num_groups_ = 4
_num_layers_ = 2
_num_frames_ = 8
_num_queries_ = 100
_topk_training_ = [4000, 16000, 64000]
_topk_testing_ = [2000, 8000, 32000]

model = dict(
    type='SparseOcc',
    data_aug=dict(
        img_color_aug=True,  # Move some augmentations to GPU
        img_norm_cfg=img_norm_cfg,
        img_pad_cfg=dict(size_divisor=32)),
    use_mask_camera=False,
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN2d', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        with_cp=True),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=_dim_,
        num_outs=4),
    pts_bbox_head=dict(
        type='SparseOccHead',
        class_names=occ_class_names,
        embed_dims=_dim_,
        occ_size=occ_size,
        pc_range=point_cloud_range,
        transformer=dict(
            type='SparseOccTransformer',
            embed_dims=_dim_,
            num_layers=_num_layers_,
            num_frames=_num_frames_,
            num_points=_num_points_,
            num_groups=_num_groups_,
            num_queries=_num_queries_,
            num_levels=4,
            num_classes=len(occ_class_names),
            pc_range=point_cloud_range,
            occ_size=occ_size,
            topk_training=_topk_training_,
            topk_testing=_topk_testing_),
        loss_cfgs=dict(
            loss_mask2former=dict(
                type='Mask2FormerLoss',
                num_classes=len(occ_class_names),
                no_class_weight=0.1,
                loss_cls_weight=2.0,
                loss_mask_weight=5.0,
                loss_dice_weight=5.0,
            ),
            loss_geo_scal=dict(
                type='GeoScalLoss',
                num_classes=len(occ_class_names),
                loss_weight=1.0
            ),
            loss_sem_scal=dict(
                type='SemScalLoss',
                num_classes=len(occ_class_names),
                loss_weight=1.0
            )
        ),
    ),
)

ida_aug_conf = {
    'resize_lim': (0.38, 0.55),
    'final_dim': (256, 704),
    'bot_pct_lim': (0.0, 0.0),
    'rot_lim': (0.0, 0.0),
    'H': 900, 'W': 1600,
    'rand_flip': True,
}

bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(1., 1.),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=True),
    dict(type='LoadOccGTFromFileSparseocc', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=True),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],  # other keys: 'mask_camera'
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar','sample_idx'))
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=False, color_type='color'),
    dict(type='LoadMultiViewImageFromMultiSweeps', sweeps_num=_num_frames_ - 1, test_mode=True),
    dict(type='BEVAug', bda_aug_conf=bda_aug_conf, classes=det_class_names, is_train=False),
    dict(type='LoadOccGTFromFileSparseocc', num_classes=len(occ_class_names)),
    dict(type='RandomTransformImage', ida_aug_conf=ida_aug_conf, training=False),
    dict(type='DefaultFormatBundle3D', class_names=det_class_names),
    dict(type='Collect3D', keys=['img', 'voxel_semantics', 'voxel_instances', 'instance_class_ids'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'lidar2img', 'img_timestamp', 'ego2lidar','sample_idx'))
]

data = dict(
    workers_per_gpu=8,
    samples_per_gpu=batch_size,
    train=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        # ann_file=dataset_root + 'nuscenes_infos_train_sweep.pkl',
        ann_file=dataset_root + 'bevdetv2-nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=det_class_names,
        modality=input_modality,
        test_mode=False
    ),
    val=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        # ann_file=dataset_root + 'nuscenes_infos_val_sweep.pkl',
        ann_file=dataset_root + 'bevdetv2-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=det_class_names,
        modality=input_modality,
        test_mode=True
    ),
    test=dict(
        type=dataset_type,
        data_root=dataset_root,
        occ_gt_root=occ_gt_root,
        # ann_file=dataset_root + 'nuscenes_infos_test_sweep.pkl',
        ann_file=dataset_root + 'bevdetv2-nuscenes_infos_val.pkl',
        pipeline=test_pipeline,
        classes=det_class_names,
        modality=input_modality,
        test_mode=True
    ),
)

optimizer = dict(
    type='AdamW',
    lr=5e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offset': dict(lr_mult=0.1),
        }),
    weight_decay=0.01
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    by_epoch=True,
    step=[22, 24],
    gamma=0.2
)


# load pretrained weights
load_from = 'ckpts/pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth'
revise_keys = [('backbone', 'img_backbone')]

# resume the last training
resume_from = None

# checkpointing
checkpoint_config = dict(interval=1, max_keep_ckpts=1)

# logging
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# log_config = dict(
#     interval=1,
#     hooks=[
#         dict(type='MyTextLoggerHook', interval=1, reset_flag=True),
#         dict(type='MyTensorboardLoggerHook', interval=500, reset_flag=True)
#     ]
# )

# evaluation
eval_config = dict(interval=total_epochs)

# other flags
debug = False

