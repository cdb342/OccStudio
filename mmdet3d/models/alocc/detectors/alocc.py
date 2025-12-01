
import torch
import torch.nn.functional as F
import torch.nn as nn
import mmcv
from mmcv.runner import BaseModule, force_fp32
from mmdet.models import DETECTORS
from mmdet3d.models import builder
from mmdet3d.models.detectors import CenterPoint
import numpy as np
import torch
from mmdet.models.backbones.resnet import ResNet
from mmdet3d.models.backbones.swin import SwinTransformer
from mmdet3d.models.backbones.swin_bev import SwinTransformerBEVFT
from mmdet3d.models.backbones.flash_intern_image import FlashInternImage
from mmdet3d.models.alocc.heads.occ_loss_utils import CustomFocalLoss
from mmdet3d.models.alocc.heads.occ_loss_utils import nusc_class_frequencies
from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import copy
import math
from ..modules.temporal_fusion import VoxelevelHistoryFusion, SceneLevelHistoryFusion, MotionHisoryFusion
import os


def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res
def positional_encoding_continual_1d(d_model, position):
    """
    :param d_model: dimension of the token
    :param position: position
    :return: (position, d_model) position embedding matrix
    """

    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros((*position.shape,d_model)).to(position.device)
    # position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model))).to(position.device)

    pe[..., 0::2] = torch.sin(position.unsqueeze(-1).float() * div_term)
    pe[..., 1::2] = torch.cos(position.unsqueeze(-1).float() * div_term)

    return pe
@DETECTORS.register_module()
class ALOCC(CenterPoint):
    def __init__(self, 
                view_transformer=None,
                img_bev_encoder_backbone=None,
                img_bev_encoder_neck=None,
                backward_projection=None,
                frpn=None,
                depth_net=None,
                context_net=None,
                occupancy_head=None,
                pre_process=None,
                use_depth_supervision=False,
                readd=False,
                fix_void=False,
                occupancy_save_path=None,
                with_cp=False,
                occ_size=[200,200,16],
                _dim_=256,
                interpolation_mode='bilinear',
                wo_pred_occ=False,
                pred_flow=False,
                downsample=16,
                depth_stereo=False,
                save_stereo=False,
                grid_config=None,
                use_focal_loss=True,
                loss_weight_cfg=None,
                balance_cls_weight=True,
                num_cls =19,
                use_another_encoder=False,
                dz=16,
                depth_loss_ce=False,
                occ_backbone_2d=False,
                occ_2d_out_channels=256,
                voxel_out_channel=48,
                dataset='nuscenes',
                occ_2d=False,
                do_history=True,
                history_cat_num=16,
                history_cat_conv_out_channels=None,
                single_bev_num_channels=80,
                alocc_head=None,
                depth2occ_intra=False,
                img_seg_weight=1.0,
                sem_sup_prototype=False,
                use_mask_net2=False,
                gts_surroundocc=False,
                cal_metric_in_model=False,
                ####################
                # GDFusion voxel-level history fusion
                not_use_history=False,
                vox_his_recurrence=False,
                pre_gen_his_grid=False,
                use_vox_his_func=False,
                not_use_time_emb=False,
                vox_his_sup_w_his=False,
                max_seqlen=50,
                vox_pre_sup_his_length=8,
                vox_his_time_emb_long=False,
                vox_his_time_emb_fixed=False,
                vox_his_learnable_lr=False,
                vox_his_base_lr=1.,
                ####################
                # GDFusion scene-level history fusion
                scene_his_fusion=False,
                scene_his_base_lr=1.,
                scene_his_learned_lr=False,
                scene_his_post_project=False,
                scene_his_pre_detach=False,
                scene_his_warm_up_start=0.1,
                scene_his_warm_up_iter=4,
                squential_length=20,
                scene_his_warm_up=False,
                scene_his_after_vox_his=False,
                scene_his_before_vox_his=True,
                scene_his_sup_w_his=False,
                scene_his_with_virtual_adverse=False,
                scene_his_with_virtual_adverse_weight=0.1,
                scene_his_decay=False,
                scene_his_decay_rate=0.1,
                scene_his_learnable_decay=False,
                scene_his_mlp=False,
                scene_his_mlp_mid_channel=80,
                ####################
                # GDFusion motion history fusion
                motion_his_fusion=False,
                motion_his_flow_bin=False,
                motion_his_sup_w_his=False,
                motion_his_base_lr=1.,
                motion_his_learnable_lr=False,
                motion_his_pred_with_his=False,
                motion_dim=2,
                ####################
                # GDFusion geometry history fusion
                geometry_his_fusion=False,
                ####################
                # CausalOcc
                class_freq=None,
                class_prob=None,
                geometry_group=False,
                learnable_pose=False,
                pose_weight=1.,
                pose_wo_bias=False,
                load_sem_gt=False,
                pose_add_noise=False,
                soft_filling_with_offset=False,
                  **kwargs):
        super(ALOCC, self).__init__(**kwargs)
        self.fix_void = fix_void
        self.view_transformer = builder.build_neck(view_transformer) if view_transformer else None
        self.img_bev_encoder_backbone = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone else None
        self.img_bev_encoder_neck = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck else None
        self.pre_process = builder.build_backbone(pre_process) if pre_process else None
        self.alocc_head = builder.build_head(alocc_head) if alocc_head else None
        self.backward_projection = builder.build_head(backward_projection) if backward_projection else None
    
        if not self.view_transformer: assert not frpn, 'frpn relies on LSS'
        self.frpn = builder.build_head(frpn) if frpn else None
        self.depth_net = builder.build_head(depth_net) if depth_net else None
        self.context_net = builder.build_head(context_net) if context_net else None
        self.occupancy_head = builder.build_head(occupancy_head) if occupancy_head else None

        
        self.readd = readd # fuse voxel features and bev features
        self.use_depth_supervision = use_depth_supervision
        self.occupancy_save_path = occupancy_save_path # for saving data\for submitting to test server

        # Deal with history
        self.single_bev_num_channels = single_bev_num_channels
        self.do_history = do_history
        self.interpolation_mode = interpolation_mode
        self.history_cat_num = history_cat_num
        self.history_cam_sweep_freq = 0.5 # seconds between each frame
        history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                         if history_cat_conv_out_channels is not None 
                                         else self.single_bev_num_channels)
       
        ################
        self.downsample=downsample
        self.grid_config=grid_config
        
        self.depth_stereo=depth_stereo
        self.save_stereo=save_stereo
        self.history_stereo=None
        self.num_frame=1 if not depth_stereo else 2
        self.temporal_frame=1
        self.extra_ref_frames=1 if  depth_stereo else 0
        
        
        self.not_use_history=not_use_history
        self.use_focal_loss = use_focal_loss
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))
            self.focal_loss_geo = builder.build_loss(dict(type='CustomFocalLoss',activated=True))
        if loss_weight_cfg is None:
            self.loss_weight_cfg = {
                "loss_voxel_ce_weight": 1.0,
                "loss_voxel_sem_scal_weight": 1.0,
                "loss_voxel_geo_scal_weight": 1.0,
                "loss_voxel_lovasz_weight": 1.0,
            }
        else:
            self.loss_weight_cfg = loss_weight_cfg
        
        # voxel losses
        self.loss_voxel_ce_weight = self.loss_weight_cfg.get('loss_voxel_ce_weight', 1.0)
        self.loss_voxel_sem_scal_weight = self.loss_weight_cfg.get('loss_voxel_sem_scal_weight', 1.0)
        self.loss_voxel_geo_scal_weight = self.loss_weight_cfg.get('loss_voxel_geo_scal_weight', 1.0)
        self.loss_voxel_lovasz_weight = self.loss_weight_cfg.get('loss_voxel_lovasz_weight', 1.0)
        if balance_cls_weight:
            if num_cls == 19:
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls] + 0.001))
                self.class_weights = torch.cat([torch.tensor([0]), self.class_weights])
                self.class_weights_geo = torch.cat([torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls][-1:] + 0.001)),\
                    torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls][:-1].sum() + 0.001)[None])])
                                                    
            else:
                if num_cls == 17: nusc_class_frequencies[0] += nusc_class_frequencies[-1]
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls] + 0.001))
        else:
            self.class_weights = torch.ones(num_cls)/num_cls  # FIXME hardcode 17
        if class_freq!=None:
            self.class_freq=class_freq 
        else:
            self.class_freq=torch.from_numpy(nusc_class_frequencies)
        if self.fix_void:
            self.class_freq=torch.cat([torch.tensor([0]), self.class_freq])
        if class_prob is not None:
            self.class_prob=torch.Tensor(class_prob)
        else:
            self.class_prob=None
        self.use_another_encoder=use_another_encoder
        
        self.img_bev_encoder_backbone2 = builder.build_backbone(img_bev_encoder_backbone) if img_bev_encoder_backbone and self.use_another_encoder else None
        self.img_bev_encoder_neck2 = builder.build_neck(img_bev_encoder_neck) if img_bev_encoder_neck and self.use_another_encoder else None
        self.depth2occ_intra=depth2occ_intra


        self.num_cls=num_cls
        self.img_seg_weight=img_seg_weight
     
        self.depth_loss_ce=depth_loss_ce
        self.sem_sup_prototype=sem_sup_prototype
        self.soft_filling_with_offset=soft_filling_with_offset
        self.use_mask_net2=use_mask_net2
        self.dataset=dataset
       
        self.occ_2d=occ_2d
        self.pred_flow=pred_flow
        self.wo_pred_occ=wo_pred_occ

        self.vox_his_recurrence=vox_his_recurrence
        self.occ_backbone_2d=occ_backbone_2d
        self.occ_2d_out_channels=occ_2d_out_channels
        self.voxel_out_channel=voxel_out_channel
        self.dz=dz
        self._dim_=_dim_
        self.scene_his_learnable_decay=scene_his_learnable_decay
        if self.occ_backbone_2d :
            self.final_conv = nn.Conv2d(occ_2d_out_channels, voxel_out_channel*dz, kernel_size=3, stride=1, padding=1)
        self.with_cp=with_cp
        
        self.pre_gen_his_grid=pre_gen_his_grid
        if grid_config is not None:
            occ_size=[int((grid_config['x'][1]-grid_config['x'][0])/grid_config['x'][2]),int((grid_config['y'][1]-grid_config['y'][0])/grid_config['y'][2]),int((grid_config['z'][1]-grid_config['z'][0])/grid_config['z'][2])]
        if pre_gen_his_grid:
            
            h,w,z=occ_size
            if self.occ_2d:
                xs = torch.linspace(0, w - 1, w).view(1, w).expand(h, w)
                ys = torch.linspace(0, h - 1, h).view(h, 1).expand(h, w)
                self.his_grid = torch.stack(
                    (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(1, h, w, 4).view(1,h,w,1,4,1)

                # This converts BEV indices to meters
                # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
                # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
                feat2bev = torch.zeros((4,4))
                # feat2bev[0, 0] = self.view_transformer.dx[0]
                # feat2bev[1, 1] = self.view_transformer.dx[1]
                # feat2bev[0, 3] = self.view_transformer.bx[0] - self.view_transformer.dx[0] / 2.
                # feat2bev[1, 3] = self.view_transformer.bx[1] - self.view_transformer.dx[1] / 2.

                feat2bev[0, 0] = self.grid_config['x'][2]
                feat2bev[1, 1] = self.grid_config['y'][2]
                feat2bev[0, 3] = self.grid_config['x'][0]
                feat2bev[1, 3] = self.grid_config['y'][0]
                feat2bev[2, 2] = 1
                feat2bev[3, 3] = 1
                self.feat2bev = feat2bev.view(1,4,4)
            else:
                xs = torch.linspace(0, w - 1, w).view(1, w, 1).expand(h, w, z)
                ys = torch.linspace(0, h - 1, h).view(h, 1, 1).expand(h, w, z)
                zs = torch.linspace(0, z - 1, z).view(1, 1, z).expand(h, w, z)
                self.his_grid = torch.stack(
                    (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(1, h, w, z, 4).view(1, h, w, z, 4, 1)
                
                # This converts BEV indices to meters
                # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
                # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
                feat2bev = torch.zeros((4,4))
                # feat2bev[0, 0] = self.view_transformer.dx[0]
                # feat2bev[1, 1] = self.view_transformer.dx[1]
                # feat2bev[2, 2] = self.view_transformer.dx[2]
                # feat2bev[0, 3] = self.view_transformer.bx[0] - self.view_transformer.dx[0] / 2.
                # feat2bev[1, 3] = self.view_transformer.bx[1] - self.view_transformer.dx[1] / 2.
                # feat2bev[2, 3] = self.view_transformer.bx[2] - self.view_transformer.dx[2] / 2.

                feat2bev[0, 0] = self.grid_config['x'][2]
                feat2bev[1, 1] = self.grid_config['y'][2]
                feat2bev[2, 2] = self.grid_config['z'][2]
                feat2bev[0, 3] = self.grid_config['x'][0]
                feat2bev[1, 3] = self.grid_config['y'][0]
                feat2bev[2, 3] = self.grid_config['z'][0]
                # feat2bev[2, 2] = 1
                feat2bev[3, 3] = 1
                self.feat2bev = feat2bev.view(1,4,4)

        self.scene_his_fusion=scene_his_fusion
        if scene_his_fusion:
            self.scene_his_sup_w_his=scene_his_sup_w_his

            self.scene_his_func_before_vox_his=SceneLevelHistoryFusion(
                    scene_his_learned_lr=scene_his_learned_lr,
                    scene_his_base_lr=scene_his_base_lr,
                    single_bev_num_channels=single_bev_num_channels,
                    scene_his_post_project=scene_his_post_project,
                    scene_his_pre_detach=scene_his_pre_detach,
                    scene_his_warm_up_start=scene_his_warm_up_start,
                    scene_his_warm_up_iter=scene_his_warm_up_iter,
                    squential_length=squential_length,
                    scene_his_warm_up=scene_his_warm_up,
                    scene_his_sup_w_his=scene_his_sup_w_his,
                    scene_his_with_virtual_adverse=scene_his_with_virtual_adverse,
                    scene_his_with_virtual_adverse_weight=scene_his_with_virtual_adverse_weight,
                    scene_his_decay=scene_his_decay,
                    scene_his_decay_rate=scene_his_decay_rate,
                    scene_his_learnable_decay=scene_his_learnable_decay,
        
                    scene_his_mlp=scene_his_mlp,
            
                    scene_his_mlp_mid_channel=scene_his_mlp_mid_channel,
        
                    )
            self.scene_his_after_vox_his=scene_his_after_vox_his
            if scene_his_after_vox_his:
                self.scene_his_func_after_vox_his=copy.deepcopy(self.scene_his_func_before_vox_his)
            self.scene_his_before_vox_his=scene_his_before_vox_his
            if not scene_his_before_vox_his:
                self.scene_his_func_before_vox_his=None
       
        self.use_vox_his_func=use_vox_his_func
        if use_vox_his_func:
            self.fuse_history_func=VoxelevelHistoryFusion(vox_his_recurrence=vox_his_recurrence,
                 view_transformer=self.view_transformer,
                 single_bev_num_channels=single_bev_num_channels,
                 history_cat_conv_out_channels=history_cat_conv_out_channels,
                 do_history=do_history,
                 interpolation_mode=interpolation_mode,
                 history_cat_num=history_cat_num,
                 occ_2d=occ_2d,
                 pre_gen_his_grid=pre_gen_his_grid,
                 occ_size=occ_size,
                 with_cp=with_cp,)
            
        
        self.motion_his_fusion=motion_his_fusion
        if motion_his_fusion:
            self.motion_his_pred_with_his=motion_his_pred_with_his
            if motion_his_pred_with_his:
                offset_net_input_channel=single_bev_num_channels*2
            else:
                offset_net_input_channel=single_bev_num_channels
            self.his_stream_offset_net=MotionHisoryFusion(offset_net_input_channel=offset_net_input_channel,conv=nn.Conv3d,bev_2d=occ_2d,
                                                       motion_his_flow_bin=motion_his_flow_bin,motion_his_sup_w_his=motion_his_sup_w_his,motion_his_base_lr=motion_his_base_lr,motion_his_learnable_lr=motion_his_learnable_lr,motion_dim=motion_dim)

        self.geometry_his_fusion=geometry_his_fusion

        self.not_use_time_emb=not_use_time_emb
        self.vox_his_sup_w_his=vox_his_sup_w_his
        self.vox_pre_sup_his_length=vox_pre_sup_his_length
        
        #######################################
        if not use_vox_his_func and not self.not_use_history :
            self.max_seqlen=max_seqlen
             ## Embed each sample with its relative temporal offset with current timestep
            # conv = nn.Conv2d if self.view_transformer.nx[-1] == 1 else nn.Conv3d
            conv = nn.Conv2d if occ_backbone_2d else nn.Conv3d
            if vox_his_recurrence:
                self.history_cat_num=1

            self.vox_his_time_emb_long=vox_his_time_emb_long
            self.vox_his_learnable_lr=vox_his_learnable_lr
            self.vox_his_base_lr=vox_his_base_lr
            
            his_bev_channel=self.single_bev_num_channels * (self.history_cat_num + 1)
            if vox_his_learnable_lr:
                self.history_keyframe_lr_conv = nn.Sequential(
                    conv(self.single_bev_num_channels * (self.history_cat_num + 1),
                            1,
                            kernel_size=1,
                            padding=0,
                            stride=1),
                    nn.Sigmoid())
            self.vox_his_time_emb_fixed=vox_his_time_emb_fixed
            if vox_his_time_emb_fixed:
                self.time_emb_net=nn.Sequential(
                    nn.Linear(self.single_bev_num_channels,self.single_bev_num_channels*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(self.single_bev_num_channels*2,self.single_bev_num_channels))
                time_emb_dim=self.single_bev_num_channels 
            else:
                time_emb_dim=self.single_bev_num_channels + 1
            
            if vox_his_sup_w_his:
                self.history_keyframe_time_conv = nn.Sequential(
                conv(time_emb_dim,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.ReLU(inplace=True))
                ## Then concatenate and send them through an MLP.
                self.history_keyframe_cat_conv = nn.Sequential(
                    conv(his_bev_channel,
                            history_cat_conv_out_channels,
                            kernel_size=1,
                            padding=0,
                            stride=1),
                    nn.ReLU(inplace=True))
            else:
                self.history_keyframe_time_conv = nn.Sequential(
                    conv(time_emb_dim,
                            self.single_bev_num_channels,
                            kernel_size=1,
                            padding=0,
                            stride=1),
                    nn.SyncBatchNorm(self.single_bev_num_channels),
                    nn.ReLU(inplace=True))
                ## Then concatenate and send them through an MLP.
                self.history_keyframe_cat_conv = nn.Sequential(
                    conv(his_bev_channel,
                            history_cat_conv_out_channels,
                            kernel_size=1,
                            padding=0,
                            stride=1),
                    nn.SyncBatchNorm(history_cat_conv_out_channels),
                    nn.ReLU(inplace=True))
        
        self.history_sweep_time = None
        self.history_bev = None
        self.history_bev_before_encoder = None
        self.history_seq_ids = None
        self.history_forward_augs = None
        self.count = 0
        
        ##############################################
        
        self.learnable_pose=learnable_pose
        self.pose_weight=pose_weight
        if learnable_pose :
            pose_input_dim=_dim_
            self.pose_net=nn.Sequential(
                    nn.Conv2d(pose_input_dim,
                            _dim_,
                            kernel_size=3,
                            padding=1,
                            stride=1),
                    nn.SyncBatchNorm(_dim_),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(_dim_,
                            _dim_,
                            kernel_size=3,
                            padding=1,
                            stride=1),
                    nn.SyncBatchNorm(_dim_),
                    nn.ReLU(inplace=True),
                    
                    )
            self.pose_head=  nn.Sequential(
                    nn.Linear(_dim_+3*3+3,
                            _dim_),
                    nn.ReLU(inplace=True),
                    nn.Linear(_dim_,3*3+3,bias=not pose_wo_bias),
                    )
       
        self.geometry_group=geometry_group
        self.load_sem_gt=load_sem_gt
        self.pose_add_noise=pose_add_noise
        self.gts_surroundocc=gts_surroundocc
        self.cal_metric_in_model=cal_metric_in_model
    def with_specific_component(self, component_name):
        """Whether the model owns a specific component"""
        return getattr(self, component_name, None) is not None

    def image_encoder(self, img, stereo=False):
        imgs = img
        B, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * N, C, imH, imW)
        
        x = self.img_backbone(imgs)
        stereo_feat = None
        if stereo:
            stereo_feat = x[0]
            if len(x)<4:
                x = x[1:]
        if self.with_img_neck:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        return x, stereo_feat

    @force_fp32()
    def bev_encoder(self, x,use_another_encoder=False):
        out=dict()
        if self.with_specific_component('img_bev_encoder_backbone'):
            if use_another_encoder:
                x = self.img_bev_encoder_backbone2(x)
            else:
                x = self.img_bev_encoder_backbone(x)
            if isinstance(x,dict):
                x=x['feats']
        if self.with_specific_component('img_bev_encoder_neck'):
            if use_another_encoder:
                x = self.img_bev_encoder_neck2(x)
            else:
                x = self.img_bev_encoder_neck(x)
        
        if type(x) not in [list, tuple]:
            x = [x]
        out['x']=x
        return out

    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda,feat_for_pred_bias=None): # align features with 3d shift
        
        if self.occ_2d:
            curr_bev = curr_bev.unsqueeze(-1)
            if feat_for_pred_bias is not None:
                feat_for_pred_bias=feat_for_pred_bias.unsqueeze(-1)
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
            if feat_for_pred_bias is not None:
                feat_for_pred_bias=feat_for_pred_bias.permute(0, 1, 4, 2, 3)
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)

        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        ## Deal with first batch
        if self.history_bev is None:
            self.history_bev = curr_bev.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1).detach()
            else:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
            # All 0s, representing current timestep.
            self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)

            if self.vox_his_sup_w_his:
                self.history_grid=[[]]*len(start_of_sequence)
                self.history_feat=[[]]*len(start_of_sequence)
                self.history_seqlen_offset=[0]*len(start_of_sequence)
                self.history_bev_his=curr_bev.repeat(1, self.history_cat_num, 1, 1, 1).detach()
            self.his_iter=torch.zeros(len(start_of_sequence)).to(curr_bev)


        self.history_bev = self.history_bev.detach()

        assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.
        assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)
        ## Replace all the new sequences' positions in history with the curr_bev information
        
        self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
        if start_of_sequence.sum()>0:
            if voxel_feat:    
                
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1).detach()
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]
            
            self.his_iter[start_of_sequence]=0

        if self.vox_his_recurrence:
            
            self.history_bev=self.history_bev[:,-self.single_bev_num_channels:,...]
        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
        
        for i in range(len(start_of_sequence)):
            history_bev_i_clone=None
            if self.vox_his_sup_w_his:

                if self.his_iter[i]==0:
                    self.history_grid[i]=[]
                    self.history_feat[i]=[]
                if self.his_iter[i]<=self.vox_pre_sup_his_length:
                    
                    if len(self.history_feat[i])>0:
                        self.history_bev[i]=self.history_feat[i][0]
                
                for j in range(len(self.history_feat[i])):
                    history_feat_j=self.history_feat[i][j]
                    grid_j=self.history_grid[i][j]
        
                    sampled_history_bev,_,tmp_bev,mc,normalize_factor=self.warp_his_(history_feat_j,self.history_bev[i:i+1],forward_augs[i:i+1],curr_to_prev_ego_rt[i:i+1],voxel_feat,start_of_sequence,feat_for_pred_bias,grid=grid_j)
                    
                    if self.vox_his_time_emb_long:
                        history_sweep_time=torch.Tensor([[j,j-1]]).to(curr_bev).clamp(0)
                    else:
                        history_sweep_time=torch.Tensor([[0.,1]]).to(curr_bev)
                    feats_to_return,_=self.his_net_func(history_feat_j, sampled_history_bev,voxel_feat,1, mc, z, h, w,c_,torch.Tensor([[0.,1]]).to(curr_bev))
                    
                    if self.occ_2d:
                        feats_to_return=feats_to_return.unsqueeze(2)
                    
                    history_bev_=self.history_bev.clone()
                    history_bev_[i:i+1]=feats_to_return
                    self.history_bev=history_bev_
                    
                    if j==0 and self.his_iter[i]>=self.vox_pre_sup_his_length:
                        history_bev_i_clone=self.history_bev[i].detach().clone()

        sampled_history_bev,grid,tmp_bev,mc,normalize_factor,rt_flow=self.warp_his_(curr_bev,self.history_bev,forward_augs,curr_to_prev_ego_rt,voxel_feat,start_of_sequence,feat_for_pred_bias)
        
        for i in range(len(start_of_sequence)):
            if self.vox_his_sup_w_his:
                self.history_grid[i].append(grid[i:i+1])
                self.history_feat[i].append(curr_bev[i:i+1].detach())
                if len(self.history_grid)>self.vox_pre_sup_his_length:
                    self.history_grid=self.history_grid[1:]
                    self.history_feat=self.history_feat[1:]
        
        ## Update history
        # Add in current frame to features & timestep
        results={}
            

        feats_to_return=None

        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)
        
        
        if self.vox_his_recurrence:
            self.history_sweep_time=self.history_sweep_time[:,:2]
            
        if self.vox_his_time_emb_long:
        
            history_sweep_time=torch.stack((self.his_iter,self.his_iter-1),dim=0).permute(1,0).clamp(0)

        else:
            history_sweep_time=self.history_sweep_time
        
        feats_to_return,feats_cat=self.his_net_func(curr_bev, sampled_history_bev,voxel_feat,n, mc, z, h, w,c_,history_sweep_time,additional_his_bev=feats_to_return)
            
        # Time conv
        if self.occ_2d:
            feats_to_return=feats_to_return.unsqueeze(2)
        
        if self.vox_his_recurrence:
            self.history_bev = feats_to_return.detach().clone()
        else:
            self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach()#.clone()
  
        if history_bev_i_clone is not None:
            self.history_bev[i]=history_bev_i_clone

        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if self.vox_his_recurrence:
            if start_of_sequence.sum()>0:
                tem=feats_to_return.clone()
                tem[start_of_sequence] = curr_bev[start_of_sequence].clone()+feats_to_return[start_of_sequence]*0.
                feats_to_return=tem
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        
        if self.occ_2d:
            feats_to_return=feats_to_return.squeeze(-1)
        results.update({'fused_bev_feat':feats_to_return.clone()})
        self.his_iter+=1
        
        return results
    def warp_his_(self,curr_bev,tmp_bev,forward_augs,curr_to_prev_ego_rt,voxel_feat,start_of_sequence,feat_for_pred_bias,normalize_factor=None,rt_flow=None,grid=None):
        n, c_, z, h, w=curr_bev.shape
        if not self.occ_2d:
            if grid is None:
                if self.pre_gen_his_grid:
                    grid=self.his_grid.to(curr_bev.device).repeat(n,1,1,1,1,1,)
                    feat2bev=self.feat2bev.to(curr_bev.device)
                else:
                    # Generate grid
                    xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
                    ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
                    zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
                    grid = torch.stack(
                        (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)
                    
                    # This converts BEV indices to meters
                    # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
                    # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
                    feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
                    # feat2bev[0, 0] = self.view_transformer.dx[0]
                    # feat2bev[1, 1] = self.view_transformer.dx[1]
                    # feat2bev[2, 2] = self.view_transformer.dx[2]
                    # feat2bev[0, 3] = self.view_transformer.bx[0] - self.view_transformer.dx[0] / 2.
                    # feat2bev[1, 3] = self.view_transformer.bx[1] - self.view_transformer.dx[1] / 2.
                    # feat2bev[2, 3] = self.view_transformer.bx[2] - self.view_transformer.dx[2] / 2.

                    feat2bev[0, 0] = self.grid_config['x'][2]
                    feat2bev[1, 1] = self.grid_config['y'][2]
                    feat2bev[2, 2] = self.grid_config['z'][2]
                    feat2bev[0, 3] = self.grid_config['x'][0]
                    feat2bev[1, 3] = self.grid_config['y'][0]
                    feat2bev[2, 3] = self.grid_config['z'][0]

                    # feat2bev[2, 2] = 1
                    feat2bev[3, 3] = 1
                    feat2bev = feat2bev.view(1,4,4)
                
                ## Get flow for grid sampling.
                # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
                # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
                # transform to previous grid locations.
                rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                        @ torch.inverse(forward_augs) @ feat2bev)

                grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid

                # normalize and sample
                normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
                grid = grid[:,:,:,:, :3,0] / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0
            
            
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
            if self.motion_his_fusion:
                rt_flow2=rt_flow
                # self.history_sample_bias=torch.tensor([[0,0],[0,4],[0,-4],[4,0],[-4,0]])
                # conv = nn.Conv2d if self.view_transformer.nx[-1] == 1 else nn.Conv3d
                if feat_for_pred_bias is None:
                    feat_for_pred_bias=curr_bev
                if self.motion_his_pred_with_his:
                    sampled_history_bev_wo_offset = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)
                    feat_for_pred_bias=torch.cat((sampled_history_bev_wo_offset,curr_bev),dim=1)
                sampled_history_bev,grid=self.his_stream_offset_net(feat_for_pred_bias,tmp_bev,grid,normalize_factor,rt_flow2,start_of_sequence=start_of_sequence)

            else:
                sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)
        
        else:
            if grid is None:
                if self.pre_gen_his_grid:
                    grid=self.his_grid.to(curr_bev.device).repeat(n,1,1,1,1,1,)
                    feat2bev=self.feat2bev.to(curr_bev.device)
                else:
                    # Generate grid
                    xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w).expand(h, w)
                    ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1).expand(h, w)
                    grid = torch.stack(
                        (xs, ys, torch.ones_like(xs), torch.ones_like(xs)), -1).view(1, h, w, 4).expand(n, h, w, 4).view(n,h,w,1,4,1)

                    # This converts BEV indices to meters
                    # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
                    # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
                    feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
                    # feat2bev[0, 0] = self.view_transformer.dx[0]
                    # feat2bev[1, 1] = self.view_transformer.dx[1]
                    # feat2bev[0, 3] = self.view_transformer.bx[0] - self.view_transformer.dx[0] / 2.
                    # feat2bev[1, 3] = self.view_transformer.bx[1] - self.view_transformer.dx[1] / 2.

                    feat2bev[0, 0] = self.grid_config['x'][2]
                    feat2bev[1, 1] = self.grid_config['y'][2]
                    feat2bev[0, 3] = self.grid_config['x'][0]
                    feat2bev[1, 3] = self.grid_config['y'][0]
                    feat2bev[2, 2] = 1
                    feat2bev[3, 3] = 1
                    feat2bev = feat2bev.view(1,4,4)
                ## Get flow for grid sampling.
                # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
                # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
                # transform to previous grid locations.
                rt_flow = (torch.inverse(feat2bev) @ self.history_forward_augs @ curr_to_prev_ego_rt
                        @ torch.inverse(forward_augs) @ feat2bev)
                
                grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
                
                # normalize and sample
                normalize_factor = torch.tensor([w - 1.0, h - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)

                grid = grid[:,:,:,:, :2,0] / normalize_factor.view(1, 1, 1, 1, 2) * 2.0 - 1.0
            
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
            
            if self.motion_his_fusion:
                # self.history_sample_bias=torch.tensor([[0,0],[0,4],[0,-4],[4,0],[-4,0]])
                # conv = nn.Conv2d if self.view_transformer.nx[-1] == 1 else nn.Conv3d
                
                rt_flow2=rt_flow
                if feat_for_pred_bias is None:
                    feat_for_pred_bias=curr_bev
                if self.motion_his_pred_with_his:
                    sampled_history_bev_wo_offset = F.grid_sample(tmp_bev[:,:,0], grid.to(curr_bev.dtype)[...,0,:], align_corners=True, mode=self.interpolation_mode)
                    
                    sampled_history_bev_wo_offset=sampled_history_bev_wo_offset.unsqueeze(2)
                    feat_for_pred_bias=torch.cat((sampled_history_bev_wo_offset,curr_bev),dim=1)
                
                sampled_history_bev,grid=self.his_stream_offset_net(feat_for_pred_bias,tmp_bev,grid,normalize_factor,rt_flow2,start_of_sequence=start_of_sequence)
                
            else:
                sampled_history_bev = F.grid_sample(tmp_bev[:,:,0], grid.to(curr_bev.dtype)[...,0,:], align_corners=True, mode=self.interpolation_mode)
         
        return sampled_history_bev,grid,tmp_bev,mc,normalize_factor,rt_flow
    def his_net_func(self,curr_bev, sampled_history_bev,voxel_feat,n, mc, z, h, w,c_,history_sweep_time,additional_his_bev=None):
        
        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        if self.vox_his_recurrence:
          
            feats_to_return = feats_cat.reshape(feats_cat.shape[0], 1+ 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        else:
            feats_to_return = feats_cat.reshape(
                    feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
        if not self.not_use_time_emb:
           
            if voxel_feat:
               
                if self.vox_his_time_emb_fixed:
                    time_emb=positional_encoding_continual_1d(self.single_bev_num_channels,history_sweep_time*self.history_cam_sweep_freq)
                    _,n_his,_=time_emb.shape
                    time_emb=time_emb.reshape(n*n_his,-1)
                    time_emb=self.time_emb_net(time_emb)
                    time_emb=time_emb.reshape(n,n_his,-1)
                    feats_to_return=feats_to_return+time_emb[:,:,:,None,None, None].repeat(1, 1, 1, *feats_to_return.shape[3:])
         
                else:
                    feats_to_return = torch.cat(
                    [feats_to_return, history_sweep_time[:, :, None, None, None, None].repeat(1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
                    ], dim=2) # B x (1 + T) x 81 x Z x H x W
            else:
                feats_to_return = torch.cat(
                [feats_to_return, history_sweep_time[:, :, None, None, None].repeat(
                    1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
                ], dim=2) # B x (1 + T) x 81 x H x W

        if self.occ_2d:
            feats_to_return=feats_to_return.squeeze(3)
        

        if self.with_cp:
            feats_to_return = checkpoint(self.history_keyframe_time_conv, feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:])
        else:
            
            feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        if self.vox_his_learnable_lr:
            lr=self.history_keyframe_lr_conv(feats_to_return.reshape(feats_to_return.shape[0], -1, *feats_to_return.shape[3:]))*self.vox_his_base_lr
            feats_to_return1,feats_to_return2=feats_to_return[:,0:1,...],feats_to_return[:,1:2,...]
            weight1,weight2=self.history_keyframe_cat_conv[0].weight[:,:feats_to_return1.shape[2]],self.history_keyframe_cat_conv[0].weight[:,feats_to_return1.shape[2]:feats_to_return1.shape[2]*2]
            
            if self.occ_2d:
                feats_to_return1_weighted=torch.einsum('ijhw,bcihw->bcjhw',weight1,feats_to_return1).squeeze(1)
                feats_to_return2_weighted=torch.einsum('ijhw,bcihw->bcjhw',weight2,feats_to_return2).squeeze(1)
                bias=self.history_keyframe_cat_conv[0].bias[:,None,None]
            else:
                feats_to_return1_weighted=torch.einsum('ijzhw,bcizhw->bcjzhw',weight1,feats_to_return1).squeeze(1)
                feats_to_return2_weighted=torch.einsum('ijzhw,bcizhw->bcjzhw',weight2,feats_to_return2).squeeze(1)
                bias=self.history_keyframe_cat_conv[0].bias[:,None,None,None]
            feats_to_return=feats_to_return2.squeeze(1)-feats_to_return2_weighted*lr+feats_to_return1_weighted*lr+bias
            

            feats_to_return=self.history_keyframe_cat_conv[1](feats_to_return)
            feats_to_return=self.history_keyframe_cat_conv[2](feats_to_return)
        else:        
            if self.with_cp:  
                feats_to_return =checkpoint(self.history_keyframe_cat_conv,feats_to_return.reshape(feats_to_return.shape[0], -1, *feats_to_return.shape[3:]))
            else:
                feats_to_return = self.history_keyframe_cat_conv(
                    feats_to_return.reshape(feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        return feats_to_return,feats_cat

    def prepare_inputs(self, inputs,inputs_stereo):
        B, N_, C, H, W = inputs[0].shape
        
        N = N_ // self.num_frame
        
        if not (( self.geometry_his_fusion) and not self.depth_stereo):
            if  self.save_stereo and not self.training:
                
                imgs=[inputs[0]]
                N=N_
            else:
                imgs = inputs[0].view(B, N, self.num_frame, C, H, W)
                imgs = torch.split(imgs, 1, 2)
                imgs = [t.squeeze(2) for t in imgs]
        else:
            imgs=None
        sensor2egos, ego2globals = inputs[1:3]
        sensor2egos_stereo, ego2globals_stereo= inputs_stereo
        sensor2egos = sensor2egos.view(B, 1, N, 4, 4).contiguous()
        ego2globals = ego2globals.view(B, 1, N, 4, 4).contiguous()
        sensor2egos_stereo = sensor2egos_stereo.view(B, 1, N, 4, 4).contiguous()
        ego2globals_stereo = ego2globals_stereo.view(B, 1, N, 4, 4).contiguous()
        
        sensor2egos_cv, ego2globals_cv = sensor2egos, ego2globals
        sensor2egos_curr = sensor2egos_cv.double()
        ego2globals_curr = ego2globals_cv.double()
        sensor2egos_adj = sensor2egos_stereo.double()
        ego2globals_adj = ego2globals_stereo.double()
        curr2adjsensor = \
            torch.inverse(ego2globals_adj @ sensor2egos_adj) \
            @ ego2globals_curr @ sensor2egos_curr
        curr2adjsensor = curr2adjsensor.float().squeeze(1)
        return imgs,curr2adjsensor
    
    def extract_stereo_ref_feat(self, x):
        
        B, N, C, imH, imW = x.shape
        x = x.view(B * N, C, imH, imW)
        if isinstance(self.img_backbone,ResNet):
            if self.img_backbone.deep_stem:
                x = self.img_backbone.stem(x)
            else:
                x = self.img_backbone.conv1(x)
                x = self.img_backbone.norm1(x)
                x = self.img_backbone.relu(x)
            x = self.img_backbone.maxpool(x)
            for i, layer_name in enumerate(self.img_backbone.res_layers):
                res_layer = getattr(self.img_backbone, layer_name)
                x = res_layer(x)
                return x
            
        elif isinstance(self.img_backbone, SwinTransformer):
            x = self.img_backbone.patch_embed(x)
            hw_shape = (self.img_backbone.patch_embed.DH,
                        self.img_backbone.patch_embed.DW)
            if self.img_backbone.use_abs_pos_embed:
                x = x + self.img_backbone.absolute_pos_embed
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                return out
        elif isinstance(self.img_backbone, SwinTransformerBEVFT):
            x, hw_shape = self.img_backbone.patch_embed(x)
            if self.img_backbone.use_abs_pos_embed:
                # x = x + self.img_backbone.absolute_pos_embed
                absolute_pos_embed = F.interpolate(self.img_backbone.absolute_pos_embed, 
                                                size=hw_shape, mode='bicubic')
                x = x + absolute_pos_embed.flatten(2).transpose(1, 2)
            x = self.img_backbone.drop_after_pos(x)

            for i, stage in enumerate(self.img_backbone.stages):
                x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
                out = out.view(-1,  *out_hw_shape,
                               self.img_backbone.num_features[i])
                out = out.permute(0, 3, 1, 2).contiguous()
                
                return out
        elif isinstance(self.img_backbone, FlashInternImage):
            x = self.img_backbone.patch_embed(x)
            N, H, W, C = x.shape
            x = x.view(N, H*W, C)

            shape=(H, W)
            for level_idx, level in enumerate(self.img_backbone.levels):
                old_shape = shape
                x, x_ , shape = level(x, return_wo_downsample=True, shape=shape, level_idx=level_idx)   
                h, w= old_shape
                return x_.reshape(N, h, w, -1).permute(0, 3, 1, 2)
    
        else:
            for i in range(4):
                x = self.img_backbone.downsample_layers[i](x)
                x = self.img_backbone.stages[i](x)
                return x
    def extract_img_bev_feat(self, img, img_metas, **kwargs):
        """Extract features of images."""
        
        return_map = {}
        
        if self.pose_add_noise:
            rot=img[1]
            tran=img[2]
            img[1]=img[1]+torch.randn_like(img[1])*self.pose_add_noise
            img[2]=img[2]+torch.randn_like(img[2])*self.pose_add_noise
        cam_params = img[1:7]
        
        if self.depth_stereo:
        
            inputs_=[img[0],*kwargs['aux_cam_params']] if self.training else [img[0],*kwargs['aux_cam_params'][0]]
            inputs_stereo_=kwargs['adj_aux_cam_params'] if self.training else kwargs['adj_aux_cam_params'][0]
            imgs, curr2adjsensor = self.prepare_inputs(inputs_,inputs_stereo_)
            context, stereo_feat = self.image_encoder(imgs[0], stereo=True)
            if self.save_stereo and not self.training:
                """
                The problem of different data augmentation in different training epochs has not been solved yet.
                """
                start_of_sequence = torch.BoolTensor([
                    single_img_metas['start_of_sequence'] 
                    for single_img_metas in img_metas]).to(stereo_feat.device)
                if self.history_stereo is None:
                    self.history_stereo=stereo_feat
                elif start_of_sequence.sum()>0:
                    
                    batch_size=imgs[0].shape[0]
                    tem=stereo_feat.reshape(batch_size,stereo_feat.shape[0]//batch_size,*stereo_feat.shape[1:])[start_of_sequence].detach()
                    self.history_stereo=self.history_stereo.reshape(batch_size,self.history_stereo.shape[0]//batch_size,*self.history_stereo.shape[1:])
                    self.history_stereo[start_of_sequence]=tem
                    self.history_stereo=self.history_stereo.reshape(*stereo_feat.shape)
                
                feat_prev_iv=self.history_stereo
                self.history_stereo=stereo_feat
                
            else:
                with torch.no_grad():
                    feat_prev_iv = self.extract_stereo_ref_feat(imgs[1])
               
            stereo_metas = dict(k2s_sensor=curr2adjsensor,
                     intrins=img[3],
                     post_rots=img[4],
                     post_trans=img[5],
                    #  frustum=self.cv_frustum.to(stereo_feat.device),
                     cv_downsample=4,
                     downsample=self.downsample,
                     grid_config=self.grid_config,
                     cv_feat_list=[feat_prev_iv, stereo_feat])
            
            #################################################################
       
            depth_occ_volumn=None

        else:
            context,_ = self.image_encoder(img[0])
            stereo_metas = None
            depth_occ_volumn=None
        if  self.learnable_pose:
             
            b,n,c,h,w=context.shape
            context_=context.reshape(b*n,c,h,w)
            rot=cam_params[0].reshape(b*n,-1)
            tran=cam_params[1].reshape(b*n,-1)
            
            pose=self.pose_net(context_)
            pose=pose.mean((2,3))
            pose=self.pose_head(torch.cat((pose,rot,tran),dim=-1))*self.pose_weight
            
            rot=rot+pose[:,:3*3]
            tran=tran+pose[:,3*3:]
            
            rot=rot.reshape(b,n,3,3)
            tran=tran.reshape(b,n,3)
            cam_params[0]=rot
            cam_params[1]=tran
        
            
        return_map['context_before_depth_net']=context
        if not self.depth_stereo and ( self.geometry_his_fusion):
            inputs_=[img[0],*kwargs['aux_cam_params']] if self.training else [img[0],*kwargs['aux_cam_params'][0]]
            inputs_stereo_=kwargs['adj_aux_cam_params'] if self.training else kwargs['adj_aux_cam_params'][0]
            imgs, curr2adjsensor = self.prepare_inputs(inputs_,inputs_stereo_)
            stereo_metas = dict(k2s_sensor=curr2adjsensor,
                     intrins=img[3],
                     post_rots=img[4],
                     post_trans=img[5],
                    #  frustum=self.cv_frustum.to(stereo_feat.device),
                     cv_downsample=4,
                     downsample=self.downsample,
                     grid_config=self.grid_config)
        
           
        if self.with_specific_component('depth_net'):
    
            depth_output = self.depth_net(context, cam_params,stereo_metas,img_metas=img_metas,cost_volumn=depth_occ_volumn,**kwargs)
            context, depth_pred,geometry=depth_output['context'],depth_output['depth_pred'],depth_output['geometry']
            if self.soft_filling_with_offset:
                coor_offsets= depth_output['coor_offsets']

            return_map['depth'] = depth_pred
            return_map['context'] = context
            
                
        else:
            # context=None
            geometry=None
            depth_output=None
        if self.with_specific_component('context_net'):
            mlp_input = self.context_net.get_mlp_input(*cam_params)
            context= self.context_net(context, mlp_input)
        
        if self.with_specific_component('view_transformer'):
            if not self.soft_filling_with_offset:
                coor_offsets=None
            
            if self.load_sem_gt :
                if self.training:
                    gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'],kwargs['gt_semantic_map'])
                else:
                    gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'][0],kwargs['gt_semantic_map'][0])
                kwargs['gt_imgseg']=gt_imgseg
                kwargs['gt_depth_disc']=gt_depth
            bev_feat = self.view_transformer(cam_params, context, geometry,coor_offsets,depth_output=depth_output, **kwargs) 
            
            return_map['cam_params'] = cam_params
        else:
            bev_feat = None


        if self.with_specific_component('frpn'): # not used in FB-OCC
            bev_mask_logit = self.frpn(bev_feat)
            bev_mask = bev_mask_logit.sigmoid() > self.frpn.mask_thre
            
            if bev_mask.requires_grad: # during training phase
                gt_bev_mask = kwargs['gt_bev_mask'].to(torch.bool)
                bev_mask = gt_bev_mask | bev_mask
            return_map['bev_mask_logit'] = bev_mask_logit    
        else:
            bev_mask = None

        if self.with_specific_component('backward_projection'):
            if self.geometry_group:
                geometry=geometry.reshape(geometry.shape[0],geometry.shape[1]//self.geometry_group,self.geometry_group,*geometry.shape[2:])
                geometry=geometry.sum(2)/self.geometry_group
            bev_feat_refined = self.backward_projection([context],
                                        img_metas,
                                        lss_bev=bev_feat.mean(-1),
                                        cam_params=cam_params,
                                        bev_mask=bev_mask,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=geometry)  
                                        
            if self.readd:
                bev_feat = bev_feat_refined[..., None] + bev_feat
            else:
                bev_feat = bev_feat_refined
 
        if self.with_specific_component('pre_process'):
            bev_feat = self.pre_process(bev_feat)[0]
        return_map['bev_feat_before_encoder']=bev_feat

        feat_for_pred_bias=None
        if self.scene_his_fusion and self.scene_his_before_vox_his:
                
                start_of_sequence = torch.BoolTensor([
                        single_img_metas['start_of_sequence'] 
                        for single_img_metas in img_metas]).to(img[0][0].device)
                bev_shape=bev_feat.shape[2:]
                bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1],-1)
                bev_feat=self.scene_his_func_before_vox_his(bev_feat,start_of_sequence)
                
                bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1],*bev_shape)
        if not self.not_use_history:
            if self.use_vox_his_func:
                bev_feat = self.fuse_history_func.forward_fuse_his(bev_feat, img_metas, img[6])['fused_bev_feat']
            else:
                bev_feat = self.fuse_history(bev_feat, img_metas, img[6],feat_for_pred_bias=feat_for_pred_bias)['fused_bev_feat']
        if self.scene_his_fusion and self.scene_his_after_vox_his:
            start_of_sequence = torch.BoolTensor([
                    single_img_metas['start_of_sequence'] 
                    for single_img_metas in img_metas]).to(img[0][0].device)
            bev_shape=bev_feat.shape[2:]
            bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1],-1)

            bev_feat=self.scene_his_func_after_vox_his(bev_feat,start_of_sequence)
            
            bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1],*bev_shape)
        
        bev_feat = self.bev_encoder(bev_feat)
        if isinstance(bev_feat,dict):
            bev_feat=bev_feat['x']
       
        if self.occ_backbone_2d :
            if self.with_cp and bev_feat[0].requires_grad:
                bev_feat = checkpoint(self.final_conv, bev_feat[0])
            else:
                bev_feat = self.final_conv(bev_feat[0])

            bev_feat=bev_feat.reshape(bev_feat.shape[0],bev_feat.shape[1]//self.dz,self.dz,*bev_feat.shape[2:])
            bev_feat=bev_feat.permute(0,1,3,4,2)
            bev_feat=[bev_feat]
        return_map['img_bev_feat'] = bev_feat
       
        return return_map

    def extract_lidar_bev_feat(self, pts, img_feats, img_metas):
        """Extract features of points."""

        voxels, num_points, coors = self.voxelize(pts)

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        bev_feat = self.pts_middle_encoder(voxel_features, coors, batch_size)
        bev_feat = self.bev_encoder(bev_feat)
        if isinstance(bev_feat,dict):
            bev_feat=bev_feat['x']
        return bev_feat


    def extract_feat(self, points, img, img_metas, **kwargs):
        """Extract features from images and points."""

        results={}
        if img is not None and self.with_specific_component('img_backbone'):
   
                    
            results.update(self.extract_img_bev_feat(img, img_metas, **kwargs))
        if points is not None and self.with_specific_component('pts_voxel_encoder'):
            results['lidar_bev_feat'] = self.extract_lidar_bev_feat(points, img, img_metas)
        

        return results


    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      gt_occupancy_flow=None,
                      **kwargs):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        
        results= self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        losses = dict()

        if  self.with_pts_bbox:
            losses_pts = self.forward_pts_train(results['img_bev_feat'], gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore)
            losses.update(losses_pts)

        if self.with_specific_component('alocc_head'):
            
            class_prototype=None
            gt_occ_label=kwargs['gt_occupancy']
            losses_occupancy,_,all_mask_preds = self.alocc_head.forward_train(results['img_bev_feat'], img_metas=img_metas, gt_labels=gt_occ_label, 
                                                                    class_prototype=class_prototype,bda=img_inputs[6],**kwargs)

            losses.update(losses_occupancy)

            if self.sem_sup_prototype:
                gt_depth,gt_imgseg=self.depth_net.get_downsampled_gt_depth_semantics(kwargs['gt_depth'],kwargs['gt_semantic_map'])
                results['gt_imgseg']=gt_imgseg

                loss_img_seg,all_cls_scores,all_mask_preds=self.alocc_head.forward_train([results['context'].permute(0,2,1,3,4)], img_metas=img_metas, gt_labels=gt_imgseg,mask_embede2=self.use_mask_net2,class_prototype=None,tag='img')

                for loss in loss_img_seg:
                    losses.update({'img_seg_'+loss:loss_img_seg[loss]* self.img_seg_weight})

        if self.with_specific_component('occupancy_head'):
            losses_occupancy ,occ_pred= self.occupancy_head.forward_train(results['img_bev_feat'], results=results,**kwargs)

            losses.update(losses_occupancy)
        
        if self.with_specific_component('frpn'):
            losses_mask = self.frpn.get_bev_mask_loss(kwargs['gt_bev_mask'], results['bev_mask_logit'])
            losses.update(losses_mask)
        
        if self.use_depth_supervision and self.with_specific_component('depth_net') :
            if self.depth_loss_ce:
                loss_depth = self.depth_net.get_depth_loss_(kwargs['gt_depth'], results['depth'])
            else:
                loss_depth = self.depth_net.get_depth_loss(kwargs['gt_depth'], results['depth'])
            losses.update(loss_depth)
        
        return losses
    
    
    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points=[None]
        if self.dataset=='kitti':
            img_inputs=[img_inputs]
            img_metas=[img_metas]
            points=[points]
        self.do_history = True
        if img_inputs is not None:
            for var, name in [(img_inputs, 'img_inputs'),
                          (img_metas, 'img_metas')]:
                if not isinstance(var, list) :
                    raise TypeError('{} must be a list, but got {}'.format(
                        name, type(var)))        
            num_augs = len(img_inputs)
            
            if num_augs != len(img_metas):
                raise ValueError(
                    'num of augmentations ({}) != num of image meta ({})'.format(
                        len(img_inputs), len(img_metas)))
            
            
            if isinstance(img_metas[0],mmcv.parallel.DataContainer):
                if num_augs==1 and not img_metas[0].data[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                    return self.simple_test(points[0], img_metas[0].data[0][0], img_inputs[0],
                                        **kwargs)
                else:
                    return self.aug_test(points, img_metas, img_inputs, **kwargs)
            else:
                if num_augs==1 and not img_metas[0][0].get('tta_config', dict(dist_tta=False))['dist_tta']:
                    return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                        **kwargs)
                else:
                    return self.aug_test(points, img_metas, img_inputs, **kwargs)
        
        elif points is not None:
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            points = [points] if points is None else points
            return self.simple_test(points[0], img_metas[0], img_inputs[0],
                                    **kwargs)
    def aug_test(self,points,
                    img_metas,
                    img_inputs=None,
                    visible_mask=[None],
                    return_raw_occ=False,
                    **kwargs):
        """Test function without augmentaiton."""
        if 'semantic_anything_map' in kwargs:
            
            kwargs['semantic_anything_map']=[torch.cat(kwargs['semantic_anything_map'],dim=0)]
        if 'depth_anything_map' in kwargs:
            kwargs['depth_anything_map']=[torch.cat(kwargs['depth_anything_map'],dim=0)]
        img_inputs=[torch.cat([img_inputs[i][j] for i in range(len(img_inputs))],dim=0) for j in range(len(img_inputs[0]))]
        
        img=img_inputs
        kwargs['aux_cam_params']=[[torch.cat([kwargs['aux_cam_params'][i][j] for i in range(len(kwargs['aux_cam_params']))],dim=0) for j in range(len(kwargs['aux_cam_params'][0]))]]
        
        kwargs['adj_aux_cam_params']=[[torch.cat([kwargs['adj_aux_cam_params'][i][j] for i in range(len(kwargs['adj_aux_cam_params']))],dim=0) for j in range(len(kwargs['adj_aux_cam_params'][0]))]]
        
        img_metas=[img_metas[i][0] for i in range(len(img_metas))]
        results = self.extract_feat(points, img=img_inputs, img_metas=img_metas, **kwargs)
     
        pred_occupancy_category=None
        pred_density=None
        pred_flow=None
        bbox_list = [dict() for _ in range(len(img_metas))]
        if  self.with_pts_bbox:
            bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        else:
            bbox_pts = [None for _ in range(len(img_metas))]
        
        if self.with_specific_component('alocc_head'):
            class_prototype=None
            
            output_test = self.alocc_head.simple_test(results['img_bev_feat'], img_metas=img_metas,class_prototype=class_prototype, bda=img[6],**kwargs)
            pred_occupancy = output_test['output_voxels'][0]
            for i in range(len(img_metas)):
                tta_config = kwargs['tta_config'][i]
                flip_dx=tta_config['flip_dx']
                flip_dy=tta_config['flip_dy']
                pred_occupancy_i = pred_occupancy[i]
                if flip_dx:
                    pred_occupancy_i = torch.flip(pred_occupancy_i, [2])
    
                if flip_dy:
                    pred_occupancy_i = torch.flip(pred_occupancy_i, [1])
                pred_occupancy[i] = pred_occupancy_i
            pred_occupancy = pred_occupancy.mean(0,keepdim=True)
            
            if self.pred_flow:
                
                pred_flow = output_test['output_flow'][0]
                for i in range(len(img_metas)):
                    tta_config = kwargs['tta_config'][i]
                    flip_dx=tta_config['flip_dx']
                    flip_dy=tta_config['flip_dy']
                    pred_flow_i = pred_flow[i]
                    
                    if flip_dx:
                        pred_flow_i = torch.flip(pred_flow_i, [1])
                        pred_flow_i[...,1]=-pred_flow_i[...,1]
        
                    if flip_dy:
                        pred_flow_i = torch.flip(pred_flow_i, [0])
                        pred_flow_i[...,0]=-pred_flow_i[...,0]
                    pred_flow[i] = pred_flow_i
                pred_flow = pred_flow.mean(0,keepdim=True)
                pred_flow = pred_flow[0]
                # pred_flow = pred_flow.permute(0, 2, 3, 4, 1)[0]
                pred_flow=pred_flow.permute(3, 2, 0, 1)
                pred_flow = torch.flip(pred_flow, [2])
                pred_flow=pred_flow[[1,0]]
                pred_flow = torch.rot90(pred_flow, -1, [2, 3])
                pred_flow = pred_flow.permute(2, 3, 1, 0)

            if not self.wo_pred_occ:
                #################
                pred_density=output_test['output_density'][0]
                for i in range(len(img_metas)):
                    tta_config = kwargs['tta_config'][i]
                    flip_dx=tta_config['flip_dx']
                    flip_dy=tta_config['flip_dy']
                    pred_density_i = pred_density[i]
                    if flip_dx:
                        pred_density_i = torch.flip(pred_density_i, [1])
                    
                    if flip_dy:
                        
                        pred_density_i = torch.flip(pred_density_i, [0])
                    pred_density[i] = pred_density_i
                pred_density = pred_density.mean(0,keepdim=True)
                
                pred_density=pred_density.unsqueeze(1)
                pred_density=pred_density.permute(0, 2, 3, 4, 1)[0]
                pred_density = pred_density.permute(3, 2, 0, 1)
                pred_density = torch.flip(pred_density, [2])
                pred_density = torch.rot90(pred_density, -1, [2, 3])
                pred_density = pred_density.permute(2, 3, 1, 0).squeeze(-1)

                ############
                if self.dataset=='kitti':
                    
                    output={}
                    output['output_voxels'] = pred_occupancy
                    output['target_voxels'] = kwargs['gt_occupancy']
                    return output
                pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
                
                if self.fix_void:
                    pred_occupancy = pred_occupancy[..., 1:]     
                    

                # convert to CVPR2023 Format
                pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
                pred_occupancy = torch.flip(pred_occupancy, [2])
                pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
                pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
                
                if 'vis_class' in kwargs:
                    mask=kwargs['gt_occupancy'][0]==kwargs['vis_class']
                    # mask=mask*(kwargs['gt_occupancy_ori'][0]!=255)
                    pred_occupancy=pred_occupancy[mask[0]].sum(0)
                    return pred_occupancy
                if return_raw_occ:
                    pred_occupancy_category = pred_occupancy
                    return pred_occupancy_category
                else:
                    pred_occupancy_category = pred_occupancy.argmax(-1) 
                    # pred_occupancy_category = pred_occupancy[...,:-1].argmax(-1) 
                    # pred_density=(pred_occupancy.argmax(-1)!=16).float()
                pred_occupancy_category = pred_occupancy_category.cpu().numpy().astype(np.uint8)

            else:
                pred_occupancy_category =  None
                

        else:
            pred_occupancy_category =  None

        if results.get('bev_mask_logit', None) is not None:
            pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5
            iou = IOU(pred_bev_mask.reshape(1, -1), kwargs['gt_bev_mask'][0].reshape(1, -1)).cpu().numpy()
        else:
            iou = None
        
        if not self.pred_flow:
            pred_flow=None
        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = iou
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
            result_dict['pred_flow'] = pred_flow.half().cpu().numpy() if pred_flow is not None else None
            # result_dict['pred_density'] = pred_density.cpu().numpy() if pred_density is not None else None

            result_dict['pred_density'] =  None

        return bbox_list

    def simple_test(self,
                    points,
                    img_metas,
                    img=None,
                    rescale=False,
                    visible_mask=[None],
                    return_raw_occ=False,
                    
                    **kwargs):
        """Test function without augmentaiton."""
        results = self.extract_feat(
            points, img=img, img_metas=img_metas, **kwargs)
        

        bbox_list = [dict() for _ in range(len(img_metas))]
        
        if  self.with_pts_bbox:
            
            bbox_pts = self.simple_test_pts(results['img_bev_feat'], img_metas, rescale=rescale)
        else:
            bbox_pts = [None for _ in range(len(img_metas))]
        pred_occupancy_category=None
        pred_density=None
        pred_flow=None
        
        if self.with_specific_component('occupancy_head'):
            if self.occupancy_head.final_two_part_loss:
                
                if self.occupancy_head.final_two_part_loss:
                    output_test=self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)
                    pred_occupancy = output_test['output_voxels'][0]
                else:
                    pred_occupancy=results['inter_occs'][0]
                pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]

                if self.fix_void:
                    pred_occupancy = pred_occupancy[..., 1:]     

                # convert to CVPR2023 Format
                pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
                pred_occupancy = torch.flip(pred_occupancy, [2])
                pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
                pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)


                pred_sem = pred_occupancy[..., :-1]
                pred_occ = pred_occupancy[..., -1:].sigmoid()
                pred_sem_category = pred_sem.argmax(-1)
                pred_free_category = (pred_occ<0.5).squeeze(-1)

                pred_sem_category[pred_free_category] = 17
                pred_occupancy_category = pred_sem_category
            else:
                output_test = self.occupancy_head(results['img_bev_feat'], results=results, **kwargs)
                pred_occupancy = output_test['output_voxels'][0]
                if self.pred_flow:
                    
                    pred_flow = output_test['output_flow'][0]
                    pred_flow = pred_flow.permute(0, 2, 3, 4, 1)[0]
                    # pred_flow = pred_flow[0]
                    pred_flow=pred_flow.permute(3, 2, 0, 1)
                    pred_flow = torch.flip(pred_flow, [2])
                    pred_flow=pred_flow[[1,0]]
                    pred_flow = torch.rot90(pred_flow, -1, [2, 3])
                    pred_flow = pred_flow.permute(2, 3, 1, 0)
                if not self.wo_pred_occ:
                
                    if self.dataset=='kitti':
                        output={}
                        output['output_voxels'] = pred_occupancy
                        output['target_voxels'] = kwargs['gt_occupancy']
                        return output
                    
                    pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
                
                    if self.fix_void:
                        pred_occupancy = pred_occupancy[..., 1:] 
                        ############################   
                   
                    ##############################
                    pred_occupancy = pred_occupancy.softmax(-1)
                    ####################
                    # convert to CVPR2023 Format
                    pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
                    pred_occupancy = torch.flip(pred_occupancy, [2])
                    pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
                    pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
                    
                    if 'vis_class' in kwargs:
                        
                        pred_occupancy=pred_occupancy.mean((0,1,2))

                        return pred_occupancy
                    if return_raw_occ:
                        pred_occupancy_category = pred_occupancy
                        
                        return pred_occupancy_category
                    else:
                        if self.gts_surroundocc:
                            pred_occupancy_category = pred_occupancy[...,1:].argmax(-1)+1
                        else:
                            pred_occupancy_category = pred_occupancy.argmax(-1) 
                
                    pred_occupancy_category= pred_occupancy_category.cpu().numpy().astype(np.uint8)



                else:
                    pred_occupancy_category =  None

        elif self.with_specific_component('alocc_head'):
            class_prototype=None
            
            output_test = self.alocc_head.simple_test(results['img_bev_feat'], img_metas=img_metas,class_prototype=class_prototype, bda=img[6],**kwargs)
            pred_occupancy = output_test['output_voxels'][0]
            if self.pred_flow:
                
                pred_flow = output_test['output_flow'][0]
                pred_flow = pred_flow[0]
                pred_flow=pred_flow.permute(3, 2, 0, 1)
                pred_flow = torch.flip(pred_flow, [2])

                pred_flow = torch.rot90(pred_flow, -1, [2, 3])

                pred_flow=pred_flow[[1,0]]
                pred_flow = pred_flow.permute(2, 3, 1, 0)
            
            if not self.wo_pred_occ:
                #
                if self.dataset=='kitti':
                    
                    output={}
                    output['output_voxels'] = pred_occupancy
                    output['target_voxels'] = kwargs['gt_occupancy']
                    return output
                pred_occupancy = pred_occupancy.permute(0, 2, 3, 4, 1)[0]
                
                if self.fix_void:
                    pred_occupancy = pred_occupancy[..., 1:]     
                    

                # convert to CVPR2023 Format
                pred_occupancy = pred_occupancy.permute(3, 2, 0, 1)
                pred_occupancy = torch.flip(pred_occupancy, [2])
                pred_occupancy = torch.rot90(pred_occupancy, -1, [2, 3])
                pred_occupancy = pred_occupancy.permute(2, 3, 1, 0)
             
                if 'vis_class' in kwargs:
                    pred_occupancy=pred_occupancy.softmax(-1)
                else:
                    pred_occupancy=pred_occupancy.sigmoid()
                if 'vis_class' in kwargs:
                    pred_occupancy=pred_occupancy.mean((0,1,2))

                    return pred_occupancy
                if return_raw_occ:
                    pred_occupancy_category = pred_occupancy
                    return pred_occupancy_category
                else:
                    if self.gts_surroundocc:
                        pred_occupancy_category = pred_occupancy[...,1:].argmax(-1)+1
                    else:
                        pred_occupancy_category = pred_occupancy.argmax(-1) 
              
                pred_occupancy_category = pred_occupancy_category.cpu().numpy().astype(np.uint8)
            else:
                pred_occupancy_category =  None
        else:
            pred_occupancy_category =  None
            
        if results.get('bev_mask_logit', None) is not None:
            pred_bev_mask = results['bev_mask_logit'].sigmoid() > 0.5
            iou = IOU(pred_bev_mask.reshape(1, -1), kwargs['gt_bev_mask'][0].reshape(1, -1)).cpu().numpy()
        else:
            iou = None
        assert len(img_metas) == 1

        if self.occupancy_save_path is not None:
            scene_name = img_metas[0]['scene_name']
            sample_token = img_metas[0]['sample_idx']
            save_dir=os.path.join(self.occupancy_save_path, 'occupancy_pred',scene_name)
            if not os.path.exists(save_dir):
                mmcv.mkdir_or_exist(save_dir)
            save_path = os.path.join(save_dir, f'{sample_token}.npz')
            np.savez_compressed(save_path,pred_occupancy_category)
      
        
        if self.cal_metric_in_model:
            
            occupancy_gt = kwargs['gt_occupancy'][0][0].permute(2, 0, 1)
            occupancy_gt = torch.flip(occupancy_gt, [1])
            occupancy_gt = torch.rot90(occupancy_gt, -1, [1, 2])
            
            occupancy_gt = occupancy_gt.permute(1, 2, 0)
            pred_occupancy_category=torch.from_numpy(pred_occupancy_category).to(occupancy_gt)
            mask = (occupancy_gt != 255)
            if self.fix_void:
                num_cls=self.num_cls-1
                occupancy_gt-=1
            else:
                num_cls=self.num_cls
            semantic_iou = torch.zeros((num_cls, 3),device=pred_occupancy_category.device)
            occ_iou = torch.zeros((1, 3),device=pred_occupancy_category.device)
            for j in range(num_cls):
                
                semantic_iou[j][0] += ((occupancy_gt[mask] == j) * (pred_occupancy_category[mask] == j)).sum()
                semantic_iou[j][1] += (occupancy_gt[mask] == j).sum()
                semantic_iou[j][2] += (pred_occupancy_category[mask] == j).sum()    
                
            semantic_iou=semantic_iou.cpu().numpy()   
            
            for j in range(1):
                occ_iou[j][0] += ((occupancy_gt[mask] !=num_cls-1) * (pred_occupancy_category[mask] !=num_cls-1)).sum()
                occ_iou[j][1] += (occupancy_gt[mask] !=num_cls-1).sum()
                occ_iou[j][2] += (pred_occupancy_category[mask] !=num_cls-1).sum()   
            occ_iou=occ_iou.cpu().numpy()   
            pred_occupancy_category=1
                
        for i, result_dict in enumerate(bbox_list):
            result_dict['pts_bbox'] = bbox_pts[i]
            result_dict['iou'] = iou
            result_dict['pred_occupancy'] = pred_occupancy_category
            result_dict['index'] = img_metas[0]['index']
            result_dict['pred_flow'] = pred_flow.half().cpu().numpy() if pred_flow is not None else None
            # result_dict['pred_density'] = pred_density.cpu().numpy() if pred_density is not None else None

            result_dict['pred_density'] =  None
            
            if self.cal_metric_in_model:
                result_dict['semantic_iou'] =  semantic_iou
                result_dict['occ_iou'] =  occ_iou
                
        return bbox_list

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      **kwargs):
        results = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas, **kwargs)
        assert self.with_pts_bbox
        outs = self.pts_bbox_head(results['img_bev_feat'])
        return outs
