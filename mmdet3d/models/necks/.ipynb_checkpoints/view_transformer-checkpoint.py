# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint

from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
from mmdet.models.backbones.resnet import BasicBlock
from ..builder import NECKS
from .zero_out_between_ones import zero_out_between_ones
from .cal_depth2occ import cal_depth2occ,cal_kernal,cal_weight,get_expanded_feature_map,cal_depth2occ_set
import time
import math
from .deformable_lift import broadcast_pred_linear_interpolation as deformable_lift
from mmcv.cnn import xavier_init, constant_init
from ..losses import SigLoss
from .. import builder
from mmdet.models.utils import build_transformer
import copy
from mmdet.models.utils.transformer import inverse_sigmoid
def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res
def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb
@NECKS.register_module()
class LSSViewTransformer(BaseModule):
    r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
        `paper <https://arxiv.org/abs/2211.17111>`

    Args:
        grid_config (dict): Config of grid alone each axis in format of
            (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
        input_size (tuple(int)): Size of input images in format of (height,
            width).
        downsample (int): Down sample factor from the input size to the feature
            size.
        in_channels (int): Channels of input feature.
        out_channels (int): Channels of transformed feature.
        accelerate (bool): Whether the view transformation is conducted with
            acceleration. Note: the intrinsic and extrinsic of cameras should
            be constant when 'accelerate' is set true.
        sid (bool): Whether to use Spacing Increasing Discretization (SID)
            depth distribution as `STS: Surround-view Temporal Stereo for
            Multi-view 3D Detection`.
        collapse_z (bool): Whether to collapse in z direction.
    """

    def __init__(
        self,
        grid_config,
        input_size,
        downsample=16,
        in_channels=512,
        out_channels=64,
        accelerate=False,
        sid=False,
        collapse_z=True,
        ############
        bev_mean_pool=False,
        SL=False,
        SL_size=[20,250],
        num_classes=18,
        torch_sparse_coor=False,
        to_set_interval=1,
        coor_expand_in_vox=None,
        toSetV2=False,
        depth_sup_CE=False,
        bev_weighted_pool=False,
        SL_max_weight=20,
        toSetV3=False,
        cumprod=False,
        max_depth=False,
        toSetV4=False,
        toSetV5=False,
        k_sqrt=3,
        num_shapes=512,
        depth_sampling=9,
        depth_continue=False,
        depth_with_temp=False,
        disc2continue_depth=False,
        coor_reproject=False,
        depth_denoising=False,
        reproject_mix=False,
        depth_denoising_ratio=0.3,
        depth_detach=False,
        denoising_with_noise=None,
        denoising_with_mixed=None,
        deformable_lift=False,
        num_points=0,
        num_anchors=0,
        sampling_offset_in_img_coor=False,
        use_cross_attention=None,
        only_add_on_depth=False,
        adaptive_depth_bin=False,
        ada_bin_self_wight=False,
        ada_bin_tau=0.005,
        init_weights3=False,
        sup_adaptive_depth=False,
        init0=False,
        num_sampling_from_depth=False,
        num_points_adap2depth=False,
        num_points_adap2depth_gai=False,
        offsets_with_MLP=False,
        only_train_depth=False,
        adaptive_bin2_continue_depth=False,
        readd=True,
        sampling_offsets_weight=None,
        disc2continue_depth_continue_sup=False,
        lift_attn=None,
        lift_attn_fix_bin=False,
        lift_attn_round=1,
        lift_pre_process=None,
        ada2fix_bin=False,
        lift_attn_norm_add=False,
        global_learnable_bin=False,
        supervise_intermedia=False,
        occ2depth=False,
        occ2_depth_use_occ=False,
        lift_attn_not_detach_depth=False,
        num_points_lift_attn=None,
        lift_attn_simple_add=False,
        simple_not_add=False,
        use_gt_occ2depth=False,
        depth_gt_not_mix=False,
        lift_attn_with_ori_feat=False,
        lift_attn_new=False,
        attn_weight_sigmoid=False,
        attn_depth_mix=False,
        use_pred_depth=False,
        occ2_depth_use_depth_sup_occ=False,
        add_ffn_norm=False,
        zero2inf=False,
        attn_weight_temp=0,
        sup_attn_weight=False,
        simple_use_post_cov=False,
        lift_attn_with_complex_conv=False,
        lift_encoder=None,
        lift_encoder_neck=None,
        downsample_add=False,
        downsample_neck=None,
        add_occ_depth_loss=False,
        lift_attn_downsample=[1,1,1],
        attn_with_pos=False,
        splat_with_occ=False,
        attn_lift_post_conv=False,
        splat_with_occ_only=False,
        sampling_weights_from_vox=False,
        #history
        ######
        fuse_his_attn=False,
        do_history=True,
        interpolation_mode='bilinear',
        history_cat_num=16,
        history_cat_conv_out_channels=None,
        single_bev_num_channels=32,
        ###############
        vox_upsample_scale=[1,1,1],
        fuse_self=False,
        depth_weight_exp=False,
        fill_all_vox=False,
        inverse_grid_index=False,
        fill_all_vox_with_occ=False,
        fill_with_gt_occ=False,
        mask_around_img=False,
        lift_attn_with_ori_feat_add=False,
        fuse_self_round=1,
        first_phase_non_all_vox=False,
        lift_attn_pre_img_conv=False,
        img_upsample=False,
        renderencoder=False,
        downsample_render=[8],
    ):
        super(LSSViewTransformer, self).__init__()
        self.grid_config = grid_config
        self.downsample = downsample
        self.create_grid_infos(**grid_config)
        self.sid = sid
        ##################
        if not SL:
            self.frustum = self.create_frustum(grid_config['depth'],
                                            input_size, downsample)
        else:
            self.frustum = self.create_frustum_SL(grid_config['depth'],
                                            input_size, SL_size)
        self.input_size_=input_size
                                            #####################
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
        self.accelerate = accelerate
        self.initial_flag = True
        self.collapse_z = collapse_z
        
        ################
        self.bev_mean_pool=bev_mean_pool
        self.SL=SL
        self.SL_size=SL_size
        if SL:
            self.predicter = nn.Sequential(
                nn.Linear(self.out_channels, self.out_channels*2),
                nn.Softplus(),
                nn.Linear(self.out_channels*2, num_classes),
            )
        self.torch_sparse_coor=torch_sparse_coor
        self.to_set_interval=to_set_interval
        self.coor_expand_in_vox=coor_expand_in_vox
        self.toSetV2=toSetV2
        self.depth_sup_CE=depth_sup_CE
        self.bev_weighted_pool=bev_weighted_pool
        self.SL_max_weight=SL_max_weight
        self.toSetV3=toSetV3
        self.cumprod=cumprod
        self.max_depth=max_depth
        self.toSetV4=toSetV4
        self.toSetV5=toSetV5
        self.k_sqrt=k_sqrt
        self.k=k_sqrt**2
        if self.toSetV5:
            self.num_shapes=num_shapes
            # self.code_book=nn.Sequential(nn.Linear(self.out_channels,self.num_shapes*self.k*self.k_sqrt,bias=False))
            self.code_book=nn.Parameter(torch.randn(self.num_shapes,self.k*self.k_sqrt,self.out_channels))
            # nn.init.kaiming_uniform_(self.code_book.weight, a=math.sqrt(5))
        self.depth_sampling=depth_sampling
        self.depth_continue=depth_continue
        self.depth_with_temp=depth_with_temp
        self.disc2continue_depth=disc2continue_depth
        self.coor_reproject=coor_reproject
        self.depth_denoising=depth_denoising
        self.reproject_mix=reproject_mix
        self.depth_denoising_ratio=depth_denoising_ratio
        self.depth_detach=depth_detach
        self.denoising_with_noise=denoising_with_noise
        self.denoising_with_mixed=denoising_with_mixed
        self.deformable_lift=deformable_lift
        self.num_points=num_points
        self.num_anchors=num_anchors
        self.only_add_on_depth=only_add_on_depth
        self.init_weights3=init_weights3
        self.init0=init0
        self.num_points_adap2depth=num_points_adap2depth
        self.num_points_adap2depth_gai=num_points_adap2depth_gai
        self.adaptive_depth_bin=adaptive_depth_bin
        self.ada_bin_self_wight=ada_bin_self_wight
        if self.ada_bin_self_wight:
            self.ada_bin_tau=ada_bin_tau
        self.offsets_with_MLP=offsets_with_MLP
        if self.num_points:
            if not num_sampling_from_depth:
                self.attention_weights = nn.Linear(out_channels, num_points)
                if self.only_add_on_depth:
                    self.sampling_offsets = nn.Linear(out_channels,  num_points)
                    
                elif self.num_points_adap2depth:
                    if self.disc2continue_depth:
                        self.sampling_offsets = nn.Linear(out_channels,  num_points*3)
                    elif self.adaptive_depth_bin:
                        self.sampling_offsets = nn.Linear(out_channels,  num_points*self.adaptive_depth_bin*3)
                        self.num_points*=self.adaptive_depth_bin
                        self.attention_weights = nn.Linear(out_channels, num_points*self.adaptive_depth_bin)
                    else:
                        self.sampling_offsets = nn.Linear(out_channels,  num_points*self.D*3)
                        self.num_points*=self.D
                        self.attention_weights = nn.Linear(out_channels, num_points*self.D)
                    self.sampling_offsets,self.attention_weights=self.init_weights_(self.sampling_offsets,self.attention_weights,self.num_points)
                elif self.offsets_with_MLP:
                    self.sampling_offsets =nn.Sequential(nn.Linear(out_channels,  out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels,  num_points*3))
                    self.attention_weights =nn.Sequential(nn.Linear(out_channels,  out_channels),
                    nn.ReLU(),
                    nn.Linear(out_channels,  num_points))
                    
                else:
                    
                    self.sampling_offsets = nn.Linear(out_channels,  num_points*3)
                if self.only_add_on_depth:
                    self.sampling_offsets,self.attention_weights=self.init_weights2_(self.sampling_offsets,self.attention_weights,self.num_points)
                elif self.init_weights3:
                    self.sampling_offsets,self.attention_weights=self.init_weights3_(self.sampling_offsets,self.attention_weights,self.num_points)
                elif self.init0:
                    self.sampling_offsets,self.attention_weights=self.init_weights0_(self.sampling_offsets,self.attention_weights,self.num_points)
                elif self.init_weights_anchors:
                    self.sampling_offsets,self.attention_weights=self.init_weights_anchors(self.sampling_offsets,self.attention_weights,self.num_points,self.num_anchors)
                else:
                    self.sampling_offsets,self.attention_weights=self.init_weights_(self.sampling_offsets,self.attention_weights,self.num_points)
                self.sampling_offsets_nets=nn.ModuleList()
                self.sampling_offsets_nets.append(self.sampling_offsets)
                self.attention_weights_nets=nn.ModuleList()
                self.attention_weights_nets.append(self.attention_weights)
        self.sampling_offset_in_img_coor=sampling_offset_in_img_coor
        self.use_cross_attention=use_cross_attention
        self.num_coor=3 if not self.only_add_on_depth else 1
        
        self.sup_adaptive_depth=sup_adaptive_depth
        # if self.sup_adaptive_depth:
        self.depth_loss=SigLoss()
        self.num_sampling_from_depth=num_sampling_from_depth
        self.only_train_depth=only_train_depth
        self.adaptive_bin2_continue_depth=adaptive_bin2_continue_depth
        if self.use_cross_attention is not None:
            self.backward_projection = builder.build_head(self.use_cross_attention) 
            self.readd=readd
        if sampling_offsets_weight is not None:
            self.sampling_offsets_weight=torch.Tensor(sampling_offsets_weight)
        else:
            self.sampling_offsets_weight=None
        self.disc2continue_depth_continue_sup=disc2continue_depth_continue_sup
        self.ada2fix_bin=ada2fix_bin
        self.lift_attn=lift_attn
        self.lift_attn_round=lift_attn_round
        self.num_points_lift_attn=num_points_lift_attn
      
        self.lift_attn_new=lift_attn_new
        if self.lift_attn is not None:
            # lift_attn=build_transformer(self.lift_attn)
            self.lift_attn=nn.ModuleList([build_transformer(self.lift_attn) for _ in range(self.lift_attn_round)])
            if self.num_points_lift_attn is not None:
                if not num_sampling_from_depth:
                    self.sampling_offsets_nets=nn.ModuleList()
                    self.attention_weights_nets=nn.ModuleList()
                    
                    for i in range(len(self.num_points_lift_attn)):
                        sampling_offsets,attention_weights,num_points=self.lift_attn_gene_num_sampling_nets(self.num_points_lift_attn[i],num_anchors[i])
                        self.sampling_offsets_nets.append(sampling_offsets)
                        self.attention_weights_nets.append(attention_weights)
                        self.num_points_lift_attn[i]=num_points
                        
                    
        self.lift_attn_fix_bin=lift_attn_fix_bin
        
        self.lift_pre_process = lift_pre_process is not None
        if self.lift_pre_process:
            # lift_pre_process_net = builder.build_backbone(lift_pre_process)
            self.lift_pre_process_nets = nn.ModuleList(
                [builder.build_backbone(lift_pre_process) for _ in range(self.lift_attn_round)])
        self.lift_attn_norm_add=lift_attn_norm_add    
        self.global_learnable_bin=global_learnable_bin
        self.supervise_intermedia=supervise_intermedia
        if self.supervise_intermedia and self.lift_attn is None:
            self.inter_predictor=nn.Sequential(
                nn.Linear(self.out_channels, self.out_channels*2),
                nn.Softplus(),
                nn.Linear(self.out_channels*2, num_classes),
            )
        self.occ2_depth_use_occ=occ2_depth_use_occ
        self.occ2depth=occ2depth
        self.lift_attn_not_detach_depth=lift_attn_not_detach_depth
        self.lift_attn_simple_add=lift_attn_simple_add
        if self.lift_attn_simple_add:
            self.inter_predictor=nn.Sequential(
                nn.Linear(self.out_channels, self.out_channels*2),
                nn.Softplus(),
                nn.Linear(self.out_channels*2, num_classes),
            )
            self.lift_attn=nn.Identity()
            self.simple_not_add=simple_not_add
        self.use_gt_occ2depth=use_gt_occ2depth
        self.depth_gt_not_mix=depth_gt_not_mix
        self.lift_attn_with_ori_feat=lift_attn_with_ori_feat
        self.lift_attn_pre_img_conv=lift_attn_pre_img_conv
        if self.lift_attn_new:
            if not splat_with_occ_only:
                self.Q_net=nn.ModuleList(nn.Linear(self.out_channels,self.out_channels,bias=False) for _ in range(self.lift_attn_round))
                self.K_net=nn.ModuleList(nn.Linear(self.out_channels,self.out_channels,bias=False) for _ in range(self.lift_attn_round))
                self.V_net=nn.ModuleList(nn.Linear(self.out_channels,self.out_channels,bias=False) for _ in range(self.lift_attn_round))
            if self.lift_attn_pre_img_conv:
                self.pre_img_conv=nn.ModuleList(nn.Sequential(
                    nn.Conv2d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                    nn.ReLU(inplace=True),
                    ) for _ in range(self.lift_attn_round+1))
            if self.supervise_intermedia:
                self.inter_predictor=nn.ModuleList(
                nn.Sequential(
                nn.Linear(self.out_channels, self.out_channels*2),
                nn.Softplus(),
                nn.Linear(self.out_channels*2, num_classes),
            ) for _ in range(self.lift_attn_round))
            self.attn_weight_sigmoid=attn_weight_sigmoid
            self.lift_attn=nn.Identity()
            self.attn_depth_mix=attn_depth_mix
        self.use_pred_depth=use_pred_depth
        self.occ2_depth_use_depth_sup_occ=occ2_depth_use_depth_sup_occ
        self.add_ffn_norm=add_ffn_norm
        if self.add_ffn_norm:
            self.pre_norm=nn.ModuleList(nn.LayerNorm(self.out_channels) for _ in range(self.lift_attn_round))
            self.pre_ffn=nn.ModuleList(nn.Linear(self.out_channels,self.out_channels) for _ in range(self.lift_attn_round))
            self.post_norm=nn.ModuleList(nn.LayerNorm(self.out_channels) for _ in range(self.lift_attn_round))
            self.post_ffn=nn.ModuleList(nn.Sequential(nn.Linear(self.out_channels,self.out_channels*2),
                                                     nn.ReLU(inplace=True),
                                                     nn.Linear(self.out_channels*2,self.out_channels)) for _ in range(self.lift_attn_round))
        self.zero2inf=zero2inf   
        self.attn_weight_temp=attn_weight_temp 
        self.sup_attn_weight=sup_attn_weight
        self.simple_use_post_cov=simple_use_post_cov
        self.lift_attn_with_complex_conv=lift_attn_with_complex_conv
        if self.lift_attn_with_complex_conv:
            self.complex_conv=nn.ModuleList(nn.Sequential(
                builder.build_backbone(lift_encoder),
                builder.build_backbone(lift_encoder_neck)) for _ in range(self.lift_attn_round))
        self.downsample_add=downsample_add
        if self.downsample_add :
            # self.downsample_net=builder.build_backbone(self.downsample_add)
            self.downsample_net=nn.Sequential(nn.Conv2d(self.out_channels,self.out_channels*2,3,stride=2,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.out_channels*2,self.out_channels*2,3,stride=1,padding=1),
                                              nn.ReLU(inplace=True),
                                              nn.Conv2d(self.out_channels*2,self.out_channels*8,3,stride=2,padding=1),
                                                nn.ReLU(inplace=True),
                                                nn.Conv2d(self.out_channels*8,self.out_channels*8,3,stride=1,padding=1),
                                                # nn.ReLU(inplace=True),
                                              )
            self.upsample_net=nn.Sequential(nn.ConvTranspose2d(self.out_channels*8,self.out_channels*2,3,stride=2,padding=1,output_padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.out_channels*2,self.out_channels*2,3,stride=1,padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.ConvTranspose2d(self.out_channels*2,self.out_channels,3,stride=2,padding=1,output_padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                                            # nn.ReLU(inplace=True),
                                            )
            self.downsample_neck=builder.build_backbone(downsample_neck)
        self.add_occ_depth_loss=add_occ_depth_loss
        self.lift_attn_downsample=torch.Tensor(lift_attn_downsample)
        if torch.any(self.lift_attn_downsample!=torch.Tensor([1,1,1])):
            self.upsample_net=nn.ModuleList(nn.Sequential(nn.ConvTranspose2d(self.out_channels*8,self.out_channels*2,3,stride=2,padding=1,output_padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.out_channels*2,self.out_channels*2,3,stride=1,padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.ConvTranspose2d(self.out_channels*2,self.out_channels,3,stride=2,padding=1,output_padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                                            # nn.ReLU(inplace=True),
                                            ) for _ in range(self.lift_attn_round))
            self.downsample_ffn=nn.ModuleList(nn.Sequential(nn.Linear(self.out_channels,self.out_channels*8),
                                                            nn.ReLU(inplace=True),
                                                            nn.Linear(self.out_channels*8,self.out_channels*8),
                                                            )
                                                            for _ in range(self.lift_attn_round))
            self.downsample_neck=nn.ModuleList(builder.build_backbone(downsample_neck) for _ in range(self.lift_attn_round))
        self.attn_with_pos=attn_with_pos
        if self.attn_with_pos:
            self.pos_embedding = nn.Sequential(
                nn.Linear(self.out_channels*3//2, self.out_channels),
                nn.ReLU(),
                nn.Linear(self.out_channels, self.out_channels),
            )
        self.splat_with_occ=splat_with_occ
        self.attn_lift_post_conv=attn_lift_post_conv
        if self.attn_lift_post_conv:
            self.attn_lift_post_net=nn.ModuleList(nn.Linear(self.out_channels*2,self.out_channels,bias=False) for _ in range(self.lift_attn_round))
        self.splat_with_occ_only=splat_with_occ_only
        self.sampling_weights_from_vox=sampling_weights_from_vox
        if self.sampling_weights_from_vox:
            self.attention_weights_nets=None
        ############
        # Deal with history
        self.fuse_his_attn=fuse_his_attn
        if self.fuse_his_attn:
            self.single_bev_num_channels = single_bev_num_channels

            self.do_history = do_history
            self.interpolation_mode = interpolation_mode
            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)
            ## Embed each sample with its relative temporal offset with current timestep
            # if self.forward_projection:
            #     conv = nn.Conv2d if self.forward_projection.nx[-1] == 1 else nn.Conv3d
            # else:
            conv=nn.Conv3d
            self.history_keyframe_time_conv = nn.Sequential(
                conv(self.single_bev_num_channels + 1,
                        self.single_bev_num_channels,
                        kernel_size=1,
                        padding=0,
                        stride=1),
                nn.SyncBatchNorm(self.single_bev_num_channels),
                nn.ReLU(inplace=True))
            ## Then concatenate and send them through an MLP.
            self.history_keyframe_cat_conv = nn.Sequential(
                conv(self.single_bev_num_channels * (self.history_cat_num + 1),
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
            
        ##############
        self.vox_upsample_scale=torch.Tensor(vox_upsample_scale)
        if (self.vox_upsample_scale!=torch.Tensor([1,1,1])).any():
            self.vox_upsample1=nn.Sequential(nn.ConvTranspose2d(self.out_channels,self.out_channels,3,stride=2,padding=1,output_padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv2d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                                            nn.ReLU(inplace=True),
                                            )
            
            self.vox_upsample2=nn.Sequential(nn.ConvTranspose3d(self.out_channels,self.out_channels,3,stride=2,padding=1,output_padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                                            nn.ReLU(inplace=True),
                                            )
        self.fuse_self=fuse_self
        if self.fuse_self:
            self.fuse_self_net=nn.Sequential(nn.Conv3d(self.out_channels*2,self.out_channels,3,stride=1,padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                                            nn.ReLU(inplace=True),
                                            )
        self.depth_weight_exp=depth_weight_exp
        self.fill_all_vox=fill_all_vox
        self.inverse_grid_index=inverse_grid_index
        self.grid_map=self.gen_grid_map(grid_size=self.grid_size//self.lift_attn_downsample)
        self.fill_all_vox_with_occ=fill_all_vox_with_occ
        self.fill_with_gt_occ=fill_with_gt_occ
        self.mask_around_img=mask_around_img
        self.lift_attn_with_ori_feat_add=lift_attn_with_ori_feat_add
        if self.lift_attn_with_ori_feat_add:
            self.fuse_self_round=fuse_self_round
            self.lift_attn=nn.ModuleList([build_transformer(lift_attn) for _ in range(self.fuse_self_round)])
            self.fuse_self_net=nn.ModuleList([nn.Sequential(nn.Conv3d(self.out_channels*2,self.out_channels,3,stride=1,padding=1),
                                            nn.ReLU(inplace=True),
                                            nn.Conv3d(self.out_channels,self.out_channels,3,stride=1,padding=1),
                                            nn.ReLU(inplace=True),
                                            ) for _ in range(self.fuse_self_round)])
            self.lift_pre_process_nets = nn.ModuleList(
                [builder.build_backbone(lift_pre_process) for _ in range(self.fuse_self_round)])
        self.first_phase_non_all_vox=first_phase_non_all_vox
        if renderencoder:
            depth_config=[[grid_config['depth'][0],grid_config['depth'][1],grid_config['depth'][2]*downsample_render[i]] for i in range(len(downsample_render))]
            self.render_frustum=[self.create_frustum(depth_config[i], input_size, downsample_render[i]*8) for i in range(len(downsample_render))]

            self.render_grid_map=[self.gen_grid_map(grid_size=self.grid_size//downsample_render[i]) for i in range(len(downsample_render))]
            self.render_grid_size=[self.grid_size//downsample_render[i] for i in range(len(downsample_render))]
            self.render_grid_interval=[self.grid_interval*downsample_render[i] for i in range(len(downsample_render))]
            self.render_grid_lower_bound=[self.grid_lower_bound for i in range(len(downsample_render))]
            self.render_config={'render_frustum':self.render_frustum,'render_grid_map':self.render_grid_map,'render_grid_size':self.render_grid_size,'render_grid_interval':self.render_grid_interval,'render_grid_lower_bound':self.render_grid_lower_bound}
            # import pdb;pdb.set_trace()
        self.img_upsample=img_upsample
    def gen_grid_map(self,grid_size):
        # w,h,z=grid_size.long().tolist()  
        # xs = torch.linspace(0, w - 1,w).view(1, w, 1).expand(h, w, z)
        # ys = torch.linspace(0, h - 1, h).view(h, 1, 1).expand(h, w, z)
        # zs = torch.linspace(0, z- 1, z).view(1, 1, z).expand(h, w, z)
        # grid = torch.stack((xs, ys, zs), -1).view(1, h, w, z, 3)+0.5
        # # import pdb;pdb.set_trace()
        
        
        # if self.inverse_grid_index:
        #     grid=grid.permute(0,3,2,1,4)
        
        # return grid
    
        w,h,z=grid_size.long().tolist()  
        xs = torch.linspace(0, h - 1, h).view(h, 1, 1).expand(h, w, z)
        ys = torch.linspace(0, w - 1,w).view(1, w, 1).expand(h, w, z)
        zs = torch.linspace(0, z- 1, z).view(1, 1, z).expand(h, w, z)
        grid = torch.stack((xs, ys, zs), -1).view(1, h, w, z, 3)+0.5
        # import pdb;pdb.set_trace()
        
        
        if self.inverse_grid_index:
            grid=grid.permute(0,3,2,1,4)
        
        return grid
    def lift_attn_gene_num_sampling_nets(self,num_points,num_anchors):
        if num_points==0:
            return None,None,0
        if not self.num_sampling_from_depth:
            attention_weights = nn.Linear(self.out_channels, num_points)
            if self.only_add_on_depth:
                sampling_offsets = nn.Linear(self.out_channels,  num_points)
                
            elif self.num_points_adap2depth:
                if self.disc2continue_depth:
                    sampling_offsets = nn.Linear(self.out_channels,  num_points*3)
                elif self.adaptive_depth_bin:
                    sampling_offsets = nn.Linear(self.out_channels,  num_points*self.adaptive_depth_bin*3)
                    num_points*=self.adaptive_depth_bin
                    attention_weights = nn.Linear(self.out_channels, num_points*self.adaptive_depth_bin)
                else:
                    sampling_offsets = nn.Linear(self.out_channels,  num_points*self.D*3)
                    num_points*=self.D
                    attention_weights = nn.Linear(self.out_channels, num_points)
                sampling_offsets,attention_weights=self.init_weights_(sampling_offsets,attention_weights,num_points)
            elif self.offsets_with_MLP:
                sampling_offsets =nn.Sequential(nn.Linear(self.out_channels,  self.out_channels),
                nn.ReLU(),
                nn.Linear(self.out_channels,  num_points*3))
                attention_weights =nn.Sequential(nn.Linear(self.out_channels,  self.out_channels),
                nn.ReLU(),
                nn.Linear(self.out_channels,  num_points))
                
            else:
                
                sampling_offsets = nn.Linear(self.out_channels,  num_points*3)
            if self.only_add_on_depth:
                sampling_offsets,attention_weights=self.init_weights2_(sampling_offsets,attention_weights,num_points)
            elif self.init_weights3:
                sampling_offsets,attention_weights=self.init_weights3_(sampling_offsets,attention_weights,num_points)
            elif self.init0:
                sampling_offsets,attention_weights=self.init_weights0_(sampling_offsets,attention_weights,num_points)
            elif self.init_weights_anchors:
                sampling_offsets,attention_weights=self.init_weights_anchors(sampling_offsets,attention_weights,num_points,num_anchors)
            else:
                sampling_offsets,attention_weights=self.init_weights_(sampling_offsets,attention_weights,num_points)
            return sampling_offsets,attention_weights,num_points
    def init_weights_anchors(self,sampling_offsets,attention_weights,num_points,num_anchors):
        """Default initialization for Parameters of Module."""
        constant_init(sampling_offsets, 0.)
        # num_anchors=1
        num_levels=1
        
        num_offsets=num_points//num_anchors

        thetas = torch.arange(
            num_offsets,
            dtype=torch.float32) * (2.0 * math.pi / num_offsets)
        grid_init = torch.stack([thetas.cos(), thetas.sin(),(thetas.cos() + thetas.sin()) / 2], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            num_offsets, 1, 1,
            3).repeat(1, num_levels, num_anchors, 1)
        for i in range(num_anchors):
            grid_init[:, :, i, :] *= i + 1

        sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(attention_weights, val=0., bias=0.)
        # import pdb;pdb.set_trace()
        return sampling_offsets,attention_weights
    def init_weights0_(self,sampling_offsets,attention_weights,num_points):
        """Default initialization for Parameters of Module."""
        constant_init(sampling_offsets, 0.)
        num_heads=1
        num_levels=1
        thetas = torch.arange(
            num_heads,
            dtype=torch.float32) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([torch.zeros(3)], -1)
        grid_init = grid_init.view(
            num_heads, 1, 1,
            3).repeat(1, num_levels, num_points, 1)
        # for i in range(num_points):
        #     grid_init[:, :, i, :] *= i + 1

        sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(attention_weights, val=0., bias=0.)
        return sampling_offsets,attention_weights
    def init_weights_(self,sampling_offsets,attention_weights,num_points):
        """Default initialization for Parameters of Module."""
        constant_init(sampling_offsets, 0.)
        num_heads=1
        num_levels=1
        thetas = torch.arange(
            num_heads,
            dtype=torch.float32) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin(),(thetas.cos() + thetas.sin()) / 2], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            num_heads, 1, 1,
            3).repeat(1, num_levels, num_points, 1)
        for i in range(num_points):
            grid_init[:, :, i, :] *= i + 1

        sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(attention_weights, val=0., bias=0.)
        return sampling_offsets,attention_weights

    def init_weights2_(self,sampling_offsets,attention_weights,num_points):
        """Default initialization for Parameters of Module."""
        constant_init(sampling_offsets, 0.)
        num_heads=1
        num_levels=1
        thetas = torch.arange(
            num_heads,
            dtype=torch.float32) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([thetas.cos()], -1)
        grid_init = grid_init.view(
            num_heads, 1, 1,
            1).repeat(1, num_levels, num_points, 1)
        for i in range(num_points):
            grid_init[:, :, i, :] *= i + 1
        grid_init-=grid_init.mean(2,keepdim=True)
        grid_init*=self.grid_config['depth'][2]

        sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(attention_weights, val=0., bias=0.) 
        return sampling_offsets,attention_weights
    def init_weights3_(self,sampling_offsets,attention_weights,num_points):
        """Default initialization for Parameters of Module."""
        constant_init(sampling_offsets, 0.)
        num_heads=1
        num_levels=1
        thetas = torch.arange(
            num_heads,
            dtype=torch.float32) * (2.0 * math.pi / num_heads)
        grid_init = torch.stack([torch.zeros(1),torch.zeros(1),thetas.cos()], -1)
        grid_init = grid_init.view(
            num_heads, 1, 1,
            3).repeat(1, num_levels, num_points, 1)
        weight=torch.linspace(self.grid_config['depth'][0],self.grid_config['depth'][1],num_points)+0.5
        for i in range(num_points):
            grid_init[:, :, i, 2] *= weight[i]
        grid_init-=grid_init.mean(2,keepdim=True)

        sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(attention_weights, val=0., bias=0.)
        return sampling_offsets,attention_weights
      

    def create_grid_infos(self, x, y, z, **kwargs):
        """Generate the grid information including the lower bound, interval,
        and size.

        Args:
            x (tuple(float)): Config of grid alone x axis in format of
                (lower_bound, upper_bound, interval).
            y (tuple(float)): Config of grid alone y axis in format of
                (lower_bound, upper_bound, interval).
            z (tuple(float)): Config of grid alone z axis in format of
                (lower_bound, upper_bound, interval).
            **kwargs: Container for other potential parameters
        """
        self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
        self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
        self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
                                       for cfg in [x, y, z]])
    def create_frustum_SL(self, depth_cfg, input_size, SL_size):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = SL_size
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

        # D x H x W x 3
        return torch.stack((x, y, d), -1)
    def create_frustum(self, depth_cfg, input_size, downsample):
        """Generate the frustum template for each image.

        Args:
            depth_cfg (tuple(float)): Config of grid alone depth axis in format
                of (lower_bound, upper_bound, interval).
            input_size (tuple(int)): Size of input images in format of (height,
                width).
            downsample (int): Down sample scale factor from the input size to
                the feature size.
        """
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float)\
            .view(-1, 1, 1).expand(-1, H_feat, W_feat)
        self.D = d.shape[0]
        if self.sid:
            d_sid = torch.arange(self.D).float()
            depth_cfg_t = torch.tensor(depth_cfg).float()
            d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
                              torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
            d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)
        # import pdb;pdb.set_trace()

        # D x H x W x 3
        return torch.stack((x, y, d), -1)
    def img_to_ego_coor(self,coor,sensor2ego,post_trans,post_rots,cam2imgs,bda):

        B, N, _, _ = sensor2ego.shape
        points = coor.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
            .matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        points = bda.view(B, 1, 1, 1, 1, 3,
                          3).matmul(points.unsqueeze(-1)).squeeze(-1)
        return points
    def ego_to_img_coor(self,coor,sensor2ego,post_trans,post_rots,cam2imgs,bda):
        B, N, _, _ = sensor2ego.shape
        combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
        coor=torch.inverse(bda).view(B, 1, 1, 1, 1, 3,3).matmul(coor.unsqueeze(-1)).squeeze(-1)
        # print(abs(coor-points1).max(),11111111111111111)
        coor=coor-sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
        coor=torch.inverse(combine).view(B, N, 1, 1, 1, 3, 3).matmul(coor.unsqueeze(-1))
        # print(abs(coor-points2).max(),2222222222222222)
        coor[...,2,:]=torch.where(coor[...,2,:]<=0,torch.ones_like(coor[...,2,:])*1e-6,coor[...,2,:])
        coor=torch.cat((coor[..., :2,:]/coor[..., 2:3,:], coor[..., 2:3,:]), 5)
        # print(abs(coor-points3).max(),333333333333333333)
        coor=post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(coor)
        # print(abs(coor.squeeze(-1)-points4).max(),4444444444444444)
        coor=coor.squeeze(-1)+post_trans.view(B, N, 1, 1, 1, 3)
        return coor
    def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda,coor_reproject=False,points=None,sampling_offsets=None,round=0):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        B, N, _, _ = sensor2ego.shape
        if coor_reproject:
            combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
            coor=torch.inverse(bda).view(B, 1, 1, 1, 1, 3,3).matmul(points.unsqueeze(-1)).squeeze(-1)
            # print(abs(coor-points1).max(),11111111111111111)
            coor=coor-sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
            coor=torch.inverse(combine).view(B, N, 1, 1, 1, 3, 3).matmul(coor.unsqueeze(-1))
            # print(abs(coor-points2).max(),2222222222222222)
            
            coor=torch.cat((coor[..., :2,:]/coor[..., 2:3,:], coor[..., 2:3,:]), 5)
            # print(abs(coor-points3).max(),333333333333333333)
            coor=post_rots.view(B, N, 1, 1, 1, 3, 3).matmul(coor)
            # print(abs(coor.squeeze(-1)-points4).max(),4444444444444444)
            coor=coor.squeeze(-1)+post_trans.view(B, N, 1, 1, 1, 3)
            # import pdb;pdb.set_trace()
            return coor

        # post-transformation
        # B x N x D x H x W x 3
        if self.toSet:
            if self.coor_expand_in_vox is None:
                if self.toSetV2:
                    frustum=self.frustum.unsqueeze(-2).repeat(1,1,1,self.k*self.k_sqrt,1)
                    # import pdb;pdb.set_trace()
                    frustum=frustum+self.coor_offset.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    frustum=frustum.reshape(frustum.shape[0],frustum.shape[1],frustum.shape[2]*frustum.shape[3],frustum.shape[4])
                else:
                    
                    frustum=self.frustum.unsqueeze(-2).repeat(1,1,1,self.k,1)
                    # import pdb;pdb.set_trace()
                    frustum[...,:2]=frustum[...,:2]+self.coor_offset.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    frustum=frustum.reshape(frustum.shape[0],frustum.shape[1],frustum.shape[2]*frustum.shape[3],frustum.shape[4])
            else:
                frustum=self.frustum
        else:
            frustum = self.frustum
        # frustum=frustum.view(B, N, 1, 1, 1, 3)
        if sampling_offsets is not None:

            frustum=frustum.to(sampling_offsets).unsqueeze(0).unsqueeze(0).repeat(B,N,1,1,1,1)
            if (self.lift_attn_new or self.lift_attn_with_ori_feat) and round>0:
                # import pdb;pdb.set_trace()
                frustum=frustum.unsqueeze(3)
                frustum=frustum+sampling_offsets
                frustum=frustum.reshape(B,N,frustum.shape[2]*frustum.shape[3],*frustum.shape[4:]).contiguous()
            else:
                if self.num_points_adap2depth:
                    # import pdb;pdb.set_trace()
                    frustum=frustum.repeat(1,1,sampling_offsets.shape[2]//frustum.shape[2],1,1,1)
                    frustum=frustum+sampling_offsets
                else:
                    frustum=frustum.unsqueeze(2)+sampling_offsets.unsqueeze(3)
                    frustum=frustum.reshape(B,N,frustum.shape[2]*frustum.shape[3],*frustum.shape[4:]).contiguous()
            
        points=self.img_to_ego_coor(frustum,sensor2ego,post_trans,post_rots,cam2imgs,bda)
            
        
        return points
    def get_lidar_coor_single_depth(self, depth,input_size, downsample,sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
                       bda,sampling_offsets=None):
        """Calculate the locations of the frustum points in the lidar
        coordinate system.

        Args:
            rots (torch.Tensor): Rotation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3, 3).
            trans (torch.Tensor): Translation from camera coordinate system to
                lidar coordinate system in shape (B, N_cams, 3).
            cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
                (B, N_cams, 3, 3).
            post_rots (torch.Tensor): Rotation in camera coordinate system in
                shape (B, N_cams, 3, 3). It is derived from the image view
                augmentation.
            post_trans (torch.Tensor): Translation in camera coordinate system
                derived from image view augmentation in shape (B, N_cams, 3).

        Returns:
            torch.tensor: Point coordinates in shape
                (B, N_cams, D, ownsample, 3)
        """
        
        H_in, W_in = input_size
        H_feat, W_feat = H_in // downsample, W_in // downsample
        
        
       
     
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float).view(1,1, 1, W_feat).expand(depth.shape[0],depth.shape[1], H_feat, W_feat).to(depth.device)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float).view(1,1, H_feat, 1).expand(depth.shape[0],depth.shape[1], H_feat, W_feat).to(depth.device)

        # D x H x W x 3
        coor=torch.stack((x, y, depth), -1)
        
    
    
        B, N, _, _ = sensor2ego.shape

        # post-transformation
        # B x N x D x H x W x 3
        coor=coor.reshape(B,N,*coor.shape[1:])
        
        if sampling_offsets is not None:
            if self.only_add_on_depth:
                # import pdb;pdb.set_trace()
                coor=coor.unsqueeze(2).repeat(1,1,sampling_offsets.shape[2],1,1,1,1)
                sampling_offsets=sampling_offsets.unsqueeze(3)
                
                coor[...,2:3]=coor[...,2:3]+sampling_offsets
                coor=coor.reshape(B,N,coor.shape[2]*coor.shape[3],*coor.shape[4:]).contiguous()
                
            else:
                if self.num_points_adap2depth:
                    
                    coor=coor.repeat(1,1,sampling_offsets.shape[2]//coor.shape[2],1,1,1)
                    coor=coor+sampling_offsets
                else:
                    coor=coor.unsqueeze(2)+sampling_offsets.unsqueeze(3)
                    coor=coor.reshape(B,N,coor.shape[2]*coor.shape[3],*coor.shape[4:]).contiguous()
        points=self.img_to_ego_coor(coor,sensor2ego,post_trans,post_rots,cam2imgs,bda)

        return points

    def init_acceleration_v2(self, coor):
        """Pre-compute the necessary information in acceleration including the
        index of points in the final feature.

        Args:
            coor (torch.tensor): Coordinate of points in lidar space in shape
                (B, N_cams, D, H, W, 3).
            x (torch.tensor): Feature of points in shape
                (B, N_cams, D, H, W, C).
        """

        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor)

        self.ranks_bev = ranks_bev.int().contiguous()
        self.ranks_feat = ranks_feat.int().contiguous()
        self.ranks_depth = ranks_depth.int().contiguous()
        self.interval_starts = interval_starts.int().contiguous()
        self.interval_lengths = interval_lengths.int().contiguous()

    def voxel_pooling_v2(self, coor, depth, feat,occ2set=None,grid_size=torch.Tensor([200,200,16]),grid_interval=torch.Tensor([0.4,0.4,0.4])):
        if self.depth_detach:
            depth=depth.detach()
        
            
        # import pdb;pdb.set_trace()
        if self.toSetV4 or self.toSetV5:
            depth=depth.squeeze(-1)
        if self.max_depth:
            gumbel_noise = -(-(torch.rand_like(depth)).log()).log()
            if self.training:
                depth_ = (depth+1e-8).log() + gumbel_noise
                if self.depth2occ:
                    depth=depth_.exp()
                else:
                    depth=torch.softmax(depth_,dim=2)

            else:
                depth_=depth
            
            
            idx=torch.topk(depth_,k=int(self.max_depth),dim=2,largest=True)[1]
            if self.depth_sampling:
                idx_down_samp=torch.arange(0,depth.shape[2],depth.shape[2]//self.depth_sampling+1).to(depth.device)
                idx_down_samp=idx_down_samp.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(depth.shape[0],depth.shape[1],1,depth.shape[3],depth.shape[4])
                idx=torch.cat([idx,idx_down_samp],dim=2)
                
            # idx=torch.argmax(depth_,dim=2,keepdim=True)
            depth=torch.gather(depth,dim=2,index=idx)
            coor=torch.gather(coor,dim=2,index=idx.unsqueeze(-1).repeat(1,1,1,1,1,3))
        

        
        if self.torch_sparse_coor:
        # # import pdb;pdb.set_trace()
            if self.toSet:
                # import pdb;pdb.set_trace()
                
                coor_=coor
                # coor_=coor.unsqueeze(-2).repeat(1,1,1,1,1,self.k_sqrt**2,1)
                
                coor_ = ((coor_ - self.grid_lower_bound.to(coor_)) /grid_interval.to(coor_))
                
                coor_ = coor_.long()
                
                if self.coor_expand_in_vox is not None:
                    if self.toSetV2:
                        # import pdb;pdb.set_trace()
                        coor_ = coor_.unsqueeze(-2).repeat(1,1,1,1,1,self.k*self.k_sqrt,1)
                        coor_=coor_+self.coor_offset.to(coor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        B, N, D, H, W,K, _ = coor_.shape
                        num_points = B * N * D * H * W * K
                    else:
                        coor_ = coor_.unsqueeze(-2).repeat(1,1,1,1,1,self.k,1)
                        coor_[...,:2]=coor_[...,:2]+self.coor_offset.to(coor.device).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
                        B, N, D, H, W,K, _ = coor_.shape
                        num_points = B * N * D * H * W * K
                else:
                    B, N, D, H, W, _ = coor.shape
                    num_points = B * N * D * H * W #* self.k_sqrt**2
                coor_ = coor_.view(num_points, 3)
                batch_idx = torch.arange(0, B ).reshape(B, 1).expand(B, num_points // B).reshape(num_points, 1).to(coor_)
                coor_ = torch.cat((coor_, batch_idx), 1)
                kept = (coor_[:, 0] >= 0) & (coor_[:, 0] < grid_size[0]) &(coor_[:, 1] >= 0) & (coor_[:, 1] < grid_size[1]) & (coor_[:, 2] >= 0) & (coor_[:, 2] < grid_size[2])
                if len(kept) == 0:
                    return None, None, None, None, None
                if self.toSetV2:
                    occ2set=occ2set.reshape(B,N,self.k*self.k_sqrt,H,-1)
                    depth_=depth.repeat(1,1,1,1,1,self.k*self.k_sqrt)
                    occ2set=occ2set.unsqueeze(2).repeat(1,1,D,1,1,1).permute(0,1,2,4,5,3)
                    depth_=depth_*occ2set
                    
                    feat_=feat.unsqueeze(-1).repeat(1,1,1,1,1,self.k*self.k_sqrt)
                else:
                    depth_=depth
                    # feat_=feat.reshape(B*N,-1,H,W)
                    # feat_=get_expanded_feature_map(feat_,self.k_sqrt,self.kernel.to(feat.device))
                    feat_=feat.unsqueeze(-1).repeat(1,1,1,1,1,self.k)
                    # feat_=feat_.reshape(B,N,self.k_sqrt**2,-1,H,W).permute(0,1,3,4,5,2)
                    # import pdb;pdb.set_trace()
                depth_=depth_.unsqueeze(3)
                feat_=feat_.unsqueeze(2)
                weighted_feat=(depth_*feat_).permute(0,1,2,4,5,6,3).reshape(num_points,-1)
                coor_, weighted_feat = coor_[kept], weighted_feat[kept]

                coor_=torch.flip(coor_,[-1])
                bev_feat_shape = (depth.shape[0], int(grid_size[2]),int(grid_size[1]), int(grid_size[0]),weighted_feat.shape[-1])  # (B, Z, Y, X, C)
                # import pdb;pdb.set_trace()
                bev_feat=torch.sparse_coo_tensor(coor_.t(),weighted_feat,bev_feat_shape).to_dense()
                bev_feat= bev_feat.permute(0, 4, 1, 2, 3).contiguous()
            else:
                
                B, N, D, H, W, _ = coor.shape
                num_points = B * N * D * H * W

                coor_ = ((coor - self.grid_lower_bound.to(coor)) /
                        grid_interval.to(coor))
                coor_ = coor_.long().view(num_points, 3)
                batch_idx = torch.arange(0, B ).reshape(B, 1). \
                    expand(B, num_points // B).reshape(num_points, 1).to(coor_)
                coor_ = torch.cat((coor_, batch_idx), 1)

                # filter out points that are outside box
                kept = (coor_[:, 0] >= 0) & (coor_[:, 0] < grid_size[0]) & \
                    (coor_[:, 1] >= 0) & (coor_[:, 1] < grid_size[1]) & \
                    (coor_[:, 2] >= 0) & (coor_[:, 2] < grid_size[2])
                if len(kept) == 0:
                    return None, None, None, None, None
                
                depth_=depth.unsqueeze(3)
                feat_=feat.unsqueeze(2)
                weighted_feat=(depth_*feat_).permute(0,1,2,4,5,3).reshape(num_points,-1)
                coor_, weighted_feat = \
                    coor_[kept], weighted_feat[kept]

                coor_=torch.flip(coor_,[-1])
                bev_feat_shape = (depth.shape[0], int(grid_size[2]),
                                int(grid_size[1]), int(grid_size[0]),
                                weighted_feat.shape[-1])  # (B, Z, Y, X, C)
                # import pdb;pdb.set_trace()
                bev_feat=torch.sparse_coo_tensor(coor_.t(),weighted_feat,bev_feat_shape).to_dense()
                bev_feat= bev_feat.permute(0, 4, 1, 2, 3).contiguous()

            if self.bev_mean_pool:
                feat_=torch.ones_like(feat).to(feat.device)
                depth_=torch.ones_like(depth).to(depth.device)

                weights = bev_pool_v2(depth_, feat_, ranks_depth, ranks_feat, ranks_bev,bev_feat_shape, interval_starts,interval_lengths)
                # import pdb;pdb.set_trace()
                weights=torch.where(weights>0,weights,torch.ones_like(weights).to(weights))
                bev_feat=bev_feat/weights
            if self.SL:
                feat_=torch.ones_like(feat).to(feat.device).unsqueeze(2)
                weighted_feat=(depth_*feat_).permute(0,1,2,4,5,3).reshape(num_points,-1)
                weighted_feat = weighted_feat[kept]

                weights=torch.sparse_coo_tensor(coor_.t(),weighted_feat,bev_feat_shape).to_dense()
                weights= weights.permute(0, 4, 1, 2, 3).contiguous()

                weights=torch.where(weights>0,weights,torch.ones_like(weights).to(weights))
                bev_feat=bev_feat/weights
            if self.collapse_z:
                bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)

            return bev_feat


        if self.toSet:
            
            if self.coor_expand_in_vox is None:
                
                if self.toSetV2 or self.toSetV4:
                    
                    B, N, D, H, W, _ = coor.shape
                    W=W//(self.k*self.k_sqrt)
                    coor=coor.reshape(B,N,D,H,W,self.k*self.k_sqrt,_).permute(0,5,1,2,3,4,6)
                    coor=coor.reshape(B,self.k*self.k_sqrt*N,*coor.shape[3:])
                   
                else:
                    # import pdb;pdb.set_trace()
                    B, N, D, H, W, _ = coor.shape
                    W=W//self.k
                    coor=coor.reshape(B,N,D,H,W,self.k,_).permute(0,5,1,2,3,4,6)
                    coor=coor.reshape(B,self.k*N,*coor.shape[3:])
                
            if self.toSetV2:
                B, N, D, H, W, _ = depth.shape
                
                occ2set=occ2set.reshape(B,N,self.k*self.k_sqrt,H,-1)
                depth=depth.repeat(1,1,1,1,1,self.k*self.k_sqrt).permute(0,5,1,2,3,4)
                occ2set=occ2set.unsqueeze(2).repeat(1,1,D,1,1,1).permute(0,3,1,2,4,5)
                depth=depth*occ2set
                depth=depth.reshape(B,self.k*self.k_sqrt*N,*depth.shape[3:])
                
                feat=feat.unsqueeze(1).repeat(1,self.k*self.k_sqrt,1,1,1,1)
                feat=feat.reshape(B,self.k*self.k_sqrt*N,*feat.shape[3:])
            elif self.toSetV4:
                B, N, D, H, W= depth.shape
                
                depth=depth.unsqueeze(-1).repeat(1,1,1,1,1,self.k*self.k_sqrt).permute(0,5,1,2,3,4)
                depth=depth.reshape(B,self.k*self.k_sqrt*N,*depth.shape[3:])
                
                feat=feat.unsqueeze(1).repeat(1,self.k*self.k_sqrt,1,1,1,1)
                feat=feat.reshape(B,self.k*self.k_sqrt*N,*feat.shape[3:])
            elif self.toSetV5:
                # import pdb;pdb.set_trace()
                B, N, D, H, W= depth.shape
                
                depth=depth.unsqueeze(-1).repeat(1,1,1,1,1,self.k*self.k_sqrt).permute(0,5,1,2,3,4)
                

                code_book_mean=self.code_book.to(depth.device).mean(dim=1)
                logits=torch.einsum('bnchw,kc->bnkhw',feat,code_book_mean)
                idx=torch.argmax(logits,dim=2)
                # loss_code_book=
                idx=idx.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,1,self.code_book.shape[1],self.out_channels)
                code_book=torch.gather(self.code_book.to(depth.device),dim=0,index=idx.reshape(-1,self.code_book.shape[1],self.out_channels)).reshape(idx.shape)
                code_book=code_book.permute(0,1,4,5,2,3)
                weight=torch.einsum('bnchw,bnkchw->bnkhw',feat,code_book).sigmoid()
                weight=weight.permute(0,2,1,3,4).unsqueeze(3)
                depth=depth*weight
                depth=depth.reshape(B,self.k*self.k_sqrt*N,*depth.shape[3:])
                
                feat=feat.unsqueeze(1).repeat(1,self.k*self.k_sqrt,1,1,1,1)
                feat=feat.reshape(B,self.k*self.k_sqrt*N,*feat.shape[3:])
            else:
                
                B, N, D, H, W, _ = depth.shape
            
                depth=depth.permute(0,5,1,2,3,4)
                depth=depth.reshape(depth.shape[0],depth.shape[1]*depth.shape[2],*depth.shape[3:])
                feat=feat.unsqueeze(1).repeat(1,self.k,1,1,1,1)
                feat=feat.reshape(feat.shape[0],feat.shape[1]*feat.shape[2],*feat.shape[3:])    



        ranks_bev, ranks_depth, ranks_feat, \
            interval_starts, interval_lengths = \
            self.voxel_pooling_prepare_v2(coor,grid_size)
        if ranks_feat is None:
            print('warning ---> no points within the predefined '
                  'bev receptive field')
            dummy = torch.zeros(size=[
                feat.shape[0], feat.shape[2],
                int(grid_size[2]),
                int(grid_size[0]),
                int(grid_size[1])
            ]).to(feat)
            dummy = torch.cat(dummy.unbind(dim=2), 1)
            return dummy
        feat = feat.permute(0, 1, 3, 4, 2)
        ##
        # feat=feat+100
        # depth=depth+100
        ##
        bev_feat_shape = (depth.shape[0], int(grid_size[2]),
                          int(grid_size[1]), int(grid_size[0]),
                          feat.shape[-1])  # (B, Z, Y, X, C)
        # import pdb;pdb.set_trace()
        if not self.bev_weighted_pool:
            bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,bev_feat_shape, interval_starts,interval_lengths)
        else:
            # import pdb;pdb.set_trace()
            weight=torch.exp(depth*self.bev_weighted_pool).detach()
            depth=depth*weight

            bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,bev_feat_shape, interval_starts,interval_lengths)
            weight_shape = (depth.shape[0], int(grid_size[2]),
                          int(grid_size[1]), int(grid_size[0]),
                          1)  # (B, Z, Y, X, C)

            weight=bev_pool_v2(weight, torch.ones_like(feat[...,:1]).to(feat.device), ranks_depth, ranks_feat, ranks_bev,weight_shape, interval_starts,interval_lengths)
            weight=torch.where(weight>0,weight,torch.ones_like(weight).to(weight))
            bev_feat=bev_feat/weight
        if self.bev_mean_pool:
            feat_=torch.ones_like(feat).to(feat.device)
            depth_=torch.ones_like(depth).to(depth.device)

            weights = bev_pool_v2(depth_, feat_, ranks_depth, ranks_feat, ranks_bev,bev_feat_shape, interval_starts,interval_lengths)
            # import pdb;pdb.set_trace()
            weights=torch.where(weights>0,weights,torch.ones_like(weights).to(weights))
            bev_feat=bev_feat/weights
        

        if self.SL:
            feat_=torch.ones_like(feat).to(feat.device)
            weights = bev_pool_v2(depth, feat_, ranks_depth, ranks_feat, ranks_bev,bev_feat_shape, interval_starts,interval_lengths)
            weights=torch.where(weights>0,weights,torch.ones_like(weights).to(weights))
            bev_feat=bev_feat/weights
            bev_feat_sem=bev_feat[:,:-1,...]

            weight=torch.exp(depth*self.SL_max_weight).detach()
            depth=depth*weight
            
            occ_shape = (depth.shape[0], int(grid_size[2]),int(grid_size[1]), int(grid_size[0]),1)  # (B, Z, Y, X, C)
            bev_feat_occ = bev_pool_v2(depth, torch.ones_like(feat[...,:1]).to(feat.device), ranks_depth, ranks_feat, ranks_bev,occ_shape, interval_starts,interval_lengths)
            
            weight=bev_pool_v2(weight, torch.ones_like(feat[...,:1]).to(feat.device), ranks_depth, ranks_feat, ranks_bev,occ_shape, interval_starts,interval_lengths)
            weight=torch.where(weight>0,weight,torch.ones_like(weight).to(weight))
            bev_feat_occ=bev_feat_occ/weight
            bev_feat=torch.cat([bev_feat_sem,bev_feat_occ],dim=1)
            
        # import pdb;pdb.set_trace()

        # collapse Z
        if self.collapse_z:
            bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
        # if self.toSetV5:
        #     return [bev_feat,]
       
        return bev_feat

    def voxel_pooling_prepare_v2(self, coor,grid_size=torch.Tensor([200,200,16])):
        """Data preparation for voxel pooling.

        Args:
            coor (torch.tensor): Coordinate of points in the lidar space in
                shape (B, N, D, H, W, 3).

        Returns:
            tuple[torch.tensor]: Rank of the voxel that a point is belong to
                in shape (N_Points); Reserved index of points in the depth
                space in shape (N_Points). Reserved index of points in the
                feature space in shape (N_Points).
        """
        

        # convert coordinate into the voxel space
     
        
        if self.coor_expand_in_vox is not None:
            if self.toSetV2 or self.toSetV4 or self.toSetV5:
                # import pdb;pdb.set_trace()
                coor = coor.unsqueeze(1).repeat(1,self.k*self.k_sqrt,1,1,1,1,1)
                coor=coor+self.coor_offset.to(coor.device).unsqueeze(0).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
                coor=coor.reshape(coor.shape[0],coor.shape[1]*coor.shape[2],*coor.shape[3:])
                
            else:
                # import pdb;pdb.set_trace()
                coor = coor.unsqueeze(1).repeat(1,self.k,1,1,1,1,1)
                coor[...,:2]=coor[...,:2]+self.coor_offset.to(coor.device).unsqueeze(0).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2).unsqueeze(-2)
                coor=coor.reshape(coor.shape[0],coor.shape[1]*coor.shape[2],*coor.shape[3:])
                
        B, N, D, H, W, _ = coor.shape
        num_points = B * N * D * H * W
        # import pdb;pdb.set_trace()
        if self.deformable_lift:
            coor = coor.view(num_points, 3)
        else:
            coor = coor.long().view(num_points, 3)


        # record the index of selected points for acceleration purpose
        
        ranks_depth = torch.arange(
            0, num_points , dtype=torch.int, device=coor.device)
        ranks_feat = torch.arange(
            0, num_points // D , dtype=torch.int, device=coor.device)
        ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
        ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()

        

        batch_idx = torch.arange(0, B ).reshape(B, 1). \
            expand(B, num_points // B).reshape(num_points, 1).to(coor)
        coor = torch.cat((coor, batch_idx), 1)

        # filter out points that are outside box
        kept = (coor[:, 0] >= 0) & (coor[:, 0] < grid_size[0]) & \
               (coor[:, 1] >= 0) & (coor[:, 1] < grid_size[1]) & \
               (coor[:, 2] >= 0) & (coor[:, 2] < grid_size[2])
        if len(kept) == 0:
            return None, None, None, None, None
        coor, ranks_depth, ranks_feat = \
            coor[kept], ranks_depth[kept], ranks_feat[kept]
        # get tensors from the same voxel next to each other
        ranks_bev = coor[:, 3] * (
            grid_size[2] * grid_size[1] * grid_size[0])
        ranks_bev += coor[:, 2] * (grid_size[1] * grid_size[0])
        ranks_bev += coor[:, 1] * grid_size[0] + coor[:, 0]
        order = ranks_bev.argsort()
        ranks_bev, ranks_depth, ranks_feat = \
            ranks_bev[order], ranks_depth[order], ranks_feat[order]

        kept = torch.ones(
            ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int()
        if len(interval_starts) == 0:
            return None, None, None, None, None
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
        ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
        ), interval_lengths.int().contiguous()

    def pre_compute(self, input):
        if self.initial_flag:
            coor = self.get_lidar_coor(*input[1:7])
            self.init_acceleration_v2(coor)
            self.initial_flag = False
    def deformable_lift_func(self,coor):
        
        B,N,K,H,W,_=coor.shape
        # import pdb;pdb.set_trace()
        # coor=coor-0.5###########################
        coor=coor.reshape(-1,3)
        weight,coor=deformable_lift(coor)
        coor=coor.reshape(B,N,K,H,W,8,3)
        coor=coor.permute(0,1,2,5,3,4,6)
        coor=coor.reshape(B,N,coor.shape[2]*coor.shape[3],*coor.shape[4:]).contiguous()
        return weight,coor
    def view_transform_fill_all_with_occ(self, input, occ, tran_feat,occ2set=None,round=0,num_points_expand=0,sampling_offsets=None,attention_weights=None,
                            grid_size=torch.Tensor([200,200,16]),grid_interval=torch.Tensor([0.4,0.4,0.4]),gt_occ=None):
        
        B, N, C, H, W = input[0].shape
        
        sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
        if self.fill_with_gt_occ:
            occ=gt_occ
            occ=(gt_occ!=17).float()
        
        w,h,z=grid_size.long().tolist()  
        if self.inverse_grid_index:
            w,h,z=z,h,w
        grid=self.grid_map.to(tran_feat.device).unsqueeze(0).repeat(B,N,1,1,1,1)
        grid=self.ego_to_img_coor(grid,sensor2ego,post_trans,post_rots,cam2imgs,bda)
        
        H_in, W_in = self.input_size_
        # grid[...,-1]-=self.grid_config['depth'][0]
        # grid/=torch.Tensor([W_in-1,H_in-1,self.grid_config['depth'][1]-self.grid_config['depth'][0]-self.grid_config['depth'][2]]).to(grid.device)
        # grid=grid*2-1
        # grid=grid.reshape(B*N,*grid.shape[2:])
        mask=(grid[...,0]>=0)&(grid[...,0]<=W_in-1)&(grid[...,1]>=0)&(grid[...,1]<=H_in-1)&(grid[...,2]>=self.grid_config['depth'][0])&(grid[...,2]<=self.grid_config['depth'][1]-self.grid_config['depth'][2])
        valid_grid=grid[mask]
        # valid_frustum=
        
        depth_range = torch.arange(*self.grid_config['depth'], dtype=torch.float,device=tran_feat.device).view(1, -1).repeat(valid_grid.shape[0],1)
        valid_grid_frustum=valid_grid.unsqueeze(1).repeat(1,depth_range.shape[1],1)
        valid_grid_frustum[...,2]=depth_range
        
        zz=[]
        grid_frustum=grid.unsqueeze(-2).repeat(1,1,1,1,1,depth_range.shape[1],1)
        grid_frustum[mask]=valid_grid_frustum
        for i in range(B):
            grid_frustum_i=self.img_to_ego_coor(grid_frustum[i:i+1].reshape(1,N,w,h,z*depth_range.shape[1],3),sensor2ego,post_trans,post_rots,cam2imgs,bda)
            zz.append(grid_frustum_i)
        zz=torch.cat(zz,dim=0)
        # import pdb;pdb.set_trace()
        valid_grid_frustum=self.img_to_ego_coor(valid_grid_frustum,sensor2ego,post_trans,post_rots,cam2imgs,bda)
        valid_grid_frustum = ((valid_grid_frustum - self.grid_lower_bound.to(valid_grid_frustum)) /grid_interval.to(valid_grid_frustum))
        valid_grid_frustum = valid_grid_frustum

    def view_transform_fill_all2(self, input, depth, tran_feat,grid_map,grid_size,grid_interval,grid_lower_bound,gt_occ=None,coor=None):
        B, N, C, H, W = input[0].shape
        # import pdb;pdb.set_trace()
        sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
        
       
        
        w,h,z=grid_size.long().tolist()  
        if self.inverse_grid_index:
            w,h,z=z,h,w
        grid=grid_map.to(depth.device).unsqueeze(0).repeat(B,N,1,1,1,1)
        
        ###################################
       
            
        
        grid=grid*grid_interval.to(grid)+self.grid_lower_bound.to(grid)
        # import pdb;pdb.set_trace()
        grid=self.ego_to_img_coor(grid,sensor2ego,post_trans,post_rots,cam2imgs,bda)
        
        H_in, W_in = self.input_size_
        grid[...,-1]-=self.grid_config['depth'][0]
        grid/=torch.Tensor([W_in-1,H_in-1,self.grid_config['depth'][1]-self.grid_config['depth'][0]-self.grid_config['depth'][2]]).to(grid.device)
        grid=grid*2-1
     
        grid=grid.reshape(B*N,*grid.shape[2:])
        
        
        # import pdb;pdb.set_trace()
        sampled_feat=F.grid_sample(tran_feat,grid[...,:-1].reshape(B*N,w*h,z,2),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(B*N,-1,w,h,z)
        sampled_depth=F.grid_sample(depth.unsqueeze(1),grid,mode='bilinear',padding_mode='zeros',align_corners=True)
        # sampled_non_zero_weight=F.grid_sample(torch.ones_like(tran_feat,device=grid.device),grid[...,:-1].reshape(B*N,w*h,z,2),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(B*N,-1,w,h,z)
        # sampled_non_zero_weight=F.grid_sample(torch.ones_like(depth.unsqueeze(1),device=grid.device),grid,mode='bilinear',padding_mode='zeros',align_corners=True)
        # import pdb;pdb.set_trace()
        bev_feat=sampled_feat*sampled_depth
        bev_feat=bev_feat.reshape(B,N,-1,w,h,z).sum(1)
        # sampled_non_zero_weight=sampled_non_zero_weight.reshape(B,N,-1,w,h,z).sum(1)
        # sampled_non_zero_weight=torch.where(sampled_non_zero_weight>0,sampled_non_zero_weight,torch.ones_like(sampled_non_zero_weight).to(sampled_non_zero_weight.device))
        # bev_feat=bev_feat/sampled_non_zero_weight
        if not self.inverse_grid_index:
            bev_feat=bev_feat.permute(0,1,4,3,2)
        
        return bev_feat
        
    def view_transform_fill_all(self, input, depth, tran_feat,occ2set=None,round=0,num_points_expand=0,sampling_offsets=None,attention_weights=None,
                            grid_size=torch.Tensor([200,200,16]),grid_interval=torch.Tensor([0.4,0.4,0.4]),gt_occ=None):
        B, N, C, H, W = input[0].shape
        # import pdb;pdb.set_trace()
        sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
        if self.fill_with_gt_occ:
            # import pdb;pdb.set_trace()
            # occ=gt_occ
            # occ=(gt_occ!=17).float()
            gt_occ=gt_occ.to(depth)
            gt_occ=gt_occ!=17
            gt_occ=gt_occ.float()
            
            coor = self.frustum
            coor = self.img_to_ego_coor(coor,sensor2ego,post_trans,post_rots,cam2imgs,bda)
            coor = ((coor - self.grid_lower_bound.to(coor)) /grid_interval.to(coor))
        
            D=coor.shape[2]
            coor=coor.reshape(B,N,D,H*W,3)
            coor=coor/(grid_size.to(coor.device)-1)
            coor=coor*2-1
            
            
            occ_weight_gt=F.grid_sample(gt_occ.unsqueeze(1), torch.flip(coor,[-1]),align_corners=True,padding_mode='zeros')
            
            free_weight_gt=1-occ_weight_gt
            cum_free_weight_gt=torch.cumprod(free_weight_gt,dim=-2)
            cum_free_weight_gt=torch.cat([torch.ones_like(cum_free_weight_gt[...,0:1,:]),cum_free_weight_gt[...,:-1,:]],dim=-2)
            weight_gt=occ_weight_gt*cum_free_weight_gt
            
            weight_gt=weight_gt.reshape(B*N,D,H,W)
            depth=weight_gt
            import pdb;pdb.set_trace()
        # coor = self.frustum
        # # import pdb;pdb.set_trace()
        # coor = self.img_to_ego_coor(coor,sensor2ego,post_trans,post_rots,cam2imgs,bda)
        # coor = ((coor - self.grid_lower_bound.to(coor)) /grid_interval.to(coor))
        ########









   #########################################################################################################
        # coor=coor*grid_interval.to(coor)+self.grid_lower_bound.to(coor)
        
        # coor=self.ego_to_img_coor(coor,sensor2ego,post_trans,post_rots,cam2imgs,bda)
        ###################
        # H_in, W_in = self.input_size_
        # coor[...,-1]-=self.grid_config['depth'][0]
        # coor/=torch.Tensor([W_in-1,H_in-1,self.grid_config['depth'][1]-self.grid_config['depth'][0]-self.grid_config['depth'][2]]).to(coor.device)
        # coor=coor*2-1
        
        
             # Generate grid
        # import pdb;pdb.set_trace()
        # w,h,z=grid_size.long().tolist()  
        # xs = torch.linspace(0, w - 1,w, dtype=depth.dtype, device=depth.device).view(1, w, 1).expand(h, w, z)
        # ys = torch.linspace(0, h - 1, h, dtype=depth.dtype, device=depth.device).view(h, 1, 1).expand(h, w, z)
        # zs = torch.linspace(0, z- 1, z, dtype=depth.dtype, device=depth.device).view(1, 1, z).expand(h, w, z)
        # grid = torch.stack((xs, ys, zs), -1).view(1, h, w, z, 3).expand(B,N, h, w, z, 3)
        
        # grid=grid*grid_interval.to(grid)+self.grid_lower_bound.to(grid)
        
        w,h,z=grid_size.long().tolist()  
        if self.inverse_grid_index:
            w,h,z=z,h,w
        grid=self.grid_map.to(depth.device).unsqueeze(0).repeat(B,N,1,1,1,1)
        
        ###################################
        if self.mask_around_img:
            # import pdb;pdb.set_trace()
            coor_=coor
            coor_=coor_.long()
            coor_remap=coor_.float()*grid_interval.to(coor)+self.grid_lower_bound.to(coor)
            coor_remap_copy=coor_remap.clone()
            # import pdb;pdb.set_trace()
            coor_remap=self.ego_to_img_coor(coor_remap.float(),sensor2ego,post_trans,post_rots,cam2imgs,bda)
            
            coor_remap[...,-1]-=self.grid_config['depth'][0]
            H_in,W_in=self.input_size_
            coor_remap/=torch.Tensor([W_in-1,H_in-1,self.grid_config['depth'][1]-self.grid_config['depth'][0]-self.grid_config['depth'][2]]).to(coor.device)
            coor_remap=coor_remap*2-1
            
            b,n,d,h,w,_=coor_remap.shape
            sampled_tran_feat=F.grid_sample(tran_feat,coor_remap[...,:-1].reshape(b*n,d,w*h,2),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,n,-1,d,h,w)
            sampled_depth=F.grid_sample(depth.unsqueeze(1),coor_remap.reshape(b*n,d,h,w,3),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,n,-1,d,h,w)
             
            # bev_feat=torch.sparse_coo_tensor(coor.t(),tran_feat_valid,(int(grid_size[0]),int(grid_size[1]),int(grid_size[2]),tran_feat.shape[1])).to_dense()
            
            
            # coor = self.img_to_ego_coor(coor,sensor2ego,post_trans,post_rots,cam2imgs,bda)
            # coor = ((coor - self.grid_lower_bound.to(coor)) /grid_interval.to(coor))
            # import pdb;pdb.set_trace()
            
            
            bev_feats=[]
            for i in range(len(coor)):
                coor=coor_[i]
                
                coor_valid=(coor[...,0]>=0)&(coor[...,0]<=grid_size[0]-1)&(coor[...,1]>=0)&(coor[...,1]<=grid_size[1]-1)&(coor[...,2]>=0)&(coor[...,2]<=grid_size[2]-1)
                coor=coor[coor_valid]
                # tran_feat_batch=tran_feat.reshape(B,N,-1,H,W)[i]
                # tran_feat_batch=tran_feat_batch.permute(0,2,3,1)
                # tran_feat_batch=tran_feat_batch.unsqueeze(1).repeat(1,coor_valid.shape[1],1,1,1)
                # # import pdb;pdb.set_trace()
                # tran_feat_batch=depth.reshape(B,N,-1,H,W)[i].unsqueeze(-1)*tran_feat_batch
                
                tran_feat_batch=sampled_tran_feat[i]*sampled_depth[i]
                tran_feat_batch=tran_feat_batch.permute(0,2,3,4,1)
                # import pdb;pdb.set_trace()
                tran_feat_valid=tran_feat_batch[coor_valid]
                bev_feat=torch.sparse_coo_tensor(coor.t(),tran_feat_valid,(int(grid_size[0]),int(grid_size[1]),int(grid_size[2]),tran_feat.shape[1])).to_dense()
                bev_feat=bev_feat.permute(3,2,1,0)
                
                
                grid_coor_=torch.sparse_coo_tensor(coor.t(),coor,(int(grid_size[0]),int(grid_size[1]),int(grid_size[2]),3)).to_dense()
                weight=torch.sparse_coo_tensor(coor.t(),torch.ones_like(coor),(int(grid_size[0]),int(grid_size[1]),int(grid_size[2]),3)).to_dense()
                weight=torch.where(weight>0,weight,torch.ones_like(weight).to(weight.device))
                grid_coor_=grid_coor_.long()/weight.float()
                # import pdb;pdb.set_trace()
                # bev_feat=bev_feat.permute(3,0,1,2)
                # mask_around_img=torch.zeros((grid_size[0].long().item(),grid_size[1].long().item(),grid_size[2].long().item()),dtype=torch.bool,device=coor.device)
                # mask_around_img[coor[...,0],coor[...,1],coor[...,2]]=1
                # mask_around_imgs.append(mask_around_img)
                bev_feats.append(bev_feat)
            bev_feat=torch.stack(bev_feats,dim=0)
           
           
        #    ############################
            mask_around_img=(bev_feat.sum(1)!=0).unsqueeze(1).repeat(1,N,1,1,1).permute(0,1,4,3,2)
            
            
            # for i in range(len(coor)):
            #     coor=coor_[i]
            #     coor=coor.long()
            #     coor_valid=(coor[...,0]>=0)&(coor[...,0]<=grid_size[0]-1)&(coor[...,1]>=0)&(coor[...,1]<=grid_size[1]-1)&(coor[...,2]>=0)&(coor[...,2]<=grid_size[2]-1)
            #     coor=coor[coor_valid]
                
            #     mask_around_img=torch.zeros((grid_size[0].long().item(),grid_size[1].long().item(),grid_size[2].long().item()),dtype=torch.bool,device=coor.device)
            #     mask_around_img[coor[...,0],coor[...,1],coor[...,2]]=1
            #     mask_around_imgs.append(mask_around_img)
            # mask_around_img=torch.stack(mask_around_imgs,dim=0)
            
            # # mask_around_img=torch.sparse_coo_tensor(coor.t(),torch.ones_like(coor[...,0]).to(coor),(grid_size[0],grid_size[1],grid_size[2])).to_dense()
            # mask_around_img=mask_around_img.unsqueeze(1).repeat(1,N,1,1,1)!=0
        grid=grid*grid_interval.to(grid)+self.grid_lower_bound.to(grid)
        # import pdb;pdb.set_trace()
        grid=self.ego_to_img_coor(grid,sensor2ego,post_trans,post_rots,cam2imgs,bda)
        # mask_around_img=mask_around_img & (grid[...,-1]>1)
        
        # coor = self.img_to_ego_coor(grid,sensor2ego,post_trans,post_rots,cam2imgs,bda)
        ################
        # if self.mask_around_img:
            # zz=grid/torch.Tensor([703/43,255/15,1]).to(grid)
            # zz_=torch.round(grid)
            # mask_around_img=(zz-zz_).abs()
            # mask_around_img=(mask_around_img[...,0]<2/43)&(mask_around_img[...,1]<2/15)&(mask_around_img[...,2]<0.3)
        
        
        #################
        # import pdb;pdb.set_trace()
        
        H_in, W_in = self.input_size_
        grid[...,-1]-=self.grid_config['depth'][0]
        grid/=torch.Tensor([W_in-1,H_in-1,self.grid_config['depth'][1]-self.grid_config['depth'][0]-self.grid_config['depth'][2]]).to(grid.device)
        grid=grid*2-1
        if self.mask_around_img:
            grid[~mask_around_img]=10
        grid=grid.reshape(B*N,*grid.shape[2:])
        
        
        # import pdb;pdb.set_trace()
        sampled_feat=F.grid_sample(tran_feat,grid[...,:-1].reshape(B*N,w*h,z,2),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(B*N,-1,w,h,z)
        sampled_depth=F.grid_sample(depth.unsqueeze(1),grid,mode='bilinear',padding_mode='zeros',align_corners=True)
        # sampled_non_zero_weight=F.grid_sample(torch.ones_like(tran_feat,device=grid.device),grid[...,:-1].reshape(B*N,w*h,z,2),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(B*N,-1,w,h,z)
        # sampled_non_zero_weight=F.grid_sample(torch.ones_like(depth.unsqueeze(1),device=grid.device),grid,mode='bilinear',padding_mode='zeros',align_corners=True)
        # import pdb;pdb.set_trace()
        bev_feat=sampled_feat*sampled_depth
        bev_feat=bev_feat.reshape(B,N,-1,w,h,z).sum(1)
        # sampled_non_zero_weight=sampled_non_zero_weight.reshape(B,N,-1,w,h,z).sum(1)
        # sampled_non_zero_weight=torch.where(sampled_non_zero_weight>0,sampled_non_zero_weight,torch.ones_like(sampled_non_zero_weight).to(sampled_non_zero_weight.device))
        # bev_feat=bev_feat/sampled_non_zero_weight
        if not self.inverse_grid_index:
            bev_feat=bev_feat.permute(0,1,4,3,2)
        # import pdb;pdb.set_trace()
        
        
        
        
        
        
        
        
        
        ######################
        # if self.mask_around_img:
        #     # import pdb;pdb.set_trace()
        #     coor_=coor
        #     bev_feats=[]
        #     for i in range(len(coor)):
        #         coor=coor_[i]
        #         coor=coor.long()
        #         coor_valid=(coor[...,0]>=0)&(coor[...,0]<=grid_size[0]-1)&(coor[...,1]>=0)&(coor[...,1]<=grid_size[1]-1)&(coor[...,2]>=0)&(coor[...,2]<=grid_size[2]-1)
        #         coor=coor[coor_valid]
        #         tran_feat_batch=tran_feat.reshape(B,N,-1,H,W)[i]
        #         tran_feat_batch=tran_feat_batch.permute(0,2,3,1)
        #         tran_feat_batch=tran_feat_batch.unsqueeze(1).repeat(1,coor_valid.shape[1],1,1,1)
        #         # import pdb;pdb.set_trace()
        #         tran_feat_batch=depth.reshape(B,N,-1,H,W)[i].unsqueeze(-1)*tran_feat_batch
        #         # import pdb;pdb.set_trace()
        #         tran_feat_valid=tran_feat_batch[coor_valid]
        #         bev_feat=torch.sparse_coo_tensor(coor.t(),tran_feat_valid,(int(grid_size[0]),int(grid_size[1]),int(grid_size[2]),tran_feat.shape[1])).to_dense()
        #         bev_feat=bev_feat.permute(3,2,1,0)
        #         # bev_feat=bev_feat.permute(3,0,1,2)
        #         # mask_around_img=torch.zeros((grid_size[0].long().item(),grid_size[1].long().item(),grid_size[2].long().item()),dtype=torch.bool,device=coor.device)
        #         # mask_around_img[coor[...,0],coor[...,1],coor[...,2]]=1
        #         # mask_around_imgs.append(mask_around_img)
        #         bev_feats.append(bev_feat)
        #     bev_feat=torch.stack(bev_feats,dim=0)
            # import pdb;pdb.set_trace()
        #     # mask_around_img=torch.sparse_coo_tensor(coor.t(),torch.ones_like(coor[...,0]).to(coor),(grid_size[0],grid_size[1],grid_size[2])).to_dense()
        #     # mask_around_img=mask_around_img.unsqueeze(1).repeat(1,N,1,1,1)!=0
        
        #####################
        
        
        
        
        #########################
        # if self.mask_around_img:
        #     # import pdb;pdb.set_trace()
        #     coor_=coor
        #     coor_=coor_.long()
        #     coor_remap=coor_.float()*grid_interval.to(coor)+self.grid_lower_bound.to(coor)
        #     coor_remap_copy=coor_remap.clone()
        #     # import pdb;pdb.set_trace()
        #     coor_remap=self.ego_to_img_coor(coor_remap.float(),sensor2ego,post_trans,post_rots,cam2imgs,bda)
            
        #     coor_remap[...,-1]-=self.grid_config['depth'][0]
        #     H_in,W_in=self.input_size_
        #     coor_remap/=torch.Tensor([W_in-1,H_in-1,self.grid_config['depth'][1]-self.grid_config['depth'][0]-self.grid_config['depth'][2]]).to(coor.device)
        #     coor_remap=coor_remap*2-1
            
        #     b,n,d,h,w,_=coor_remap.shape
        #     sampled_tran_feat=F.grid_sample(tran_feat,coor_remap[...,:-1].reshape(b*n,d,w*h,2),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,n,-1,d,h,w)
        #     sampled_depth=F.grid_sample(depth.unsqueeze(1),coor_remap.reshape(b*n,d,h,w,3),mode='bilinear',padding_mode='zeros',align_corners=True).reshape(b,n,-1,d,h,w)
             
        #     # bev_feat=torch.sparse_coo_tensor(coor.t(),tran_feat_valid,(int(grid_size[0]),int(grid_size[1]),int(grid_size[2]),tran_feat.shape[1])).to_dense()
            
            
        #     # coor = self.img_to_ego_coor(coor,sensor2ego,post_trans,post_rots,cam2imgs,bda)
        #     # coor = ((coor - self.grid_lower_bound.to(coor)) /grid_interval.to(coor))
        #     # import pdb;pdb.set_trace()
            
            
        #     bev_feats=[]
        #     for i in range(len(coor)):
        #         coor=coor_[i]
                
        #         coor_valid=(coor[...,0]>=0)&(coor[...,0]<=grid_size[0]-1)&(coor[...,1]>=0)&(coor[...,1]<=grid_size[1]-1)&(coor[...,2]>=0)&(coor[...,2]<=grid_size[2]-1)
        #         coor=coor[coor_valid]
        #         # tran_feat_batch=tran_feat.reshape(B,N,-1,H,W)[i]
        #         # tran_feat_batch=tran_feat_batch.permute(0,2,3,1)
        #         # tran_feat_batch=tran_feat_batch.unsqueeze(1).repeat(1,coor_valid.shape[1],1,1,1)
        #         # # import pdb;pdb.set_trace()
        #         # tran_feat_batch=depth.reshape(B,N,-1,H,W)[i].unsqueeze(-1)*tran_feat_batch
                
        #         tran_feat_batch=sampled_tran_feat[i]*sampled_depth[i]
        #         tran_feat_batch=tran_feat_batch.permute(0,2,3,4,1)
        #         # import pdb;pdb.set_trace()
        #         tran_feat_valid=tran_feat_batch[coor_valid]
        #         bev_feat=torch.sparse_coo_tensor(coor.t(),tran_feat_valid,(int(grid_size[0]),int(grid_size[1]),int(grid_size[2]),tran_feat.shape[1])).to_dense()
        #         bev_feat=bev_feat.permute(3,2,1,0)
        #         # bev_feat=bev_feat.permute(3,0,1,2)
        #         # mask_around_img=torch.zeros((grid_size[0].long().item(),grid_size[1].long().item(),grid_size[2].long().item()),dtype=torch.bool,device=coor.device)
        #         # mask_around_img[coor[...,0],coor[...,1],coor[...,2]]=1
        #         # mask_around_imgs.append(mask_around_img)
        #         bev_feats.append(bev_feat)
        #     bev_feat=torch.stack(bev_feats,dim=0)
        #     # import pdb;pdb.set_trace()
        #     # mask_around_img=torch.sparse_coo_tensor(coor.t(),torch.ones_like(coor[...,0]).to(coor),(grid_size[0],grid_size[1],grid_size[2])).to_dense()
        #     # mask_around_img=mask_around_img.unsqueeze(1).repeat(1,N,1,1,1)!=0
        
        
        
        
        ##################
        return bev_feat,[depth]
    
    def view_transform_core(self, input, depth, tran_feat,occ2set=None,round=0,num_points_expand=0,sampling_offsets=None,attention_weights=None,
                            grid_size=torch.Tensor([200,200,16]),grid_interval=torch.Tensor([0.4,0.4,0.4])):
        B, N, C, H, W = input[0].shape
        # import pdb;pdb.set_trace()
        if self.adaptive_depth_bin:
            # import pdb;pdb.set_trace()
            
            depth,depth_weight=depth[0],depth[1]
            if self.depth_weight_exp:
                # import pdb;pdb.set_trace()
                depth_weight=depth_weight.exp()
            else:
                depth_weight=depth_weight.softmax(1)
            # depth_copy=depth.clone()
            # depth_weight_copy=depth_weight.clone()
        # Lift-Splat
        if self.accelerate:
            feat = tran_feat.view(B, N, self.out_channels, H, W)
            feat = feat.permute(0, 1, 3, 4, 2)
            depth = depth.view(B, N, self.D, H, W)
            bev_feat_shape = (depth.shape[0], int(grid_size[2]),
                              int(grid_size[1]), int(grid_size[0]),
                              feat.shape[-1])  # (B, Z, Y, X, C)
            bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
                                   self.ranks_feat, self.ranks_bev,
                                   bev_feat_shape, self.interval_starts,
                                   self.interval_lengths)

            bev_feat = bev_feat.squeeze(2)
        else:
      
            if self.disc2continue_depth or self.adaptive_depth_bin:
                if self.adaptive_depth_bin:
                    # import pdb;pdb.set_trace()
                    #ada_bin
                    cum_depth=torch.cat((torch.zeros_like(depth[:,:1,...]),torch.cumsum(depth,1)[:,:-1]),dim=1)
                    bin_center=self.grid_config['depth'][0]+(self.grid_config['depth'][1]-self.grid_config['depth'][0])*(depth/2+cum_depth)#BN,n_bin,H,W
                    if self.ada2fix_bin:
                        bin_center-=0.25
                    depth_continue=bin_center
                    
                    # bin_widths = (self.grid_config['depth'][1]-self.grid_config['depth'][0]) * depth  # .shape = N, dim_out
                    
                    # # bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.grid_config['depth'][0])
                    # bin_widths=torch.cat((torch.ones_like(depth[:,:1,...])*self.grid_config['depth'][0],bin_widths),dim=1)
                    
                    # bin_edges = torch.cumsum(bin_widths, dim=1)
                    # centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
                   
   
                
                    depth=depth_weight
                    if self.adaptive_bin2_continue_depth:
                        # import pdb;pdb.set_trace()
                        depth_continue= torch.sum(depth_weight/depth_weight.sum(1,keepdim=True) *bin_center, dim=1).unsqueeze(1)
                        depth=torch.ones_like(depth_continue)#BN,D,H,W
                    if self.lift_attn:
                        pos_depth= torch.sum(depth_weight/depth_weight.sum(1,keepdim=True) *bin_center, dim=1).unsqueeze(1)
                    
                else:
                    depth_map=torch.arange(self.D).to(depth.device)+1
                    depth_map=depth_map* self.grid_config['depth'][2]+(self.grid_config['depth'][0] -self.grid_config['depth'][2])
                    depth_continue=torch.einsum('bdwh,dl->blwh',depth,depth_map.reshape(-1,1))#BN,D,H,W
                    depth=torch.ones_like(depth_continue)#BN,D,H,W
                    if self.lift_attn:
                        pos_depth= depth_continue
                
                
                if self.sampling_offset_in_img_coor:
                    if num_points_expand:
                        if self.num_sampling_from_depth:
                            import pdb;pdb.set_trace()
                        if sampling_offsets is None:
                            sampling_offsets=self.sampling_offsets_nets[round](tran_feat.permute(0,2,3,1))#BN,H,W,n_P*3
                        
                            # sampling_offsets=torch.randn(B,N,H,W,num_points_expand*3).to(coor.device)

                        sampling_offsets=sampling_offsets.reshape(B,N,H,W,num_points_expand,self.num_coor)
                        if self.sampling_offsets_weight is not None:
                            sampling_offsets*=self.sampling_offsets_weight.to(sampling_offsets.device)
                        sampling_offsets=sampling_offsets.permute(0,1,4,2,3,5)#B,N,n_P,H,W,3
                    else:
                        sampling_offsets=None
                else:
                    sampling_offsets=None    
                coor=self.get_lidar_coor_single_depth(depth_continue,self.input_size_,self.downsample,*input[1:7],sampling_offsets)
                # else:
                #     coor=self.get_lidar_coor_single_depth(depth_continue,self.input_size_,self.downsample,*input[1:7])

            else:
                if self.lift_attn_fix_bin:
                    depth_map=torch.arange(self.D).to(depth.device)+1
                    depth_map=depth_map* self.grid_config['depth'][2]+(self.grid_config['depth'][0] -self.grid_config['depth'][2])
                    pos_depth=torch.einsum('bdwh,dl->blwh',depth,depth_map.reshape(-1,1))#BN,D,H,W
                    
                if self.sampling_offset_in_img_coor and num_points_expand:
                    if sampling_offsets is None:
                        sampling_offsets=self.sampling_offsets_nets[round](tran_feat.permute(0,2,3,1))#BN,H,W,n_P*3
                        # sampling_offsets=torch.randn(B,N,H,W,num_points_expand*3).to(coor.device)
                    if (self.lift_attn_new or self.lift_attn_with_ori_feat) and round>0:
                        
                        sampling_offsets=sampling_offsets.permute(0,1,2,5,3,4,6)
                    else:
                        sampling_offsets=sampling_offsets.reshape(B,N,H,W,num_points_expand,self.num_coor)
                        sampling_offsets=sampling_offsets.permute(0,1,4,2,3,5)#B,N,n_P,H,W,3
                    if self.sampling_offsets_weight is not None:
                        sampling_offsets*=self.sampling_offsets_weight.to(sampling_offsets.device)

                    
                    coor=self.get_lidar_coor(*input[1:7],sampling_offsets=sampling_offsets,round=round)
                else:
                    coor = self.get_lidar_coor(*input[1:7])
                
            if self.deformable_lift:
                
                if num_points_expand:
                    if not self.sampling_offset_in_img_coor:
                        if sampling_offsets is None:
                            sampling_offsets=self.sampling_offsets_nets[round](tran_feat.permute(0,2,3,1))
                        # sampling_offsets=torch.randn(B,N,H,W,num_points_expand*3).to(coor.device)
                        
                        
                        sampling_offsets=sampling_offsets.reshape(B,N,H,W,num_points_expand,self.num_coor)
                        if self.sampling_offsets_weight is not None:
                            sampling_offsets*=self.sampling_offsets_weight.to(sampling_offsets.device)
                        sampling_offsets=sampling_offsets.permute(0,1,4,2,3,5)
                        
                        coor=coor.unsqueeze(2)+sampling_offsets.unsqueeze(3)
                        coor=coor.reshape(B,N,coor.shape[2]*coor.shape[3],*coor.shape[4:]).contiguous()
                    # import pdb;pdb.set_trace()
                    if attention_weights is None:
                        attention_weights=self.attention_weights_nets[round](tran_feat.permute(0,2,3,1))
 
                    # attention_weights=torch.randn(B*N,H,W,num_points_expand).to(coor.device)
                    
                    if (self.lift_attn_new or self.lift_attn_with_ori_feat) and round>0:
                        
                        
                        attention_weights=attention_weights.permute(0,1,4,2,3)
                        depth=depth.unsqueeze(2)*attention_weights
                        depth=depth.reshape(B*N,depth.shape[1]*depth.shape[2],*depth.shape[3:]).contiguous() 
                    else:
                        if self.num_points_adap2depth:
                            if self.num_points_adap2depth_gai:
                                attention_weights=attention_weights.reshape(*attention_weights.shape[:-1],-1,self.D)
                                attention_weights = attention_weights.softmax(-2).flatten(-2)
                            else:
                                attention_weights=attention_weights.reshape(*attention_weights.shape[:-1],self.D,-1)
                                attention_weights = attention_weights.softmax(-1).flatten(-2)
                            attention_weights=attention_weights.permute(0,3,1,2)
                            depth=depth.repeat(1,attention_weights.shape[1]//depth.shape[1],1,1)*attention_weights
                            
                        else:
                            attention_weights = attention_weights.softmax(-1)
                            attention_weights=attention_weights.permute(0,3,1,2)
                            depth=depth.unsqueeze(1)*attention_weights.unsqueeze(2)
                            depth=depth.reshape(B*N,depth.shape[1]*depth.shape[2],*depth.shape[3:]).contiguous() 
                    
                
                _,_,K,_,_,_=coor.shape
                
                
                # coor=coor-0.5###########################
                # coor=coor.reshape(-1,3)
                # weight,coor=deformable_lift(coor)
                # coor=coor.reshape(B,N,K,H,W,8,3)
                # coor=coor.permute(0,1,2,5,3,4,6)
                # coor=coor.reshape(B,N,coor.shape[2]*coor.shape[3],*coor.shape[4:]).contiguous()
            coor = ((coor - self.grid_lower_bound.to(coor)) /grid_interval.to(coor))
            
            
            
            
            if self.deformable_lift:
                coor_before_deform=coor.clone()
                weight,coor=self.deformable_lift_func(coor)
                
                weight=weight.reshape(B*N,K,H,W,8)
                depth=depth.unsqueeze(-1)*weight
                depth=depth.permute(0,1,4,2,3)
                depth=depth.reshape(B*N,depth.shape[1]*depth.shape[2],*depth.shape[3:]).contiguous()
            else:
                coor_before_deform=coor
            
            if self.coor_reproject:
                
                coor_ = ((coor - self.grid_lower_bound.to(coor)) /grid_interval.to(coor))
                coor_ = coor_.long()
                coor_=coor_*grid_interval.to(coor)+self.grid_lower_bound.to(coor)
                
                coor_=self.get_lidar_coor(*input[1:7],self.coor_reproject,coor_)
                mask=(coor_[...,0]>=0)&(coor_[...,0]<self.input_size_[1])&(coor_[...,1]>=0)&(coor_[...,1]<self.input_size_[0])
                
                tran_feat=tran_feat.unsqueeze(1).repeat(1,coor_.shape[2],1,1,1)
                tran_feat=tran_feat.reshape(B*N*coor_.shape[2],*tran_feat.shape[2:])
                # coor_reproject=coor_[mask].reshape(1,1,1,3)[...,:2]
                # import pdb;pdb.set_trace()
                coor_norm=coor_[...,:2]
                coor_norm[..., 0] = coor_[..., 0] / (self.input_size_[1] - 1.0) * 2.0 - 1.0
                coor_norm[..., 1] = coor_[..., 1] / (self.input_size_[0] - 1.0) * 2.0 - 1.0
                coor_norm=coor_norm.reshape(B*N*coor_.shape[2],*coor_norm.shape[3:])
                
                tran_feat_samp = F.grid_sample(tran_feat, coor_norm,align_corners=True,padding_mode='zeros')
                tran_feat_samp=tran_feat_samp.reshape(B,N,coor_.shape[2],*tran_feat_samp.shape[1:])
                
                if self.reproject_mix:
                    tran_feat=tran_feat.reshape(B,N,coor.shape[2],*tran_feat.shape[1:]).permute(0,1,2,4,5,3)
                    tran_feat_samp=tran_feat_samp.permute(0,1,2,4,5,3)
                    tran_feat_samp[~mask]=tran_feat[~mask]
                    tran_feat_samp=tran_feat_samp.permute(0,1,2,5,3,4)
                tran_feat=tran_feat_samp.permute(0,1,3,2,4,5)
                
                tran_feat=tran_feat.reshape(tran_feat.shape[0],tran_feat.shape[1],tran_feat.shape[2],tran_feat.shape[3]*tran_feat.shape[4],tran_feat.shape[5])
                depth=depth.reshape(B,N,1,depth.shape[1]*depth.shape[2],depth.shape[-1])
                coor=coor.reshape(B,N,1,coor.shape[2]*coor.shape[3],*coor.shape[4:])
                bev_feat = self.voxel_pooling_v2(coor, depth,tran_feat,grid_size=grid_size)
                return bev_feat, depth

            if self.toSet:
                bev_feat = self.voxel_pooling_v2(
                coor, depth.view(B, N, depth.shape[1], H, W,-1),
                tran_feat.view(B, N, self.out_channels, H, W),occ2set=occ2set,grid_size=grid_size)
            else:
                bev_feat = self.voxel_pooling_v2(
                    coor, depth.view(B, N,  depth.shape[1], H, W),
                    tran_feat.view(B, N, tran_feat.shape[1],tran_feat.shape[2], tran_feat.shape[3]),grid_size=grid_size)
        if self.use_cross_attention:
        
            # import pdb;pdb.set_trace()
            bev_mask=None
            bev_feat = bev_feat.permute(0, 1, 3, 4, 2)
            bev_feat_refined = self.backward_projection([tran_feat.view(B, N, tran_feat.shape[1],tran_feat.shape[2], tran_feat.shape[3])],
                                        img_metas=None,
                                        lss_bev=bev_feat.mean(-1),
                                        cam_params=input[1:7],
                                        bev_mask=bev_mask,
                                        gt_bboxes_3d=None, # debug
                                        pred_img_depth=depth.view(B, N,  depth.shape[1], H, W))  
                                        
            if self.readd:
                bev_feat = bev_feat_refined[..., None] + bev_feat
                
            else:
                bev_feat = bev_feat_refined  
            bev_feat=bev_feat.permute(0,1,4,2,3) 
        if self.lift_attn is not None:
            if 'pos_depth' not in locals().keys():
                pos_depth=None
            return bev_feat,depth,coor,coor_before_deform,pos_depth
        else:

            return bev_feat, [depth]

    def view_transform(self, input, depth, tran_feat,occ2set=None,gt_occ=None,grid_interval=torch.Tensor([0.4,0.4,0.4]),img_metas=None,key_frame=True,
                       self_bev_feat=None,fuse_self_round=0,bev_feat_args_last_phase=None):
        if self.accelerate:
            self.pre_compute(input)
        # import pdb;pdb.set_trace()
        fuse_args=[input, depth, tran_feat]
        if self.lift_attn is not None:
            # import pdb;pdb.set_trace()
            if self.lift_attn_new:
                
                if self.supervise_intermedia:
                    inter_occs=[]
                B, N, C, H, W = input[0].shape
                
                if self.depth2occ:
                    occ_weight_,depth,occ=depth
                    weight_trans=occ
                else:
                    weight_trans=depth
                depths=[depth]
                num_points_expand=self.num_points_lift_attn[0] if self.num_points_lift_attn is not None else 0
                
                # import pdb;pdb.set_trace()
                if self.lift_attn_pre_img_conv:
                    V=self.pre_img_conv[0](tran_feat)
                else:
                    V=tran_feat
                bev_feat,weight,coor,coor_before_deform,pos_depth=self.view_transform_core(input, weight_trans,V,occ2set,round=0,num_points_expand=num_points_expand)
                
                for i in range(self.lift_attn_round):
                    bev_feat=self.lift_pre_process_nets[i](bev_feat)
                
                    bev_feat=bev_feat[0]
                    if self.supervise_intermedia:
                        inter_occ=self.inter_predictor[i](bev_feat.permute(0, 4, 3, 2, 1))
                        if self.splat_with_occ:
                            density=inter_occ[...,-1:].sigmoid().permute(0, 4, 3, 2, 1)
                        inter_occs.append(inter_occ)
                    
                    
                    D=coor_before_deform.shape[2]
                    coor_before_deform=coor_before_deform.reshape(B,N,D,H*W,3)
               
                    coor_before_deform=coor_before_deform/(torch.flip(torch.tensor(bev_feat.shape[2:]).to(coor_before_deform.device),[-1])-1)
                    coor_before_deform=coor_before_deform*2-1
                    if self.splat_with_occ:
                        occ_weight = F.grid_sample(density, coor_before_deform,align_corners=True,padding_mode='zeros')#torch.Size([2, 1, 6, 88, 704])
                
                        free_weight=1-occ_weight
                        cum_free_weight=torch.cumprod(free_weight,dim=-2)
                        cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[...,0:1,:]),cum_free_weight[...,:-1,:]],dim=-2)
                        splat_weight=occ_weight*cum_free_weight
                        
                        splat_weight=splat_weight.reshape(B*N,D,H,W)
                    
                    vox_sampled=F.grid_sample(bev_feat, coor_before_deform,align_corners=True,padding_mode='zeros')#B,C,N,D,H*W
                    

                                       
                    vox_sampled=vox_sampled.permute(0,2,3,4,1).reshape(B*N,D,H,W,-1)#torch.Size([24, 88, 16, 44, 32])
                    if self.lift_attn_pre_img_conv:
                        V=self.pre_img_conv[i+1](tran_feat)
                    else:
                        V=tran_feat
                    if not self.splat_with_occ_only:
                        if self.attn_with_pos:
                            # import pdb;pdb.set_trace()
                            img_pos=self.get_lidar_coor_single_depth(pos_depth,self.input_size_,self.downsample,*input[1:7])
                            img_pos = (img_pos - self.grid_lower_bound.to(img_pos)) /(self.grid_interval*self.lift_attn_downsample).to(img_pos)
                            img_pos/=(self.grid_interval*self.lift_attn_downsample).to(img_pos)
                            img_pos=pos2posemb3d(img_pos,num_pos_feats=vox_sampled.shape[-1]//2)
                            pos_embed=self.pos_embedding(img_pos.reshape(-1,img_pos.shape[-1])).reshape(B*N,H,W,-1).permute(0,3,1,2)
                            vox_pos=pos2posemb3d((coor_before_deform+1)/2,num_pos_feats=vox_sampled.shape[-1]//2)
                            vox_embed=self.pos_embedding(vox_pos.reshape(-1,vox_pos.shape[-1])).reshape(B*N,D,H,W,-1)
                            Q=self.Q_net[i](vox_sampled+vox_embed)
                            K=self.K_net[i]((V+pos_embed).permute(0,2,3,1))
                            V=self.V_net[i]((V+pos_embed).permute(0,2,3,1)).permute(0,3,1,2)
                            ###################################################
                            
                        else:
                            Q=self.Q_net[i](vox_sampled)
                            K=self.K_net[i](V.permute(0,2,3,1))
                            V=self.V_net[i](V.permute(0,2,3,1)).permute(0,3,1,2)
                        attn_weight=torch.einsum('bdwhc,bwhc->bdwh',Q,K)
                    
                        # import pdb;pdb.set_trace()
                        if self.zero2inf:
                            attn_weight[attn_weight==0]=-float('inf')
                        if self.attn_weight_temp:
                            attn_weight*=self.attn_weight_temp
                        if self.attn_weight_sigmoid:
                            attn_weight=attn_weight.sigmoid()
                        else:
                            attn_weight=attn_weight.softmax(1)
                    
                    if self.splat_with_occ:
                        if self.splat_with_occ_only:
                            attn_weight=splat_weight
                        else:
                            attn_weight=(attn_weight+splat_weight)/2
                    if self.attn_depth_mix:
                        weight_trans=(attn_weight+depth)/2
                    else:
                        weight_trans=attn_weight
                    if self.sup_attn_weight:
                        depths.append(attn_weight)
                        
                    num_points_expand=self.num_points_lift_attn[i+1] if self.num_points_lift_attn is not None else 0
                    if num_points_expand:
                        # import pdb;pdb.set_trace()
                        sampling_offsets=self.sampling_offsets_nets[i+1](vox_sampled)
                        sampling_offsets=sampling_offsets.reshape(B,N,self.D,H,W,num_points_expand,self.num_coor)
                        
                        if self.sampling_weights_from_vox:
                            density=inter_occ[...,-1:].sigmoid().permute(0, 4, 3, 2, 1)
                            offsets=sampling_offsets.reshape(B,N,self.D,H*W,num_points_expand,self.num_coor)
                            if self.sampling_offsets_weight is not None:
                                offsets=offsets*self.sampling_offsets_weight.to(offsets.device)
                            sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
                            offsets=self.img_to_ego_coor(offsets,sensor2ego,post_trans,post_rots,cam2imgs,bda)
                            
                        # offsets=coor_before_deform.reshape(B,N,D,H*W,3)
               
                            offsets=offsets/(torch.flip(torch.tensor(bev_feat.shape[2:]).to(offsets.device),[-1])-1)
                            offsets=offsets*2-1
                            offsets=offsets+coor_before_deform.unsqueeze(-2)
                            offsets=offsets.reshape(B,N,self.D,H*W*num_points_expand,self.num_coor)
                            sampling_weights=F.grid_sample(density, offsets,align_corners=True,padding_mode='zeros')
                            sampling_weights=sampling_weights.reshape(B*N,self.D,H,W,num_points_expand)
                        
                        else:
                            sampling_weights=self.attention_weights_nets[i+1](vox_sampled)
                            sampling_weights = sampling_weights.softmax(-1)
                        # import pdb;pdb.set_trace()
                    else:
                        sampling_offsets=None
                        sampling_weights=None
                        
                    
                    bev_feat_refined,weight,coor,coor_before_deform,pos_depth=self.view_transform_core(input, weight_trans, V,occ2set,round=i+1,num_points_expand=num_points_expand,
                                                                                                       sampling_offsets=sampling_offsets,attention_weights=sampling_weights,
                                                                                                       grid_size=self.grid_size//self.lift_attn_downsample,grid_interval=self.grid_interval*self.lift_attn_downsample)
                    # import pdb;pdb.set_trace()
                    if torch.any(self.lift_attn_downsample!= torch.Tensor([1,1,1])):
                        # import pdb;pdb.set_trace()
                        b,c,d,h,w=bev_feat_refined.shape
                        bev_feat_downsamp=self.downsample_ffn[i](bev_feat_refined.permute(0,4,3,2,1)).permute(0,4,3,2,1)
                        bev_feat_downsamp=bev_feat_downsamp.permute(0,2,1,3,4).reshape(b*d,-1,h,w)
                        # bev_feat_downsamp=self.downsample_net(bev_feat_downsamp)
                        bev_feat_downsamp=self.downsample_neck[i](bev_feat_downsamp.reshape(b,d,-1,h,w).permute(0,2,1,3,4))
                        bev_feat_upsample=self.upsample_net[i](bev_feat_downsamp[0].permute(0,2,1,3,4).reshape(b*d,-1,h,w))
                        bev_feat_refined=bev_feat_upsample.reshape(b,bev_feat.shape[2],bev_feat.shape[1],*bev_feat.shape[3:]).permute(0,2,1,3,4)
                   
                
                    if self.add_ffn_norm:
                        # import pdb;pdb.set_trace()
                        bev_feat_refined=bev_feat_refined.permute(0, 4,3,2,1)
                        bev_feat_refined=self.pre_ffn[i](bev_feat_refined)
                        bev_feat=bev_feat_refined+bev_feat.permute(0, 4,3,2,1)
                        bev_feat=self.pre_norm[i](bev_feat)
                        bev_feat=self.post_ffn[i](bev_feat)
                        bev_feat=self.post_norm[i](bev_feat)
                        bev_feat=bev_feat.permute(0, 4,3,2,1)
                    elif self.attn_lift_post_conv:
                        bev_feat=self.attn_lift_post_net[i](torch.cat([bev_feat_refined,bev_feat],dim=1).permute(0, 4,3,2,1)).permute(0, 4,3,2,1)
                           
                    else:
                        bev_feat=bev_feat_refined+bev_feat
                    
                if self.supervise_intermedia:
                    depth=[depths,inter_occs]
                else:
                    depth=depths
            
                
            
            elif self.lift_attn_with_ori_feat_add:
                
                if self.supervise_intermedia:
                    inter_occs=[]
                B, N, C, H, W = input[0].shape
                # import pdb;pdb.set_trace()
                if self.depth2occ:
                    occ_weight_,depth,occ=depth
                    weight_trans=occ
                else:
                    weight_trans=depth
                num_points_expand=self.num_points_lift_attn[0] if self.num_points_lift_attn is not None else 0
                
                if fuse_self_round==0:
                    # import pdb;pdb.set_trace()
                    if self.fill_all_vox and not self.first_phase_non_all_vox:
                        # import pdb;pdb.set_trace()
                        bev_feat,_=self.view_transform_fill_all(input, depth, tran_feat,occ2set,num_points_expand=self.num_points,grid_size=self.grid_size//self.vox_upsample_scale,gt_occ=gt_occ)
                        weight=depth
                        coor = self.frustum
                        # import pdb;pdb.set_trace()
                        sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
                        coor = self.img_to_ego_coor(coor,sensor2ego,post_trans,post_rots,cam2imgs,bda)
                        coor = ((coor - self.grid_lower_bound.to(coor)) /grid_interval.to(coor))
                        coor_before_deform=coor
                        pos_depth=None
                        
                    else:
                        bev_feat,weight,coor,coor_before_deform,pos_depth=self.view_transform_core(input, weight_trans, tran_feat,occ2set,round=0,num_points_expand=num_points_expand)
                    # if self.lift_attn_with_ori_feat:
                    # if self.fuse_self and self_bev_feat is None:
                    depth=[depth,None]
                    bev_feat_args=[bev_feat,weight,coor,coor_before_deform,pos_depth,gt_occ]
                    fuse_args.append(bev_feat_args)
                    depth=[depth,fuse_args]
                    return bev_feat,depth
                else:
                    # import pdb;pdb.set_trace()
                    depths=[depth]
                    bev_feat,weight,coor,coor_before_deform,pos_depth,gt_occ=bev_feat_args_last_phase
                    # for i in range(self.lift_attn_round):
                        # if self.lift_attn_simple_add:
                        #     bev_feat_inter=self.lift_pre_process_nets[i](bev_feat)[0]
                        #     inter_occs=[self.inter_predictor(bev_feat_inter.permute(0, 4, 3, 2, 1))]
                        #     if self.simple_not_add:
                        #         bev_feat=bev_feat_inter+bev_feat
                        #     elif self.simple_use_post_cov:
                        #         bev_feat=bev_feat_inter
                        #     return bev_feat,[depths,inter_occs]
                        
                    bev_feat=self.lift_pre_process_nets[fuse_self_round-1](bev_feat)
                    if self.lift_attn_with_complex_conv:
                        bev_feat=self.complex_conv[fuse_self_round-1](bev_feat[0])
                    else:
                        bev_feat=bev_feat[0]
                    # import pdb;pdb.set_trace()
                    if self.fuse_his_attn and key_frame:
                        bev_feat=self.fuse_history(bev_feat,img_metas,input[6],update_history=False)
                    if self.fuse_self and self_bev_feat is not None:
                        # import pdb;pdb.set_trace()
                        bev_feat=self.fuse_self_net[fuse_self_round-1](torch.cat([bev_feat,self_bev_feat],dim=1))
                        
                    bev_mask=None
                    # import pdb;pdb.set_trace()
                    
                        
                    # img_pos=self.get_lidar_coor_single_depth(pos_depth,self.input_size_,self.downsample,*input[1:7])
                    # # import pdb;pdb.set_trace()
                    # img_pos = ((img_pos - self.grid_lower_bound.to(img_pos)) /(self.grid_interval*self.lift_attn_downsample).to(img_pos))
                    # img_pos/=(self.grid_interval*self.lift_attn_downsample).to(img_pos)#####################
                    # import pdb;pdb.set_trace()
                    tran_feat = self.lift_attn[fuse_self_round-1](tran_feat.view(B, N, tran_feat.shape[1],tran_feat.shape[2], tran_feat.shape[3]),
                                                img_pos=None,
                                                vox_feats=bev_feat,
                                                weight=weight,
                                                coor=coor,
                                                coor_before_deform=coor_before_deform,
                                                img_metas=None,
                                                lss_bev=bev_feat,
                                                cam_params=input[1:7],
                                                bev_mask=bev_mask,
                                                gt_occ=gt_occ,
                                                key_frame=key_frame,
                                                # gt_bboxes_3d=None, # debug
                                                # pred_img_depth=depth.view(B, N,  depth.shape[1], H, W)
                                                )  
                    if self.supervise_intermedia:
                        tran_feat,inter_occ=tran_feat
                        if self.lift_attn_with_ori_feat:
                            inter_occ,depth_weight=inter_occ
                            if self.add_occ_depth_loss:
                                depth_weight,depth_weight_gt=depth_weight
                        inter_occs.append(inter_occ)
                        if self.lift_attn_with_ori_feat:
                            num_points_expand=self.num_points_lift_attn[fuse_self_round] if self.num_points_lift_attn is not None else 0
                            if num_points_expand:
                                # import pdb;pdb.set_trace()
                                D=coor_before_deform.shape[2]
                                coor_before_deform=coor_before_deform.reshape(B,N,D,H*W,3)
                        
                                coor_before_deform=coor_before_deform/(torch.flip(torch.tensor(bev_feat.shape[2:]).to(coor_before_deform.device),[-1])-1)
                                coor_before_deform=coor_before_deform*2-1
                                if self.splat_with_occ:
                                    occ_weight = F.grid_sample(density, coor_before_deform,align_corners=True,padding_mode='zeros')#torch.Size([2, 1, 6, 88, 704])
                            
                                    free_weight=1-occ_weight
                                    cum_free_weight=torch.cumprod(free_weight,dim=-2)
                                    cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[...,0:1,:]),cum_free_weight[...,:-1,:]],dim=-2)
                                    splat_weight=occ_weight*cum_free_weight
                                    
                                    splat_weight=splat_weight.reshape(B*N,D,H,W)
                                
                                vox_sampled=F.grid_sample(bev_feat, coor_before_deform,align_corners=True,padding_mode='zeros')#B,C,N,D,H*W
                                

                                                
                                vox_sampled=vox_sampled.permute(0,2,3,4,1).reshape(B*N,D,H,W,-1)#torch.Size([24, 88, 16, 44, 32])
                                sampling_offsets=self.sampling_offsets_nets[fuse_self_round](vox_sampled)
                                sampling_offsets=sampling_offsets.reshape(B,N,self.D,H,W,num_points_expand,self.num_coor)
                                
                                if self.sampling_weights_from_vox:
                                    density=inter_occ[...,-1:].sigmoid().permute(0, 4, 3, 2, 1)
                                    offsets=sampling_offsets.reshape(B,N,self.D,H*W,num_points_expand,self.num_coor)
                                    if self.sampling_offsets_weight is not None:
                                        offsets=offsets*self.sampling_offsets_weight.to(offsets.device)
                                    sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
                                    offsets=self.img_to_ego_coor(offsets,sensor2ego,post_trans,post_rots,cam2imgs,bda)
                                    
                                # offsets=coor_before_deform.reshape(B,N,D,H*W,3)
                    
                                    offsets=offsets/(torch.flip(torch.tensor(bev_feat.shape[2:]).to(offsets.device),[-1])-1)
                                    offsets=offsets*2-1
                                    offsets=offsets+coor_before_deform.unsqueeze(-2)
                                    offsets=offsets.reshape(B,N,self.D,H*W*num_points_expand,self.num_coor)
                                    sampling_weights=F.grid_sample(density, offsets,align_corners=True,padding_mode='zeros')
                                    sampling_weights=sampling_weights.reshape(B*N,self.D,H,W,num_points_expand)
                                    # import pdb;pdb.set_trace()
                                else:
                                    sampling_weights=self.attention_weights_nets[fuse_self_round](vox_sampled)
                                    sampling_weights = sampling_weights.softmax(-1)
                                # import pdb;pdb.set_trace()
                            else:
                                sampling_offsets=None
                                sampling_weights=None
                            if self.fill_all_vox:
                                # import pdb;pdb.set_trace()
                                bev_feat,_=self.view_transform_fill_all(input, depth_weight, tran_feat.reshape(B*N,*tran_feat.shape[2:]),occ2set,num_points_expand=self.num_points,grid_size=self.grid_size//self.vox_upsample_scale,gt_occ=gt_occ)
                                weight=depth
                                coor = self.frustum
                                # import pdb;pdb.set_trace()
                                sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
                                coor = self.img_to_ego_coor(coor,sensor2ego,post_trans,post_rots,cam2imgs,bda)
                                coor = ((coor - self.grid_lower_bound.to(coor)) /grid_interval.to(coor))
                                coor_before_deform=coor
                                pos_depth=None
                            else:
                                # import pdb;pdb.set_trace()
                                bev_feat,weight,coor,coor_before_deform,pos_depth=self.view_transform_core(input, depth_weight, tran_feat.reshape(B*N,*tran_feat.shape[2:]),occ2set,round=fuse_self_round,num_points_expand=num_points_expand,
                                                                                                            sampling_offsets=sampling_offsets,attention_weights=sampling_weights,
                                                                                                            grid_size=self.grid_size//self.lift_attn_downsample,grid_interval=self.grid_interval*self.lift_attn_downsample)
                            # fuse_args.append(bev_feat)
                            if self.add_occ_depth_loss:
                                depth=[depths,inter_occs,depth_weight,depth_weight_gt]
                                if self.fuse_his_attn and key_frame:
                                    depth=[depth,input[6]]
                                if self.fuse_self :
                                    bev_feat_args=[bev_feat,weight,coor,coor_before_deform,pos_depth]
                                    fuse_args.append(bev_feat_args)
                                    depth=[depth,fuse_args]
                                return bev_feat,depth
                            else:
                                depth=[depths,inter_occs]
                                if self.fuse_his_attn and key_frame:
                                    depth=[depth,input[6]]
                                if self.fuse_self :
                                #     depth=[depth,fuse_args]
                                # import pdb;pdb.set_trace()
                                
                                # depth=[depth,None]
                                    bev_feat_args=[bev_feat,weight,coor,coor_before_deform,pos_depth]
                                    fuse_args.append(bev_feat_args)
                                    depth=[depth,fuse_args]
                                
                                return bev_feat,depth
                            
                            
                    # depth_before=depth if self.lift_attn_not_detach_depth else depth.detach()
                    # if self.occ2_depth_use_occ:
                        
                    #     depth=(inverse_sigmoid(depth_before)+self.simple_depth_nets[i](tran_feat)).sigmoid()
                    # else:
                    #     depth=(depth_before.log()+self.simple_depth_nets[i](tran_feat)).softmax(1)
                    # # import pdb;pdb.set_trace()
                    # num_points_expand=self.num_points_lift_attn[i+1] if self.num_points_lift_attn is not None else 0
                    # if num_points_expand:
                    #     # import pdb;pdb.set_trace()
                    #     sampling_offsets=self.sampling_offsets_nets[i+1](vox_sampled)
                    #     sampling_offsets=sampling_offsets.reshape(B,N,self.D,H,W,num_points_expand,self.num_coor)
                        
                    #     if self.sampling_weights_from_vox:
                    #         density=inter_occ[...,-1:].sigmoid().permute(0, 4, 3, 2, 1)
                    #         offsets=sampling_offsets.reshape(B,N,self.D,H*W,num_points_expand,self.num_coor)
                    #         if self.sampling_offsets_weight is not None:
                    #             offsets=offsets*self.sampling_offsets_weight.to(offsets.device)
                    #         sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
                    #         offsets=self.img_to_ego_coor(offsets,sensor2ego,post_trans,post_rots,cam2imgs,bda)
                            
                    #     # offsets=coor_before_deform.reshape(B,N,D,H*W,3)
                
                    #         offsets=offsets/(torch.flip(torch.tensor(bev_feat.shape[2:]).to(offsets.device),[-1])-1)
                    #         offsets=offsets*2-1
                    #         offsets=offsets+coor_before_deform.unsqueeze(-2)
                    #         offsets=offsets.reshape(B,N,self.D,H*W*num_points_expand,self.num_coor)
                    #         sampling_weights=F.grid_sample(density, offsets,align_corners=True,padding_mode='zeros')
                    #         sampling_weights=sampling_weights.reshape(B*N,self.D,H,W,num_points_expand)
                        
                    #     else:
                    #         sampling_weights=self.attention_weights_nets[i+1](vox_sampled)
                    #         sampling_weights = sampling_weights.softmax(-1)
                    #     # import pdb;pdb.set_trace()
                    # else:
                    #     sampling_offsets=None
                    #     sampling_weights=None
                        
                    # if self.depth2occ:
                    #     occ=cal_depth2occ(occ_weight_,depth)
                    #     weight_trans=occ
                    # else:
                    #     weight_trans=depth
                    # bev_feat_refined,weight,coor,coor_before_deform,pos_depth=self.view_transform_core(input, weight_trans, tran_feat,occ2set,round=i+1,num_points_expand=num_points_expand,
                    #                                                                                     grid_size=self.grid_size//self.lift_attn_downsample,grid_interval=self.grid_interval*self.lift_attn_downsample)
                    
                    # if torch.any(self.lift_attn_downsample!= torch.Tensor([1,1,1])):
                        
                    #     b,c,d,h,w=bev_feat_refined.shape
                    #     bev_feat_downsamp=self.downsample_ffn[i](bev_feat_refined.permute(0,4,3,2,1)).permute(0,4,3,2,1)
                    #     bev_feat_downsamp=bev_feat_downsamp.permute(0,2,1,3,4).reshape(b*d,-1,h,w)
                    #     # bev_feat_downsamp=self.downsample_net(bev_feat_downsamp)
                    #     bev_feat_downsamp=self.downsample_neck[i](bev_feat_downsamp.reshape(b,d,-1,h,w).permute(0,2,1,3,4))
                    #     bev_feat_upsample=self.upsample_net[i](bev_feat_downsamp[0].permute(0,2,1,3,4).reshape(b*d,-1,h,w))
                    #     bev_feat=bev_feat_upsample.reshape(b,bev_feat.shape[2],bev_feat.shape[1],*bev_feat.shape[3:]).permute(0,2,1,3,4)+bev_feat
                    # else:
                    #     if self.lift_attn_norm_add:
                    #         norm_refined=bev_feat_refined.norm(dim=1,keepdim=True)
                    #         norm=bev_feat.norm(dim=1,keepdim=True)
                    #         bev_feat_refined=bev_feat_refined/(norm+norm_refined+1e-6)*norm_refined
                    #         bev_feat=bev_feat/(norm+norm_refined+1e-6)*norm
                    #     bev_feat=bev_feat_refined+bev_feat
                        
                    # depths.append(depth)
                    
                    # if self.supervise_intermedia:
                    #     depth=[depths,inter_occs]
                    # else:
                    #     depth=depths
            
            
            else:
                if self.supervise_intermedia:
                    inter_occs=[]
                B, N, C, H, W = input[0].shape
                # import pdb;pdb.set_trace()
                if self.depth2occ:
                    occ_weight_,depth,occ=depth
                    weight_trans=occ
                else:
                    weight_trans=depth
                num_points_expand=self.num_points_lift_attn[0] if self.num_points_lift_attn is not None else 0
                
                if self.fill_all_vox:
                    # import pdb;pdb.set_trace()
                    bev_feat,_=self.view_transform_fill_all(input, depth, tran_feat,occ2set,num_points_expand=self.num_points,grid_size=self.grid_size//self.vox_upsample_scale,gt_occ=gt_occ)
                    weight=depth
                else:
                    bev_feat,weight,coor,coor_before_deform,pos_depth=self.view_transform_core(input, weight_trans, tran_feat,occ2set,round=0,num_points_expand=num_points_expand)
                if self.lift_attn_with_ori_feat:
                    if self.fuse_self and self_bev_feat is None:
                        depth=[depth,None]
                        depth=[depth,fuse_args]
                        return bev_feat,depth
                    
                depths=[depth]
                for i in range(self.lift_attn_round):
                    if self.lift_attn_simple_add:
                        bev_feat_inter=self.lift_pre_process_nets[i](bev_feat)[0]
                        inter_occs=[self.inter_predictor(bev_feat_inter.permute(0, 4, 3, 2, 1))]
                        if self.simple_not_add:
                            bev_feat=bev_feat_inter+bev_feat
                        elif self.simple_use_post_cov:
                            bev_feat=bev_feat_inter
                        return bev_feat,[depths,inter_occs]

                    
                    bev_feat=self.lift_pre_process_nets[i](bev_feat)
                    if self.lift_attn_with_complex_conv:
                        bev_feat=self.complex_conv[i](bev_feat[0])
                    else:
                        bev_feat=bev_feat[0]
                    # import pdb;pdb.set_trace()
                    if self.fuse_his_attn and key_frame:
                        bev_feat=self.fuse_history(bev_feat,img_metas,input[6],update_history=False)
                    if self.fuse_self and self_bev_feat is not None:
                        bev_feat=self.fuse_self_net(torch.cat([bev_feat,self_bev_feat],dim=1))
                        
                    bev_mask=None
                    # import pdb;pdb.set_trace()
                    
                        
                    img_pos=self.get_lidar_coor_single_depth(pos_depth,self.input_size_,self.downsample,*input[1:7])
                    # import pdb;pdb.set_trace()
                    img_pos = ((img_pos - self.grid_lower_bound.to(img_pos)) /(self.grid_interval*self.lift_attn_downsample).to(img_pos))
                    img_pos/=(self.grid_interval*self.lift_attn_downsample).to(img_pos)#####################
                    tran_feat = self.lift_attn[i](tran_feat.view(B, N, tran_feat.shape[1],tran_feat.shape[2], tran_feat.shape[3]),
                                                img_pos=img_pos,
                                                vox_feats=bev_feat,
                                                weight=weight,
                                                coor=coor,
                                                coor_before_deform=coor_before_deform,
                                                img_metas=None,
                                                lss_bev=bev_feat,
                                                cam_params=input[1:7],
                                                bev_mask=bev_mask,
                                                gt_occ=gt_occ,
                                                # gt_bboxes_3d=None, # debug
                                                # pred_img_depth=depth.view(B, N,  depth.shape[1], H, W)
                                                )  
                    if self.supervise_intermedia:
                        tran_feat,inter_occ=tran_feat
                        if self.lift_attn_with_ori_feat:
                            inter_occ,depth_weight=inter_occ
                            if self.add_occ_depth_loss:
                                depth_weight,depth_weight_gt=depth_weight
                        inter_occs.append(inter_occ)
                        if self.lift_attn_with_ori_feat:
                            num_points_expand=self.num_points_lift_attn[i+1] if self.num_points_lift_attn is not None else 0
                            if num_points_expand:
                                # import pdb;pdb.set_trace()
                                D=coor_before_deform.shape[2]
                                coor_before_deform=coor_before_deform.reshape(B,N,D,H*W,3)
                        
                                coor_before_deform=coor_before_deform/(torch.flip(torch.tensor(bev_feat.shape[2:]).to(coor_before_deform.device),[-1])-1)
                                coor_before_deform=coor_before_deform*2-1
                                if self.splat_with_occ:
                                    occ_weight = F.grid_sample(density, coor_before_deform,align_corners=True,padding_mode='zeros')#torch.Size([2, 1, 6, 88, 704])
                            
                                    free_weight=1-occ_weight
                                    cum_free_weight=torch.cumprod(free_weight,dim=-2)
                                    cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[...,0:1,:]),cum_free_weight[...,:-1,:]],dim=-2)
                                    splat_weight=occ_weight*cum_free_weight
                                    
                                    splat_weight=splat_weight.reshape(B*N,D,H,W)
                                
                                vox_sampled=F.grid_sample(bev_feat, coor_before_deform,align_corners=True,padding_mode='zeros')#B,C,N,D,H*W
                                

                                                
                                vox_sampled=vox_sampled.permute(0,2,3,4,1).reshape(B*N,D,H,W,-1)#torch.Size([24, 88, 16, 44, 32])
                                sampling_offsets=self.sampling_offsets_nets[i+1](vox_sampled)
                                sampling_offsets=sampling_offsets.reshape(B,N,self.D,H,W,num_points_expand,self.num_coor)
                                
                                if self.sampling_weights_from_vox:
                                    density=inter_occ[...,-1:].sigmoid().permute(0, 4, 3, 2, 1)
                                    offsets=sampling_offsets.reshape(B,N,self.D,H*W,num_points_expand,self.num_coor)
                                    if self.sampling_offsets_weight is not None:
                                        offsets=offsets*self.sampling_offsets_weight.to(offsets.device)
                                    sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
                                    offsets=self.img_to_ego_coor(offsets,sensor2ego,post_trans,post_rots,cam2imgs,bda)
                                    
                                # offsets=coor_before_deform.reshape(B,N,D,H*W,3)
                    
                                    offsets=offsets/(torch.flip(torch.tensor(bev_feat.shape[2:]).to(offsets.device),[-1])-1)
                                    offsets=offsets*2-1
                                    offsets=offsets+coor_before_deform.unsqueeze(-2)
                                    offsets=offsets.reshape(B,N,self.D,H*W*num_points_expand,self.num_coor)
                                    sampling_weights=F.grid_sample(density, offsets,align_corners=True,padding_mode='zeros')
                                    sampling_weights=sampling_weights.reshape(B*N,self.D,H,W,num_points_expand)
                                    # import pdb;pdb.set_trace()
                                else:
                                    sampling_weights=self.attention_weights_nets[i+1](vox_sampled)
                                    sampling_weights = sampling_weights.softmax(-1)
                                # import pdb;pdb.set_trace()
                            else:
                                sampling_offsets=None
                                sampling_weights=None
                            bev_feat,weight,coor,coor_before_deform,pos_depth=self.view_transform_core(input, depth_weight, tran_feat.reshape(B*N,*tran_feat.shape[2:]),occ2set,round=i+1,num_points_expand=num_points_expand,
                                                                                                       sampling_offsets=sampling_offsets,attention_weights=sampling_weights,
                                                                                                       grid_size=self.grid_size//self.lift_attn_downsample,grid_interval=self.grid_interval*self.lift_attn_downsample)
                            if self.add_occ_depth_loss:
                                depth=[depths,inter_occs,depth_weight,depth_weight_gt]
                                if self.fuse_his_attn and key_frame:
                                    depth=[depth,input[6]]
                                if self.fuse_self and self_bev_feat is None:
                                    depth=[depth,fuse_args]
                                return bev_feat,depth
                            else:
                                depth=[depths,inter_occs]
                                if self.fuse_his_attn and key_frame:
                                    depth=[depth,input[6]]
                                if self.fuse_self and self_bev_feat is None:
                                    depth=[depth,fuse_args]
                                # import pdb;pdb.set_trace()
                                return bev_feat,depth
                    depth_before=depth if self.lift_attn_not_detach_depth else depth.detach()
                    if self.occ2_depth_use_occ:
                        
                        depth=(inverse_sigmoid(depth_before)+self.simple_depth_nets[i](tran_feat)).sigmoid()
                    else:
                        depth=(depth_before.log()+self.simple_depth_nets[i](tran_feat)).softmax(1)
                    # import pdb;pdb.set_trace()
                    num_points_expand=self.num_points_lift_attn[i+1] if self.num_points_lift_attn is not None else 0
                    if num_points_expand:
                        # import pdb;pdb.set_trace()
                        sampling_offsets=self.sampling_offsets_nets[i+1](vox_sampled)
                        sampling_offsets=sampling_offsets.reshape(B,N,self.D,H,W,num_points_expand,self.num_coor)
                        
                        if self.sampling_weights_from_vox:
                            density=inter_occ[...,-1:].sigmoid().permute(0, 4, 3, 2, 1)
                            offsets=sampling_offsets.reshape(B,N,self.D,H*W,num_points_expand,self.num_coor)
                            if self.sampling_offsets_weight is not None:
                                offsets=offsets*self.sampling_offsets_weight.to(offsets.device)
                            sensor2ego, ego2global, cam2imgs, post_rots, post_trans,bda=input[1:7]
                            offsets=self.img_to_ego_coor(offsets,sensor2ego,post_trans,post_rots,cam2imgs,bda)
                            
                        # offsets=coor_before_deform.reshape(B,N,D,H*W,3)
               
                            offsets=offsets/(torch.flip(torch.tensor(bev_feat.shape[2:]).to(offsets.device),[-1])-1)
                            offsets=offsets*2-1
                            offsets=offsets+coor_before_deform.unsqueeze(-2)
                            offsets=offsets.reshape(B,N,self.D,H*W*num_points_expand,self.num_coor)
                            sampling_weights=F.grid_sample(density, offsets,align_corners=True,padding_mode='zeros')
                            sampling_weights=sampling_weights.reshape(B*N,self.D,H,W,num_points_expand)
                        
                        else:
                            sampling_weights=self.attention_weights_nets[i+1](vox_sampled)
                            sampling_weights = sampling_weights.softmax(-1)
                        # import pdb;pdb.set_trace()
                    else:
                        sampling_offsets=None
                        sampling_weights=None
                        
                    if self.depth2occ:
                        occ=cal_depth2occ(occ_weight_,depth)
                        weight_trans=occ
                    else:
                        weight_trans=depth
                    bev_feat_refined,weight,coor,coor_before_deform,pos_depth=self.view_transform_core(input, weight_trans, tran_feat,occ2set,round=i+1,num_points_expand=num_points_expand,
                                                                                                       grid_size=self.grid_size//self.lift_attn_downsample,grid_interval=self.grid_interval*self.lift_attn_downsample)
                    
                    if torch.any(self.lift_attn_downsample!= torch.Tensor([1,1,1])):
                        
                        b,c,d,h,w=bev_feat_refined.shape
                        bev_feat_downsamp=self.downsample_ffn[i](bev_feat_refined.permute(0,4,3,2,1)).permute(0,4,3,2,1)
                        bev_feat_downsamp=bev_feat_downsamp.permute(0,2,1,3,4).reshape(b*d,-1,h,w)
                        # bev_feat_downsamp=self.downsample_net(bev_feat_downsamp)
                        bev_feat_downsamp=self.downsample_neck[i](bev_feat_downsamp.reshape(b,d,-1,h,w).permute(0,2,1,3,4))
                        bev_feat_upsample=self.upsample_net[i](bev_feat_downsamp[0].permute(0,2,1,3,4).reshape(b*d,-1,h,w))
                        bev_feat=bev_feat_upsample.reshape(b,bev_feat.shape[2],bev_feat.shape[1],*bev_feat.shape[3:]).permute(0,2,1,3,4)+bev_feat
                    else:
                        if self.lift_attn_norm_add:
                            norm_refined=bev_feat_refined.norm(dim=1,keepdim=True)
                            norm=bev_feat.norm(dim=1,keepdim=True)
                            bev_feat_refined=bev_feat_refined/(norm+norm_refined+1e-6)*norm_refined
                            bev_feat=bev_feat/(norm+norm_refined+1e-6)*norm
                        bev_feat=bev_feat_refined+bev_feat
                    depths.append(depth)
                if self.supervise_intermedia:
                    depth=[depths,inter_occs]
                else:
                    depth=depths
                
        elif self.fill_all_vox:
            # import pdb;pdb.set_trace()
            bev_feat,_=self.view_transform_fill_all(input, depth, tran_feat,occ2set,num_points_expand=self.num_points,grid_size=self.grid_size//self.vox_upsample_scale,gt_occ=gt_occ)
        elif self.fill_all_vox_with_occ:
            # import pdb;pdb.set_trace()
            bev_feat,_=self.view_transform_fill_all_with_occ(input, depth, tran_feat,occ2set,num_points_expand=self.num_points,grid_size=self.grid_size//self.vox_upsample_scale,gt_occ=gt_occ)
        else:   
            bev_feat,_=self.view_transform_core(input, depth, tran_feat,occ2set,num_points_expand=self.num_points,grid_size=self.grid_size//self.vox_upsample_scale)
            if self.supervise_intermedia:
                # import pdb;pdb.set_trace()
                inter_occ=self.inter_predictor(bev_feat.permute(0, 4, 3, 2, 1))
                depth=[depth,[inter_occ]]
        # if (self.vox_upsample_scale!=torch.Tensor([1,1,1])).any():
        #     import pdb;pdb.set_trace()
            # self.vox_upsample1=
        if self.downsample_add:
        
            B,N,D,H,W=bev_feat.shape
            bev_feat_downsamp=bev_feat.permute(0,2,1,3,4).reshape(B*D,N,H,W)
            bev_feat_downsamp=self.downsample_net(bev_feat_downsamp)
            bev_feat_downsamp=self.downsample_neck(bev_feat_downsamp.reshape(B,D,-1,H//4,W//4).permute(0,2,1,3,4))
            bev_feat_upsample=self.upsample_net(bev_feat_downsamp[0].permute(0,2,1,3,4).reshape(B*D,-1,H//4,W//4))
            bev_feat=bev_feat_upsample.reshape(B,D,N,H,W).permute(0,2,1,3,4)+bev_feat
        if self.fuse_his_attn and key_frame:
            depth=[depth,input[6]]
        if self.fuse_self and self_bev_feat is None:
            depth=[depth,fuse_args]
        # import pdb;pdb.set_trace()
        return bev_feat, depth

    def forward(self, input):
        """Transform image-view feature into bird-eye-view feature.

        Args:
            input (list(torch.tensor)): of (image-view feature, rots, trans,
                intrins, post_rots, post_trans)

        Returns:
            torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
        """
        x = input[0]
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depth_net(x)

        depth_digit = x[:, :self.D, ...]
        tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        depth = depth_digit.softmax(dim=1)
        return self.view_transform(input, depth, tran_feat)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None
    
    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda,update_history=False): # align features with 3d shift
        
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        # if voxel_feat:
        #     curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
        
        seq_ids = torch.LongTensor([
            single_img_metas['sequence_group_idx'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        start_of_sequence = torch.BoolTensor([
            single_img_metas['start_of_sequence'] 
            for single_img_metas in img_metas]).to(curr_bev.device)
        forward_augs = generate_forward_transformation_matrix(bda)
        # import pdb;pdb.set_trace()
        curr_to_prev_ego_rt = torch.stack([
            single_img_metas['curr_to_prev_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        # import pdb;pdb.set_trace()
        #print(seq_ids,start_of_sequence,self.history_seq_ids)
        ## Deal with first batch
        if not update_history:
            if self.history_bev is None:
                self.history_bev = curr_bev.clone()
                self.history_seq_ids = seq_ids.clone()
                self.history_forward_augs = forward_augs.clone()

                # Repeat the first frame feature to be history
                if voxel_feat:
                    self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
                else:
                    self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1)
                # All 0s, representing current timestep.
                self.history_sweep_time = curr_bev.new_zeros(curr_bev.shape[0], self.history_cat_num)

        if not update_history or self.do_history:
            self.history_bev = self.history_bev.detach()

            assert self.history_bev.dtype == torch.float32

        ## Deal with the new sequences
        # First, sanity check. For every non-start of sequence, history id and seq id should be same.

            assert (self.history_seq_ids != seq_ids)[~start_of_sequence].sum() == 0, \
                    "{}, {}, {}".format(self.history_seq_ids, seq_ids, start_of_sequence)

        ## Replace all the new sequences' positions in history with the curr_bev information
        if not update_history:
            self.history_sweep_time += 1 # new timestep, everything in history gets pushed back one.
            if start_of_sequence.sum()>0:
                if voxel_feat:    
                    self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
                else:
                    self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
                
                self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
                self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
                self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]

        if not update_history or self.do_history:
            ## Get grid idxs & grid2bev first.
            if voxel_feat:
                n, c_, z, h, w = curr_bev.shape

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
            # if self.forward_projection:
            #     feat2bev[0, 0] = self.forward_projection.dx[0]
            #     feat2bev[1, 1] = self.forward_projection.dx[1]
            #     feat2bev[2, 2] = self.forward_projection.dx[2]
            #     feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
            #     feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
            #     feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
            # else:
            feat2bev[0, 0] = self.grid_config['x'][2]#ddx[0]
            feat2bev[1, 1] = self.grid_config['y'][2]#ddx[1]
            feat2bev[2, 2] = self.grid_config['z'][2]#ddx[2]
            feat2bev[0, 3] = self.grid_config['x'][0]#dbx[0] - ddx[0] / 2.
            feat2bev[1, 3] = self.grid_config['y'][0]#dbx[1] - ddx[1] / 2.
            feat2bev[2, 3] = self.grid_config['z'][0]#dbx[2] - ddx[2] / 2.
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
            

            tmp_bev = self.history_bev
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
    
            sampled_history_bev = F.grid_sample(tmp_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode=self.interpolation_mode)

        ## Update history
        # Add in current frame to features & timestep
        if not update_history:
            self.history_sweep_time = torch.cat(
                [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
                dim=1) # B x (1 + T)
        if not update_history or self.do_history:
            if voxel_feat:
                sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
                curr_bev = curr_bev.reshape(n, c_, z, h, w)
            feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        if not update_history:
            # Reshape and concatenate features and timestep
            feats_to_return = feats_cat.reshape(
                    feats_cat.shape[0], self.history_cat_num + 1, self.single_bev_num_channels, *feats_cat.shape[2:]) # B x (1 + T) x 80 x H x W
            if voxel_feat:
                feats_to_return = torch.cat(
                [feats_to_return, self.history_sweep_time[:, :, None, None, None, None].repeat(
                    1, 1, 1, *feats_to_return.shape[3:]) * self.history_cam_sweep_freq
                ], dim=2) # B x (1 + T) x 81 x Z x H x W
            else:
                feats_to_return = torch.cat(
                [feats_to_return, self.history_sweep_time[:, :, None, None, None].repeat(
                    1, 1, 1, feats_to_return.shape[3], feats_to_return.shape[4]) * self.history_cam_sweep_freq
                ], dim=2) # B x (1 + T) x 81 x H x W

            # Time conv
            feats_to_return = self.history_keyframe_time_conv(
                feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                    feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

            # Cat keyframes & conv
            feats_to_return = self.history_keyframe_cat_conv(
                feats_to_return.reshape(
                    feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        
        if update_history:
            if self.do_history:
                self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
            self.history_sweep_time = self.history_sweep_time[:, :-1]
            self.history_forward_augs = forward_augs.clone()
            return 
        # if voxel_feat:
        #     feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        return feats_to_return.clone()

class _ASPPModule(nn.Module):

    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):

    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(
            int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = BatchNorm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(
            x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):

    def __init__(self,
                 in_channels,
                 mid_channels,
                 context_channels,
                 depth_channels,
                 use_dcn=True,
                 use_aspp=True,
                 with_cp=False,
                 stereo=False,
                 bias=0.0,
                 aspp_mid_channels=-1,
                 depth2occ=False,
                 gaussion=False,
                 monoscene=False,
                 length=22,
                 k_sqrt=3,
                 toSet=False,
                 toSetV2=False,
                 toSetV3=False,
                 depth_continue=False,
                 adaptive_depth_bin=False,
                 num_sampling_from_depth=False,
                 depth_bin_pooling=False,
                 ada2fix_bin=False,
                 ada_bin_self_wight=False,
                 ):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(27)
        
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        depth_conv_input_channels = mid_channels
        downsample = None
        #######################
        self.monoscene=monoscene
        self.gaussion=gaussion
        self.depth2occ=depth2occ
        self.length=length

        self.k_sqrt=k_sqrt

        self.toSet=toSet
        self.toSetV2=toSetV2
        self.toSetV3=toSetV3
        cost_volumn_channels=depth_channels
        if self.toSetV3:
            depth_channels*=self.k_sqrt**2
        self.depth_continue=depth_continue
        if self.depth_continue:
            depth_channels=2
        self.adaptive_depth_bin=adaptive_depth_bin
        self.ada_bin_self_wight=ada_bin_self_wight
        
        if self.adaptive_depth_bin and not self.ada_bin_self_wight:
            depth_channels=self.adaptive_depth_bin*2
        self.num_sampling_from_depth=num_sampling_from_depth
        self.depth_bin_pooling=depth_bin_pooling
        self.ada2fix_bin=ada2fix_bin
        #####################
        if not monoscene:
            self.depth_mlp = Mlp(27, mid_channels, mid_channels)
            self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
            if stereo:
                depth_conv_input_channels += cost_volumn_channels
                downsample = nn.Conv2d(depth_conv_input_channels,
                                        mid_channels, 1, 1, 0)
                cost_volumn_net = []
                for stage in range(int(2)):
                    cost_volumn_net.extend([
                        nn.Conv2d(cost_volumn_channels, cost_volumn_channels, kernel_size=3,
                                stride=2, padding=1),
                        nn.BatchNorm2d(cost_volumn_channels)])
                self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
                self.bias = bias
            depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                        downsample=downsample),
                            BasicBlock(mid_channels, mid_channels),
                            BasicBlock(mid_channels, mid_channels)]
            if use_aspp:
                if aspp_mid_channels<0:
                    aspp_mid_channels = mid_channels
                depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
            if use_dcn:
                depth_conv_list.append(
                    build_conv_layer(
                        cfg=dict(
                            type='DCN',
                            in_channels=mid_channels,
                            out_channels=mid_channels,
                            kernel_size=3,
                            padding=1,
                            groups=4,
                            im2col_step=128,
                        )))
            depth_conv_list.append(
                nn.Conv2d(
                    mid_channels,
                    depth_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0))
            self.depth_conv = nn.Sequential(*depth_conv_list)

        ############################
        
        if self.toSetV2:
            set_channels=self.k_sqrt**3
            set_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                downsample=downsample),
                    BasicBlock(mid_channels, mid_channels),
                    BasicBlock(mid_channels, mid_channels)]
            if use_aspp:
                if aspp_mid_channels<0:
                    aspp_mid_channels = mid_channels
                set_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
            if use_dcn:
                set_conv_list.append(
                    build_conv_layer(
                        cfg=dict(
                            type='DCN',
                            in_channels=mid_channels,
                            out_channels=mid_channels,
                            kernel_size=3,
                            padding=1,
                            groups=4,
                            im2col_step=128,
                        )))
            set_conv_list.append(
            nn.Conv2d(
                mid_channels,
                set_channels,
                kernel_size=1,
                stride=1,
                padding=0))
            self.occToSet_conv = nn.Sequential(*set_conv_list)
        
        if self.depth2occ:
            occ_channels=self.length
            if self.toSet and not self.toSetV2:
                occ_channels*=self.k_sqrt**2
            occ_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                        downsample=downsample),
                            BasicBlock(mid_channels, mid_channels),
                            BasicBlock(mid_channels, mid_channels)]
            if use_aspp:
                if aspp_mid_channels<0:
                    aspp_mid_channels = mid_channels
                occ_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
            if use_dcn:
                occ_conv_list.append(
                    build_conv_layer(
                        cfg=dict(
                            type='DCN',
                            in_channels=mid_channels,
                            out_channels=mid_channels,
                            kernel_size=3,
                            padding=1,
                            groups=4,
                            im2col_step=128,
                        )))
            if self.gaussion:
                occ_conv_list.append(
                    nn.Conv2d(
                        mid_channels,
                        occ_channels*2,
                        kernel_size=1,
                        stride=1,
                        padding=0))
            else:    
                occ_conv_list.append(
                    nn.Conv2d(
                        mid_channels,
                        occ_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0))
            self.occ_conv = nn.Sequential(*occ_conv_list)
        if self.num_sampling_from_depth:
            if self.num_points:
                if self.disc2continue_depth:
                    sampling_offsets_channels=self.num_points*3
                else:
                    sampling_offsets_channels=self.num_points*depth_channels*3
                    
            sampling_offsets_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                downsample=downsample),
                    BasicBlock(mid_channels, mid_channels),
                    BasicBlock(mid_channels, mid_channels)]
            if use_aspp:
                if aspp_mid_channels<0:
                    aspp_mid_channels = mid_channels
                sampling_offsets_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
            if use_dcn:
                sampling_offsets_conv_list.append(
                    build_conv_layer(
                        cfg=dict(
                            type='DCN',
                            in_channels=mid_channels,
                            out_channels=mid_channels,
                            kernel_size=3,
                            padding=1,
                            groups=4,
                            im2col_step=128,
                        )))
            sampling_offsets_conv_list.append(
            nn.Conv2d(
                mid_channels,
                sampling_offsets_channels,
                kernel_size=1,
                stride=1,
                padding=0))
            self.sampling_offsets_channels_conv = nn.Sequential(*set_conv_list)
        ################################
        self.with_cp = with_cp
        self.depth_channels = depth_channels

    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        frustum = metas['frustum']
        points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
        points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
            .matmul(points.unsqueeze(-1))
        points = torch.cat(
            (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

        rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
        trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
        combine = rots.matmul(torch.inverse(metas['intrins']))

        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points += trans.view(B, N, 1, 1, 1, 3, 1)
        neg_mask = points[..., 2, 0] < 1e-3
        points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
        points = points[..., :2, :] / points[..., 2:3, :]

        points = metas['post_rots'][...,:2,:2].view(B, N, 1, 1, 1, 2, 2).matmul(
            points).squeeze(-1)
        points += metas['post_trans'][...,:2].view(B, N, 1, 1, 1, 2)

        px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
        py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
        px[neg_mask] = -2
        py[neg_mask] = -2
        grid = torch.stack([px, py], dim=-1)
        grid = grid.view(B * N, D * H, W, 2)
        return grid

    def calculate_cost_volumn(self, metas):
        prev, curr = metas['cv_feat_list']
        # import pdb;pdb.set_trace()
        group_size = 4
        _, c, hf, wf = curr.shape
        hi, wi = hf * 4, wf * 4
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = metas['frustum'].shape
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)

        prev = prev.view(B * N, -1, H, W)
        curr = curr.view(B * N, -1, H, W)
        cost_volumn = 0
        # process in group wise to save memory
        for fid in range(curr.shape[1] // group_size):
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            wrap_prev = F.grid_sample(prev_curr, grid,
                                      align_corners=True,
                                      padding_mode='zeros')
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              wrap_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)
            cost_volumn += cost_volumn_tmp
        if not self.bias == 0:
            invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        return cost_volumn

    def forward(self, x, mlp_input, stereo_metas=None):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        if self.monoscene:
            return context
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth_ = self.depth_se(x, depth_se)
        # import pdb;pdb.set_trace()
        if not stereo_metas is None:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/\
                               stereo_metas['cv_downsample']
                cost_volumn = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    cost_volumn = self.calculate_cost_volumn(stereo_metas)
            cost_volumn = self.cost_volumn_net(cost_volumn)
            # import pdb;pdb.set_trace()
            depth_ = torch.cat([depth_, cost_volumn], dim=1)
        if self.with_cp:
            depth = checkpoint(self.depth_conv, depth_)
        else:
            depth = self.depth_conv(depth_)
        if self.depth_bin_pooling:
            # import pdb;pdb.set_trace()
            bin,weight=depth[:,:depth.shape[1]//2,...],depth[:,depth.shape[1]//2:,...]
            bin=bin.mean((2,3),keepdim=True).repeat(1,1,*depth.shape[2:])
            depth=torch.cat((bin,weight),dim=1)
        if self.ada2fix_bin:
            weight=depth[:,depth.shape[1]//2:,...]
            bin=torch.ones_like(weight)
            depth=torch.cat((bin,weight),dim=1)
        results=[depth,context]
        #######################
        if self.depth2occ:
            if self.with_cp:
                occ = checkpoint(self.occ_conv, depth_)
            else:
                occ = self.occ_conv(depth_)
            if self.gaussion:
                return torch.cat(results, dim=1),occ
            results.append(occ)
        if self.toSetV2:
            if self.with_cp:
                occToSet = checkpoint(self.occToSet_conv, depth_)
            else:
                occToSet = self.occToSet_conv(depth_)
            results.append(occToSet)
        if self.num_sampling_from_depth:
            if self.with_cp:
                sampling_offsets = checkpoint(self.sampling_offsets_channels_conv, depth_)
            else:
                sampling_offsets = self.sampling_offsets_channels_conv(depth_)
            
        #######################
        if self.num_sampling_from_depth:
            return [torch.cat(results, dim=1),sampling_offsets]
        else:
            return torch.cat(results, dim=1)


class DepthAggregation(nn.Module):
    """pixel cloud feature extraction."""

    def __init__(self, in_channels, mid_channels, out_channels):
        super(DepthAggregation, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    @autocast(False)
    def forward(self, x):
        x = checkpoint(self.reduce_conv, x)
        short_cut = x
        x = checkpoint(self.conv, x)
        x = short_cut + x
        x = self.out_conv(x)
        return x


@NECKS.register_module()
class LSSViewTransformerBEVDepth(LSSViewTransformer):

    def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(),depth2occ=False,hidden_supervise=True,use_gt_depth=False,
    direct_learn_occ=False,only_supervise_front_part=False,use_gt_occ=False,gaussion=False,length=22,monoscene=False,toSet=False,
    max_pooling=False,num_front=21,bs=4,
     **kwargs):
        super(LSSViewTransformerBEVDepth, self).__init__(**kwargs)
        self.loss_depth_weight = loss_depth_weight
        self.monoscene=monoscene
        
        
        self.depth_net = DepthNet(self.in_channels, self.in_channels,
                                self.out_channels, self.D, **depthnet_cfg)
        self.depth2occ=depth2occ                   
        self.hidden_supervise=hidden_supervise  
        self.use_gt_depth=use_gt_depth      
        self.use_gt_occ=use_gt_occ
        self.direct_learn_occ=direct_learn_occ
        self.only_supervise_front_part=only_supervise_front_part
        self.gaussion=gaussion

        self.length=length
        self.toSet=toSet
        if self.toSet:
            k_sqrt=self.k_sqrt
            
            self.max_pooling=max_pooling
            self.num_front=num_front
            self.kernel=cal_kernal(k_sqrt)
        
            h,w=kwargs['input_size'][0]//self.downsample,kwargs['input_size'][1]//self.downsample
            # self.weight=cal_weight(h,w,k_sqrt,self.kernel)
            self.weight=None
            d=self.D
            
            self.z1=torch.zeros(6*bs*h*w*k_sqrt*k_sqrt,1)
            self.z2=torch.zeros(6*bs*h*w*k_sqrt*k_sqrt,d-num_front)
            self.z3=torch.zeros(6*bs*h*w*k_sqrt*k_sqrt,d+num_front-length+1)
            w_bias=torch.arange(0-k_sqrt//2,k_sqrt//2+1).repeat(k_sqrt,1)
            h_bias=torch.arange(0-k_sqrt//2,k_sqrt//2+1).reshape(-1,1).repeat(1,k_sqrt)
            if self.coor_expand_in_vox is not None:
                self.coor_expand_in_vox=torch.tensor(self.coor_expand_in_vox)
                if self.toSetV2 or self.toSetV4 or self.toSetV5:
                    w_bias=torch.arange(0-k_sqrt//2,k_sqrt//2+1).reshape(-1,1,1).repeat(1,k_sqrt,k_sqrt)
                    h_bias=torch.arange(0-k_sqrt//2,k_sqrt//2+1).reshape(1,-1,1).repeat(k_sqrt,1,k_sqrt)
                    d_bias=torch.arange(0-k_sqrt//2,k_sqrt//2+1).reshape(1,1,-1).repeat(k_sqrt,k_sqrt,1)
                    self.coor_offset=torch.stack([w_bias,h_bias,d_bias],dim=-1).reshape(-1,3)*self.coor_expand_in_vox
                else:
                    self.coor_offset=torch.stack([w_bias,h_bias],dim=-1).reshape(-1,2)*self.coor_expand_in_vox
            else:
                self.to_set_interval=torch.tensor(self.to_set_interval)
                if self.toSetV2 or self.toSetV4 or self.toSetV5:
                    
                    w_bias=torch.arange(0-k_sqrt//2,k_sqrt//2+1).reshape(-1,1,1).repeat(1,k_sqrt,k_sqrt)
                    h_bias=torch.arange(0-k_sqrt//2,k_sqrt//2+1).reshape(1,-1,1).repeat(k_sqrt,1,k_sqrt)
                    d_bias=torch.arange(0-k_sqrt//2,k_sqrt//2+1).reshape(1,1,-1).repeat(k_sqrt,k_sqrt,1)
                    self.coor_offset=torch.stack([w_bias,h_bias,d_bias],dim=-1).reshape(-1,3)*self.to_set_interval
                else:
                    self.coor_offset=torch.stack([w_bias,h_bias],dim=-1).reshape(-1,2)*self.to_set_interval
        

    def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
        B, N, _, _ = sensor2ego.shape
        bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
        mlp_input = torch.stack([
            intrin[:, :, 0, 0],
            intrin[:, :, 1, 1],
            intrin[:, :, 0, 2],
            intrin[:, :, 1, 2],
            post_rot[:, :, 0, 0],
            post_rot[:, :, 0, 1],
            post_tran[:, :, 0],
            post_rot[:, :, 1, 0],
            post_rot[:, :, 1, 1],
            post_tran[:, :, 1],
            bda[:, :, 0, 0],
            bda[:, :, 0, 1],
            bda[:, :, 1, 0],
            bda[:, :, 1, 1],
            bda[:, :, 2, 2],], dim=-1)
        sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        return mlp_input
    def get_downsampled_gt_continue_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        # if not self.training:
        #     gt_depths=gt_depths[0]
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_depths = torch.where(gt_depths == 1e5,
                                     torch.zeros_like(gt_depths),
                                    gt_depths)    
        ###############
        # zz=(gt_depths < self.grid_config['depth'][1]) & (gt_depths >= self.grid_config['depth'][0]-self.grid_config['depth'][2])
        # zzz=((~zz).float() * (gt_depths!=0).float()).sum()
        # print(zzz,1111111111111,(gt_depths!=0).float().sum(),(~zz).float().sum(),((~zz).float() * (gt_depths==0).float()).sum())
        gt_depths = torch.where((gt_depths < self.grid_config['depth'][1]) & (gt_depths >= self.grid_config['depth'][0]-self.grid_config['depth'][2]),
                                gt_depths, torch.zeros_like(gt_depths))
        ###################
        # if not self.sid:
        #     gt_depths = (gt_depths - (self.grid_config['depth'][0] -
        #                               self.grid_config['depth'][2])) / \
        #                 self.grid_config['depth'][2]
        # else:
        #     gt_depths = torch.log(gt_depths) - torch.log(
        #         torch.tensor(self.grid_config['depth'][0]).float())
        #     gt_depths = gt_depths * (self.D - 1) / torch.log(
        #         torch.tensor(self.grid_config['depth'][1] - 1.).float() /
        #         self.grid_config['depth'][0])
        #     gt_depths = gt_depths + 1.
        # gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
        #                         gt_depths, torch.zeros_like(gt_depths))
        # gt_depths = F.one_hot(
        #     gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                        #    1:]
        return gt_depths.float()
    def get_downsampled_gt_depth2(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths_ = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_depths_ = torch.where(gt_depths_ == 1e5,
                                     torch.zeros_like(gt_depths_),
                                    gt_depths_)                           

        if not self.sid:
            gt_depths = (gt_depths_ - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths_) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths_,gt_depths.float()
    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        
        # if not self.training:
        #     gt_depths=gt_depths[0]###########################
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
        return gt_depths.float()
    def get_downsampled_gt_hidden(self, gt_depths,gt_hiddens,direct_learn_occ=False):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        # import pdb; pdb.set_trace()
        if not self.training:  
            gt_depths=gt_depths[0]
            gt_hiddens=gt_hiddens[0]
        B, N, H, W = gt_depths.shape
        # import pdb; pdb.set_trace()
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        gt_hiddens=gt_hiddens.view(B * N, H // self.downsample,
                                      self.downsample, W // self.downsample,
                                      self.downsample, 1)                           
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_hiddens=gt_hiddens.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        gt_hiddens=gt_hiddens.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths,idx = torch.min(gt_depths_tmp, dim=-1)
        gt_hiddens=gt_hiddens[torch.arange(gt_hiddens.shape[0]).to(gt_hiddens.device).long(),idx]
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        gt_hiddens=gt_hiddens.view(B * N, H // self.downsample,
                                      W // self.downsample)

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
            gt_hiddens = (gt_hiddens - (self.grid_config['depth'][0] -
                                        self.grid_config['depth'][2])) / \
                            self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_hiddens = torch.where((gt_hiddens < self.D + 1) & (gt_hiddens >= 0.0),
                                gt_hiddens, torch.zeros_like(gt_hiddens))
        
        disc=gt_hiddens-gt_depths
        disc=disc.reshape(-1)
        if not direct_learn_occ:
            max_len=self.length
            indices = torch.arange(max_len).to(disc.device).unsqueeze(0).repeat(len(disc), 1)

            # 利用广播机制，生成一个对比数组
            occ= indices < disc.unsqueeze(1)
            gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
            return gt_depths.float(),occ.float(),disc.float()                                                               
        else:
            max_len=self.D
            

            gt_depths = F.one_hot(
                gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
            gt_hiddens = F.one_hot(
                gt_hiddens.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
            gt_occ=zero_out_between_ones(gt_depths,gt_hiddens) 
            return gt_depths.float(),gt_occ.float(),disc.float()    
    def get_downsampled_gt_hidden_SL(self, gt_depths,gt_hiddens,direct_learn_occ=False):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        # import pdb; pdb.set_trace()
        if not self.training:  
            gt_depths=gt_depths[0]
            gt_hiddens=gt_hiddens[0]
        B, N, H, W = gt_depths.shape
        # import pdb; pdb.set_trace()
        # gt_depths = gt_depths.view(B * N, H // self.downsample,
        #                            self.downsample, W // self.downsample,
        #                            self.downsample, 1)
        # gt_hiddens=gt_hiddens.view(B * N, H // self.downsample,
        #                               self.downsample, W // self.downsample,
        #                               self.downsample, 1)                           
        # gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        # gt_hiddens=gt_hiddens.permute(0, 1, 3, 5, 2, 4).contiguous()
        # gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        # gt_hiddens=gt_hiddens.view(-1, self.downsample * self.downsample)
        # gt_depths_tmp = torch.where(gt_depths == 0.0,
        #                             1e5 * torch.ones_like(gt_depths),
        #                             gt_depths)
        # gt_depths,idx = torch.min(gt_depths_tmp, dim=-1)
        # gt_hiddens=gt_hiddens[torch.arange(gt_hiddens.shape[0]).to(gt_hiddens.device).long(),idx]
        gt_depths = gt_depths.view(B * N, H,W )
        gt_hiddens=gt_hiddens.view(B * N, H ,W )

        if not self.sid:
            gt_depths = (gt_depths - (self.grid_config['depth'][0] -
                                      self.grid_config['depth'][2])) / \
                        self.grid_config['depth'][2]
            gt_hiddens = (gt_hiddens - (self.grid_config['depth'][0] -
                                        self.grid_config['depth'][2])) / \
                            self.grid_config['depth'][2]
        else:
            gt_depths = torch.log(gt_depths) - torch.log(
                torch.tensor(self.grid_config['depth'][0]).float())
            gt_depths = gt_depths * (self.D - 1) / torch.log(
                torch.tensor(self.grid_config['depth'][1] - 1.).float() /
                self.grid_config['depth'][0])
            gt_depths = gt_depths + 1.
        gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_hiddens = torch.where((gt_hiddens < self.D + 1) & (gt_hiddens >= 0.0),
                                gt_hiddens, torch.zeros_like(gt_hiddens))
        
        disc=gt_hiddens-gt_depths
        disc=disc.reshape(-1)
        if not direct_learn_occ:
            max_len=self.length
            indices = torch.arange(max_len).to(disc.device).unsqueeze(0).repeat(len(disc), 1)

            # 利用广播机制，生成一个对比数组
            occ= indices < disc.unsqueeze(1)
            gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
            return gt_depths.float(),occ.float(),disc.float()                                                               
        else:
            max_len=self.D
            

            gt_depths = F.one_hot(
                gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
            gt_hiddens = F.one_hot(
                gt_hiddens.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
                                                                           1:]
            gt_occ=zero_out_between_ones(gt_depths,gt_hiddens) 

            return gt_depths.float(),gt_occ.float(),disc.float()                                                                                                                
    def get_downsampled_gt_hidden_semantics(self, gt_depths,depth_semantics,direct_learn_occ=False):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        # import pdb; pdb.set_trace()
        if not self.training:  
            gt_depths=gt_depths[0]
            # gt_hiddens=gt_hiddens[0]
            depth_semantics=depth_semantics[0]
        B, N, H, W = gt_depths.shape
        # import pdb; pdb.set_trace()
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   self.downsample, W // self.downsample,
                                   self.downsample, 1)
        # gt_hiddens=gt_hiddens.view(B * N, H // self.downsample,
        #                               self.downsample, W // self.downsample,
        #                               self.downsample, 1)           
        depth_semantics=depth_semantics.view(B * N, H // self.downsample,
                                        self.downsample, W // self.downsample,
                                        self.downsample, 1)

        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        # gt_hiddens=gt_hiddens.permute(0, 1, 3, 5, 2, 4).contiguous()
        depth_semantics=depth_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
        # gt_hiddens=gt_hiddens.view(-1, self.downsample * self.downsample)
        depth_semantics=depth_semantics.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths,idx = torch.min(gt_depths_tmp, dim=-1)
        # gt_hiddens=gt_hiddens[torch.arange(gt_hiddens.shape[0]).to(gt_hiddens.device).long(),idx]
        depth_semantics=depth_semantics[torch.arange(depth_semantics.shape[0]).to(depth_semantics.device).long(),idx]
        gt_depths = gt_depths.view(B * N, H // self.downsample,
                                   W // self.downsample)
        # gt_hiddens=gt_hiddens.view(B * N, H // self.downsample,
                                    #   W // self.downsample)
        depth_semantics=depth_semantics.view(B * N, H // self.downsample,
                                        W // self.downsample)

        # if not self.sid:
        #     gt_depths = (gt_depths - (self.grid_config['depth'][0] -
        #                               self.grid_config['depth'][2])) / \
        #                 self.grid_config['depth'][2]
        #     gt_hiddens = (gt_hiddens - (self.grid_config['depth'][0] -
        #                                 self.grid_config['depth'][2])) / \
        #                     self.grid_config['depth'][2]
        # else:
        #     gt_depths = torch.log(gt_depths) - torch.log(
        #         torch.tensor(self.grid_config['depth'][0]).float())
        #     gt_depths = gt_depths * (self.D - 1) / torch.log(
        #         torch.tensor(self.grid_config['depth'][1] - 1.).float() /
        #         self.grid_config['depth'][0])
        #     gt_depths = gt_depths + 1.
        # gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
        #                         gt_depths, torch.zeros_like(gt_depths))
        # gt_hiddens = torch.where((gt_hiddens < self.D + 1) & (gt_hiddens >= 0.0),
        #                         gt_hiddens, torch.zeros_like(gt_hiddens))
        
        # disc=gt_hiddens-gt_depths
        # disc=disc.reshape(-1)
        # if not direct_learn_occ:
        #     max_len=self.length
        #     indices = torch.arange(max_len).to(disc.device).unsqueeze(0).repeat(len(disc), 1)

        #     # 利用广播机制，生成一个对比数组
        #     occ= indices < disc.unsqueeze(1)
        #     gt_depths = F.one_hot(
        #     gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
        #                                                                    1:]
        #     return gt_depths.float(),occ.float(),disc.float()                                                               
        # else:
        #     max_len=self.D
            

        #     gt_depths = F.one_hot(
        #         gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
        #                                                                    1:]
        #     gt_hiddens = F.one_hot(
        #         gt_hiddens.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
        #                                                                    1:]
        #     gt_occ=zero_out_between_ones(gt_depths,gt_hiddens) 
        return depth_semantics
    def get_depth_loss_gt_occ2depth(self,depth_preds,depth_labels):
        # import pdb;pdb.set_trace()
        # if self.occ2_depth_use_depth_sup_occ:
            # occ_weight=depth_labels
            # free_weight=1-occ_weight
            # cum_free_weight=torch.cumprod(free_weight,dim=1)
            # cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[...,0:1]),cum_free_weight[...,:-1]],dim=-1)
            # weight=occ_weight*cum_free_weight
            
            
            # depth_labels=weight
            
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        depth_loss=-(depth_preds.log()*depth_labels).sum(1).mean()
        return self.loss_depth_weight * depth_loss
    def bin2depth(self,depth_bin_prob,depth_weight):
        # depth,depth_weight=depth_preds[0],depth_preds[1]
        # depth_weight=depth_weight.softmax(1)
        cum_depth=torch.cat((torch.zeros_like(depth_bin_prob[:,:1,...]),torch.cumsum(depth_bin_prob,1)[:,:-1]),dim=1)
        bin_center=self.grid_config['depth'][0]+(self.grid_config['depth'][1]-self.grid_config['depth'][0])*(depth_bin_prob/2+cum_depth)#BN,n_bin,H,W
        if self.ada2fix_bin:
            bin_center-=0.25
        
        depth_preds = torch.sum(depth_weight/depth_weight.sum(1,keepdim=True) *bin_center, dim=1)
        return bin_center,depth_preds
    @force_fp32()
    def get_continue_gt_depth_and_pred_depth(self, depth_labels, depth_preds):
        if self.adaptive_depth_bin:
            depth_bin_prob,depth_weight=depth_preds[0],depth_preds[1]
            depth_weight=depth_weight.softmax(1)
            
            # cum_depth=torch.cat((torch.zeros_like(depth[:,:1,...]),torch.cumsum(depth,1)[:,:-1]),dim=1)
            # bin_center=self.grid_config['depth'][0]+(self.grid_config['depth'][1]-self.grid_config['depth'][0])*(depth/2+cum_depth)#BN,n_bin,H,W
           
            # depth_preds = torch.sum(depth_weight *bin_center, dim=1)
            _,depth_preds=self.bin2depth(depth_bin_prob,depth_weight)
        elif self.disc2continue_depth_continue_sup:
            # depth_pred_=depth_preds
            depth_map=torch.arange(self.D).to(depth_preds.device)+1
            depth_map=depth_map* self.grid_config['depth'][2]+(self.grid_config['depth'][0] -self.grid_config['depth'][2])
            depth_continue=torch.einsum('bdwh,dl->blwh',depth_preds,depth_map.reshape(-1,1))#BN,D,H,W
            depth_preds=depth_continue.squeeze(1)#BN,H,W
            # import pdb;pdb.set_trace()
        depth_labels = self.get_downsampled_gt_continue_depth(depth_labels)
        

        
        depth_loss=self.depth_loss(depth_preds,depth_labels)
        valid_mask = depth_labels > 0
        depth_labels=depth_labels[valid_mask]
        depth_preds=depth_preds[valid_mask]
        return self.loss_depth_weight * depth_loss,depth_preds,depth_labels
    @force_fp32()
    def get_continue_depth_loss(self, depth_labels, depth_preds):
     
        if self.adaptive_depth_bin:
            depth_bin_prob,depth_weight=depth_preds[0],depth_preds[1]
            
            depth_weight=depth_weight.softmax(1)
            # cum_depth=torch.cat((torch.zeros_like(depth[:,:1,...]),torch.cumsum(depth,1)[:,:-1]),dim=1)
            # bin_center=self.grid_config['depth'][0]+(self.grid_config['depth'][1]-self.grid_config['depth'][0])*(depth/2+cum_depth)#BN,n_bin,H,W
           
            # depth_preds = torch.sum(depth_weight *bin_center, dim=1)
            _,depth_preds=self.bin2depth(depth_bin_prob,depth_weight)
        elif self.disc2continue_depth_continue_sup:
            if self.occ2_depth_use_occ:
                free_weight=1-depth_preds
                        
                cum_free_weight=torch.cumprod(free_weight,dim=1)
                cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[:,0:1,...]),cum_free_weight[:,:-1,...]],dim=1)
                depth_preds=depth_preds*cum_free_weight
            depth_map=torch.arange(self.D).to(depth_preds.device)+1
            depth_map=depth_map* self.grid_config['depth'][2]+(self.grid_config['depth'][0] -self.grid_config['depth'][2])
            depth_continue=torch.einsum('bdwh,dl->blwh',depth_preds,depth_map.reshape(-1,1))#BN,D,H,W
            depth_preds=depth_continue.squeeze(1)#BN,H,W
            # import pdb;pdb.set_trace()
        depth_labels = self.get_downsampled_gt_continue_depth(depth_labels)
       

        
        depth_loss=self.depth_loss(depth_preds,depth_labels)*10
        return self.loss_depth_weight * depth_loss
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):

        if self.occ2_depth_use_occ:
            
            free_weight=1-depth_preds
                    
            cum_free_weight=torch.cumprod(free_weight,dim=1)
            cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[:,0:1,...]),cum_free_weight[:,:-1,...]],dim=1)
            depth_preds=depth_preds*cum_free_weight
        
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        if self.depth_sup_CE:
            depth_labels=depth_labels.argmax(dim=1)
            with autocast(enabled=False):
                depth_loss =F.cross_entropy(
                    depth_preds,
                    depth_labels,
                    reduction='none',
                ).sum() / max(1.0, fg_mask.sum())
        else:
            with autocast(enabled=False):
                depth_loss = F.binary_cross_entropy(
                    depth_preds,
                    depth_labels,
                    reduction='none',
                ).sum() / max(1.0, fg_mask.sum())
        return self.loss_depth_weight * depth_loss
    @force_fp32()
    def get_hidden_depth_loss_SL(self, depth_labels, depth_preds,hidden_labels,occ_preds):
        # import pdb;pdb.set_trace()
        depth_labels,occ_labels,disc = self.get_downsampled_gt_hidden_SL(depth_labels,hidden_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        if self.hidden_supervise:
            occ_preds = occ_preds.permute(0, 2, 3,
                                            1).contiguous().view(-1, self.length)
                                     
            valid_mask=(disc>0)*fg_mask
            # print(occ_preds.shape,occ_labels.shape,disc.shape,valid_mask.shape,222222222222222222)      
            occ_labels=occ_labels[valid_mask]
            occ_preds=occ_preds[valid_mask]

            with autocast(enabled=False):
                occ_loss = F.binary_cross_entropy(
                    occ_preds,
                    occ_labels,
                    reduction='none',
                ).sum() / max(1.0, valid_mask.sum())
        else:
            occ_loss=torch.tensor(0).to(depth_preds.device)
        return self.loss_depth_weight * depth_loss,self.loss_depth_weight * occ_loss
    @force_fp32()
    def get_hidden_depth_loss(self, depth_labels, depth_preds,hidden_labels,occ_preds):
        depth_labels,occ_labels,disc = self.get_downsampled_gt_hidden(depth_labels,hidden_labels)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0

        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        if self.hidden_supervise:
            occ_preds = occ_preds.permute(0, 2, 3,
                                            1).contiguous().view(-1, self.length)
                                     
            valid_mask=(disc>0)*fg_mask
            # print(occ_preds.shape,occ_labels.shape,disc.shape,valid_mask.shape,222222222222222222)      
            occ_labels=occ_labels[valid_mask]
            occ_preds=occ_preds[valid_mask]

            with autocast(enabled=False):
                occ_loss = F.binary_cross_entropy(
                    occ_preds,
                    occ_labels,
                    reduction='none',
                ).sum() / max(1.0, valid_mask.sum())
        else:
            occ_loss=torch.tensor(0).to(depth_preds.device)
        return self.loss_depth_weight * depth_loss,self.loss_depth_weight * occ_loss

    @force_fp32()
    def get_occ_depth_loss(self, depth_labels,depth_preds,hidden_labels, occ_preds):
        depth_labels,occ_labels,disc = self.get_downsampled_gt_hidden(depth_labels,hidden_labels,self.direct_learn_occ)
        depth_preds = depth_preds.permute(0, 2, 3,
                                          1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        if self.hidden_supervise:
            occ_preds = occ_preds.permute(0, 2, 3,
                                            1).contiguous().view(-1, self.D)
            valid_mask=(disc>0)*fg_mask
            
            
            if valid_mask.sum()>0:
                occ_labels=occ_labels[valid_mask]
                occ_preds=occ_preds[valid_mask]
                occ_preds=torch.where(occ_preds<0 ,torch.zeros_like(occ_preds),occ_preds)
                occ_preds=torch.where(occ_preds>1 ,torch.ones_like(occ_preds),occ_preds)
                try:    
                    with autocast(enabled=False):
                        occ_loss = F.binary_cross_entropy(
                            occ_preds,
                            occ_labels,
                            reduction='none',
                        )
                        if self.only_supervise_front_part:
                            front_mask=occ_labels.flip(dims=[1]).cummax(dim=1)[0].flip(dims=[1])
                            occ_loss=occ_loss*front_mask.float()
                        
                            occ_loss=occ_loss.sum() / max(1.0, valid_mask.sum())
                except Exception as e:
                    occ_loss=torch.tensor(0).to(depth_preds.device)
            else:
                occ_loss=torch.tensor(0).to(depth_preds.device)
        else:
            occ_loss=torch.tensor(0).to(depth_preds.device)
        return self.loss_depth_weight * depth_loss,self.loss_depth_weight * occ_loss

    
    def forward(self, input, stereo_metas=None):
        
        (x, rots, trans, intrins, post_rots, post_trans, bda,
         mlp_input,gt_depth,gt_hidden,gt_occ,mask_camera,img_metas,key_frame) = input[:14]
        # import pdb;pdb.set_trace()
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if gt_occ is not None:
            if not self.training:
                gt_occ=torch.cat(gt_occ,0)########################
        #插值
        # import pdb;pdb.set_trace()
        if self.toSetV3:
            D=self.D*self.k
        elif self.adaptive_depth_bin and not self.ada_bin_self_wight:
            D=self.D*2
        else:
            D=self.D
        if not self.monoscene:
            if self.gaussion:
                x ,occ_weight_= self.depth_net(x, mlp_input, stereo_metas)
                if self.cumprod:
                    occ_weight_=torch.cumprod(occ_weight_,dim=1)
            elif self.num_sampling_from_depth:
                x,sampling_offsets=self.depth_net(x, mlp_input, stereo_metas)
            else:
                x = self.depth_net(x, mlp_input, stereo_metas)
            
            depth_digit = x[:, :D, ...]
            # import pdb;pdb.set_trace()
            if self.adaptive_depth_bin and not self.ada_bin_self_wight:
                
                depth_weight=depth_digit[:, D//2:D, ...]
                depth_digit=depth_digit[:, :D//2, ...]
                if self.global_learnable_bin is not None:
                    depth_weight=self.global_learnable_bin.to(depth_weight).reshape(1,D//2,1,1).expand_as(depth_weight)
                
            if self.toSetV3:
                depth_digit=depth_digit.reshape(B*N,self.k,self.D,H,W).reshape(B*N*self.k,self.D,H,W)
        
            tran_feat = x[:, D:D + self.out_channels, ...]
        else:
            depth_digit=torch.ones((x.shape[0],self.D,x.shape[2],x.shape[3])).to(x.device)
            tran_feat=self.depth_net(x, mlp_input, stereo_metas)
       
        if not self.depth_with_temp:
            if self.occ2depth:
                depth=depth_digit.sigmoid()
                if not self.occ2_depth_use_occ and not self.occ2_depth_use_depth_sup_occ:
                   
                    free_weight=1-depth
                    
                    cum_free_weight=torch.cumprod(free_weight,dim=1)
                    cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[:,0:1,...]),cum_free_weight[:,:-1,...]],dim=1)
                    depth=depth*cum_free_weight
                    
                    
            else:
                depth = depth_digit.softmax(dim=1)
        else:
            if self.occ2depth:
                depth = (depth_digit*self.depth_with_temp).sigmoid()
                if not self.occ2_depth_use_occ:
                    free_weight=1-depth
                    
                    cum_free_weight=torch.cumprod(free_weight,dim=1)
                    cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[:,0:1,...]),cum_free_weight[:,:-1,...]],dim=1)
                    depth=depth*cum_free_weight
            else:
                depth = (depth_digit*self.depth_with_temp).softmax(dim=1)
        if self.ada_bin_self_wight:#differential in sampling, weight, and position of depth
            # import pdb;pdb.set_trace()
            cum_depth=torch.cumsum(depth,1)-depth/2
            cum_depth_disc=torch.zeros_like(cum_depth).to(cum_depth.device)
            cum_depth_disc[:,1:-1,...]=(abs(cum_depth[:,1:-1,...]-cum_depth[:,2:,...])+abs(cum_depth[:,1:-1,...]-cum_depth[:,:-2,...]))/2
            
            cum_depth_disc[:,0,...]=abs(cum_depth[:,0,...]-cum_depth[:,1,...])
            cum_depth_disc[:,-1,...]=abs(cum_depth[:,-2,...]-cum_depth[:,-1,...])
            mean_pos=cum_depth_disc.min(1)[1].unsqueeze(1)#B,1,H,W
            mean=cum_depth.gather(1,mean_pos)
            
            depth_weight=-(cum_depth-mean).pow(2)/self.ada_bin_tau#-math.log(2*math.pi)/2-(abs(self.ada_bin_tau)+1e-8).log()
            # depth_weight=-(cum_depth-mean).pow(2)/((self.ada_bin_tau**2)*2+1e-8)#-math.log(2*math.pi)/2-(abs(self.ada_bin_tau)+1e-8).log()
            # import pdb;pdb.set_trace()

        if self.SL:
            tran_feat=self.predicter(tran_feat.permute(0,2,3,1).reshape(-1,self.out_channels))
            tran_feat=tran_feat.softmax(dim=1)
           
            tran_feat=tran_feat.reshape(B*N,*self.SL_size,-1).permute(0,3,1,2)
        if self.num_sampling_from_depth:
            tran_feat=[tran_feat,sampling_offsets]
        # import pdb; pdb.set_trace()
        #####################
        if self.use_gt_depth:
            
                
                # vox_sampled=F.grid_sample(vox_feats, coor,align_corners=True,padding_mode='zeros')
            
            if self.depth2occ:
                gt_depth,gt_occ,disc=self.get_downsampled_gt_hidden(gt_depth,gt_hidden,self.direct_learn_occ)
            elif self.use_gt_occ2depth:
                # import pdb;pdb.set_trace()
                coor = self.get_lidar_coor(*input[1:7])
                coor = ((coor - self.grid_lower_bound.to(coor)) /self.grid_interval.to(coor))
                
                
                gt_occ_density=gt_occ!=17
                gt_occ_density=gt_occ_density.float()
                
                
                
                
                DDD=coor.shape[2]
                coor=coor.reshape(B,N,DDD,H*W,3)
                
                coor=coor/(torch.tensor(gt_occ_density.shape[1:]).to(coor.device)-1)
                coor=torch.flip(coor,[-1])
                coor=coor*2-1
                
                occ_weight = F.grid_sample(gt_occ_density.unsqueeze(1), coor,align_corners=True,padding_mode='zeros')#torch.Size([2, 1, 6, 88, 704])
                
                if self.occ2_depth_use_depth_sup_occ:
                    gt_depth=occ_weight.reshape(B*N,DDD,H,W).permute(0, 2, 3,1).contiguous().view(-1, self.D)
                else:
                    free_weight=1-occ_weight
                    cum_free_weight=torch.cumprod(free_weight,dim=-2)
                    cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[...,0:1,:]),cum_free_weight[...,:-1,:]],dim=-2)
                    weight=occ_weight*cum_free_weight
                    
                    weight=weight.reshape(B*N,DDD,H,W)
                    gt_depth=weight.permute(0, 2, 3,1).contiguous().view(-1, self.D)
                # import pdb;pdb.set_trace()
            else:
                gt_depth=self.get_downsampled_gt_depth(gt_depth)
            # depth_labels = self.get_downsampled_gt_depth(depth_labels)
            depth_preds=depth.clone()
            b,d,h,w=depth_preds.shape
            
 
            depth_gt_mixed = depth_preds.permute(0, 2, 3,1).contiguous().view(-1, self.D)
            fg_mask = torch.max(gt_depth, dim=1).values > 0.0
            
            if self.depth_denoising:
                fg_mask=fg_mask.nonzero().squeeze(1)
                
                random_indices = torch.randperm(fg_mask.shape[0])
                # 从1到10中采样
                sampled_values = random_indices[:int(fg_mask.shape[0]*self.depth_denoising_ratio)]
                if self.denoising_with_noise is not None:   
                    depth_gt_mixed[fg_mask[sampled_values]] = gt_depth[fg_mask[sampled_values]]+torch.randn_like(gt_depth[fg_mask[sampled_values]])*self.denoising_with_noise
                else:
                    depth_gt_mixed[fg_mask[sampled_values]] = gt_depth[fg_mask[sampled_values]]
                if self.denoising_with_mixed is not None:
                    depth_gt_mixed[fg_mask[sampled_values]] = (1-self.denoising_with_mixed)*gt_depth[fg_mask[sampled_values]]+depth_gt_mixed[fg_mask[sampled_values]]*self.denoising_with_mixed
                else:
                    depth_gt_mixed[fg_mask[sampled_values]] = gt_depth[fg_mask[sampled_values]]
            elif self.depth_gt_not_mix:
                depth_gt_mixed = gt_depth
            elif self.use_pred_depth:
                depth_gt_mixed = depth.permute(0, 2, 3,1).contiguous().view(-1, self.D)

                if self.occ2_depth_use_depth_sup_occ:
                    # depth_gt_mixed = depth
                    free_weight=1-depth_gt_mixed
                    cum_free_weight=torch.cumprod(free_weight,dim=1)
                    cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[...,0:1]),cum_free_weight[...,:-1]],dim=1)
                    depth_gt_mixed=depth_gt_mixed*cum_free_weight
                    # depth_gt_mixed =depth.permute(0, 2, 3,1).contiguous().view(-1, self.D)
                
            else:
                depth_gt_mixed[fg_mask] = gt_depth[fg_mask]
            depth_gt_mixed=depth_gt_mixed.reshape(b,h,w,d).permute(0,3,1,2)
            if self.depth_denoising:
                if not self.training:
                    depth_gt_mixed=depth
###################
        if self.toSetV2:
            if self.depth2occ:
                occ2set=x[:,self.D+self.out_channels+self.length:self.D+self.out_channels+self.length+self.k_sqrt**3,...]
                occ2set=occ2set.sigmoid()
            else:
                occ2set=x[:,self.D+self.out_channels:self.D+self.out_channels+self.k_sqrt**3,...]
                occ2set=occ2set.sigmoid()
        else:
            occ2set=None
        
        if self.depth2occ:
            if not self.gaussion:
                occ_weight_=x[:,D+self.out_channels:D+self.out_channels+self.length,...]
            occ_weight_=occ_weight_.sigmoid()
            if self.cumprod:
                occ_weight_=torch.cumprod(occ_weight_,dim=1)


            # bn,d,h,w=depth.shape
            # len_occ=occ_weight_.shape[1]
            # occ_weight=occ_weight_.permute(0,2,3,1).reshape(bn*h*w,len_occ)
            # ###############
            # # import pdb; pdb.set_trace()
            # # depth_2_occ=torch.eye(d,d).to(depth.device).unsqueeze(0).repeat(bn*h*w,1,1)
            # # for i in range(len_occ):
            # #     depth_2_occ[:,:d-1-i,i+1:]+=torch.diag_embed(occ_weight[:,i:i+1].repeat(1,d-1-i))
            # ######################
            # depth_2_occ=torch.cat([torch.zeros_like(occ_weight[:,:1]).to(depth.device),occ_weight,torch.zeros(occ_weight.shape[0],d-len_occ).to(depth.device)],dim=1).repeat(1,d).reshape(bn*h*w,d+1,d)[:,:d,:]
            # depth_2_occ=depth_2_occ.triu()+torch.eye(d,d).to(depth.device).unsqueeze(0)
            # # print(abs(cal_depth2occ(occ_weight_,depth)!=depth_2_occ).sum(),111111111111111111111)
            # ##################
            # if self.use_gt_depth:
            #     occ=torch.matmul(depth_gt_mixed.permute(0,2,3,1).reshape(bn*h*w,1,d),depth_2_occ).reshape(bn,h,w,d).permute(0,3,1,2)
            # else:
            #     occ=torch.matmul(depth.permute(0,2,3,1).reshape(bn*h*w,1,d),depth_2_occ).reshape(bn,h,w,d).permute(0,3,1,2)
            
            if self.toSet and not self.toSetV2:
                # weight=self.weight.to(depth.device)
                # kernel=self.kernel.to(depth.device)
                occ_weight_=x[:,D+self.out_channels:D+self.out_channels+self.length*self.k,...]
                occ_weight_=occ_weight_.sigmoid()
                if self.cumprod:
                    occ_weight_=torch.cumprod(occ_weight_,dim=1)
                if self.toSetV3:
                    occ_weight_=occ_weight_.reshape(B*N,self.k,self.length,H,W).reshape(B*N*self.k,self.length,H,W)
                    if self.cumprod:
                        occ_weight_=torch.cumprod(occ_weight_,dim=1)
                    occ=cal_depth2occ(occ_weight_,depth)
                    depth=depth.reshape(B*N,self.k,self.D,H,W)[:,self.k//2,...]
                    occ=occ.reshape(B*N,self.k,self.D,H,W).permute(0,2,3,4,1)
                else:
                    if self.use_gt_depth:
                        # expanded_depth=get_expanded_feature_map(depth_gt_mixed,self.k_sqrt,kernel)
                        expanded_depth=depth_gt_mixed.unsqueeze(1).repeat(1,self.k,1,1,1)
                    else:
                        # expanded_depth=get_expanded_feature_map(depth,self.k_sqrt,kernel)
                        expanded_depth=depth.unsqueeze(1).repeat(1,self.k,1,1,1)
    
                    occ=cal_depth2occ_set(occ_weight_,expanded_depth,self.k_sqrt,self.num_front,self.max_pooling,None,self.z1,self.z2,self.z3)
                # ss=time.time()
                # del kernel,weight#,expanded_depth
                # torch.cuda.empty_cache()
                # print(time.time()-ss,11111111111)
            else:        

                if self.use_gt_depth:
                    occ=cal_depth2occ(occ_weight_,depth_gt_mixed)
                else:
                    # import pdb;pdb.set_trace()
                    occ=cal_depth2occ(occ_weight_,depth)
            # import pdb; pdb.set_trace()
            if self.lift_attn is not None:
                
                bev_feat, occ_ = self.view_transform(input,[occ_weight_,depth,occ], tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
                # occ_,inter_occs=occ_
                # depths=[depth]
                # for i in range(len(occ_)):
                #     if i >0:
                #         depths.append(cal_depth2occ(occ_weight_,depth))
                return bev_feat, occ_
            if self.supervise_intermedia:
                bev_feat, depth = self.view_transform(input,occ, tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
                return bev_feat, depth
            if self.use_gt_occ:
                occ_gt_mixed=occ.clone().permute(0,2,3,1).contiguous().view(-1, self.D)
                fg_mask = (disc > 0.0)*fg_mask
                if self.only_supervise_front_part:
                    # import pdb; pdb.set_trace()
                    front_mask=gt_occ.flip(dims=[1]).cummax(dim=1)[0].flip(dims=[1])
                    fg_mask=fg_mask[:,None]*front_mask.long()
                occ_gt_mixed[fg_mask] = gt_occ[fg_mask]
                occ_gt_mixed=occ_gt_mixed.reshape(b,h,w,d).permute(0,3,1,2)
                if self.adaptive_depth_bin:
                    occ_gt_mixed=[occ_gt_mixed,depth_weight]
                if self.toSetV2:
                    bev_feat, occ_ = self.view_transform(input, occ_gt_mixed, tran_feat,occ2set,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
                else:
                    bev_feat, occ_ = self.view_transform(input, occ_gt_mixed, tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)

            else:
                if self.adaptive_depth_bin:
                    occ=[occ,depth_weight]
                if self.toSetV2:
                    bev_feat, occ_ = self.view_transform(input, occ, tran_feat,occ2set,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
                else:
                    bev_feat, occ_ = self.view_transform(input, occ, tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
            # if self.direct_learn_occ:
            #     return bev_feat, [depth,occ,occ_weight_]
            return bev_feat, [depth,occ,occ_weight_]
#############################
        if self.lift_attn is not None:
            if self.use_gt_depth:
                bev_feat, depth = self.view_transform(input,depth_gt_mixed, tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
            else:
                bev_feat, depth = self.view_transform(input,depth, tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
            return bev_feat, depth
        if self.supervise_intermedia:
            if self.use_gt_depth:
                bev_feat, depth = self.view_transform(input,depth_gt_mixed, tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
            else:
                bev_feat, depth = self.view_transform(input,depth, tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
            return bev_feat, depth
        if self.use_gt_depth:
            if self.adaptive_depth_bin:
                depth_gt_mixed=[depth_gt_mixed,depth_weight]
            if self.toSetV2:
                bev_feat, depth_ = self.view_transform(input, depth_gt_mixed, tran_feat,occ2set,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
            else:
                bev_feat, depth_ = self.view_transform(input, depth_gt_mixed, tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
        else:
            if self.adaptive_depth_bin:
                
                depth=[depth,depth_weight]
                if self.only_train_depth:
                    return [None],depth
            if self.toSetV2:
                bev_feat, depth_ = self.view_transform(input, depth, tran_feat,occ2set,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
            else:
                bev_feat, depth_ = self.view_transform(input,depth, tran_feat,gt_occ=gt_occ,img_metas=img_metas,key_frame=key_frame)
        if self.use_gt_occ2depth:
            return bev_feat,[depth,gt_depth]
        return bev_feat, depth
    def occ2depth_func(self,  depth, occ,coor=None,cam_params=None):#occ:[16,200,200]
        
    #     # import pdb;pdb.set_trace()
        B,N,D,H,W=depth.shape
        # gt_occ=kwargs['gt_occupancy_ori'].permute(0,3,1,2)
        # if coor is None:
        #     coor = self.get_lidar_coor(*cam_params)
        #     coor = ((coor - self.grid_lower_bound.to(coor)) /self.grid_interval.to(coor))
        # else:
        #     # import pdb;pdb.set_trace()
        #     coor = ((coor - self.grid_lower_bound.to(coor)) /self.grid_interval.to(coor))
        
        
        # gt_occ_density=gt_occ!=18
        # gt_occ_density=gt_occ_density.float()
        
        
        # DDD=coor.shape[2]
        coor=coor.reshape(B,N,D,H*W,3)
        ######################
        coor=coor/(torch.tensor(occ.shape[1:]).to(coor.device)-1)
        coor=torch.flip(coor,[-1])

        coor=coor*2-1
        
        occ_weight = F.grid_sample(occ.unsqueeze(1), coor,align_corners=True,padding_mode='zeros')#torch.Size([2, 1, 6, 88, 704])
        
        # if self.occ2_depth_use_depth_sup_occ:
        #     gt_depth=occ_weight.reshape(B*N,D,H,W).permute(0, 2, 3,1).contiguous().view(-1, self.D)
        # else:
        free_weight=1-occ_weight
        cum_free_weight=torch.cumprod(free_weight,dim=-2)
        cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[...,0:1,:]),cum_free_weight[...,:-1,:]],dim=-2)
        weight=occ_weight*cum_free_weight
        # import pdb;pdb.set_trace()
        weight=weight.reshape(B,N,D,H,W)
        
        return weight

@NECKS.register_module()
class LSSViewTransformerBEVStereo(LSSViewTransformerBEVDepth):

    def __init__(self,  **kwargs):
        super(LSSViewTransformerBEVStereo, self).__init__(**kwargs)
        if self.img_upsample:
            self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                                kwargs['input_size'],
                                                downsample=2)
        else:    
            if not self.SL:
                self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
                                                    kwargs['input_size'],
                                                    downsample=4)
            else:
                self.cv_frustum = self.create_frustum_SL(kwargs['grid_config']['depth'],
                                                    kwargs['input_size'],
                                                    [self.SL_size[0]*4,self.SL_size[1]*4])
        if self.adaptive_depth_bin:
            self.D=self.adaptive_depth_bin
        if self.lift_attn is not None and not self.lift_attn_simple_add and not self.lift_attn_with_ori_feat and not self.lift_attn_new:
            # simple_depth_net=nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=1, padding=0),
            #                                     nn.ReLU(),
            #                                     nn.Conv2d(self.out_channels*2, self.D, kernel_size=1, padding=0),
            #                                     )
            # self.simple_depth_nets = nn.ModuleList(
            #     [simple_depth_net for _ in range(self.lift_attn_round)])
            self.simple_depth_nets=nn.ModuleList()
            for i in range(self.lift_attn_round):
                simple_depth_net=nn.Sequential(nn.Conv2d(self.out_channels, self.out_channels*2, kernel_size=1, padding=0),
                                                nn.ReLU(),
                                                nn.Conv2d(self.out_channels*2, self.D, kernel_size=1, padding=0),
                                                )
                self.simple_depth_nets.append(simple_depth_net)
        if self.global_learnable_bin:
            self.global_learnable_bin=nn.Parameter(torch.ones(self.D)/self.D)
        else:
            self.global_learnable_bin=None
            






###############################################################################################################################################################################
# # Copyright (c) OpenMMLab. All rights reserved.
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from mmcv.cnn import build_conv_layer
# from mmcv.runner import BaseModule, force_fp32
# from torch.cuda.amp.autocast_mode import autocast
# from torch.utils.checkpoint import checkpoint

# from mmdet3d.ops.bev_pool_v2.bev_pool import bev_pool_v2
# from mmdet.models.backbones.resnet import BasicBlock
# from ..builder import NECKS
# from .zero_out_between_ones import zero_out_between_ones


# @NECKS.register_module()
# class LSSViewTransformer(BaseModule):
#     r"""Lift-Splat-Shoot view transformer with BEVPoolv2 implementation.

#     Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_ and
#         `paper <https://arxiv.org/abs/2211.17111>`

#     Args:
#         grid_config (dict): Config of grid alone each axis in format of
#             (lower_bound, upper_bound, interval). axis in {x,y,z,depth}.
#         input_size (tuple(int)): Size of input images in format of (height,
#             width).
#         downsample (int): Down sample factor from the input size to the feature
#             size.
#         in_channels (int): Channels of input feature.
#         out_channels (int): Channels of transformed feature.
#         accelerate (bool): Whether the view transformation is conducted with
#             acceleration. Note: the intrinsic and extrinsic of cameras should
#             be constant when 'accelerate' is set true.
#         sid (bool): Whether to use Spacing Increasing Discretization (SID)
#             depth distribution as `STS: Surround-view Temporal Stereo for
#             Multi-view 3D Detection`.
#         collapse_z (bool): Whether to collapse in z direction.
#     """

#     def __init__(
#         self,
#         grid_config,
#         input_size,
#         downsample=16,
#         in_channels=512,
#         out_channels=64,
#         accelerate=False,
#         sid=False,
#         collapse_z=True,
#     ):
#         super(LSSViewTransformer, self).__init__()
#         self.grid_config = grid_config
#         self.downsample = downsample
#         self.create_grid_infos(**grid_config)
#         self.sid = sid
#         self.frustum = self.create_frustum(grid_config['depth'],
#                                            input_size, downsample)
#         self.out_channels = out_channels
#         self.in_channels = in_channels
#         self.depth_net = nn.Conv2d(
#             in_channels, self.D + self.out_channels, kernel_size=1, padding=0)
#         self.accelerate = accelerate
#         self.initial_flag = True
#         self.collapse_z = collapse_z

#     def create_grid_infos(self, x, y, z, **kwargs):
#         """Generate the grid information including the lower bound, interval,
#         and size.

#         Args:
#             x (tuple(float)): Config of grid alone x axis in format of
#                 (lower_bound, upper_bound, interval).
#             y (tuple(float)): Config of grid alone y axis in format of
#                 (lower_bound, upper_bound, interval).
#             z (tuple(float)): Config of grid alone z axis in format of
#                 (lower_bound, upper_bound, interval).
#             **kwargs: Container for other potential parameters
#         """
#         self.grid_lower_bound = torch.Tensor([cfg[0] for cfg in [x, y, z]])
#         self.grid_interval = torch.Tensor([cfg[2] for cfg in [x, y, z]])
#         self.grid_size = torch.Tensor([(cfg[1] - cfg[0]) / cfg[2]
#                                        for cfg in [x, y, z]])

#     def create_frustum(self, depth_cfg, input_size, downsample):
#         """Generate the frustum template for each image.

#         Args:
#             depth_cfg (tuple(float)): Config of grid alone depth axis in format
#                 of (lower_bound, upper_bound, interval).
#             input_size (tuple(int)): Size of input images in format of (height,
#                 width).
#             downsample (int): Down sample scale factor from the input size to
#                 the feature size.
#         """
#         H_in, W_in = input_size
#         H_feat, W_feat = H_in // downsample, W_in // downsample
#         d = torch.arange(*depth_cfg, dtype=torch.float)\
#             .view(-1, 1, 1).expand(-1, H_feat, W_feat)
#         self.D = d.shape[0]
#         if self.sid:
#             d_sid = torch.arange(self.D).float()
#             depth_cfg_t = torch.tensor(depth_cfg).float()
#             d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
#                               torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
#             d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
#         x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
#             .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
#         y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
#             .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)

#         # D x H x W x 3
#         return torch.stack((x, y, d), -1)

    # def get_lidar_coor(self, sensor2ego, ego2global, cam2imgs, post_rots, post_trans,
    #                    bda):
    #     """Calculate the locations of the frustum points in the lidar
    #     coordinate system.

    #     Args:
    #         rots (torch.Tensor): Rotation from camera coordinate system to
    #             lidar coordinate system in shape (B, N_cams, 3, 3).
    #         trans (torch.Tensor): Translation from camera coordinate system to
    #             lidar coordinate system in shape (B, N_cams, 3).
    #         cam2imgs (torch.Tensor): Camera intrinsic matrixes in shape
    #             (B, N_cams, 3, 3).
    #         post_rots (torch.Tensor): Rotation in camera coordinate system in
    #             shape (B, N_cams, 3, 3). It is derived from the image view
    #             augmentation.
    #         post_trans (torch.Tensor): Translation in camera coordinate system
    #             derived from image view augmentation in shape (B, N_cams, 3).

    #     Returns:
    #         torch.tensor: Point coordinates in shape
    #             (B, N_cams, D, ownsample, 3)
    #     """
    #     B, N, _, _ = sensor2ego.shape

    #     # post-transformation
    #     # B x N x D x H x W x 3
    #     points = self.frustum.to(sensor2ego) - post_trans.view(B, N, 1, 1, 1, 3)
    #     points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3)\
    #         .matmul(points.unsqueeze(-1))

    #     # cam_to_ego
    #     points = torch.cat(
    #         (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)
    #     combine = sensor2ego[:,:,:3,:3].matmul(torch.inverse(cam2imgs))
    #     points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
    #     points += sensor2ego[:,:,:3, 3].view(B, N, 1, 1, 1, 3)
    #     points = bda.view(B, 1, 1, 1, 1, 3,
    #                       3).matmul(points.unsqueeze(-1)).squeeze(-1)
    #     return points

#     def init_acceleration_v2(self, coor):
#         """Pre-compute the necessary information in acceleration including the
#         index of points in the final feature.

#         Args:
#             coor (torch.tensor): Coordinate of points in lidar space in shape
#                 (B, N_cams, D, H, W, 3).
#             x (torch.tensor): Feature of points in shape
#                 (B, N_cams, D, H, W, C).
#         """

#         ranks_bev, ranks_depth, ranks_feat, \
#             interval_starts, interval_lengths = \
#             self.voxel_pooling_prepare_v2(coor)

#         self.ranks_bev = ranks_bev.int().contiguous()
#         self.ranks_feat = ranks_feat.int().contiguous()
#         self.ranks_depth = ranks_depth.int().contiguous()
#         self.interval_starts = interval_starts.int().contiguous()
#         self.interval_lengths = interval_lengths.int().contiguous()

#     def voxel_pooling_v2(self, coor, depth, feat):
#         ranks_bev, ranks_depth, ranks_feat, \
#             interval_starts, interval_lengths = \
#             self.voxel_pooling_prepare_v2(coor)
#         if ranks_feat is None:
#             print('warning ---> no points within the predefined '
#                   'bev receptive field')
#             dummy = torch.zeros(size=[
#                 feat.shape[0], feat.shape[2],
#                 int(self.grid_size[2]),
#                 int(self.grid_size[0]),
#                 int(self.grid_size[1])
#             ]).to(feat)
#             dummy = torch.cat(dummy.unbind(dim=2), 1)
#             return dummy
#         feat = feat.permute(0, 1, 3, 4, 2)
#         bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
#                           int(self.grid_size[1]), int(self.grid_size[0]),
#                           feat.shape[-1])  # (B, Z, Y, X, C)
#         bev_feat = bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
#                                bev_feat_shape, interval_starts,
#                                interval_lengths)
#         # collapse Z
#         if self.collapse_z:
#             bev_feat = torch.cat(bev_feat.unbind(dim=2), 1)
#         return bev_feat

#     def voxel_pooling_prepare_v2(self, coor):
#         """Data preparation for voxel pooling.

#         Args:
#             coor (torch.tensor): Coordinate of points in the lidar space in
#                 shape (B, N, D, H, W, 3).

#         Returns:
#             tuple[torch.tensor]: Rank of the voxel that a point is belong to
#                 in shape (N_Points); Reserved index of points in the depth
#                 space in shape (N_Points). Reserved index of points in the
#                 feature space in shape (N_Points).
#         """
#         B, N, D, H, W, _ = coor.shape
#         num_points = B * N * D * H * W
#         # record the index of selected points for acceleration purpose
#         ranks_depth = torch.range(
#             0, num_points - 1, dtype=torch.int, device=coor.device)
#         ranks_feat = torch.range(
#             0, num_points // D - 1, dtype=torch.int, device=coor.device)
#         ranks_feat = ranks_feat.reshape(B, N, 1, H, W)
#         ranks_feat = ranks_feat.expand(B, N, D, H, W).flatten()
#         # convert coordinate into the voxel space
#         coor = ((coor - self.grid_lower_bound.to(coor)) /
#                 self.grid_interval.to(coor))
#         coor = coor.long().view(num_points, 3)
#         batch_idx = torch.range(0, B - 1).reshape(B, 1). \
#             expand(B, num_points // B).reshape(num_points, 1).to(coor)
#         coor = torch.cat((coor, batch_idx), 1)

#         # filter out points that are outside box
#         kept = (coor[:, 0] >= 0) & (coor[:, 0] < self.grid_size[0]) & \
#                (coor[:, 1] >= 0) & (coor[:, 1] < self.grid_size[1]) & \
#                (coor[:, 2] >= 0) & (coor[:, 2] < self.grid_size[2])
#         if len(kept) == 0:
#             return None, None, None, None, None
#         coor, ranks_depth, ranks_feat = \
#             coor[kept], ranks_depth[kept], ranks_feat[kept]
#         # get tensors from the same voxel next to each other
#         ranks_bev = coor[:, 3] * (
#             self.grid_size[2] * self.grid_size[1] * self.grid_size[0])
#         ranks_bev += coor[:, 2] * (self.grid_size[1] * self.grid_size[0])
#         ranks_bev += coor[:, 1] * self.grid_size[0] + coor[:, 0]
#         order = ranks_bev.argsort()
#         ranks_bev, ranks_depth, ranks_feat = \
#             ranks_bev[order], ranks_depth[order], ranks_feat[order]

#         kept = torch.ones(
#             ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
#         kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
#         interval_starts = torch.where(kept)[0].int()
#         if len(interval_starts) == 0:
#             return None, None, None, None, None
#         interval_lengths = torch.zeros_like(interval_starts)
#         interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
#         interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
#         return ranks_bev.int().contiguous(), ranks_depth.int().contiguous(
#         ), ranks_feat.int().contiguous(), interval_starts.int().contiguous(
#         ), interval_lengths.int().contiguous()

#     def pre_compute(self, input):
#         if self.initial_flag:
#             coor = self.get_lidar_coor(*input[1:7])
#             self.init_acceleration_v2(coor)
#             self.initial_flag = False

#     def view_transform_core(self, input, depth, tran_feat):
#         B, N, C, H, W = input[0].shape

#         # Lift-Splat
#         if self.accelerate:
#             feat = tran_feat.view(B, N, self.out_channels, H, W)
#             feat = feat.permute(0, 1, 3, 4, 2)
#             depth = depth.view(B, N, self.D, H, W)
#             bev_feat_shape = (depth.shape[0], int(self.grid_size[2]),
#                               int(self.grid_size[1]), int(self.grid_size[0]),
#                               feat.shape[-1])  # (B, Z, Y, X, C)
#             bev_feat = bev_pool_v2(depth, feat, self.ranks_depth,
#                                    self.ranks_feat, self.ranks_bev,
#                                    bev_feat_shape, self.interval_starts,
#                                    self.interval_lengths)

#             bev_feat = bev_feat.squeeze(2)
#         else:
#             coor = self.get_lidar_coor(*input[1:7])
#             bev_feat = self.voxel_pooling_v2(
#                 coor, depth.view(B, N, self.D, H, W),
#                 tran_feat.view(B, N, self.out_channels, H, W))
#         return bev_feat, depth

#     def view_transform(self, input, depth, tran_feat):
#         if self.accelerate:
#             self.pre_compute(input)
#         return self.view_transform_core(input, depth, tran_feat)

#     def forward(self, input):
#         """Transform image-view feature into bird-eye-view feature.

#         Args:
#             input (list(torch.tensor)): of (image-view feature, rots, trans,
#                 intrins, post_rots, post_trans)

#         Returns:
#             torch.tensor: Bird-eye-view feature in shape (B, C, H_BEV, W_BEV)
#         """
#         x = input[0]
#         B, N, C, H, W = x.shape
#         x = x.view(B * N, C, H, W)
#         x = self.depth_net(x)

#         depth_digit = x[:, :self.D, ...]
#         tran_feat = x[:, self.D:self.D + self.out_channels, ...]
#         depth = depth_digit.softmax(dim=1)
#         return self.view_transform(input, depth, tran_feat)

#     def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
#         return None


# class _ASPPModule(nn.Module):

#     def __init__(self, inplanes, planes, kernel_size, padding, dilation,
#                  BatchNorm):
#         super(_ASPPModule, self).__init__()
#         self.atrous_conv = nn.Conv2d(
#             inplanes,
#             planes,
#             kernel_size=kernel_size,
#             stride=1,
#             padding=padding,
#             dilation=dilation,
#             bias=False)
#         self.bn = BatchNorm(planes)
#         self.relu = nn.ReLU()

#         self._init_weight()

#     def forward(self, x):
#         x = self.atrous_conv(x)
#         x = self.bn(x)

#         return self.relu(x)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# class ASPP(nn.Module):

#     def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
#         super(ASPP, self).__init__()

#         dilations = [1, 6, 12, 18]

#         self.aspp1 = _ASPPModule(
#             inplanes,
#             mid_channels,
#             1,
#             padding=0,
#             dilation=dilations[0],
#             BatchNorm=BatchNorm)
#         self.aspp2 = _ASPPModule(
#             inplanes,
#             mid_channels,
#             3,
#             padding=dilations[1],
#             dilation=dilations[1],
#             BatchNorm=BatchNorm)
#         self.aspp3 = _ASPPModule(
#             inplanes,
#             mid_channels,
#             3,
#             padding=dilations[2],
#             dilation=dilations[2],
#             BatchNorm=BatchNorm)
#         self.aspp4 = _ASPPModule(
#             inplanes,
#             mid_channels,
#             3,
#             padding=dilations[3],
#             dilation=dilations[3],
#             BatchNorm=BatchNorm)

#         self.global_avg_pool = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
#             BatchNorm(mid_channels),
#             nn.ReLU(),
#         )
#         self.conv1 = nn.Conv2d(
#             int(mid_channels * 5), inplanes, 1, bias=False)
#         self.bn1 = BatchNorm(inplanes)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)
#         self._init_weight()

#     def forward(self, x):
#         x1 = self.aspp1(x)
#         x2 = self.aspp2(x)
#         x3 = self.aspp3(x)
#         x4 = self.aspp4(x)
#         x5 = self.global_avg_pool(x)
#         x5 = F.interpolate(
#             x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
#         x = torch.cat((x1, x2, x3, x4, x5), dim=1)

#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)

#         return self.dropout(x)

#     def _init_weight(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()


# class Mlp(nn.Module):

#     def __init__(self,
#                  in_features,
#                  hidden_features=None,
#                  out_features=None,
#                  act_layer=nn.ReLU,
#                  drop=0.0):
#         super().__init__()
#         out_features = out_features or in_features
#         hidden_features = hidden_features or in_features
#         self.fc1 = nn.Linear(in_features, hidden_features)
#         self.act = act_layer()
#         self.drop1 = nn.Dropout(drop)
#         self.fc2 = nn.Linear(hidden_features, out_features)
#         self.drop2 = nn.Dropout(drop)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.act(x)
#         x = self.drop1(x)
#         x = self.fc2(x)
#         x = self.drop2(x)
#         return x


# class SELayer(nn.Module):

#     def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
#         super().__init__()
#         self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
#         self.act1 = act_layer()
#         self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
#         self.gate = gate_layer()

#     def forward(self, x, x_se):
#         x_se = self.conv_reduce(x_se)
#         x_se = self.act1(x_se)
#         x_se = self.conv_expand(x_se)
#         return x * self.gate(x_se)


# class DepthNet(nn.Module):

#     def __init__(self,
#                  in_channels,
#                  mid_channels,
#                  context_channels,
#                  depth_channels,
#                  use_dcn=True,
#                  use_aspp=True,
#                  with_cp=False,
#                  stereo=False,
#                  bias=0.0,
#                  aspp_mid_channels=-1,
#                  depth2occ=False,
#                  gaussion=False,
#                  length=22,):
#         super(DepthNet, self).__init__()
#         self.reduce_conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#         )
#         self.context_conv = nn.Conv2d(
#             mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
#         self.bn = nn.BatchNorm1d(27)
#         self.depth_mlp = Mlp(27, mid_channels, mid_channels)
#         self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
#         self.context_mlp = Mlp(27, mid_channels, mid_channels)
#         self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
#         depth_conv_input_channels = mid_channels
#         downsample = None

#         if stereo:
#             depth_conv_input_channels += depth_channels
#             downsample = nn.Conv2d(depth_conv_input_channels,
#                                     mid_channels, 1, 1, 0)
#             cost_volumn_net = []
#             for stage in range(int(2)):
#                 cost_volumn_net.extend([
#                     nn.Conv2d(depth_channels, depth_channels, kernel_size=3,
#                               stride=2, padding=1),
#                     nn.BatchNorm2d(depth_channels)])
#             self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
#             self.bias = bias
#         depth_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
#                                       downsample=downsample),
#                            BasicBlock(mid_channels, mid_channels),
#                            BasicBlock(mid_channels, mid_channels)]
#         if use_aspp:
#             if aspp_mid_channels<0:
#                 aspp_mid_channels = mid_channels
#             depth_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
#         if use_dcn:
#             depth_conv_list.append(
#                 build_conv_layer(
#                     cfg=dict(
#                         type='DCN',
#                         in_channels=mid_channels,
#                         out_channels=mid_channels,
#                         kernel_size=3,
#                         padding=1,
#                         groups=4,
#                         im2col_step=128,
#                     )))
#         depth_conv_list.append(
#             nn.Conv2d(
#                 mid_channels,
#                 depth_channels,
#                 kernel_size=1,
#                 stride=1,
#                 padding=0))
#         self.depth_conv = nn.Sequential(*depth_conv_list)

#         ############################
#         self.gaussion=gaussion
#         self.depth2occ=depth2occ
#         self.length=length
#         if self.depth2occ:
#             occ_channels=self.length
#             occ_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
#                                         downsample=downsample),
#                             BasicBlock(mid_channels, mid_channels),
#                             BasicBlock(mid_channels, mid_channels)]
#             if use_aspp:
#                 if aspp_mid_channels<0:
#                     aspp_mid_channels = mid_channels
#                 occ_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
#             if use_dcn:
#                 occ_conv_list.append(
#                     build_conv_layer(
#                         cfg=dict(
#                             type='DCN',
#                             in_channels=mid_channels,
#                             out_channels=mid_channels,
#                             kernel_size=3,
#                             padding=1,
#                             groups=4,
#                             im2col_step=128,
#                         )))
#             if self.gaussion:
#                 occ_conv_list.append(
#                     nn.Conv2d(
#                         mid_channels,
#                         occ_channels*2,
#                         kernel_size=1,
#                         stride=1,
#                         padding=0))
#             else:    
#                 occ_conv_list.append(
#                     nn.Conv2d(
#                         mid_channels,
#                         occ_channels,
#                         kernel_size=1,
#                         stride=1,
#                         padding=0))
#             self.occ_conv = nn.Sequential(*occ_conv_list)
#         ################################
#         self.with_cp = with_cp
#         self.depth_channels = depth_channels

#     def gen_grid(self, metas, B, N, D, H, W, hi, wi):
#         frustum = metas['frustum']
#         points = frustum - metas['post_trans'].view(B, N, 1, 1, 1, 3)
#         points = torch.inverse(metas['post_rots']).view(B, N, 1, 1, 1, 3, 3) \
#             .matmul(points.unsqueeze(-1))
#         points = torch.cat(
#             (points[..., :2, :] * points[..., 2:3, :], points[..., 2:3, :]), 5)

#         rots = metas['k2s_sensor'][:, :, :3, :3].contiguous()
#         trans = metas['k2s_sensor'][:, :, :3, 3].contiguous()
#         combine = rots.matmul(torch.inverse(metas['intrins']))

#         points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points)
#         points += trans.view(B, N, 1, 1, 1, 3, 1)
#         neg_mask = points[..., 2, 0] < 1e-3
#         points = metas['intrins'].view(B, N, 1, 1, 1, 3, 3).matmul(points)
#         points = points[..., :2, :] / points[..., 2:3, :]

#         points = metas['post_rots'][...,:2,:2].view(B, N, 1, 1, 1, 2, 2).matmul(
#             points).squeeze(-1)
#         points += metas['post_trans'][...,:2].view(B, N, 1, 1, 1, 2)

#         px = points[..., 0] / (wi - 1.0) * 2.0 - 1.0
#         py = points[..., 1] / (hi - 1.0) * 2.0 - 1.0
#         px[neg_mask] = -2
#         py[neg_mask] = -2
#         grid = torch.stack([px, py], dim=-1)
#         grid = grid.view(B * N, D * H, W, 2)
#         return grid

#     def calculate_cost_volumn(self, metas):
#         prev, curr = metas['cv_feat_list']
#         group_size = 4
#         _, c, hf, wf = curr.shape
#         hi, wi = hf * 4, wf * 4
#         B, N, _ = metas['post_trans'].shape
#         D, H, W, _ = metas['frustum'].shape
#         grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)

#         prev = prev.view(B * N, -1, H, W)
#         curr = curr.view(B * N, -1, H, W)
#         cost_volumn = 0
#         # process in group wise to save memory
#         for fid in range(curr.shape[1] // group_size):
#             prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
#             wrap_prev = F.grid_sample(prev_curr, grid,
#                                       align_corners=True,
#                                       padding_mode='zeros')
#             curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
#             cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
#                               wrap_prev.view(B * N, -1, D, H, W)
#             cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)
#             cost_volumn += cost_volumn_tmp
#         if not self.bias == 0:
#             invalid = wrap_prev[:, 0, ...].view(B * N, D, H, W) == 0
#             cost_volumn[invalid] = cost_volumn[invalid] + self.bias
#         cost_volumn = - cost_volumn
#         cost_volumn = cost_volumn.softmax(dim=1)
#         return cost_volumn

#     def forward(self, x, mlp_input, stereo_metas=None):
#         mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
#         x = self.reduce_conv(x)
#         context_se = self.context_mlp(mlp_input)[..., None, None]
#         context = self.context_se(x, context_se)
#         context = self.context_conv(context)
#         depth_se = self.depth_mlp(mlp_input)[..., None, None]
#         depth_ = self.depth_se(x, depth_se)

#         if not stereo_metas is None:
#             if stereo_metas['cv_feat_list'][0] is None:
#                 BN, _, H, W = x.shape
#                 scale_factor = float(stereo_metas['downsample'])/\
#                                stereo_metas['cv_downsample']
#                 cost_volumn = \
#                     torch.zeros((BN, self.depth_channels,
#                                  int(H*scale_factor),
#                                  int(W*scale_factor))).to(x)
#             else:
#                 with torch.no_grad():
#                     cost_volumn = self.calculate_cost_volumn(stereo_metas)
#             cost_volumn = self.cost_volumn_net(cost_volumn)
#             depth_ = torch.cat([depth_, cost_volumn], dim=1)
#         if self.with_cp:
#             depth = checkpoint(self.depth_conv, depth_)
#         else:
#             depth = self.depth_conv(depth_)
#         #######################
#         if self.depth2occ:
#             if self.with_cp:
#                 occ = checkpoint(self.occ_conv, depth_)
#             else:
#                 occ = self.occ_conv(depth_)
#             if self.gaussion:
#                 return torch.cat([depth, context], dim=1),occ
#             return torch.cat([depth, context, occ], dim=1)
#         #######################
#         return torch.cat([depth, context], dim=1)


# class DepthAggregation(nn.Module):
#     """pixel cloud feature extraction."""

#     def __init__(self, in_channels, mid_channels, out_channels):
#         super(DepthAggregation, self).__init__()

#         self.reduce_conv = nn.Sequential(
#             nn.Conv2d(
#                 in_channels,
#                 mid_channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#         )

#         self.conv = nn.Sequential(
#             nn.Conv2d(
#                 mid_channels,
#                 mid_channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(
#                 mid_channels,
#                 mid_channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=False),
#             nn.BatchNorm2d(mid_channels),
#             nn.ReLU(inplace=True),
#         )

#         self.out_conv = nn.Sequential(
#             nn.Conv2d(
#                 mid_channels,
#                 out_channels,
#                 kernel_size=3,
#                 stride=1,
#                 padding=1,
#                 bias=True),
#             # nn.BatchNorm3d(out_channels),
#             # nn.ReLU(inplace=True),
#         )

#     @autocast(False)
#     def forward(self, x):
#         x = checkpoint(self.reduce_conv, x)
#         short_cut = x
#         x = checkpoint(self.conv, x)
#         x = short_cut + x
#         x = self.out_conv(x)
#         return x


# @NECKS.register_module()
# class LSSViewTransformerBEVDepth(LSSViewTransformer):

#     def __init__(self, loss_depth_weight=3.0, depthnet_cfg=dict(),depth2occ=False,hidden_supervise=True,use_gt_depth=False,
#     direct_learn_occ=False,only_supervise_front_part=False,use_gt_occ=False,gaussion=False,length=22,
#      **kwargs):
#         super(LSSViewTransformerBEVDepth, self).__init__(**kwargs)
#         self.loss_depth_weight = loss_depth_weight
#         self.depth_net = DepthNet(self.in_channels, self.in_channels,
#                                   self.out_channels, self.D, **depthnet_cfg)
#         self.depth2occ=depth2occ                   
#         self.hidden_supervise=hidden_supervise  
#         self.use_gt_depth=use_gt_depth      
#         self.use_gt_occ=use_gt_occ
#         self.direct_learn_occ=direct_learn_occ
#         self.only_supervise_front_part=only_supervise_front_part
#         self.gaussion=gaussion

#         self.length=length

#     def get_mlp_input(self, sensor2ego, ego2global, intrin, post_rot, post_tran, bda):
#         B, N, _, _ = sensor2ego.shape
#         bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
#         mlp_input = torch.stack([
#             intrin[:, :, 0, 0],
#             intrin[:, :, 1, 1],
#             intrin[:, :, 0, 2],
#             intrin[:, :, 1, 2],
#             post_rot[:, :, 0, 0],
#             post_rot[:, :, 0, 1],
#             post_tran[:, :, 0],
#             post_rot[:, :, 1, 0],
#             post_rot[:, :, 1, 1],
#             post_tran[:, :, 1],
#             bda[:, :, 0, 0],
#             bda[:, :, 0, 1],
#             bda[:, :, 1, 0],
#             bda[:, :, 1, 1],
#             bda[:, :, 2, 2],], dim=-1)
#         sensor2ego = sensor2ego[:,:,:3,:].reshape(B, N, -1)
#         mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
#         return mlp_input

#     def get_downsampled_gt_depth(self, gt_depths):
#         """
#         Input:
#             gt_depths: [B, N, H, W]
#         Output:
#             gt_depths: [B*N*h*w, d]
#         """
#         if not self.training:
#             gt_depths=gt_depths[0]
#         B, N, H, W = gt_depths.shape
#         gt_depths = gt_depths.view(B * N, H // self.downsample,
#                                    self.downsample, W // self.downsample,
#                                    self.downsample, 1)
#         gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
#         gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
#         gt_depths_tmp = torch.where(gt_depths == 0.0,
#                                     1e5 * torch.ones_like(gt_depths),
#                                     gt_depths)
#         gt_depths = torch.min(gt_depths_tmp, dim=-1).values
#         gt_depths = gt_depths.view(B * N, H // self.downsample,
#                                    W // self.downsample)

#         if not self.sid:
#             gt_depths = (gt_depths - (self.grid_config['depth'][0] -
#                                       self.grid_config['depth'][2])) / \
#                         self.grid_config['depth'][2]
#         else:
#             gt_depths = torch.log(gt_depths) - torch.log(
#                 torch.tensor(self.grid_config['depth'][0]).float())
#             gt_depths = gt_depths * (self.D - 1) / torch.log(
#                 torch.tensor(self.grid_config['depth'][1] - 1.).float() /
#                 self.grid_config['depth'][0])
#             gt_depths = gt_depths + 1.
#         gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
#                                 gt_depths, torch.zeros_like(gt_depths))
#         gt_depths = F.one_hot(
#             gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
#                                                                            1:]
#         return gt_depths.float()
#     def get_downsampled_gt_hidden(self, gt_depths,gt_hiddens,direct_learn_occ=False):
#         """
#         Input:
#             gt_depths: [B, N, H, W]
#         Output:
#             gt_depths: [B*N*h*w, d]
#         """
#         # import pdb; pdb.set_trace()
#         if not self.training:  
#             gt_depths=gt_depths[0]
#             gt_hiddens=gt_hiddens[0]
#         B, N, H, W = gt_depths.shape
#         # import pdb; pdb.set_trace()
#         gt_depths = gt_depths.view(B * N, H // self.downsample,
#                                    self.downsample, W // self.downsample,
#                                    self.downsample, 1)
#         gt_hiddens=gt_hiddens.view(B * N, H // self.downsample,
#                                       self.downsample, W // self.downsample,
#                                       self.downsample, 1)                           
#         gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
#         gt_hiddens=gt_hiddens.permute(0, 1, 3, 5, 2, 4).contiguous()
#         gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
#         gt_hiddens=gt_hiddens.view(-1, self.downsample * self.downsample)
#         gt_depths_tmp = torch.where(gt_depths == 0.0,
#                                     1e5 * torch.ones_like(gt_depths),
#                                     gt_depths)
#         gt_depths,idx = torch.min(gt_depths_tmp, dim=-1)
#         gt_hiddens=gt_hiddens[torch.arange(gt_hiddens.shape[0]).to(gt_hiddens.device).long(),idx]
#         gt_depths = gt_depths.view(B * N, H // self.downsample,
#                                    W // self.downsample)
#         gt_hiddens=gt_hiddens.view(B * N, H // self.downsample,
#                                       W // self.downsample)

#         if not self.sid:
#             gt_depths = (gt_depths - (self.grid_config['depth'][0] -
#                                       self.grid_config['depth'][2])) / \
#                         self.grid_config['depth'][2]
#             gt_hiddens = (gt_hiddens - (self.grid_config['depth'][0] -
#                                         self.grid_config['depth'][2])) / \
#                             self.grid_config['depth'][2]
#         else:
#             gt_depths = torch.log(gt_depths) - torch.log(
#                 torch.tensor(self.grid_config['depth'][0]).float())
#             gt_depths = gt_depths * (self.D - 1) / torch.log(
#                 torch.tensor(self.grid_config['depth'][1] - 1.).float() /
#                 self.grid_config['depth'][0])
#             gt_depths = gt_depths + 1.
#         gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
#                                 gt_depths, torch.zeros_like(gt_depths))
#         gt_hiddens = torch.where((gt_hiddens < self.D + 1) & (gt_hiddens >= 0.0),
#                                 gt_hiddens, torch.zeros_like(gt_hiddens))
        
#         disc=gt_hiddens-gt_depths
#         disc=disc.reshape(-1)
#         if not direct_learn_occ:
#             max_len=self.length
#             indices = torch.arange(max_len).to(disc.device).unsqueeze(0).repeat(len(disc), 1)

#             # 利用广播机制，生成一个对比数组
#             occ= indices < disc.unsqueeze(1)
#             gt_depths = F.one_hot(
#             gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
#                                                                            1:]
#             return gt_depths.float(),occ.float(),disc.float()                                                               
#         else:
#             max_len=self.D
            

#             gt_depths = F.one_hot(
#                 gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
#                                                                            1:]
#             gt_hiddens = F.one_hot(
#                 gt_hiddens.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
#                                                                            1:]
#             gt_occ=zero_out_between_ones(gt_depths,gt_hiddens) 
#             return gt_depths.float(),gt_occ.float(),disc.float()                                                                                                        
#     def get_downsampled_gt_hidden_semantics(self, gt_depths,depth_semantics,direct_learn_occ=False):
#         """
#         Input:
#             gt_depths: [B, N, H, W]
#         Output:
#             gt_depths: [B*N*h*w, d]
#         """
#         # import pdb; pdb.set_trace()
#         if not self.training:  
#             gt_depths=gt_depths[0]
#             # gt_hiddens=gt_hiddens[0]
#             depth_semantics=depth_semantics[0]
#         B, N, H, W = gt_depths.shape
#         # import pdb; pdb.set_trace()
#         gt_depths = gt_depths.view(B * N, H // self.downsample,
#                                    self.downsample, W // self.downsample,
#                                    self.downsample, 1)
#         # gt_hiddens=gt_hiddens.view(B * N, H // self.downsample,
#         #                               self.downsample, W // self.downsample,
#         #                               self.downsample, 1)           
#         depth_semantics=depth_semantics.view(B * N, H // self.downsample,
#                                         self.downsample, W // self.downsample,
#                                         self.downsample, 1)

#         gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
#         # gt_hiddens=gt_hiddens.permute(0, 1, 3, 5, 2, 4).contiguous()
#         depth_semantics=depth_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
#         gt_depths = gt_depths.view(-1, self.downsample * self.downsample)
#         # gt_hiddens=gt_hiddens.view(-1, self.downsample * self.downsample)
#         depth_semantics=depth_semantics.view(-1, self.downsample * self.downsample)
#         gt_depths_tmp = torch.where(gt_depths == 0.0,
#                                     1e5 * torch.ones_like(gt_depths),
#                                     gt_depths)
#         gt_depths,idx = torch.min(gt_depths_tmp, dim=-1)
#         # gt_hiddens=gt_hiddens[torch.arange(gt_hiddens.shape[0]).to(gt_hiddens.device).long(),idx]
#         depth_semantics=depth_semantics[torch.arange(depth_semantics.shape[0]).to(depth_semantics.device).long(),idx]
#         gt_depths = gt_depths.view(B * N, H // self.downsample,
#                                    W // self.downsample)
#         # gt_hiddens=gt_hiddens.view(B * N, H // self.downsample,
#                                     #   W // self.downsample)
#         depth_semantics=depth_semantics.view(B * N, H // self.downsample,
#                                         W // self.downsample)

#         # if not self.sid:
#         #     gt_depths = (gt_depths - (self.grid_config['depth'][0] -
#         #                               self.grid_config['depth'][2])) / \
#         #                 self.grid_config['depth'][2]
#         #     gt_hiddens = (gt_hiddens - (self.grid_config['depth'][0] -
#         #                                 self.grid_config['depth'][2])) / \
#         #                     self.grid_config['depth'][2]
#         # else:
#         #     gt_depths = torch.log(gt_depths) - torch.log(
#         #         torch.tensor(self.grid_config['depth'][0]).float())
#         #     gt_depths = gt_depths * (self.D - 1) / torch.log(
#         #         torch.tensor(self.grid_config['depth'][1] - 1.).float() /
#         #         self.grid_config['depth'][0])
#         #     gt_depths = gt_depths + 1.
#         # gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0),
#         #                         gt_depths, torch.zeros_like(gt_depths))
#         # gt_hiddens = torch.where((gt_hiddens < self.D + 1) & (gt_hiddens >= 0.0),
#         #                         gt_hiddens, torch.zeros_like(gt_hiddens))
        
#         # disc=gt_hiddens-gt_depths
#         # disc=disc.reshape(-1)
#         # if not direct_learn_occ:
#         #     max_len=self.length
#         #     indices = torch.arange(max_len).to(disc.device).unsqueeze(0).repeat(len(disc), 1)

#         #     # 利用广播机制，生成一个对比数组
#         #     occ= indices < disc.unsqueeze(1)
#         #     gt_depths = F.one_hot(
#         #     gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
#         #                                                                    1:]
#         #     return gt_depths.float(),occ.float(),disc.float()                                                               
#         # else:
#         #     max_len=self.D
            

#         #     gt_depths = F.one_hot(
#         #         gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
#         #                                                                    1:]
#         #     gt_hiddens = F.one_hot(
#         #         gt_hiddens.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:,
#         #                                                                    1:]
#         #     gt_occ=zero_out_between_ones(gt_depths,gt_hiddens) 
#         return depth_semantics
#     @force_fp32()
#     def get_depth_loss(self, depth_labels, depth_preds):
#         depth_labels = self.get_downsampled_gt_depth(depth_labels)
#         depth_preds = depth_preds.permute(0, 2, 3,
#                                           1).contiguous().view(-1, self.D)
#         fg_mask = torch.max(depth_labels, dim=1).values > 0.0
#         depth_labels = depth_labels[fg_mask]
#         depth_preds = depth_preds[fg_mask]
#         with autocast(enabled=False):
#             depth_loss = F.binary_cross_entropy(
#                 depth_preds,
#                 depth_labels,
#                 reduction='none',
#             ).sum() / max(1.0, fg_mask.sum())
#         return self.loss_depth_weight * depth_loss
#     @force_fp32()
#     def get_hidden_depth_loss(self, depth_labels, depth_preds,hidden_labels,occ_preds):
#         depth_labels,occ_labels,disc = self.get_downsampled_gt_hidden(depth_labels,hidden_labels)
#         depth_preds = depth_preds.permute(0, 2, 3,
#                                           1).contiguous().view(-1, self.D)
#         fg_mask = torch.max(depth_labels, dim=1).values > 0.0
#         # import pdb; pdb.set_trace()
#         depth_labels = depth_labels[fg_mask]
#         depth_preds = depth_preds[fg_mask]
#         with autocast(enabled=False):
#             depth_loss = F.binary_cross_entropy(
#                 depth_preds,
#                 depth_labels,
#                 reduction='none',
#             ).sum() / max(1.0, fg_mask.sum())
#         if self.hidden_supervise:
#             occ_preds = occ_preds.permute(0, 2, 3,
#                                             1).contiguous().view(-1, self.length)
                                     
#             valid_mask=(disc>0)*fg_mask
#             # print(occ_preds.shape,occ_labels.shape,disc.shape,valid_mask.shape,222222222222222222)      
#             occ_labels=occ_labels[valid_mask]
#             occ_preds=occ_preds[valid_mask]

#             with autocast(enabled=False):
#                 occ_loss = F.binary_cross_entropy(
#                     occ_preds,
#                     occ_labels,
#                     reduction='none',
#                 ).sum() / max(1.0, valid_mask.sum())
#         else:
#             occ_loss=torch.tensor(0).to(depth_preds.device)
#         return self.loss_depth_weight * depth_loss,self.loss_depth_weight * occ_loss

#     @force_fp32()
#     def get_occ_depth_loss(self, depth_labels,depth_preds,hidden_labels, occ_preds):
#         depth_labels,occ_labels,disc = self.get_downsampled_gt_hidden(depth_labels,hidden_labels,self.direct_learn_occ)
#         depth_preds = depth_preds.permute(0, 2, 3,
#                                           1).contiguous().view(-1, self.D)
#         fg_mask = torch.max(depth_labels, dim=1).values > 0.0
#         depth_labels = depth_labels[fg_mask]
#         depth_preds = depth_preds[fg_mask]
#         with autocast(enabled=False):
#             depth_loss = F.binary_cross_entropy(
#                 depth_preds,
#                 depth_labels,
#                 reduction='none',
#             ).sum() / max(1.0, fg_mask.sum())
#         if self.hidden_supervise:
#             occ_preds = occ_preds.permute(0, 2, 3,
#                                             1).contiguous().view(-1, self.D)
#             valid_mask=(disc>0)*fg_mask
            
            
#             if valid_mask.sum()>0:
#                 occ_labels=occ_labels[valid_mask]
#                 occ_preds=occ_preds[valid_mask]
#                 occ_preds=torch.where(occ_preds<0 ,torch.zeros_like(occ_preds),occ_preds)
#                 occ_preds=torch.where(occ_preds>1 ,torch.ones_like(occ_preds),occ_preds)
#                 try:    
#                     with autocast(enabled=False):
#                         occ_loss = F.binary_cross_entropy(
#                             occ_preds,
#                             occ_labels,
#                             reduction='none',
#                         )
#                         if self.only_supervise_front_part:
#                             front_mask=occ_labels.flip(dims=[1]).cummax(dim=1)[0].flip(dims=[1])
#                             occ_loss=occ_loss*front_mask.float()
                        
#                             occ_loss=occ_loss.sum() / max(1.0, valid_mask.sum())
#                 except Exception as e:
#                     occ_loss=torch.tensor(0).to(depth_preds.device)
#             else:
#                 occ_loss=torch.tensor(0).to(depth_preds.device)
#         else:
#             occ_loss=torch.tensor(0).to(depth_preds.device)
#         return self.loss_depth_weight * depth_loss,self.loss_depth_weight * occ_loss

#     def forward(self, input, stereo_metas=None):
#         (x, rots, trans, intrins, post_rots, post_trans, bda,
#          mlp_input,gt_depth,gt_hidden) = input[:10]

#         B, N, C, H, W = x.shape
#         x = x.view(B * N, C, H, W)
#         if self.gaussion:
#             x ,occ_weight_= self.depth_net(x, mlp_input, stereo_metas)
#         else:
#             x = self.depth_net(x, mlp_input, stereo_metas)
#         depth_digit = x[:, :self.D, ...]
#         tran_feat = x[:, self.D:self.D + self.out_channels, ...]
        
#         depth = depth_digit.softmax(dim=1)
#         # import pdb; pdb.set_trace()
#         #####################
#         if self.use_gt_depth:
#             if self.depth2occ:
#                 gt_depth,gt_occ,disc=self.get_downsampled_gt_hidden(gt_depth,gt_hidden,self.direct_learn_occ)
#             else:
#                 gt_depth=self.get_downsampled_gt_depth(gt_depth)
#             # depth_labels = self.get_downsampled_gt_depth(depth_labels)
#             depth_preds=depth.clone()
#             b,d,h,w=depth_preds.shape
#             depth_gt_mixed = depth_preds.permute(0, 2, 3,
#                                             1).contiguous().view(-1, self.D)
#             fg_mask = torch.max(gt_depth, dim=1).values > 0.0
#             depth_gt_mixed[fg_mask] = gt_depth[fg_mask]
#             depth_gt_mixed=depth_gt_mixed.reshape(b,h,w,d).permute(0,3,1,2)
# ###################
#         if self.depth2occ:
#             if not self.gaussion:
#                 occ_weight_=x[:,self.D+self.out_channels:,...]
#             occ_weight_=occ_weight_.sigmoid()
#             bn,d,h,w=depth.shape
#             len_occ=occ_weight_.shape[1]
#             occ_weight=occ_weight_.permute(0,2,3,1).reshape(bn*h*w,len_occ)
#             ###############
#             # import pdb; pdb.set_trace()
#             # depth_2_occ=torch.eye(d,d).to(depth.device).unsqueeze(0).repeat(bn*h*w,1,1)
#             # for i in range(len_occ):
#             #     depth_2_occ[:,:d-1-i,i+1:]+=torch.diag_embed(occ_weight[:,i:i+1].repeat(1,d-1-i))
#             ######################
#             depth_2_occ=torch.cat([torch.zeros_like(occ_weight[:,:1]).to(depth.device),occ_weight,torch.zeros(occ_weight.shape[0],d-len_occ).to(depth.device)],dim=1).repeat(1,d).reshape(bn*h*w,d+1,d)[:,:d,:]
#             depth_2_occ=depth_2_occ.triu()+torch.eye(d,d).to(depth.device).unsqueeze(0)
#             ##################
#             if self.use_gt_depth:
#                 occ=torch.matmul(depth_gt_mixed.permute(0,2,3,1).reshape(bn*h*w,1,d),depth_2_occ).reshape(bn,h,w,d).permute(0,3,1,2)
#             else:
#                 occ=torch.matmul(depth.permute(0,2,3,1).reshape(bn*h*w,1,d),depth_2_occ).reshape(bn,h,w,d).permute(0,3,1,2)
#             if self.use_gt_occ:
#                 occ_gt_mixed=occ.clone().permute(0,2,3,1).contiguous().view(-1, self.D)
#                 fg_mask = (disc > 0.0)*fg_mask
#                 if self.only_supervise_front_part:
#                     # import pdb; pdb.set_trace()
#                     front_mask=gt_occ.flip(dims=[1]).cummax(dim=1)[0].flip(dims=[1])
#                     fg_mask=fg_mask[:,None]*front_mask.long()
#                 occ_gt_mixed[fg_mask] = gt_occ[fg_mask]
#                 occ_gt_mixed=occ_gt_mixed.reshape(b,h,w,d).permute(0,3,1,2)
#                 bev_feat, occ_ = self.view_transform(input, occ_gt_mixed, tran_feat)

#             else:
#                 bev_feat, occ_ = self.view_transform(input, occ, tran_feat)
#             # if self.direct_learn_occ:
#             #     return bev_feat, [depth,occ,occ_weight_]
#             return bev_feat, [depth,occ,occ_weight_]
# #############################
#         if self.use_gt_depth:
#             bev_feat, depth_ = self.view_transform(input, depth_gt_mixed, tran_feat)
#         else:
#             bev_feat, depth_ = self.view_transform(input,depth, tran_feat)
#         return bev_feat, depth


# @NECKS.register_module()
# class LSSViewTransformerBEVStereo(LSSViewTransformerBEVDepth):

#     def __init__(self,  **kwargs):
#         super(LSSViewTransformerBEVStereo, self).__init__(**kwargs)
#         self.cv_frustum = self.create_frustum(kwargs['grid_config']['depth'],
#                                               kwargs['input_size'],
#                                               downsample=4)
