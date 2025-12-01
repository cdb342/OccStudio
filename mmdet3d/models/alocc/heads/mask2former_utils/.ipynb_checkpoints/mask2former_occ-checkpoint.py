# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Conv3d, caffe2_xavier_init
from mmcv.cnn.bricks.transformer import (build_positional_encoding,
                                         build_transformer_layer_sequence)
from mmcv.runner import ModuleList, force_fp32

from mmdet.core import build_assigner, build_sampler, reduce_mean, multi_apply
from mmdet.models.builder import HEADS, build_loss

from .base.mmdet_utils import (sample_valid_coords_with_frequencies,
                          get_uncertain_point_coords_3d_with_frequency,
                          preprocess_occupancy_gt, point_sample_3d)

from .base.anchor_free_head import AnchorFreeHead
from .base.maskformer_head import MaskFormerHead
from ..utils.semkitti import semantic_kitti_class_frequencies
from mmdet3d.models.fbbev.modules.occ_loss_utils import nusc_class_frequencies, nusc_class_names
import pdb
from concurrent.futures import ThreadPoolExecutor
from mmdet3d.models import builder
from mmdet3d.models.fbbev.modules.occ_loss_utils import geo_scal_loss, sem_scal_loss, CE_ssc_loss
from mmdet3d.models.fbbev.modules.occ_loss_utils import lovasz_softmax, CustomFocalLoss
from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.datasets.semkitti import semantic_kitti_class_frequencies
from mmdet3d.models.losses import SigLoss
import math
import os
from mmdet3d.models.necks.deformable_lift import broadcast_pred_linear_interpolation as deformable_lift
def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res
# 定义旋转矩阵
R_ = torch.tensor([
    [1, 0, 0],
    [0, 0, -1],
    [0, 1, 0]
], dtype=torch.float32)

# 定义翻转矩阵
F_ = torch.tensor([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
], dtype=torch.float32)

# 定义第一次 permute 的矩阵
P1 = torch.tensor([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
], dtype=torch.float32)

# 定义第二次 permute 的矩阵
P2 = torch.tensor([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
], dtype=torch.float32)

# 组合变换矩阵
T_ = torch.mm(P2, torch.mm(F_, torch.mm(R_, P1)))
dvr = torch.utils.cpp_extension.load("dvr", sources=["tools/ray_iou/lib/dvr/dvr.cpp", "tools/ray_iou/lib/dvr/dvr.cu"], verbose=True, extra_cuda_cflags=['-allow-unsupported-compiler'])
_pc_range = [-40, -40, -1.0, 40, 40, 5.4]
_voxel_size = 0.4
def generate_lidar_rays():

    # prepare lidar ray angles
    pitch_angles = []
    for k in range(10):
        angle = math.pi / 2 - math.atan(k + 1)
        pitch_angles.append(-angle)

    # nuscenes lidar fov: [0.2107773983152201, -0.5439104895672159] (rad)
    while pitch_angles[-1] < 0.21:
        delta = pitch_angles[-1] - pitch_angles[-2]
        pitch_angles.append(pitch_angles[-1] + delta)

    lidar_rays = []
    for pitch_angle in pitch_angles:
        for azimuth_angle in np.arange(0, 360, 1):
            azimuth_angle = np.deg2rad(azimuth_angle)

            x = np.cos(pitch_angle) * np.cos(azimuth_angle)
            y = np.cos(pitch_angle) * np.sin(azimuth_angle)
            z = np.sin(pitch_angle)

            lidar_rays.append((x, y, z))

    return np.array(lidar_rays, dtype=np.float32)

# Mask2former for 3D Occupancy Segmentation
@HEADS.register_module()
class Mask2FormerOccHead(MaskFormerHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`mmcv.ConfigDict` | dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`mmcv.ConfigDict` | dict): Config for
            transformer decoder position encoding. Defaults to None.
        loss_cls (:obj:`mmcv.ConfigDict` | dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`mmcv.ConfigDict` | dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`mmcv.ConfigDict` | dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`mmcv.ConfigDict` | dict): Training config of
            Mask2Former head.
        test_cfg (:obj:`mmcv.ConfigDict` | dict): Testing config of
            Mask2Former head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 feat_channels,
                 out_channels,
                 num_occupancy_classes=20,
                 num_queries=100,
                 num_transformer_feat_level=3,
                 enforce_decoder_input_project=False,
                 transformer_decoder=None,
                 positional_encoding=None,
                 pooling_attn_mask=True,
                 sample_weight_gamma=0.25,
                 align_corners=True,
                 loss_cls=None,
                 loss_mask=None,
                 loss_dice=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None,
                 CE_loss_only=False,
                 use_focal_loss=True,
                 focal_loss_only=False,
                 use_class_weight=True,
                 flash_occ=False,
                 external_in_channels=256,
                    external_out_channels=512,
                    Dz=16,
                flash_occ_vae=False,
                balance_cls_weight=True,
                vq_mf_head=False,
                mask_embed2=False,
                wo_assign=False,
                mask_loss_softmax=False,  
                mask_loss_softmax_adaptive=False,  
                semantic_cluster=False,
                out_channels_embed2=48,
                num_points_img=12544,
                dataset='nusc',
                open_occ=False,
                pred_flow=False,
                flow_l2_loss=False,
                context_post_process=None,
                flow_post_process=None,
                render_depth_sup=False,
                flow_loss_weight=1.0,
                render_depth_sup_st=False,
                flow_class_balance=False,
                ray_former=False,
                sup_occupy_only=False,
                do_history=True,
                history_cat_num=1,
                history_cat_conv_out_channels=None,
                single_bev_num_channels=80,
                interpolation_mode='bilinear',
                flow_with_his=False,
                use_adabin_flow_decoder=False,
                flow_multi_layer=False,
                flow_scale=1.0,
                flow_render_loss=False,
                flow_render_loss_weight=1.0,
                pred_flow_only=False,
                flow_prev2curr=False,
                flow_curr2future=False,
                flow_cosine_loss=False,
                flow_out_channels=2,
                
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        #######
        self.use_adabin_flow_decoder=use_adabin_flow_decoder
        if self.use_adabin_flow_decoder:
            # flow_out_channels=88
            self.bin_invertal=[-22,22,0.5]
            flow_out_channels=len(np.arange(self.bin_invertal[0],self.bin_invertal[1],self.bin_invertal[2]))
            self.flow_out_channels=flow_out_channels
            self.adabin_bin_decoder = nn.Sequential(
                        nn.Linear(feat_channels, feat_channels),
                        nn.ReLU(inplace=True),
                        nn.Linear(feat_channels, flow_out_channels*2),

                    )
            

        self.semantic_cluster=semantic_cluster
        self.open_occ=open_occ
        self.pred_flow=pred_flow
        self.flow_scale=flow_scale
        if context_post_process!=None and not pred_flow_only:
            self.context_post_conv=nn.Sequential(
                            nn.Conv3d(
                                feat_channels,
                                feat_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1
                            ),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(
                                feat_channels,
                                feat_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0
                            )
                )
        else:
            self.context_post_conv=builder.build_backbone(context_post_process) if context_post_process is not None else None
        if self.pred_flow:
            if flow_post_process is not None:
                self.flow_post_conv = builder.build_backbone(flow_post_process)
            else:
                if flow_multi_layer:
                    pass
                else:
                    self.flow_post_conv =  nn.Sequential(
                                nn.Conv3d(
                                    feat_channels,
                                    feat_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1
                                ),
                    nn.ReLU(inplace=True),
                    )
            self.flow_predicter = nn.Sequential(
                nn.Conv3d(
                                feat_channels,
                                feat_channels*2,
                                kernel_size=1,
                                stride=1,
                                padding=0
                            ),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                                feat_channels*2,
                                flow_out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0
                            )
            )
        self.flow_l2_loss=flow_l2_loss
        self.render_depth_sup=render_depth_sup
        if self.render_depth_sup:
            self.sigloss=SigLoss(loss_weight=10.0)
            # generate lidar rays
        lidar_rays = generate_lidar_rays()
        self.lidar_rays = torch.from_numpy(lidar_rays)
        self.flow_loss_weight=flow_loss_weight
        self.render_depth_sup_st=render_depth_sup_st
        self.flow_class_balance=flow_class_balance
        self.sup_occupy_only=sup_occupy_only
        
        self.flow_render_loss=flow_render_loss

        self.flow_render_loss_weight=flow_render_loss_weight
        self.pred_flow_only=pred_flow_only
        self.flow_prev2curr=flow_prev2curr
        if flow_prev2curr:
            x = torch.linspace(0, 199, 200)
            y = torch.linspace(0, 199, 200)
            z = torch.linspace(0, 199, 200)
            X, Y, Z = torch.meshgrid(x, y, z)
            self.vv = torch.stack([X, Y, Z], dim=-1)
        self.flow_curr2future=flow_curr2future
        if flow_curr2future:
            self.future_occ_predictor = nn.Sequential(
                nn.Conv3d(
                    feat_channels,
                    feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1
                ),
                nn.ReLU(inplace=True),
                nn.Conv3d(
                    feat_channels,
                    feat_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )
            )
        self.flow_cosine_loss=flow_cosine_loss
       
        #############
        
        self.num_occupancy_classes = num_occupancy_classes
        self.num_classes = self.num_occupancy_classes
        self.num_queries = num_queries
        # import pdb; pdb.set_trace()
        ''' Transformer Decoder Related '''
        # number of multi-scale features for masked attention
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.transformerlayers.attn_cfgs.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        # if  not pred_flow_only:
        if self.num_transformer_decoder_layers>0:
            self.transformer_decoder = build_transformer_layer_sequence(
                transformer_decoder)
            self.decoder_embed_dims = self.transformer_decoder.embed_dims
        else:
            self.decoder_embed_dims=feat_channels
            self.post_norm=nn.LayerNorm(self.decoder_embed_dims)
        
        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution, align the channel of input features
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv3d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
                
                
        self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
        if self.num_transformer_decoder_layers>0:
            self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        if not self.semantic_cluster:
            self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        if  self.num_transformer_decoder_layers>0:
            self.level_embed = nn.Embedding(self.num_transformer_feat_level, feat_channels)

        ''' Pixel Decoder Related, skipped '''
        if not wo_assign:
            self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        if not pred_flow_only:
            self.mask_embed = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, out_channels))
        
            ################
        
        if mask_embed2:
            self.mask_embed2 = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, out_channels_embed2))
                #####################

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.num_points_img=num_points_img
            self.ori_num_points = self.num_points
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        # create class_weights for semantic_kitti
        self.class_weight = loss_cls.class_weight
        # import pdb; pdb.set_trace()
        if not use_class_weight:
            self.class_weight = np.ones_like(nusc_class_frequencies)
            
        # import pdb;pdb.set_trace()
        if vq_mf_head:
            nusc_class_frequencies=np.ones(self.num_classes)
        else:
            from mmdet3d.models.fbbev.modules.occ_loss_utils import nusc_class_frequencies

        if dataset=='kitti':
            nusc_class_frequencies = semantic_kitti_class_frequencies
        if self.open_occ:
            nusc_class_frequencies=nusc_class_frequencies[np.array([4,10,9,3,5,2,6,7,8,1,11,12,13,14,15,16,17])]
        nusc_class_weights = 1 / np.log(nusc_class_frequencies[:self.num_classes]+0.001)
        norm_nusc_class_weights = nusc_class_weights / nusc_class_weights[0]
        norm_nusc_class_weights = norm_nusc_class_weights.tolist()
        # append the class_weight for background
        if not wo_assign:
            norm_nusc_class_weights.append(self.class_weight[-1])
        if not vq_mf_head and dataset!='kitti':
            norm_nusc_class_weights = [0.0]+norm_nusc_class_weights
        self.class_weight = norm_nusc_class_weights
        
        loss_cls.class_weight = self.class_weight
        # import pdb;pdb.set_trace()
        # frequencies=np.ones_like(nusc_class_frequencies)
        # computing sampling weight        
        sample_weights = 1 / nusc_class_frequencies
        sample_weights = sample_weights / sample_weights.min()
        if not vq_mf_head and dataset!='kitti':
            sample_weights=np.concatenate([[0.0],sample_weights])


        self.baseline_sample_weights = sample_weights
        self.sample_weight_gamma = sample_weight_gamma
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.pooling_attn_mask = pooling_attn_mask
        
        # align_corners
        self.align_corners = align_corners
        self.CE_loss_only=CE_loss_only
        self.focal_loss_only=focal_loss_only
        if self.focal_loss_only:
            self.CE_loss_only=False
        self.use_focal_loss=use_focal_loss
        if self.focal_loss_only:
            self.use_focal_loss=True
        if self.use_focal_loss:
            self.focal_loss = builder.build_loss(dict(type='CustomFocalLoss'))

        ###################################
        loss_weight_cfg=None
        balance_cls_weight=balance_cls_weight
        num_cls=19
        self.empty_idx=18
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
                # import pdb;pdb.set_trace()
                self.class_weights_geo = torch.cat([torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls][-1:] + 0.001)),\
                    torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls][:-1].sum() + 0.001)[None])])
                                                    
            else:
                if num_cls == 17: nusc_class_frequencies[0] += nusc_class_frequencies[-1]
                self.class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:num_cls] + 0.001))
        else:
            self.class_weights = torch.ones(num_cls)/num_cls  # FIXME hardcode 17
        self.flash_occ=flash_occ

        self.Dz=Dz
        self.num_cls=num_cls
        if self.flash_occ:
            self.final_conv = ConvModule(
                            external_in_channels,
                            external_out_channels,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                            conv_cfg=dict(type='Conv2d')
                        )
        self.flash_occ_vae=flash_occ_vae
        self.wo_assign=wo_assign
        self.mask_loss_softmax=mask_loss_softmax
        self.mask_loss_softmax_adaptive=mask_loss_softmax_adaptive
        self.flow_with_his=flow_with_his
        if flow_with_his:
                    #######################################
                # Deal with history
            self.single_bev_num_channels = single_bev_num_channels
            self.do_history = do_history
            self.interpolation_mode = interpolation_mode
            self.history_cat_num = history_cat_num
            self.history_cam_sweep_freq = 0.5 # seconds between each frame
            history_cat_conv_out_channels = (history_cat_conv_out_channels 
                                            if history_cat_conv_out_channels is not None 
                                            else self.single_bev_num_channels)

             ## Embed each sample with its relative temporal offset with current timestep
            conv =  nn.Conv3d
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
    # def adabin_flow_decoder(self, x):
    #     x = self.flow_post_conv(x)
    #     x = self.flow_predicter(x)
    #     cum_depth=torch.cat((torch.zeros_like(depth[:,:1,...]),torch.cumsum(depth,1)[:,:-1]),dim=1)
    #     bin_center=self.grid_config['depth'][0]+(self.grid_config['depth'][1]-self.grid_config['depth'][0])*(depth/2+cum_depth)#BN,n_bin,H,W
        
    #     return x
        
    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda,update_history=True): # align features with 3d shift
        # import pdb;pdb.set_trace()
        if self.flash_occ:
            curr_bev = curr_bev.unsqueeze(-1)
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
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
        # print(seq_ids,self.history_seq_ids)
        # import pdb;pdb.set_trace()
        ## Deal with first batch

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
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1)
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
        if not self.flash_occ:
            # Generate grid
            xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
            ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
            zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
            grid = torch.stack(
                (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)
            # import pdb;pdb.set_trace()
            # This converts BEV indices to meters
            # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
            # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
            feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
            # feat2bev[0, 0] = self.forward_projection.dx[0]
            # feat2bev[1, 1] = self.forward_projection.dx[1]
            # feat2bev[2, 2] = self.forward_projection.dx[2]
            # feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
            # feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
            # feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
            feat2bev[0, 0] = 0.4
            feat2bev[1, 1] =0.4
            feat2bev[2, 2] =0.4
            feat2bev[0, 3] =-40.
            feat2bev[1, 3] =-40.
            feat2bev[2, 3] = -1.
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
            feat2bev[0, 0] = self.forward_projection.dx[0]
            feat2bev[1, 1] = self.forward_projection.dx[1]
            feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
            feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
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

            tmp_bev = self.history_bev
            if voxel_feat: 
                n, mc, z, h, w = tmp_bev.shape
                tmp_bev = tmp_bev.reshape(n, mc, z, h, w)
        
            sampled_history_bev = F.grid_sample(tmp_bev[:,:,0], grid.to(curr_bev.dtype)[...,0,:], align_corners=True, mode=self.interpolation_mode)

        
        # import pdb;pdb.set_trace()

        ## Update history
        # Add in current frame to features & timestep
        self.history_sweep_time = torch.cat(
            [self.history_sweep_time.new_zeros(self.history_sweep_time.shape[0], 1), self.history_sweep_time],
            dim=1) # B x (1 + T)

        if voxel_feat:
            sampled_history_bev = sampled_history_bev.reshape(n, mc, z, h, w)
            curr_bev = curr_bev.reshape(n, c_, z, h, w)
        feats_cat = torch.cat([curr_bev, sampled_history_bev], dim=1) # B x (1 + T) * 80 x H x W or B x (1 + T) * 80 xZ x H x W 

        # Reshape and concatenate features and timestep
        # import pdb;pdb.set_trace()
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

        if self.flash_occ:
            feats_to_return=feats_to_return.squeeze(3)
        # Time conv
        feats_to_return = self.history_keyframe_time_conv(
            feats_to_return.reshape(-1, *feats_to_return.shape[2:])).reshape(
                feats_to_return.shape[0], feats_to_return.shape[1], -1, *feats_to_return.shape[3:]) # B x (1 + T) x 80 xZ x H x W

        # Cat keyframes & conv
        feats_to_return = self.history_keyframe_cat_conv(
            feats_to_return.reshape(
                feats_to_return.shape[0], -1, *feats_to_return.shape[3:])) # B x C x H x W or B x C x Z x H x W
        if self.flash_occ:
            feats_to_return=feats_to_return.unsqueeze(3)
        self.history_bev = feats_cat[:, :-self.single_bev_num_channels, ...].detach().clone()
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        if self.flash_occ:
            feats_to_return=feats_to_return.squeeze(2)
        if self.flow_prev2curr:
            return feats_to_return, grid
        return feats_to_return.clone()


    # @force_fp32()
    # def curr2next_with_flow(self, curr_bev, pred_flow,img_metas, bda,update_history=True): # align features with 3d shift
        
    #     curr_bev = curr_bev.permute(0, 1, 4, 2, 3)
    #     # pred_flow=pred_flow.permute(0, 3, 1,2, 4)
    #     # pred_flow=torch.flip(pred_flow,[-1])

    #     forward_augs = generate_forward_transformation_matrix(bda)

    #     # import pdb;pdb.set_trace()
    #     curr_to_next_ego_rt = torch.stack([
    #         single_img_metas['curr_to_next_ego_rt']
    #         for single_img_metas in img_metas]).to(curr_bev)
    #     # print(seq_ids,self.history_seq_ids)
    #     # Generate grid
    #     n, c_, z, h, w = curr_bev.shape

    #     xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
    #     ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
    #     zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
    #     grid = torch.stack(
    #         (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)
    #     # import pdb;pdb.set_trace()
    #     # This converts BEV indices to meters
    #     # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
    #     # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
    #     feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
    #     # feat2bev[0, 0] = self.forward_projection.dx[0]
    #     # feat2bev[1, 1] = self.forward_projection.dx[1]
    #     # feat2bev[2, 2] = self.forward_projection.dx[2]
    #     # feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
    #     # feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
    #     # feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
    #     feat2bev[0, 0] = 0.4
    #     feat2bev[1, 1] =0.4
    #     feat2bev[2, 2] =0.4
    #     feat2bev[0, 3] =-40.
    #     feat2bev[1, 3] =-40.
    #     feat2bev[2, 3] = -1.
    #     # feat2bev[2, 2] = 1
    #     feat2bev[3, 3] = 1
    #     feat2bev = feat2bev.view(1,4,4)
    #     ## Get flow for grid sampling.
    #     # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
    #     # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
    #     # transform to previous grid locations.
    #     rt_flow = (torch.inverse(feat2bev)  @ forward_augs
    #             @ torch.inverse(curr_to_next_ego_rt) @ feat2bev)


    #     grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
    #     grid = grid[:,:,:,:, :3,0]
    #     # import pdb;pdb.set_trace()
    #     grid[...,:2]=grid[...,:2]-pred_flow#.permute(0, 1,3,2, 4)

    #     # normalize and sample
    #     normalize_factor = torch.tensor([w - 1.0, h - 1.0, z - 1.0], dtype=curr_bev.dtype, device=curr_bev.device)
    #     grid = grid / normalize_factor.view(1, 1, 1, 1, 3) * 2.0 - 1.0

    #     valid_grid=(grid[...,0]>=-1)*(grid[...,0]<=1)*(grid[...,1]>=-1)*(grid[...,1]<=1)*(grid[...,2]>=-1)*(grid[...,2]<=1)
    #     # import pdb;pdb.set_trace()
    #     sampled_next_bev = F.grid_sample(curr_bev, grid.to(curr_bev.dtype).permute(0, 3, 1, 2, 4), align_corners=True, mode='bilinear')
        
    #     # import pdb;pdb.set_trace(

    #     feats_to_return = sampled_next_bev.permute(0, 1, 3, 4, 2)
    #     # valid_grid=valid_grid.permute(0, 2,3,1)
    #     return feats_to_return,valid_grid
    def curr2next_with_flow( self,curr_bev, pred_flow,img_metas, bda,update_history=True): # align features with 3d shift


        curr_bev = curr_bev.permute(0, 1, 4, 2, 3)
        # pred_flow=pred_flow.permute(0, 3, 1,2, 4)
        # pred_flow=torch.flip(pred_flow,[-1])

        forward_augs = generate_forward_transformation_matrix(bda)

        # import pdb;pdb.set_trace()
        curr_to_next_ego_rt = torch.stack([
            single_img_metas['curr_to_next_ego_rt']
            for single_img_metas in img_metas]).to(curr_bev)
        # print(seq_ids,self.history_seq_ids)
        # Generate grid
        n, c_, z, h, w = curr_bev.shape

        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
        grid = torch.stack(
            (xs, ys, zs, torch.ones_like(xs)), -1).view(1, h, w, z, 4).expand(n, h, w, z, 4).view(n, h, w, z, 4, 1)
        # import pdb;pdb.set_trace()
        # This converts BEV indices to meters
        # IMPORTANT: the feat2bev[0, 3] is changed from feat2bev[0, 2] because previous was 2D rotation
        # which has 2-th index as the hom index. Now, with 3D hom, 3-th is hom
        feat2bev = torch.zeros((4,4),dtype=grid.dtype).to(grid)
        # feat2bev[0, 0] = self.forward_projection.dx[0]
        # feat2bev[1, 1] = self.forward_projection.dx[1]
        # feat2bev[2, 2] = self.forward_projection.dx[2]
        # feat2bev[0, 3] = self.forward_projection.bx[0] - self.forward_projection.dx[0] / 2.
        # feat2bev[1, 3] = self.forward_projection.bx[1] - self.forward_projection.dx[1] / 2.
        # feat2bev[2, 3] = self.forward_projection.bx[2] - self.forward_projection.dx[2] / 2.
        feat2bev[0, 0] = 0.4
        feat2bev[1, 1] =0.4
        feat2bev[2, 2] =0.4
        feat2bev[0, 3] =-40.
        feat2bev[1, 3] =-40.
        feat2bev[2, 3] = -1.
        # feat2bev[2, 2] = 1
        feat2bev[3, 3] = 1
        feat2bev = feat2bev.view(1,4,4)
        ## Get flow for grid sampling.
        # The flow is as follows. Starting from grid locations in curr bev, transform to BEV XY11,
        # backward of current augmentations, curr lidar to prev lidar, forward of previous augmentations,
        # transform to previous grid locations.

        rt_flow = (torch.inverse(feat2bev)  @ forward_augs
                @ torch.inverse(curr_to_next_ego_rt) @ feat2bev)
        # import pdb;pdb.set_trace()
        grid[...,:2,0] = grid[...,:2,0]+pred_flow.permute(0, 2,1,3, 4)


        grid = rt_flow.view(n, 1, 1, 1, 4, 4) @ grid
        grid = grid[:,:,:,:, :3,0]#.contiguous()
        

        grid=grid.reshape(-1,3)
        weight,coor=deformable_lift(grid)

        coor=coor.reshape(-1,3)#.contiguous()
        num_points=coor.shape[0]
        batch_idx = torch.arange(0, n ).reshape(n, 1).expand(n, num_points // n).reshape(num_points, 1).to(coor)

        kept = (coor[:, 0] >= 0) & (coor[:, 0] < w) &(coor[:, 1] >= 0) & (coor[:, 1] < h) & (coor[:, 2] >= 0) & (coor[:, 2] < z)
        coor = torch.cat((batch_idx,coor), 1)
        
        coor = coor[kept]
                    
        
        # import pdb;pdb.set_trace()
        curr_bev=curr_bev.permute(0,4,3,2,1)#.contiguous()
        bev_feat_shape=curr_bev.shape
        curr_bev=curr_bev.reshape(-1,curr_bev.shape[-1])#.contiguous()
        # import pdb;pdb.set_trace()
        weighted_feat=curr_bev.unsqueeze(1)*weight.unsqueeze(-1)

        weighted_feat=weighted_feat.reshape(-1,curr_bev.shape[1])
        weighted_feat=weighted_feat[kept]#.contiguous()
        # import pdb;pdb.set_trace()
        bev_feat=torch.sparse_coo_tensor(coor.t(),weighted_feat,bev_feat_shape).to_dense()#.contiguous()
        # import pdb;pdb.set_trace()
        bev_feat=bev_feat.permute(0,4,1,2,3).contiguous()

        return bev_feat
    def get_sampling_weights(self):
        if type(self.sample_weight_gamma) is list:
            # dynamic sampling weights
            min_gamma, max_gamma = self.sample_weight_gamma
            sample_weight_gamma = np.random.uniform(low=min_gamma, high=max_gamma)
        else:
            sample_weight_gamma = self.sample_weight_gamma
        
        self.sample_weights = self.baseline_sample_weights ** sample_weight_gamma
    
    def set_num_points(self,target_size):
        # import pdb;pdb.set_trace()
        
        if torch.cumprod(torch.tensor(target_size[1:]),0)[-1] < self.ori_num_points:

            self.num_points = torch.cumprod(torch.tensor(target_size[1:]),0)[-1]//4
        else:
            self.num_points = self.ori_num_points

    def init_weights(self):
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)
        
        if hasattr(self, "pixel_decoder"):
            self.pixel_decoder.init_weights()
        if self.num_transformer_decoder_layers>0:
            for p in self.transformer_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_normal_(p)
    
    def get_targets(self, cls_scores_list, mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """
   
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
        pos_inds_list,
        neg_inds_list) = multi_apply(self._get_target_single, cls_scores_list,
                                    mask_preds_list, gt_labels_list,
                                    gt_masks_list, img_metas)

        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)
    def get_targets_wo_assign(self,  mask_preds_list, gt_labels_list,
                    gt_masks_list, img_metas):
        """Compute classification and mask targets for all images for a decoder
        layer.

        Args:
            cls_scores_list (list[Tensor]): Mask score logits from a single
                decoder layer for all images. Each with shape (num_queries,
                cls_out_channels).
            mask_preds_list (list[Tensor]): Mask logits from a single decoder
                layer for all images. Each with shape (num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for all
                images. Each with shape (n, ), n is the sum of number of stuff
                type and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[list[Tensor]]: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels of all images.\
                    Each with shape (num_queries, ).
                - label_weights_list (list[Tensor]): Label weights\
                    of all images. Each with shape (num_queries, ).
                - mask_targets_list (list[Tensor]): Mask targets of\
                    all images. Each with shape (num_queries, h, w).
                - mask_weights_list (list[Tensor]): Mask weights of\
                    all images. Each with shape (num_queries, ).
                - num_total_pos (int): Number of positive samples in\
                    all images.
                - num_total_neg (int): Number of negative samples in\
                    all images.
        """

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
        pos_inds_list,
        neg_inds_list) = multi_apply(self._get_target_single_wo_assign,
                                    mask_preds_list, gt_labels_list,
                                    gt_masks_list, img_metas)

        num_total_pos = None
        num_total_neg = None
        return (labels_list, label_weights_list, mask_targets_list,
                mask_weights_list, num_total_pos, num_total_neg)

    def _get_target_single(self, cls_score, mask_pred, gt_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, x, y, z).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, x, y, z).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        gt_labels = gt_labels.long()
        
        # create sampling weights
        # import pdb;pdb.set_trace()
        point_indices, point_coords = sample_valid_coords_with_frequencies(self.num_points, 
                gt_labels=gt_labels, gt_masks=gt_masks, sample_weights=self.sample_weights)
        
        point_coords = point_coords[..., [2, 1, 0]]
        mask_points_pred = point_sample_3d(
            mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1), align_corners=self.align_corners).squeeze(1)
        
        # shape (num_gts, num_points)
        gt_points_masks = gt_masks.view(num_gts, -1)[:, point_indices]
        
        assign_result = self.assigner.assign(cls_score, mask_points_pred,
                                             gt_labels, gt_points_masks,
                                             img_metas)
        
        sampling_result = self.sampler.sample(assign_result, mask_pred,
                                              gt_masks)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        # label target
        labels = gt_labels.new_full((self.num_queries, ), self.num_classes, dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = labels.new_ones(self.num_queries).type_as(cls_score)
        class_weights_tensor = torch.tensor(self.class_weight).type_as(cls_score)

        # mask target
        mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_weights = mask_pred.new_zeros((self.num_queries, ))
        mask_weights[pos_inds] = class_weights_tensor[labels[pos_inds]]
        # import pdb;pdb.set_trace()
        
        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
    
    def _get_target_single_wo_assign(self, mask_pred, gt_labels, gt_masks, img_metas):
        """Compute classification and mask targets for one image.

        Args:
            cls_score (Tensor): Mask score logits from a single decoder layer
                for one image. Shape (num_queries, cls_out_channels).
            mask_pred (Tensor): Mask logits for a single decoder layer for one
                image. Shape (num_queries, x, y, z).
            gt_labels (Tensor): Ground truth class indices for one image with
                shape (num_gts, ).
            gt_masks (Tensor): Ground truth mask for each image, each with
                shape (num_gts, x, y, z).
            img_metas (dict): Image informtation.

        Returns:
            tuple[Tensor]: A tuple containing the following for one image.

                - labels (Tensor): Labels of each image. \
                    shape (num_queries, ).
                - label_weights (Tensor): Label weights of each image. \
                    shape (num_queries, ).
                - mask_targets (Tensor): Mask targets of each image. \
                    shape (num_queries, h, w).
                - mask_weights (Tensor): Mask weights of each image. \
                    shape (num_queries, ).
                - pos_inds (Tensor): Sampled positive indices for each \
                    image.
                - neg_inds (Tensor): Sampled negative indices for each \
                    image.
        """
        # sample points
        # num_queries = cls_score.shape[0]
        num_gts = gt_labels.shape[0]
        gt_labels = gt_labels.long()
        
        # create sampling weights
        # import pdb;pdb.set_trace()
        # point_indices, point_coords = sample_valid_coords_with_frequencies(self.num_points, 
        #         gt_labels=gt_labels, gt_masks=gt_masks, sample_weights=self.sample_weights)
        
        # point_coords = point_coords[..., [2, 1, 0]]
        # mask_points_pred = point_sample_3d(
        #     mask_pred.unsqueeze(1), point_coords.repeat(num_queries, 1, 1), align_corners=self.align_corners).squeeze(1)
        
        # # shape (num_gts, num_points)
        # gt_points_masks = gt_masks.view(num_gts, -1)[:, point_indices]

        # import pdb;pdb.set_trace()
        
        # assign_result = self.assigner.assign(cls_score, mask_points_pred,
        #                                      gt_labels, gt_points_masks,
        #                                      img_metas)
        
        # sampling_result = self.sampler.sample(assign_result, mask_pred,
        #                                       gt_masks)
        
        # pos_inds = sampling_result.pos_inds
        # neg_inds = sampling_result.neg_inds
        pos_inds=None
        neg_inds=None
        
        # label target
        # labels = gt_labels.new_full((self.num_queries, ), self.num_classes, dtype=torch.long)
        # labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        # label_weights = labels.new_ones(self.num_queries).type_as(cls_score)
        labels=None
        label_weights=None
        class_weights_tensor = torch.tensor(self.class_weight).type_as(mask_pred)

        # mask target
        # mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_targets=gt_masks
        # import pdb;pdb.set_trace()
        mask_weights = torch.zeros_like(class_weights_tensor).to(mask_pred)
        mask_weights[gt_labels] = class_weights_tensor[gt_labels]
        # mask_weights=class_weights_tensor
        # import pdb;pdb.set_trace()
        
        return (labels, label_weights, mask_targets, mask_weights, pos_inds, neg_inds)
    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss(self, all_cls_scores, all_mask_preds, gt_labels_list,
                gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # import pdb;pdb.set_trace()
        
        
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)
        
        



    # # # 使用线程池并行执行 matcher 函数
    #     with ThreadPoolExecutor(max_workers=7) as executor:
    #         results = list(executor.map(self.loss_single, all_cls_scores, all_mask_preds,
    #             all_gt_labels_list, all_gt_masks_list, img_metas_list))
    #     losses_cls, losses_mask, losses_dice=list(zip(*results))
        # import pdb; pdb.set_trace()
        loss_dict = dict()
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_cls[:-1], losses_mask[:-1], losses_dice[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        
        return loss_dict

    @force_fp32(apply_to=('all_cls_scores', 'all_mask_preds'))
    def loss_wo_assign(self, all_cls_scores, all_mask_preds, gt_labels_list,
                gt_masks_list, img_metas):
        """Loss function.

        Args:
            all_cls_scores (Tensor): Classification scores for all decoder
                layers with shape (num_decoder, batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            all_mask_preds (Tensor): Mask scores for all decoder layers with
                shape (num_decoder, batch_size, num_queries, h, w).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (n, ). n is the sum of number of stuff type
                and number of instance in a image.
            gt_masks_list (list[Tensor]): Ground truth mask for each image with
                shape (n, h, w).
            img_metas (list[dict]): List of image meta information.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_dec_layers = len(all_cls_scores)
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]
        # import pdb;pdb.set_trace()
        
        losses_mask, losses_dice = multi_apply(
            self.loss_single_wo_assign, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)
        
        



    # # # 使用线程池并行执行 matcher 函数
    #     with ThreadPoolExecutor(max_workers=7) as executor:
    #         results = list(executor.map(self.loss_single, all_cls_scores, all_mask_preds,
    #             all_gt_labels_list, all_gt_masks_list, img_metas_list))
    #     losses_cls, losses_mask, losses_dice=list(zip(*results))
        # import pdb; pdb.set_trace()
        loss_dict = dict()
        # loss from the last decoder layer
        # loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_mask'] = losses_mask[-1]
        loss_dict['loss_dice'] = losses_dice[-1]
        
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_mask_i, loss_dice_i in zip(
                losses_mask[:-1], losses_dice[:-1]):
            # loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_mask'] = loss_mask_i
            loss_dict[f'd{num_dec_layer}.loss_dice'] = loss_dice_i
            num_dec_layer += 1
        
        return loss_dict

    def loss_single(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, x, y, z).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, x, y, z).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = cls_scores.size(0)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        
        # import pdb;pdb.set_trace()
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                gt_labels_list, gt_masks_list, img_metas)

        # import pdb;pdb.set_trace()
        # shape (batch_size, num_queries)
        labels = torch.stack(labels_list, dim=0)
        # shape (batch_size, num_queries)
        label_weights = torch.stack(label_weights_list, dim=0)
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)

        # classfication loss
        # shape (batch_size * num_queries, )
        cls_scores = cls_scores.flatten(0, 1)
        labels = labels.flatten(0, 1)
        label_weights = label_weights.flatten(0, 1)
        class_weight = cls_scores.new_tensor(self.class_weight)
        # import pdb;pdb.set_trace()
        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum(),
        )
        # import pdb;pdb.set_trace()
        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        mask_preds = mask_preds[mask_weights > 0]
        mask_weights = mask_weights[mask_weights > 0]
        
        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        ''' 
        randomly sample K points for supervision, which can largely improve the 
        efficiency and preserve the performance. oversample_ratio = 3.0, importance_sample_ratio = 0.75
        '''
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            point_indices, point_coords = get_uncertain_point_coords_3d_with_frequency(
                mask_preds.unsqueeze(1), None, gt_labels_list, gt_masks_list, 
                self.sample_weights, self.num_points, self.oversample_ratio, 
                self.importance_sample_ratio)
            
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = torch.gather(mask_targets.view(mask_targets.shape[0], -1), 
                                        dim=1, index=point_indices)
        
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample_3d(
            mask_preds.unsqueeze(1), point_coords[..., [2, 1, 0]], align_corners=self.align_corners).squeeze(1)
        
        # dice loss
        num_total_mask_weights = reduce_mean(mask_weights.sum())
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, 
                        weight=mask_weights, avg_factor=num_total_mask_weights)

        # mask loss
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        mask_point_weights = mask_weights.view(-1, 1).repeat(1, self.num_points)
        mask_point_weights = mask_point_weights.reshape(-1)
        
        num_total_mask_point_weights = reduce_mean(mask_point_weights.sum())
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            weight=mask_point_weights,
            avg_factor=num_total_mask_point_weights)

        return loss_cls, loss_mask, loss_dice
    def loss_single_wo_assign(self, cls_scores, mask_preds, gt_labels_list,
                    gt_masks_list, img_metas):
        """Loss function for outputs from a single decoder layer.

        Args:
            cls_scores (Tensor): Mask score logits from a single decoder layer
                for all images. Shape (batch_size, num_queries,
                cls_out_channels). Note `cls_out_channels` should includes
                background.
            mask_preds (Tensor): Mask logits for a pixel decoder for all
                images. Shape (batch_size, num_queries, x, y, z).
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image, each with shape (num_gts, ).
            gt_masks_list (list[Tensor]): Ground truth mask for each image,
                each with shape (num_gts, x, y, z).
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple[Tensor]: Loss components for outputs from a single \
                decoder layer.
        """
        num_imgs = mask_preds.size(0)

        cls_scores_list = None
        mask_preds_list = [mask_preds[i] for i in range(num_imgs)]
        
        if self.sup_occupy_only:
            gt_masks_list=[gt_masks_list[i][:-1] for i in range(len(gt_masks_list))]
            gt_labels_list=[gt_labels_list[i][:-1] for i in range(len(gt_labels_list))]


        # import pdb;pdb.set_trace()
        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets_wo_assign( mask_preds_list,
                                gt_labels_list, gt_masks_list, img_metas)
        # import pdb;pdb.set_trace()
        
        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        if self.sup_occupy_only:
            mask_weights[:,-1]=0

 
        # extract positive ones
        # shape (batch_size, num_queries, h, w) -> (num_total_gts, h, w)
        # import pdb;pdb.set_trace()
        if self.mask_loss_softmax:
            # import pdb;pdb.set_trace()
            mask_preds = F.softmax(mask_preds, dim=1)

            mask_preds = mask_preds[mask_weights > 0]
            mask_weights = mask_weights[mask_weights > 0]
        elif self.mask_loss_softmax_adaptive:
            mask_preds_valid_list=[]
            mask_weights_valid_list=[]
            for i in range(mask_preds.size(0)):
                mask_preds_valid=mask_preds[i][mask_weights[i]>0]
                mask_preds_valid=F.softmax(mask_preds_valid, dim=0)
                mask_preds_valid_list.append(mask_preds_valid)
                mask_weights_valid=mask_weights[i][mask_weights[i]>0]
                mask_weights_valid_list.append(mask_weights_valid)
            mask_preds=torch.cat(mask_preds_valid_list,dim=0)
            mask_weights=torch.cat(mask_weights_valid_list,dim=0)
        else:
            mask_preds = mask_preds[mask_weights > 0]
            mask_weights = mask_weights[mask_weights > 0]
        
        if mask_targets.shape[0] == 0:
            # zero match
            loss_dice = mask_preds.sum()
            loss_mask = mask_preds.sum()
            return loss_cls, loss_mask, loss_dice

        ''' 
        randomly sample K points for supervision, which can largely improve the 
        efficiency and preserve the performance. oversample_ratio = 3.0, importance_sample_ratio = 0.75
        '''
        # import pdb;pdb.set_trace()
        with torch.no_grad():
            point_indices, point_coords = get_uncertain_point_coords_3d_with_frequency(
                mask_preds.unsqueeze(1), None, gt_labels_list, gt_masks_list, 
                self.sample_weights, self.num_points, self.oversample_ratio, 
                self.importance_sample_ratio)
            
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = torch.gather(mask_targets.view(mask_targets.shape[0], -1), 
                                        dim=1, index=point_indices)
        
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample_3d(
            mask_preds.unsqueeze(1), point_coords[..., [2, 1, 0]], align_corners=self.align_corners).squeeze(1)
        
        # dice loss
        num_total_mask_weights = reduce_mean(mask_weights.sum())
        # import pdb;pdb.set_trace()
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, 
                        weight=mask_weights, avg_factor=num_total_mask_weights)

        if self.mask_loss_softmax_adaptive or self.mask_loss_softmax:
            mask_point_preds=inverse_sigmoid(mask_point_preds)
            # loss_mask=F.binary_cross_entropy(mask_point_preds,mask_point_targets.float(),mask_weights.reshape(-1,1))
            # loss_mask=0
            # num=0
            # import pdb;pdb.set_trace()



            # for i in range(len(mask_preds_valid_list)):
            #     mask_preds_valid_i=mask_point_preds[num:num+mask_preds_valid_list[i].size(0)].t()
            #     mask_weights_valid_i=mask_weights_valid_list[i]
            #     mask_targets_valid_i=mask_point_targets[num:num+mask_preds_valid_list[i].size(0)].t().float()
            #     # mask_targets_valid_i=mask_targets_valid_i.argmax(dim=1)
            #     # mask_targets_valid_i=F.one_hot(mask_targets_valid_i.to(torch.int64),mask_targets_valid_i.shape[1]).float()
            #     # loss_mask_i = F.nll_loss((mask_preds_valid_i+1e-8).log(), mask_targets_valid_i,weight=mask_weights_valid_i)
            #     # import pdb;pdb.set_trace()
            #     loss_mask_i = F.binary_cross_entropy(mask_preds_valid_i, mask_targets_valid_i,weight=mask_weights_valid_i)
                
            #     loss_mask+=loss_mask_i
            #     num+=mask_preds_valid_list[i].size(0)
            # loss_mask=loss_mask/len(mask_preds_valid_list)
        # mask loss
        # else:
        # shape (num_queries, num_points) -> (num_queries * num_points, )
        mask_point_preds = mask_point_preds.reshape(-1)
        # shape (num_total_gts, num_points) -> (num_total_gts * num_points, )
        mask_point_targets = mask_point_targets.reshape(-1)
        mask_point_weights = mask_weights.view(-1, 1).repeat(1, self.num_points)
        mask_point_weights = mask_point_weights.reshape(-1)
        
        num_total_mask_point_weights = reduce_mean(mask_point_weights.sum())
        loss_mask = self.loss_mask(
            mask_point_preds,
            mask_point_targets,
            weight=mask_point_weights,
            avg_factor=num_total_mask_point_weights)
        # import pdb;pdb.set_trace()
        return loss_mask, loss_dice

    def forward_head(self, decoder_out, mask_feature, attn_mask_target_size,mask_embede2=False):
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (num_queries, batch_size, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

            - cls_pred (Tensor): Classification scores in shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred (Tensor): Mask scores in shape \
                (batch_size, num_queries, x, y, z).
            - attn_mask (Tensor): Attention mask in shape \
                (batch_size * num_heads, num_queries, h, w).
        """
        if self.num_transformer_decoder_layers>0:
            decoder_out = self.transformer_decoder.post_norm(decoder_out)
        else:
            decoder_out = self.post_norm(decoder_out)
        decoder_out = decoder_out.transpose(0, 1)
        # shape (batch_size, num_queries, c)
        if self.wo_assign:
            cls_pred = None
        else:
            cls_pred = self.cls_embed(decoder_out)
        # shape (batch_size, num_queries, c)
        if mask_embede2:
            mask_embed = self.mask_embed2(decoder_out)
        else:
            mask_embed = self.mask_embed(decoder_out)
        # shape (batch_size, num_queries, h, w)
        

        mask_pred = torch.einsum('bqc,bcxyz->bqxyz', mask_embed, mask_feature)
        
        ''' 对于一些样本数量较少的类别来说，经过 trilinear 插值 + 0.5 阈值，正样本直接消失 '''
        
        if attn_mask_target_size is not None:
            if self.pooling_attn_mask:
                # however, using max-pooling can save more positive samples, which is quite important for rare classes
                attn_mask = F.adaptive_max_pool3d(mask_pred.float(), attn_mask_target_size)
            else:
                # by default, we use trilinear interp for downsampling
                attn_mask = F.interpolate(mask_pred, attn_mask_target_size, mode='trilinear', align_corners=self.align_corners)
        
            # merge the dims of [x, y, z]
            attn_mask = attn_mask.flatten(2).detach() # detach the gradients back to mask_pred
            attn_mask = attn_mask.sigmoid() < 0.5
            
            # repeat for the num_head axis, (batch_size, num_queries, num_seq) -> (batch_size * num_head, num_queries, num_seq)
            attn_mask = attn_mask.unsqueeze(1).repeat((1, self.num_heads, 1, 1)).flatten(0, 1)
        else:
            attn_mask=None
        return cls_pred, mask_pred, attn_mask

    def preprocess_gt(self, gt_labels, img_metas):
        
        """Preprocess the ground truth for all images.

        Args:
            gt_labels_list (list[Tensor]): Each is ground truth
                labels of each bbox, with shape (num_gts, ).
            gt_masks_list (list[BitmapMasks]): Each is ground truth
                masks of each instances of a image, shape
                (num_gts, h, w).
            gt_semantic_seg (Tensor | None): Ground truth of semantic
                segmentation with the shape (batch_size, n, h, w).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            img_metas (list[dict]): List of image meta information.

        Returns:
            tuple: a tuple containing the following targets.
                - labels (list[Tensor]): Ground truth class indices\
                    for all images. Each with shape (n, ), n is the sum of\
                    number of stuff type and number of instance in a image.
                - masks (list[Tensor]): Ground truth mask for each\
                    image, each with shape (n, h, w).
        """
        
        num_class_list = [self.num_occupancy_classes] * len(img_metas)
        targets = multi_apply(preprocess_occupancy_gt, gt_labels, num_class_list, img_metas)
        
        labels, masks = targets
        return labels, masks
    def bin2depth(self,depth_bin_prob,depth_weight):
        # depth,depth_weight=depth_preds[0],depth_preds[1]
        # depth_weight=depth_weight.softmax(1)
        cum_depth=torch.cat((torch.zeros_like(depth_bin_prob[:,:1,...]),torch.cumsum(depth_bin_prob,1)[:,:-1]),dim=1)
        bin_center=self.grid_config['depth'][0]+(self.grid_config['depth'][1]-self.grid_config['depth'][0])*(depth_bin_prob/2+cum_depth)#BN,n_bin,H,W
        if self.ada2fix_bin:
            bin_center-=0.25
        
        depth_preds = torch.sum(depth_weight/depth_weight.sum(1,keepdim=True) *bin_center, dim=1)
        return bin_center,depth_preds
    def flow_decoder(self, x,img_metas, **kwargs):
        """Flow decoder.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output feature map.
        """
        if self.flow_with_his:
            bev_feat = self.fuse_history(x, img_metas, kwargs['bda'])
            if  self.flow_prev2curr:
                bev_feat,grid=bev_feat
        else:
            bev_feat = x

        

        flow_pred_ = self.flow_post_conv(bev_feat)
        if isinstance(flow_pred_, list):
            flow_pred_ = flow_pred_[0]
        # output['flow'] = [flow_pred]
        
        flow_pred = self.flow_predicter(flow_pred_)*self.flow_scale

        if self.use_adabin_flow_decoder:
            # import pdb;pdb.set_trace()
            bin_prob=self.adabin_bin_decoder(flow_pred_.mean(dim=[2, 3, 4]))

            bin_prob=bin_prob.reshape(1,self.flow_out_channels,2)
            bin_prob=bin_prob.softmax(1)
            bin_weight=flow_pred.softmax(1)

            cum_bin_prob=torch.cat((torch.zeros_like(bin_prob[:,:1,...]),torch.cumsum(bin_prob,1)[:,:-1]),dim=1)
            bin_center=self.bin_invertal[0]+(self.bin_invertal[1]-self.bin_invertal[0])*(bin_prob/2+cum_bin_prob)#BN,n_bin,H,W
            # import pdb;pdb.set_trace()
            flow_pred = torch.sum(bin_weight[:,:,None,...] *bin_center[...,None,None,None], dim=1)

        flow_pred=flow_pred.permute(0, 2, 3, 4, 1)
        # import pdb;pdb.set_trace()

        return flow_pred
    def forward_train(self,
            feats,
            img_metas,
            gt_labels,
            mask_embede2=False,
            class_prototype=None,
            tag='vox',
            **kwargs,
        ):
        """Forward function for training mode.

        Args:
            feats (list[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            img_metas (list[Dict]): List of image information.
            gt_bboxes (list[Tensor]): Each element is ground truth bboxes of
                the image, shape (num_gts, 4). Not used here.
            gt_labels (list[Tensor]): Each element is ground truth labels of
                each box, shape (num_gts,).
            gt_masks (list[BitmapMasks]): Each element is masks of instances
                of a image, shape (num_gts, h, w).
            gt_semantic_seg (list[tensor] | None): Each element is the ground
                truth of semantic segmentation with the shape (N, H, W).
                [0, num_thing_class - 1] means things,
                [num_thing_class, num_class-1] means stuff,
                255 means VOID. It's None when training instance segmentation.
            gt_bboxes_ignore (list[Tensor]): Ground truth bboxes to be
                ignored. Defaults to None.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses={}
        all_cls_scores=[None]
        all_mask_preds=[None]
        if not self.pred_flow_only or tag!='vox':
        # reset the sampling weights
            self.get_sampling_weights()

            # forward
            all_cls_scores, all_mask_preds = self(feats, img_metas,mask_embede2=mask_embede2,class_prototype=class_prototype,tag=tag)

            if self.training:

                # self.set_num_points(gt_labels.shape)
                # import pdb;pdb.set_trace()
                if mask_embede2:
                    self.num_points=self.num_points_img
                else:
                    self.num_points=self.ori_num_points
                    
                
                # preprocess ground truth
                gt_labels, gt_masks = self.preprocess_gt(gt_labels, img_metas)

                # loss
                if self.wo_assign:
                    losses = self.loss_wo_assign(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)
                else:
                    losses = self.loss(all_cls_scores, all_mask_preds, gt_labels, gt_masks, img_metas)
            else:
                losses=None
        # import pdb;pdb.set_trace()
        if self.pred_flow and tag=='vox':
            # if self.flow_with_his:
            #     bev_feat = self.fuse_history(feats[0], img_metas, kwargs['bda'])
            #     if  self.flow_prev2curr:
            #         bev_feat,grid=bev_feat
            # else:
            #     bev_feat = feats[0]

            # flow_pred = self.flow_post_conv(bev_feat)
            # if isinstance(flow_pred, list):
            #     flow_pred = flow_pred[0]
            flow_pred =self.flow_decoder(feats[0],img_metas, **kwargs)
            # # output['flow'] = [flow_pred]
            
            
            
            
            gt_occ_flow=kwargs['gt_occ_flow']
            # save_dir='./debug_save'
            # if not os.path.exists(save_dir):
            #     os.makedirs(save_dir)
            # import pdb;pdb.set_trace()
            # save_path=os.path.join(save_dir,'gt_occ_flow.npz')
            # np.save(gt_occ_flow.numpy(),save_path)
            # save_path_occ=os.path.join(save_dir,'gt_occ.npz')
            # np.save(kwargs['gt_occupancy'].numpy(),save_path_occ)

            mask=gt_occ_flow!=255
            # import pdb;pdb.set_trace()
            if  self.flow_l2_loss:
                
                if self.flow_class_balance:
                    loss_flow=0
                    for i in range(8):
                        # import pdb;pdb.set_trace()
                        mask_i=(kwargs['gt_occupancy_ori'][...,None]==i+1)*mask
                        if mask_i.sum()==0:
                            continue
                        loss_flow_i=F.mse_loss(flow_pred[mask_i],gt_occ_flow[mask_i])*self.flow_loss_weight
                        loss_flow+=loss_flow_i
                    if loss_flow==0:
                        loss_flow=0.*flow_pred[0,0,0,0].sum()
                    #     loss_flow.append(loss_flow_i)
                    # if len(loss_flow)==0:
                    #     loss_flow=torch.tensor(0.0).to(flow_pred.device)
                    # else:
                    #     loss_flow=sum(loss_flow)/len(loss_flow)

                else:
                    if mask.sum()==0:
                        loss_flow=0.*flow_pred[0,0,0,0].sum()
                    else:
                        loss_flow= F.mse_loss(flow_pred[mask],gt_occ_flow[mask])*self.flow_loss_weight
                losses.update({'loss_flow_l2':loss_flow})
            elif self.flow_render_loss:
                    
                # import pdb;pdb.set_trace()
                bdas=[kwargs['bda'][i] for i in range(len(flow_pred))]
                gt_pred_pcds_t=kwargs['gt_pred_pcds_t']
                # import pdb; pdb.set_trace()
                if not self.pred_flow_only:
                    sem_pred=all_mask_preds[0].max(1)[1]
                    sem_preds=[sem_pred[i] for i in range(len(sem_pred))]
                
                    density=sem_pred!=all_mask_preds[0].shape[1]-1
                    density=density.float()
                    densitys=[density[i] for i in range(len(density))]
                    origins=kwargs['ray_origin']
                    loss_render_flow=multi_apply(self.render_depth_single,kwargs['gt_render_depth'],densitys,origins,bdas,gt_pred_pcds_t,flow_pred,sem_preds)[0]
                else:
                   loss_render_flow=multi_apply(self. sup_flow_swin_single,kwargs['gt_swin_pred_pcds_t'],bdas,gt_pred_pcds_t,flow_pred)[0]
                loss_render_flow=sum(loss_render_flow)/len(loss_render_flow)
                loss_flow=loss_render_flow
                losses.update({'loss_render_flow':loss_flow})
            elif self.flow_prev2curr:
                prev_occ_gt=kwargs['gt_prev_swin_pred_occ']
                curr_occ_gt=kwargs['gt_swin_pred_occ']
                loss_flows=[]
                flow_pred=flow_pred.permute(0, 3, 1,2, 4)
                flow_pred=torch.flip(flow_pred,[-1])
                for i in range( len(curr_occ_gt)):
                    # import pdb;pdb.set_trace()
                    if not torch.isnan(prev_occ_gt[i]).any() and not torch.isnan(curr_occ_gt[i]).any():
                        grid_i=grid[i:i+1]
                        occ_shape=prev_occ_gt[i].shape
                        curr_bev=F.one_hot(curr_occ_gt[i].reshape(-1).long(),num_classes=18).float()
                        curr_bev=curr_bev.reshape(1,*occ_shape,18).permute(0,4,1,2,3)
                        curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
                        tmp_bev=F.one_hot(prev_occ_gt[i].reshape(-1).long(),num_classes=18).float()
                        tmp_bev=tmp_bev.reshape(1,*occ_shape,18).permute(0,4,1,2,3)
                       


                        tmp_bev = tmp_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w

                        n, c_, z, h, w = curr_bev.shape
                        # if not self.flash_occ:
                            # Generate grid
                        xs = torch.linspace(0, w - 1, w, dtype=curr_bev.dtype, device=curr_bev.device).view(1, w, 1).expand(h, w, z)
                        ys = torch.linspace(0, h - 1, h, dtype=curr_bev.dtype, device=curr_bev.device).view(h, 1, 1).expand(h, w, z)
                        zs = torch.linspace(0, z - 1, z, dtype=curr_bev.dtype, device=curr_bev.device).view(1, 1, z).expand(h, w, z)
                        coords_curr = torch.stack((xs, ys, zs), -1).view(1, h, w, z, 3).expand(n, h, w, z, 3)

                        # import pdb;pdb.set_trace()
                        grid_i=grid_i.permute(0, 3, 1, 2, 4)
                        normalize_factor =  torch.tensor([199,199,15], dtype=flow_pred.dtype, device=flow_pred.device).view(1, 1, 1, 1, 3)
                        grid_i=(grid_i+1.0)*2.0*normalize_factor
                        grid_i[...,:2]=grid_i[...,:2]-flow_pred[i:i+1]

                         
                        grid_i = grid_i/ normalize_factor* 2.0 - 1.0
                        
                        sampled_history_bev = F.grid_sample(tmp_bev, grid_i, align_corners=True, mode=self.interpolation_mode)

                        sampled_history_bev=sampled_history_bev.permute(0, 1, 3, 4, 2)
                        sampled_history_bev=sampled_history_bev.permute(0, 2,3,4,1)
                        curr_bev=curr_bev.permute(0, 1, 3, 4, 2)
                        curr_bev=curr_bev.permute(0, 2,3,4,1)
                        foreground_mask=(curr_occ_gt[i]<9).unsqueeze(0)


                        loss_flow= F.l1_loss(sampled_history_bev[foreground_mask],curr_bev[foreground_mask])*self.flow_loss_weight
                        loss_flows.append(loss_flow)
                if len(loss_flows)==0:
                    loss_flow=0.*flow_pred[0,0,0,0].sum()
                else:
                    loss_flow=sum(loss_flows)/len(loss_flows)
                        # point_prev_coord=self.vv[prev_occ_gt[i]<9]
                        # point_curr_coord=self.vv[curr_occ_gt[i]<9]
                losses.update({'loss_flow_l1':loss_flow})
            else:
                if mask.sum()==0:
                    loss_flow=0.*flow_pred[0,0,0,0].sum()
                else:
                    loss_flow= F.l1_loss(flow_pred[mask],gt_occ_flow[mask])*self.flow_loss_weight
                losses.update({'loss_flow_prev2curr':loss_flow})

            if self.flow_cosine_loss:
                mask=gt_occ_flow!=255
                if mask.sum()==0:
                    loss_flow_cosine=0.*flow_pred[0,0,0,0].sum()
                else:
                    if self.flow_class_balance:
                        loss_flow_cosine=0
                        for i in range(8):
                            # import pdb;pdb.set_trace()
                            mask_i=(kwargs['gt_occupancy_ori'][...,None]==i+1)*mask
                            if mask_i.sum()==0:
                                continue
                            loss_flow_cosine_i=-F.cosine_similarity(flow_pred[mask_i],gt_occ_flow[mask_i],dim=-1)*self.flow_loss_weight

                            loss_flow_cosine+=loss_flow_cosine_i
                        if loss_flow_cosine==0:
                            loss_flow_cosine=0.*flow_pred[0,0,0,0].sum()
                    #     loss_flow.append(loss_flow_i)
                    # if len(loss_flow)==0:
                    #     loss_flow=torch.tensor(0.0).to(flow_pred.device)
                    # else:
                    #     loss_flow=sum(loss_flow)/len(loss_flow)

                    else:
                        flow_pred=flow_pred[mask]
                        gt_occ_flow=gt_occ_flow[mask]
                        loss_flow_cosine=-F.cosine_similarity(flow_pred,gt_occ_flow,dim=-1)*self.flow_loss_weight
                losses.update({'loss_flow_cosine':loss_flow_cosine})
            # # import pdb;pdb.set_trace()
            
            # curr_bev=F.one_hot(kwargs['gt_occupancy_ori'].long().reshape(-1).contiguous(),18).contiguous().t().float().reshape(1,18,200,200,16).contiguous()
            # # import pdb;pdb.set_trace()
            # flow_pred=kwargs['gt_occ_flow'].contiguous()
            
            # # flow_pred=kwargs['gt_occ_flow'][...,[1,0]].contiguous()

            # future_occ_feat=self.curr2next_with_flow(curr_bev,flow_pred, img_metas, kwargs['bda'])
                    
            # save_path_future=os.path.join('./debug_save','occ.npz')
            # future_occ_feat=future_occ_feat[:,1:].argmax(dim=1)
            # np.savez_compressed(save_path_future,g1=kwargs['gt_occupancy_ori'].cpu().numpy(), g2=kwargs['future_gt_occ'][0].cpu().numpy(),g3=future_occ_feat.cpu().numpy(),g4=kwargs['gt_occ_flow'].cpu().numpy())
            
            # # save_path_future=os.path.join('./debug_save','occ_metas.npz')
            # # np.savez_compressed(save_path_future,curr_bev=curr_bev.cpu().numpy(), pred_flow=flow_pred.cpu().numpy(),curr_to_next_ego_rt=img_metas[0]['curr_to_next_ego_rt'].cpu().numpy(),bda=kwargs['bda'].cpu().numpy())
            
            # # import pdb;pdb.set_trace()

            
            if self.flow_curr2future:
                # prev_occ_gt=kwargs['gt_prev_swin_pred_occ']
                # curr_occ_gt=kwargs['gt_swin_pred_occ']
                
      
                end_of_sequence = torch.BoolTensor([
                    single_img_metas['end_of_sequence'] 
                    for single_img_metas in img_metas]).to(flow_pred.device)
                # import pdb;pdb.set_trace()


                if not sum(~end_of_sequence)==0:
                    future_occ_gt=[kwargs['future_gt_occ'][i] for i in range(len(end_of_sequence)) if ~end_of_sequence[i]]
                    future_occ_gt=torch.stack(future_occ_gt,dim=0)
                    curr_bev=feats[0][~end_of_sequence]

                    future_occ_feat=self.curr2next_with_flow(curr_bev,flow_pred, img_metas, kwargs['bda'])
                    
                    # save_path_future=os.path.join('./debug_save','occ.npz')
                    # future_occ_feat=future_occ_feat.argmax(dim=1)
                    # np.save(g1=kwargs['gt_semantic_map'].cpu().numpy(), g2=kwargs['future_gt_occ'].cpu().numpy(),\
                    #     g3=future_occ_feat.cpu().numpy())
                    # import pdb;pdb.set_trace()
                    
                    
                    future_occ_feat = self.future_occ_predictor(future_occ_feat)
                    # import pdb;pdb.set_trace()
                    mask=(future_occ_gt<=8)
                    # print(valid_grid.sum(),mask.sum(),11111111111111)
                    if mask.sum()>0:
                        loss_future_occ = F.cross_entropy(future_occ_feat.permute(0,2,3,4,1)[mask], future_occ_gt[mask].long(), ignore_index=255)
                    else:
                        loss_future_occ=0.*future_occ_feat[0,0,0,0,0].sum()
                    losses.update({'loss_future_occ':loss_future_occ})
                else:
                    loss_future_occ=0.*self.future_occ_predictor(feats[0])[0,0,0,0].sum()
                    losses.update({'loss_future_occ':loss_future_occ})

            
        if self.render_depth_sup and tag=='vox':
            
            pred_mask=all_mask_preds[0].sigmoid()
            density1=pred_mask[:,:-1].max(1)[0]
            density2=1-pred_mask[:,-1]
            density=(1-1/self.num_classes)*density1+1/self.num_classes*density2
            if self.render_depth_sup_st:
                #straight through estimator
                density_binary=(density>0.5).float()
                density=density_binary-density.detach()+density
            densitys=[density[i] for i in range(len(density))]
            # origins=[kwargs['ray_origin'][i][None] for i in range(len(density))]
            origins=kwargs['ray_origin']
            # import pdb;pdb.set_trace()
            bdas=[kwargs['bda'][i] for i in range(len(density))]
            gt_pred_pcds_t=kwargs['gt_pred_pcds_t']

            
            loss_render_depth=multi_apply(self.render_depth_single,kwargs['gt_render_depth'],densitys,origins,bdas,gt_pred_pcds_t)[0]
            # import pdb;pdb.set_trace()
            # multi_apply(self.render_depth_single,kwargs['gt_render_depth'],[kwargs['gt_occupancy'][i] for i in range(len(kwargs['gt_occupancy']))],origins,bdas,gt_pred_pcds_t)[0]
            # for i in range(len(density)):
            #     rendered_depth=self.render_depth_loss_single(densitys[i],origins[i],bdas[i])
            #     # import pdb;pdb.set_trace()
            #     loss_render_depth+=self.sigloss(rendered_depth,kwargs['gt_render_depth'][i])
            # multi_apply(self.render_depth_single,densitys,origins,bdas)
            # import pdb;pdb.set_trace()
            # rendered_depths=torch.stack([torch.stack(rendered_depths[i],dim=0) for i in range(len(rendered_depths))],dim=0)
            # rendered_depths=torch.stack(rendered_depths,dim=0)
            # import pdb;pdb.set_trace()
            # rendered_depths=rendered_depths.permute(1,0,2)
            # loss_render_depth=self.sigloss(rendered_depths,kwargs['gt_render_depth'])*10.0
            loss_render_depth=sum(loss_render_depth)/len(loss_render_depth)
            losses.update({'loss_render_depth':loss_render_depth})
            # print('loss_render_depth:',loss_render_depth,111111111111111111111)
            # if loss_render_depth.isnan():
            #     print('loss_render_depth:',loss_render_depth,111111111111111111111)

            
        return losses,all_cls_scores,all_mask_preds
    def render_func(self,occ_pred, lidar_origin,lidar_rays, offset,scaler):
        # lidar_origin = output_origin[:, t:t+1, :]  # [1, 1, 3]
        lidar_endpts = lidar_rays[None] + lidar_origin  # [1, 15840, 3]

        output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
        output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
        output_tindex_render = torch.zeros([1, lidar_rays.shape[0]]).to(occ_pred.device)  # [1, N], all zeros

        # import pdb; pdb.set_trace()
        # with torch.no_grad():
        pred_dist, _, coord_index = dvr.render_forward(
            occ_pred,
            output_origin_render,
            output_points_render,
            output_tindex_render,
            [1, 16, 200, 200],
            "train"
        )
        # import pdb;pdb.set_trace()
        return pred_dist,coord_index
    def render_depth_single(self,gt_render_depth,occ_pred, output_origin,bda, gt_pred_pcds_t,flow_pred=None,sem_pred=None, return_xyz=False):
        output_origin=output_origin.unsqueeze(0)


        bda_ori=bda.clone()

        output_origin_=output_origin.clone()

        T = output_origin.shape[1]
        pred_pcds_t = []
        bda=torch.mm(bda,T_.to(bda.device))
        lidar_rays =bda[None].matmul(self.lidar_rays.to(bda.device).unsqueeze(-1)).squeeze(-1)
        # lidar_rays = (bda@self.lidar_rays.to(bda.device).T).T
        # import pdb;pdb.set_trace()
        output_origin=bda.unsqueeze(0).unsqueeze(0).matmul(output_origin.float().unsqueeze(-1)).squeeze(-1)
        # output_origin=(bda@output_origin.T).T
        # free_id =self.num_classes - 1 
        # occ_pred = copy.deepcopy(sem_pred)
        # occ_pred[sem_pred < free_id] = 1
        # occ_pred[sem_pred == free_id] = 0
        occ_pred = occ_pred.permute(2, 1, 0)
        occ_pred = occ_pred[None, None, :].contiguous().float()

        offset = torch.Tensor(_pc_range[:3])[None, None, :].to(occ_pred.device)
        scaler = torch.Tensor([_voxel_size] * 3)[None, None, :].to(occ_pred.device)

        # lidar_tindex = torch.zeros([1, lidar_rays.shape[0]])
        ray_masks=[]
        ray_mask2s=[]
        pred_dists=[]
        # import pdb;pdb.set_trace()
        pred_dists,coords=multi_apply(self.render_func,[occ_pred]*T, output_origin.unbind(1),[lidar_rays]*T, [offset]*T,[scaler]*T)
        # F.mse_loss(self.render_func((occ_pred!=17).float(), output_origin.unbind(1)[0],lidar_rays, offset,scaler)[0],gt_render_depth[:14040])
        # for t in range(T): 
            
        #     pred_dists.append(pred_dist)
        # import pdb;pdb.set_trace()
        if self.render_depth_sup:
            pred_dists = torch.cat(pred_dists, dim=0).reshape(-1)
            valid_mask = (gt_pred_pcds_t[:, 0] != self.num_classes-1)
            pred_dists = pred_dists[valid_mask]
            gt_render_depth = gt_render_depth[valid_mask]
            render_depth_loss=self.sigloss(pred_dists,gt_render_depth)
            # print('pred_dists:',pred_dists.shape,pred_dists.min(),pred_dists.max(),22222222222222222)
            # if gt_render_depth.min()<0:
            #         print('loss_render_depth:',render_depth_loss,pred_dists.min(),pred_dists.max(),'occ_pred',occ_pred.min(),occ_pred.max(),output_origin,output_origin_,gt_render_depth.min(),gt_render_depth.max(),222222222222222222222)   

            return (render_depth_loss,render_depth_loss)
        elif self.flow_render_loss:
            depth_pred=torch.cat(pred_dists, dim=0).reshape(-1)*0.4
            coords=torch.cat(coords, dim=0).reshape(-1,3).long()
            depth_gt = gt_pred_pcds_t[:, 1]
            l1_error = torch.abs(depth_pred - depth_gt)
            threshold=2
            tp_dist_mask = (l1_error < threshold)

            pred_label = sem_pred[coords[:, 0], coords[:, 1], coords[:, 2]]

            tp_label_mask = (pred_label == gt_pred_pcds_t[:, 0] )

            foreground_mask = (gt_pred_pcds_t[:, 0] <8)
            tp_mask = tp_dist_mask & tp_label_mask & foreground_mask

            
            pred_flow = flow_pred[coords[:, 0], coords[:, 1], coords[:, 2]][tp_mask]
            gt_flow = gt_pred_pcds_t[tp_mask, 2:4]
            # import pdb;pdb.set_trace()
            if tp_mask.sum()==0:
                loss_render_flow=0.*flow_pred[0].sum()
            else:
                loss_render_flow = F.mse_loss(pred_flow, gt_flow)*self.flow_render_loss_weight
            return (loss_render_flow,loss_render_flow)

    def sup_flow_swin_single(self,swin_pred_pcds_t,bda, gt_pred_pcds_t,flow_pred=None,return_xyz=False):

        # output_origin=output_origin.unsqueeze(0)


        # bda_ori=bda.clone()

        # output_origin_=output_origin.clone()

        # T = output_origin.shape[1]
        # import pdb;pdb.set_trace()
        if torch.isnan(swin_pred_pcds_t).any():
            loss_render_flow=0.*flow_pred[0].sum()
            return (loss_render_flow,loss_render_flow)
        pred_pcds_t = []
        bda=torch.mm(bda,T_.to(bda.device))

        coor=swin_pred_pcds_t[:,2:5]
        coor=coor.float()+0.5
        coor=coor*_voxel_size+ torch.Tensor(_pc_range[:3]).to(coor.device)
        coor=bda.unsqueeze(0).unsqueeze(0).matmul(coor.float().unsqueeze(-1)).squeeze(-1).squeeze(0)
        # coor=(bda@coor.T).T
        # import pdb;pdb.set_trace()
        coor=coor- torch.Tensor(_pc_range[:3]).to(coor.device)
        coor=coor/_voxel_size-0.5
        coor=coor.round().long()


     
        depth_pred=swin_pred_pcds_t[:,1]
        coords=coor
        depth_gt = gt_pred_pcds_t[:, 1]
        l1_error = torch.abs(depth_pred - depth_gt)
        threshold=2
        tp_dist_mask = (l1_error < threshold)

        pred_label = swin_pred_pcds_t[:,0]

        tp_label_mask = (pred_label == gt_pred_pcds_t[:, 0] )

        foreground_mask = (gt_pred_pcds_t[:, 0] <8)
        tp_mask = tp_dist_mask & tp_label_mask & foreground_mask

        # import pdb;pdb.set_trace()
        pred_flow = flow_pred[coords[:, 0], coords[:, 1], coords[:, 2]][tp_mask]
        gt_flow = gt_pred_pcds_t[tp_mask, 2:4]
        # import pdb;pdb.set_trace()
        if tp_mask.sum()==0:
            loss_render_flow=0.*flow_pred[0].sum()
        else:
            loss_render_flow = F.mse_loss(pred_flow, gt_flow)*self.flow_render_loss_weight
        return (loss_render_flow,loss_render_flow)
        
    def forward(self, 
            feats,
            img_metas,
            mask_embede2=False,
            class_prototype=None,
            tag='vox',
            **kwargs,
        ):
        """Forward function.

        Args:
            feats (list[Tensor]): Multi scale Features from the
                upstream network, each is a 5D-tensor (B, C, X, Y, Z).
            img_metas (list[dict]): List of image information.

        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        """
        # import pdb;pdb.set_trace()
        if self.flash_occ:
            feats = self.final_conv(feats[0])
            # import pdb;pdb.set_trace()
            # feats = feats.permute(0, 3, 2, 1)
            if  self.flash_occ_vae:
                feats=feats.reshape(feats.shape[0],self.Dz,feats.shape[1]//self.Dz,*feats.shape[2:])
                feats=feats.permute(0,2,3,4,1)
            else:
                feats=feats.reshape(feats.shape[0],feats.shape[1]//self.Dz,self.Dz,*feats.shape[2:])
                feats=feats.permute(0,1,3,4,2)
            feats=[feats]
        # import pdb;pdb.set_trace()
        batch_size = len(img_metas)
        mask_features = feats[0]
        if self.context_post_conv is not None:
            if tag=='vox':
                mask_features = self.context_post_conv(mask_features)
                if isinstance(mask_features, list):
                    mask_features = mask_features[0]

        multi_scale_memorys = feats[:0:-1]
        
        decoder_inputs = []
        decoder_positional_encodings = []
        # import pdb;pdb.set_trace()
        if self.num_transformer_decoder_layers>0:
            for i in range(self.num_transformer_feat_level):
                ''' with flatten features '''
                # projection for input features
                # import pdb;pdb.set_trace()
                decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
                # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
                decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
                ''' with level embeddings '''
                level_embed = self.level_embed.weight[i].view(1, 1, -1)
                decoder_input = decoder_input + level_embed
                ''' with positional encodings '''
                # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
                mask = decoder_input.new_zeros((batch_size, ) + multi_scale_memorys[i].shape[-3:], dtype=torch.bool)
                # import pdb;pdb.set_trace()
                decoder_positional_encoding = self.decoder_positional_encoding(mask)
                decoder_positional_encoding = decoder_positional_encoding.flatten(2).permute(2, 0, 1)
                
                decoder_inputs.append(decoder_input)
                decoder_positional_encodings.append(decoder_positional_encoding)
        
        # shape (num_queries, c) -> (num_queries, batch_size, c)
        if class_prototype is not None:
            query_feat = class_prototype.unsqueeze(1).repeat((1, batch_size, 1))
        else:
            query_feat = self.query_feat.weight.unsqueeze(1).repeat((1, batch_size, 1))
        if self.num_transformer_decoder_layers>0:
            query_embed = self.query_embed.weight.unsqueeze(1).repeat((1, batch_size, 1))
        
        ''' directly deocde the learnable queries, as simple proposals '''
        cls_pred_list = []
        mask_pred_list = []
        # import pdb;pdb.set_trace()
        if len(multi_scale_memorys)>0:
            cls_pred, mask_pred, attn_mask = self.forward_head(query_feat, 
                mask_features, multi_scale_memorys[0].shape[-3:],mask_embede2=mask_embede2)
        else:

            cls_pred, mask_pred, attn_mask = self.forward_head(query_feat, 
                        mask_features, None,mask_embede2=mask_embede2)
    
        cls_pred_list.append(cls_pred)
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
                
            '''
            if the attn_mask is all True (ignore everywhere), simply change it to all False (attend everywhere) 
            '''
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            attn_masks = [attn_mask, None]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                attn_masks=attn_masks,
                query_key_padding_mask=None,
                key_padding_mask=None)
            
            cls_pred, mask_pred, attn_mask = self.forward_head(
                query_feat, mask_features, 
                multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-3:],mask_embede2=mask_embede2)

            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)
        
        '''
        Returns:
            tuple: A tuple contains two elements.

            - cls_pred_list (list[Tensor)]: Classification logits \
                for each decoder layer. Each is a 3D-tensor with shape \
                (batch_size, num_queries, cls_out_channels). \
                Note `cls_out_channels` should includes background.
            - mask_pred_list (list[Tensor]): Mask logits for each \
                decoder layer. Each with shape (batch_size, num_queries, \
                 X, Y, Z).
        '''
        
        return cls_pred_list, mask_pred_list

    def format_results(self, mask_cls_results, mask_pred_results):
        if self.wo_assign:
            return mask_pred_results

        mask_cls = F.softmax(mask_cls_results, dim=-1)[..., :-1]

        mask_pred = mask_pred_results.sigmoid()
        output_voxels = torch.einsum("bqc, bqxyz->bcxyz", mask_cls, mask_pred)

        
        
        return output_voxels

    def simple_test(self, 
            feats,
            img_metas,
            class_prototype,
            **kwargs,
        ):
        if not self.pred_flow_only:
            all_cls_scores, all_mask_preds = self(feats, img_metas,class_prototype=class_prototype)
            
            # import pdb;pdb.set_trace()
            
            mask_cls_results = all_cls_scores[-1]
            mask_pred_results = all_mask_preds[-1]

            # mask_cls_results = all_cls_scores[5]
            # mask_pred_results = all_mask_preds[5]
            
            # # rescale mask prediction
            # mask_pred_results = F.interpolate(
            #     mask_pred_results,
            #     size=tuple(img_metas[0]['occ_size']),
            #     mode='trilinear',
            #     align_corners=self.align_corners,
            # )
            # import pdb;pdb.set_trace()
            output_voxels = self.format_results(mask_cls_results, mask_pred_results)
            ###########
            # pred_mask=mask_pred_results.sigmoid()
            # density1=pred_mask[:,:-1].max(1)[0]
            # density2=1-pred_mask[:,-1]
            # density=(1-1/self.num_classes)*density1+1/self.num_classes*density2

            ###################
            # import pdb;pdb.set_trace()
            pred_mask=mask_pred_results.sigmoid()
            density1=pred_mask[:,:-1].max(1)[0]

            density=(density1>0.5).float()
        else:
            output_voxels=None
            density=None
        #################
        # import pdb;pdb.set_trace()
        res = {
            'output_voxels': [output_voxels],
            'output_points': None,
            'output_density':[density]
        }
        if self.pred_flow:
           
            flow_pred =self.flow_decoder(feats[0],img_metas, **kwargs)
            # import pdb;pdb.set_trace()
            # flow_pred=flow_pred.permute(0, 2, 3, 4, 1)
            # gt_occ_flow=kwargs['gt_occ_flow']
            # mask=gt_occ_flow!=255
        
            # loss_flow= F.l1_loss(flow_pred[mask],gt_occ_flow[mask])
            # losses.update({'loss_flow':loss_flow})
            res['output_flow'] = [flow_pred]

        return res
    def train_with_multi_layer_ce_loss(self, 
            feats,
            img_metas,
            **kwargs,
        ):
        all_cls_scores, all_mask_preds = self(feats, img_metas)
        # import pdb;pdb.set_trace()
        # mask_cls_results = all_cls_scores[-1]
        # mask_pred_results = all_mask_preds[-1]
        
        # # rescale mask prediction
        # mask_pred_results = F.interpolate(
        #     mask_pred_results,
        #     size=tuple(img_metas[0]['occ_size']),
        #     mode='trilinear',
        #     align_corners=self.align_corners,
        # )
        # import pdb;pdb.set_trace()
        output_voxels=[]
        for index, mask_cls_results in enumerate(all_cls_scores):
            output_voxels.append((self.format_results(mask_cls_results, all_mask_preds[index])+1e-8).log())
        # output_voxels = self.format_results(mask_cls_results, mask_pred_results)
        res = {
            'output_voxels': output_voxels,
            'output_points': None,
        }

        return res
    @force_fp32() 
    def loss_voxel(self, output_voxels, target_voxels,gt_occupancy_ori, tag):
        # import pdb;pdb.set_trace()
        # resize gt                       
        B, C, H, W, D = output_voxels.shape
        ratio = target_voxels.shape[2] // H
        # import pdb;pdb.set_trace()
        if ratio != 1:
            target_voxels = target_voxels.reshape(B, H, ratio, W, ratio, D, ratio).permute(0,1,3,5,2,4,6).reshape(B, H, W, D, ratio**3)
            empty_mask = target_voxels.sum(-1) == self.empty_idx
            target_voxels = target_voxels.to(torch.int64)
            occ_space = target_voxels[~empty_mask]
            occ_space[occ_space==0] = -torch.arange(len(occ_space[occ_space==0])).to(occ_space.device) - 1
            target_voxels[~empty_mask] = occ_space
            target_voxels = torch.mode(target_voxels, dim=-1)[0]
            target_voxels[target_voxels<0] = 255
            target_voxels = target_voxels.long()
        
        # output_voxels = torch.log(output_voxels * 0) + output_voxels/0 # debug !!!!!!!!

        output_voxels[torch.isnan(output_voxels)] = 0
        output_voxels[torch.isinf(output_voxels)] = 0
        assert torch.isnan(output_voxels).sum().item() == 0
        assert torch.isnan(target_voxels).sum().item() == 0

        loss_dict = {}

        # igore 255 = ignore noise. we keep the loss bascward for the label=0 (free voxels)
        # if self.final_two_part_loss:
        #     loss_dict.update(self.TwoPart_loss(output_voxels.permute(0, 2, 3, 4, 1), gt_occupancy_ori,mask=target_voxels!=255))
        # else:
        # import pdb;pdb.set_trace()
        if self.CE_loss_only:
            # import pdb;pdb.set_trace()
            ##########################
            preds=output_voxels.permute(0, 2, 3, 4, 1)
            mask=target_voxels!=255
            preds=preds[mask]
            target_voxels=target_voxels[mask]
            loss_dict['loss_voxel_ce_{}'.format(tag)] = torch.nn.CrossEntropyLoss()(preds, target_voxels)
            ##################
            # loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, None, ignore_index=255)
        else:
            if self.use_focal_loss:
                # import pdb;pdb.set_trace()
                loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * self.focal_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            else:
                loss_dict['loss_voxel_ce_{}'.format(tag)] = self.loss_voxel_ce_weight * CE_ssc_loss(output_voxels, target_voxels, self.class_weights.type_as(output_voxels), ignore_index=255)
            if not self.focal_loss_only:
                loss_dict['loss_voxel_sem_scal_{}'.format(tag)] = self.loss_voxel_sem_scal_weight * sem_scal_loss(output_voxels, target_voxels, ignore_index=255)
                loss_dict['loss_voxel_geo_scal_{}'.format(tag)] = self.loss_voxel_geo_scal_weight * geo_scal_loss(output_voxels, target_voxels, ignore_index=255, non_empty_idx=self.empty_idx)
                loss_dict['loss_voxel_lovasz_{}'.format(tag)] = self.loss_voxel_lovasz_weight * lovasz_softmax(torch.softmax(output_voxels, dim=1), target_voxels, ignore=255)


        # if self.use_dice_loss:
        #     visible_mask = target_voxels!=255
        #     visible_pred_voxels = output_voxels.permute(0, 2, 3, 4, 1)[visible_mask]
        #     visible_target_voxels = target_voxels[visible_mask]
        #     visible_target_voxels = F.one_hot(visible_target_voxels.to(torch.long), 19)
        #     loss_dict['loss_voxel_dice_{}'.format(tag)] = self.dice_loss(visible_pred_voxels, visible_target_voxels)

        return loss_dict

    @force_fp32() 
    def loss_ce(self, output_voxels=None,
                output_coords_fine=None, output_voxels_fine=None, 
                target_voxels=None,gt_occupancy_ori=None, visible_mask=None, **kwargs):
        loss_dict = {}
        # import pdb;pdb.set_trace()
        for index, output_voxel in enumerate(output_voxels):
            loss_dict.update(self.loss_voxel(output_voxel, target_voxels,gt_occupancy_ori,  tag='c_{}'.format(index)))
        return loss_dict
