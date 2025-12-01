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

from .mask2former_utils.base.mmdet_utils import (sample_valid_coords_with_frequencies,
                          get_uncertain_point_coords_3d_with_frequency,
                          preprocess_occupancy_gt, point_sample_3d)

from .mask2former_utils.base.anchor_free_head import AnchorFreeHead
from .mask2former_utils.base.maskformer_head import MaskFormerHead
from .utils.semkitti import semantic_kitti_class_frequencies
from mmdet3d.models.alocc.heads.occ_loss_utils import nusc_class_frequencies, nusc_class_names

from mmdet3d.models import builder

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.datasets.semkitti import semantic_kitti_class_frequencies

def generate_forward_transformation_matrix(bda, img_meta_dict=None):
    b = bda.size(0)
    hom_res = torch.eye(4)[None].repeat(b, 1, 1).to(bda.device)
    for i in range(b):
        hom_res[i, :3, :3] = bda[i]
    return hom_res


# Modified from `Masked-attention Mask Transformer for Universal Image Segmentation <https://arxiv.org/pdf/2112.01527>`
@HEADS.register_module()
class ALOccHead(MaskFormerHead):
    """
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
                 init_cfg=None,
                 use_class_weight=True,
                balance_cls_weight=True,
                mask_embed2=False,
                wo_assign=False,
                out_channels_embed2=48,
                num_points_img=12544,
                dataset='nusc',
                open_occ=False,
                pred_flow=False,
                flow_l2_loss=False,
                context_post_process=None,
                flow_post_process=None,
                flow_loss_weight=1.0,
                sup_occupy_only=False,
                do_history=True,
                history_cat_num=1,
                interpolation_mode='bilinear',
                use_flow_bin_decoder=False,
                flow_bin_fixed=False,
                sup_bin=False,
                flow_scale=1.0,
                pred_flow_only=False,
                flow_cosine_loss=False,
                flow_out_channels=2,
                flow_bev=False,
                flow_post_neck=None,
                bev_out_channels=256,
                leanable_scale=False,
                freeze_occ=False,
                flow_with_his=False,
                flow_his_kernal_size=11,
                flow_his_dilation=2,
                bev_3_9=True,
                flow_gt_denoise=False,
                flow_gt_denoise_rate=1.,
                cost_volum_temprature=0.1,
                class_weights=None,
                num_cls=19,
                empty_idx=18,
                class_freq=None,
                 **kwargs):
        super(AnchorFreeHead, self).__init__(init_cfg)
        #######
        self.use_flow_bin_decoder=use_flow_bin_decoder
        if self.use_flow_bin_decoder:
            # flow_out_channels=88
            self.bin_invertal=[-22,22,0.5]
            flow_out_channels=len(np.arange(self.bin_invertal[0],self.bin_invertal[1],self.bin_invertal[2]))*2
            self.flow_out_channels=flow_out_channels
            self.flow_bin_fixed=flow_bin_fixed
            self.flow_bin=torch.from_numpy(np.arange(self.bin_invertal[0],self.bin_invertal[1],self.bin_invertal[2]))+0.5
            if flow_bin_fixed:
                self.flow_bin=torch.from_numpy(np.arange(self.bin_invertal[0],self.bin_invertal[1],self.bin_invertal[2]))
             
            else:
                if flow_bev:
                    self.flow_bin_decoder = nn.Sequential(
                            nn.Linear(bev_out_channels, bev_out_channels),
                            nn.ReLU(inplace=True),
                            nn.Linear(bev_out_channels, flow_out_channels),

                        )
                else:
                    self.flow_bin_decoder = nn.Sequential(
                                nn.Linear(feat_channels, feat_channels),
                                nn.ReLU(inplace=True),
                                nn.Linear(feat_channels, flow_out_channels),

                            )
            self.sup_bin=sup_bin

        self.open_occ=open_occ
        self.pred_flow=pred_flow
        self.flow_scale=flow_scale
        if leanable_scale:
            self.flow_scale=nn.Parameter(torch.tensor(flow_scale))
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
            if flow_post_neck is not None:
                self.flow_post_neck = builder.build_backbone(flow_post_neck)
            else:
                self.flow_post_neck = None

            flow_post_conv_channels=int(feat_channels*1.5)
            flow_post_conv_mid_channels=feat_channels

        
            if flow_post_process is not None:
                self.flow_post_conv = builder.build_backbone(flow_post_process)
            else:
                self.flow_post_conv =  nn.Sequential(
                            nn.Conv3d(
                                flow_post_conv_channels,
                                flow_post_conv_mid_channels,
                                kernel_size=3,
                                stride=1,
                                padding=1
                            ),
                nn.ReLU(inplace=True),
                )
                    
            flow_predicter_in_channels=feat_channels

            if not flow_bev:
                self.flow_predicter = nn.Sequential(
                    nn.Conv3d(
                                    flow_predicter_in_channels,
                                    flow_predicter_in_channels*2,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0
                                ),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(
                                    flow_predicter_in_channels*2,
                                    flow_out_channels,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0
                                )
                )
            else:
                self.flow_predicter = nn.Sequential(
                    nn.Conv2d(
                                    bev_out_channels,
                                    bev_out_channels*2,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0
                                ),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                                    bev_out_channels*2,
                                    flow_out_channels*16,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0
                                )
                )
        self.flow_l2_loss=flow_l2_loss
        

        self.flow_loss_weight=flow_loss_weight
        self.sup_occupy_only=sup_occupy_only
        
        self.pred_flow_only=pred_flow_only

        
        self.flow_cosine_loss=flow_cosine_loss
        if pred_flow:
            self.flow_with_his=flow_with_his
            if flow_with_his:

                self.do_history = do_history
                self.interpolation_mode = interpolation_mode
                self.history_cat_num = history_cat_num

                self.history_sweep_time = None
                self.history_bev = None
                self.history_seq_ids = None
                self.history_forward_augs = None
                cost_volumn_channel=feat_channels//2

                self.flow_his_kernal_size=flow_his_kernal_size
                self.flow_his_dilation=flow_his_dilation
                self.cost_volum_net=nn.Sequential(
                        nn.Conv2d(flow_his_kernal_size*flow_his_kernal_size, cost_volumn_channel, kernel_size=3,
                                    stride=1, padding=1),
                            nn.BatchNorm2d(cost_volumn_channel))
        self.flow_bev=flow_bev
        self.freeze_occ=freeze_occ

        self.bev_3_9=bev_3_9
        self.flow_gt_denoise=flow_gt_denoise
        self.flow_gt_denoise_rate=flow_gt_denoise_rate
        if flow_gt_denoise:
            self.history_bev_gt=None

        self.num_occupancy_classes = num_occupancy_classes
        self.num_classes = self.num_occupancy_classes
        self.num_queries = num_queries
        ''' Transformer Decoder Related '''
        # number of multi-scale features for masked attention
        self.num_transformer_feat_level = num_transformer_feat_level
        if transformer_decoder.transformerlayers is not None:
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
                
        if positional_encoding is not None:
            self.decoder_positional_encoding = build_positional_encoding(positional_encoding)
        if self.num_transformer_decoder_layers>0:
            self.query_embed = nn.Embedding(self.num_queries, feat_channels)

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

        if mask_embed2:
            self.mask_embed2 = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, out_channels_embed2))

        self.train_cfg = train_cfg
        if train_cfg:
            if self.train_cfg.assigner is not None:
                self.assigner = build_assigner(self.train_cfg.assigner)
            if self.train_cfg.sampler is not None:
                self.sampler = build_sampler(self.train_cfg.sampler, context=self)
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.num_points_img=num_points_img
            self.ori_num_points = self.num_points
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)
        ##################################################
        #class weight and sample weight
        ####################################################
        # create class_weights for semantic_kitti
        self.class_weight = loss_cls.class_weight

        if not use_class_weight:
            self.class_weight = np.ones_like(nusc_class_frequencies)
        else:
            if class_weights is None:
                if dataset=='nusc':
                    from mmdet3d.models.alocc.heads.occ_loss_utils import nusc_class_frequencies
                    if class_freq is None:
                        class_frequencies=nusc_class_frequencies
                    else:
                        class_frequencies=np.array(class_freq)
                    
                    if self.open_occ:
                        class_frequencies=class_frequencies[np.array([4,10,9,3,5,2,6,7,8,1,11,12,13,14,15,16,17])]
                elif dataset=='kitti':
                    class_frequencies = semantic_kitti_class_frequencies
                    
                elif dataset=='waymo':
                    from mmdet3d.models.alocc.heads.occ_loss_utils import waymo_class_frequencies
                    class_frequencies=waymo_class_frequencies
            
                class_weights = 1 / np.log(class_frequencies[:self.num_classes]+0.001)
                sample_weights = 1 / class_frequencies
                    
            else:
                class_weights=np.array(class_weights)
                
                sample_weights=np.array(class_weights)

            norm_class_weights = class_weights / class_weights[0]
            norm_class_weights = norm_class_weights.tolist()
            # append the class_weight for background
            
            if dataset!='kitti':
                norm_class_weights = [0.0]+norm_class_weights
            
            if not wo_assign:
                norm_class_weights.append(self.class_weight[-1])
                
            self.class_weight = norm_class_weights
            loss_cls.class_weight = self.class_weight
            
            sample_weights = sample_weights / sample_weights.min()
            if dataset!='kitti':
                sample_weights=np.concatenate([[0.0],sample_weights])

            self.baseline_sample_weights = sample_weights

        self.sample_weight_gamma = sample_weight_gamma
        
        self.loss_cls = build_loss(loss_cls)
        self.loss_mask = build_loss(loss_mask)
        self.loss_dice = build_loss(loss_dice)
        self.pooling_attn_mask = pooling_attn_mask
        
        # align_corners
        self.align_corners = align_corners


        ###################################
        loss_weight_cfg=None
        balance_cls_weight=balance_cls_weight
        num_cls=num_occupancy_classes
        self.empty_idx=empty_idx
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
            if dataset=='nusc':
                if num_cls == 19:
                    self.class_weights = torch.from_numpy(1 / np.log(class_frequencies[:num_cls] + 0.001))
                    self.class_weights = torch.cat([torch.tensor([0]), self.class_weights])
                else:
                    if num_cls == 17: class_frequencies[0] += class_frequencies[-1]
                    self.class_weights = torch.from_numpy(1 / np.log(class_frequencies[:num_cls] + 0.001))
            elif dataset=='waymo':
                if num_cls == 17:
                    self.class_weights = torch.from_numpy(class_frequencies/max(class_frequencies))
                    self.class_weights = torch.cat([torch.tensor([0]), self.class_weights])                                 
                else:
                    pass
        else:
            self.class_weights = torch.ones(num_cls)/num_cls  # FIXME hardcode 17

        self.num_cls=num_cls
        self.wo_assign=wo_assign
        self.cost_volum_temprature=cost_volum_temprature

    @force_fp32()
    def fuse_history(self, curr_bev, img_metas, bda,curr2=None,bev_2d=False): # align features with 3d shift
        if bev_2d:
            curr_bev = curr_bev.unsqueeze(-1)
            if curr2 is not None:
                curr2 = curr2.unsqueeze(-1)
        voxel_feat = True  if len(curr_bev.shape) == 5 else False
        if voxel_feat:
            curr_bev = curr_bev.permute(0, 1, 4, 2, 3) # n, c, z, h, w
            if curr2 is not None:
                curr2 = curr2.permute(0, 1, 4, 2, 3) # n, c, z, h, w
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
            if curr2 is not None:
                self.history_bev_gt=curr2.clone()
            self.history_seq_ids = seq_ids.clone()
            self.history_forward_augs = forward_augs.clone()

            # Repeat the first frame feature to be history
            if voxel_feat:
                self.history_bev = curr_bev.repeat(1, self.history_cat_num, 1, 1, 1) 
                if curr2 is not None:
                    self.history_bev_gt=curr2.repeat(1, self.history_cat_num, 1, 1, 1) 
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
                if curr2 is not None:
                    self.history_bev_gt[start_of_sequence]=curr2[start_of_sequence].repeat(1, self.history_cat_num, 1, 1, 1) 
            else:
                self.history_bev[start_of_sequence] = curr_bev[start_of_sequence].repeat(1, self.history_cat_num, 1, 1)
            
            self.history_sweep_time[start_of_sequence] = 0 # zero the new sequence timestep starts
            self.history_seq_ids[start_of_sequence] = seq_ids[start_of_sequence]
            self.history_forward_augs[start_of_sequence] = forward_augs[start_of_sequence]


        ## Get grid idxs & grid2bev first.
        if voxel_feat:
            n, c_, z, h, w = curr_bev.shape
        if not bev_2d:
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
            feat2bev[0, 0] = 0.4
            feat2bev[1, 1] = 0.4
            feat2bev[0, 3] = -40.
            feat2bev[1, 3] = -40.
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

            if curr2 is not None and  self.training:
                sampled_history_bev_gt = F.grid_sample(self.history_bev_gt[:,:,0], grid.to(curr_bev.dtype)[...,0,:], align_corners=True, mode=self.interpolation_mode)
                
        feats_cat=sampled_history_bev
        feats_to_return=feats_cat
        if bev_2d:
            feats_to_return=feats_to_return.unsqueeze(2)
            if curr2 is not None and  self.training:
                sampled_history_bev_gt=sampled_history_bev_gt.unsqueeze(2)
        self.history_bev = curr_bev
        self.history_bev_gt=curr2
        
        self.history_sweep_time = self.history_sweep_time[:, :-1]
        self.history_forward_augs = forward_augs.clone()
        if voxel_feat:
            feats_to_return = feats_to_return.permute(0, 1, 3, 4, 2)
            if curr2 is not None and  self.training:
                sampled_history_bev_gt=sampled_history_bev_gt.permute(0, 1, 3, 4, 2)
        if not self.do_history:
            self.history_bev = None
        if bev_2d:
            feats_to_return=feats_to_return.squeeze(-1)
            if curr2 is not None and  self.training:
                sampled_history_bev_gt=sampled_history_bev_gt.squeeze(-1)
        if curr2 is not None and  self.training:
            feats_to_return=[feats_to_return.clone()   ,sampled_history_bev_gt.clone()   ]
        else:
            feats_to_return=feats_to_return.clone()

        return feats_to_return
    

    def get_sampling_weights(self):
        if type(self.sample_weight_gamma) is list:
            # dynamic sampling weights
            min_gamma, max_gamma = self.sample_weight_gamma
            sample_weight_gamma = np.random.uniform(low=min_gamma, high=max_gamma)
        else:
            sample_weight_gamma = self.sample_weight_gamma
        self.sample_weights = self.baseline_sample_weights ** sample_weight_gamma
    
    def set_num_points(self,target_size):

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
        
    
        pos_inds=None
        neg_inds=None
        
 
        labels=None
        label_weights=None
        class_weights_tensor = torch.tensor(self.class_weight).type_as(mask_pred)

        # mask target
        # mask_targets = gt_masks[sampling_result.pos_assigned_gt_inds]
        mask_targets=gt_masks
        
        mask_weights = torch.zeros_like(class_weights_tensor).to(mask_pred)
        mask_weights[gt_labels] = class_weights_tensor[gt_labels]
        # mask_weights=class_weights_tensor
        
        
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
 
        
        losses_cls, losses_mask, losses_dice = multi_apply(
            self.loss_single, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)
        
        
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
        
        
        losses_mask, losses_dice = multi_apply(
            self.loss_single_wo_assign, all_cls_scores, all_mask_preds,
            all_gt_labels_list, all_gt_masks_list, img_metas_list)
        
        


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

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets(cls_scores_list, mask_preds_list,
                                gt_labels_list, gt_masks_list, img_metas)

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

        loss_cls = self.loss_cls(
            cls_scores,
            labels,
            label_weights,
            avg_factor=class_weight[labels].sum(),
        )
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

        (labels_list, label_weights_list, mask_targets_list, mask_weights_list,
         num_total_pos,
         num_total_neg) = self.get_targets_wo_assign( mask_preds_list,
                                gt_labels_list, gt_masks_list, img_metas)


        # shape (num_total_gts, h, w)
        mask_targets = torch.cat(mask_targets_list, dim=0)
        # shape (batch_size, num_queries)
        mask_weights = torch.stack(mask_weights_list, dim=0)
        if self.sup_occupy_only:
            mask_weights[:,-1]=0

 
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
        mask_preds_=mask_preds.unsqueeze(1)
        with torch.no_grad():
            point_indices, point_coords = get_uncertain_point_coords_3d_with_frequency(
                mask_preds_, None, gt_labels_list, gt_masks_list, 
                self.sample_weights, self.num_points, self.oversample_ratio, 
                self.importance_sample_ratio)
            
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = torch.gather(mask_targets.view(mask_targets.shape[0], -1), dim=1, index=point_indices)

        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample_3d(mask_preds.unsqueeze(1), point_coords[..., [2, 1, 0]], align_corners=self.align_corners).squeeze(1)

        # dice loss
        num_total_mask_weights = reduce_mean(mask_weights.sum())
   
        loss_dice = self.loss_dice(mask_point_preds, mask_point_targets, 
                            weight=mask_weights, avg_factor=num_total_mask_weights)

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
    def occ2bev(self,feat,occ_weight):
        
                    
        occ_weight=occ_weight.flip([-1])
        free_weight=1-occ_weight
        
        cum_free_weight=torch.cumprod(free_weight,dim=-1)
        cum_free_weight=torch.cat([torch.ones_like(cum_free_weight[...,0:1]),cum_free_weight[...,:-1]],dim=-1)
        weight=occ_weight*cum_free_weight
        weight=weight.flip([-1])
        out=(feat*weight.unsqueeze(1)).sum(-1)
        return out
    def flow_decoder(self, x,img_metas, occ_pred=None,num_cls=18,**kwargs):
        """Flow decoder.

        Args:
            x (Tensor): Input feature map.

        Returns:
            Tensor: Output feature map.
        """
        if self.flow_with_his:
            
            with torch.no_grad():
                occ_weight=(occ_pred!=num_cls-1).float()
                if self.flow_gt_denoise and  self.training:
                    gt_occ=kwargs['gt_occupancy_ori']
                    gt_occ[gt_occ==255]=num_cls
                    occ_shape=gt_occ.shape
                    gt_occ_onehot=F.one_hot(gt_occ.long(),num_cls+1).reshape(*occ_shape,num_cls+1)
                    gt_occ_onehot=gt_occ_onehot.permute(0,4,1,2,3).float()
                    if self.bev_3_9:
                        gt_occ_onehot_curr=gt_occ_onehot[...,3:9].mean(-1)

                if self.bev_3_9:
                    bev_feat_curr=x[...,3:9].mean(-1)
                else:
                    bev_feat_curr=self.occ2bev(x,occ_weight)

                if self.flow_gt_denoise and  self.training:
                    bev_feat = self.fuse_history(bev_feat_curr, img_metas, kwargs['bda'],curr2=gt_occ_onehot_curr,bev_2d=True)
                    bev_feat,bev_feat_gt=bev_feat
                    gt_occ_onehot_curr= F.interpolate(gt_occ_onehot_curr, [gt_occ_onehot_curr.shape[-2]//2,gt_occ_onehot_curr.shape[-1]//2], mode='bilinear', align_corners=True)
                    bev_feat_gt= F.interpolate(bev_feat_gt, [bev_feat_gt.shape[-2]//2,bev_feat_gt.shape[-1]//2], mode='bilinear', align_corners=True)
                    bev_feat_gt_unfold=F.unfold(bev_feat_gt,self.flow_his_kernal_size,dilation=self.flow_his_dilation,padding=self.flow_his_kernal_size+self.flow_his_dilation-3)
                    
                    bev_feat_gt_unfold=bev_feat_gt_unfold.reshape(*bev_feat_gt.shape[:2],-1,*bev_feat_gt.shape[-2:])
                    
                    cost_volumn_gt=-(F.normalize(gt_occ_onehot_curr,dim=1).unsqueeze(2)*F.normalize(bev_feat_gt_unfold,dim=1)).sum(1)/self.cost_volum_temprature
                    invalid=bev_feat_gt_unfold.sum(1)==0
                    cost_volumn_gt[invalid] = cost_volumn_gt[invalid] + 5.

                else:
                    bev_feat = self.fuse_history(bev_feat_curr, img_metas, kwargs['bda'],bev_2d=True)
        
                bev_feat_curr= F.interpolate(bev_feat_curr, [bev_feat_curr.shape[-2]//2,bev_feat_curr.shape[-1]//2], mode='bilinear', align_corners=True)
                bev_feat= F.interpolate(bev_feat, [bev_feat.shape[-2]//2,bev_feat.shape[-1]//2], mode='bilinear', align_corners=True)
                bev_feat_unfold=F.unfold(bev_feat,self.flow_his_kernal_size,dilation=self.flow_his_dilation,padding=self.flow_his_kernal_size+self.flow_his_dilation-3)
                
                bev_feat_unfold=bev_feat_unfold.reshape(*bev_feat.shape[:2],-1,*bev_feat.shape[-2:])

                cost_volumn=-(F.normalize(bev_feat_curr,dim=1).unsqueeze(2)*F.normalize(bev_feat_unfold,dim=1)).sum(1)/self.cost_volum_temprature
                invalid=bev_feat_unfold.sum(1)==0
                # cost_volumn=cost_volumn.permute(0,2,3,1)
                cost_volumn[invalid] = cost_volumn[invalid] + 5.

                if self.flow_gt_denoise and  self.training:
                    cost_volumn=self.flow_gt_denoise_rate*cost_volumn_gt+cost_volumn*(1-self.flow_gt_denoise_rate)

                cost_volumn = - cost_volumn
                cost_volumn = cost_volumn.softmax(dim=1)
            cost_volumn=self.cost_volum_net(cost_volumn)
            cost_volumn= F.interpolate(cost_volumn, [cost_volumn.shape[-2]*2,cost_volumn.shape[-1]*2], mode='bilinear', align_corners=True)
            cost_volumn=cost_volumn.unsqueeze(-1).repeat(1,1,1,1,x.shape[-1])
            bev_feat=torch.cat((cost_volumn,x),dim=1)
        else:
            bev_feat = x

        flow_pred_ = self.flow_post_conv(bev_feat)
        if self.flow_post_neck is not None:
            flow_pred_=self.flow_post_neck(flow_pred_)
        if isinstance(flow_pred_, list):
            flow_pred_ = flow_pred_[0]
            # output['flow'] = [flow_pred]

        flow_pred = self.flow_predicter(flow_pred_)
        if self.flow_bev:
            flow_pred = flow_pred.reshape(flow_pred.shape[0], -1,16, flow_pred.shape[2], flow_pred.shape[3])
            flow_pred=flow_pred.permute(0, 1, 3,4,2)

        if self.use_flow_bin_decoder:
            bin_weight=flow_pred[:,:,None,...]
            if self.flow_bin_fixed:
                bin_center=self.flow_bin.to(flow_pred.device).float()
                bin_center=bin_center.unsqueeze(-1).unsqueeze(0).repeat(flow_pred.shape[0],1,2)
            else:    
                if self.flow_bev:
                    bin_prob=self.flow_bin_decoder(flow_pred_.mean(dim=[2, 3]))
                else:
                    bin_prob=self.flow_bin_decoder(flow_pred_.mean(dim=[2, 3, 4]))

                bin_prob=bin_prob.reshape(bin_prob.shape[0],self.flow_out_channels//2,2)
                bin_prob=bin_prob.softmax(1)
            

                cum_bin_prob=torch.cat((torch.zeros_like(bin_prob[:,:1,...]),torch.cumsum(bin_prob,1)[:,:-1]),dim=1)
                bin_center=self.bin_invertal[0]+(self.bin_invertal[1]-self.bin_invertal[0])*(bin_prob/2+cum_bin_prob)#BN,n_bin,H,W
            bin_weight=bin_weight.reshape(bin_weight.shape[0],bin_weight.shape[1]//2,2,*bin_weight.shape[3:])
            bin_ce_loss=None
            if self.flow_bin_fixed and self.sup_bin and self.training:
                gt_flow=kwargs['gt_occ_flow']
                valid_mask=gt_flow!=float('inf')
                if valid_mask.sum()!=0:
                    gt_flow=gt_flow[valid_mask]
                    gt_flow=((gt_flow-self.bin_invertal[0])/self.bin_invertal[2]).round()
                    gt_flow=torch.clamp(gt_flow,0,bin_center.shape[1]-1)
                    gt_flow=gt_flow.long()
                    
                    bin_weight_valid=bin_weight.permute(0,3,4,5,2,1)[valid_mask]
                    bin_ce_loss=F.cross_entropy(bin_weight_valid,gt_flow)
                else:
                    bin_ce_loss=torch.zeros(1).to(gt_flow)
                
                
            bin_weight=bin_weight.softmax(1)
            flow_pred = torch.sum(bin_weight *bin_center[...,None,None,None], dim=1)

        flow_pred=flow_pred.permute(0, 2, 3, 4, 1)

        out=flow_pred*self.flow_scale
        if self.use_flow_bin_decoder and self.flow_bin_fixed and self.sup_bin:
            out=[out,bin_ce_loss]
        
        return out
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
        if not self.freeze_occ and (not self.pred_flow_only or tag!='vox'):
        # reset the sampling weights
            self.get_sampling_weights()

            # forward
            all_cls_scores, all_mask_preds = self(feats, img_metas,mask_embede2=mask_embede2,class_prototype=class_prototype,tag=tag)
            

            if self.training:
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

        if self.pred_flow and tag=='vox':

            flow_pred =self.flow_decoder(feats[0],img_metas, all_mask_preds[0].max(1)[1],num_cls=all_mask_preds[0].shape[1],**kwargs)
            if self.use_flow_bin_decoder and self.flow_bin_fixed and self.sup_bin:
                flow_pred,bin_ce_loss=flow_pred
                losses.update({'bin_ce_loss':bin_ce_loss})

            gt_occ_flow=kwargs['gt_occ_flow']

            mask=gt_occ_flow!=float('inf')
            
            
            if  self.flow_l2_loss:
                self.flow_criterion=F.mse_loss
            else: 
                self.flow_criterion=F.l1_loss
                
            
            if mask.sum()==0:
                loss_flow=0.*flow_pred[0,0,0,0].sum()
            else:
                
                mask=mask[...,0]
                loss_flow= self.flow_criterion(flow_pred[mask],gt_occ_flow[mask])*self.flow_loss_weight
            losses.update({'loss_flow':loss_flow})

            if self.flow_cosine_loss:
                mask=gt_occ_flow!=float('inf')
                if mask.sum()==0:
                    loss_flow_cosine=0.*flow_pred[0,0,0,0].sum()
                else:
                    mask=mask[...,0]
                    flow_pred_=flow_pred[mask]
                    gt_occ_flow_=gt_occ_flow[mask]
                    loss_flow_cosine=-(F.cosine_similarity(flow_pred_,gt_occ_flow_,dim=-1)).mean()*self.flow_loss_weight
                losses.update({'loss_flow_cosine':loss_flow_cosine})

                
        return losses,all_cls_scores,all_mask_preds

        
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
        
        if self.num_transformer_decoder_layers>0:
            for i in range(self.num_transformer_feat_level):
                ''' with flatten features '''
                # projection for input features
                
                decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
                # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
                decoder_input = decoder_input.flatten(2).permute(2, 0, 1)
                ''' with level embeddings '''
                level_embed = self.level_embed.weight[i].view(1, 1, -1)
                decoder_input = decoder_input + level_embed
                ''' with positional encodings '''
                # shape (batch_size, c, x, y, z) -> (x * y * z, batch_size, c)
                mask = decoder_input.new_zeros((batch_size, ) + multi_scale_memorys[i].shape[-3:], dtype=torch.bool)
                
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

            mask_cls_results = all_cls_scores[-1]
            mask_pred_results = all_mask_preds[-1]


            output_voxels = self.format_results(mask_cls_results, mask_pred_results)

            pred_mask=mask_pred_results.sigmoid()
            density1=pred_mask[:,:-1].max(1)[0]

            density=(density1>0.5).float()
        else:
            output_voxels=None
            density=None
   
        res = {
            'output_voxels': [output_voxels],
            'output_points': None,
            'output_density':[density]
        }
        if self.pred_flow:

            flow_pred =self.flow_decoder(feats[0],img_metas, output_voxels.max(1)[1],num_cls=all_mask_preds[0].shape[1],**kwargs)
            if self.use_flow_bin_decoder and self.flow_bin_fixed and self.sup_bin:
                flow_pred,_=flow_pred
        
            res['output_flow'] = [flow_pred]

        return res
