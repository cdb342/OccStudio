import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.checkpoint import checkpoint
from mmdet.models.backbones.resnet import BasicBlock
from mmdet.models import HEADS
import torch.utils.checkpoint as cp
from mmdet3d.models import builder
from mmcv.runner import force_fp32, auto_fp16
import torch
from torchvision.utils import make_grid
import torchvision
import matplotlib.pyplot as plt
import cv2
import math
from mmcv.cnn import xavier_init, constant_init
import time
from itertools import product
from .cal_depth2occ import cal_depth2occ
from ..modules.temporal_fusion import GeometryHistoryFusion

z=list(product([-1,0,1],[-1,0,1]))

@HEADS.register_module()
class NaiveDepthNet(BaseModule):
    r"""Naive depthnet used in Lift-Splat-Shoot 

    Please refer to the `paper <https://arxiv.org/abs/2008.05711>`_

    Args:
        in_channels (int): Channels of input feature.
        context_channels (int): Channels of transformed feature.
    """

    def __init__(
        self,
        in_channels=512,
        context_channels=64,
        depth_channels=118,
        downsample=16,
        uniform=False,
        with_cp=False
    ):
        super(NaiveDepthNet, self).__init__()
        self.uniform = uniform
        self.with_cp = with_cp     
        self.context_channels = context_channels
        self.in_channels = in_channels
        self.D =depth_channels
        self.downsample=downsample,
        self.depth_net = nn.Conv2d(
            in_channels, self.D + self.context_channels, kernel_size=1, padding=0)
    
    @force_fp32()
    def forward(self, x, mlp_input=None):
        """
        """
       
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.depth_net, x)
        else:
            x = self.depth_net(x)            

        depth_digit = x[:, :self.D, ...]
        context = x[:, self.D:self.D + self.context_channels, ...]
        if self.uniform:
            depth_digit = depth_digit * 0
            depth = depth_digit.softmax(dim=1)
        else:
            depth = depth_digit.softmax(dim=1)
        context = context.view(B, N,  self.context_channels, H, W)
        depth = depth.view(B, N,  self.D, H, W)
        return context, depth

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        return None



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
    
    @force_fp32()
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
    
    @force_fp32()
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
    
    @force_fp32()
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
    
    @force_fp32()
    def forward(self, x, x_se):
 
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)

        return x * self.gate(x_se)

@HEADS.register_module()
class CM_DepthNet(BaseModule):
    """
        Camera parameters aware depth net
    """
    def __init__(self,
                in_channels=512,
                context_channels=64,
                depth_channels=118,
                mid_channels=512,
                use_dcn=True,
                downsample=16,
                grid_config=None,
                loss_depth_weight=3.0,
                with_cp=False,
                se_depth_map=False,
                sid=False,
                bias=0.0,
                input_size=None,
                use_aspp=True,
                stereo=False,
                aspp_mid_channels=512,
                depth2occ_intra=False,
                depth2occ_intra_post_norm=False,
                length=44,
                train_depth_only=False,
                num_cls=17,
                soft_filling_with_offset=False,
                cam_channels=27,
                occlude_tau=1.0,
                depth2occ_inter=False,
                depth2occ_inter_kernal_size=3,
                depth2occ_inter_downsample=4,
                use_context_post_ln=False,
                geometry_denoise=False,
                geometry_denoise_rate=1.,
                ####################
                # GDFusion geometry his fusion
                geometry_his_fusion=False,
                geometry_his_fusion_with_gt=False,
                geometry_his_fusion_after_sup=True,
                depth_his_sup_w_his=False,
                depth_conv_head_split=False,
                per_pixel_weight=False,
                gate_net_with_his=False, 
                gate_net_mlp=False,
                ####################
                depth_gt=False,
                depth_sigmoid=False,
                geometry_group=False,
                 ):
        super(CM_DepthNet, self).__init__()
        self.fp16_enable=False
        self.sid=sid
        self.with_cp = with_cp
        self.downsample = downsample
        self.grid_config = grid_config
        self.loss_depth_weight = loss_depth_weight
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_channels = context_channels
        
        self.se_depth_map = se_depth_map

        
        self.bn = nn.BatchNorm1d(cam_channels)
        self.depth_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        context_conv_input_channels=mid_channels
        if not train_depth_only:
            self.context_mlp = Mlp(cam_channels, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
            self.context_conv = nn.Conv2d(
                        context_conv_input_channels, context_channels, kernel_size=1, stride=1, padding=0)
            self.use_context_post_ln=use_context_post_ln
            if use_context_post_ln:
                self.context_post_ln=nn.LayerNorm(context_channels)
        self.geometry_group=geometry_group
        
        if geometry_group:
            depth_channels=depth_channels*geometry_group
        depth_conv_input_channels = mid_channels
        downsample_net = None
        self.geometry_denoise=geometry_denoise
        self.geometry_denoise_rate=geometry_denoise_rate

        self.geometry_his_fusion=geometry_his_fusion
        self.geometry_his_fusion_with_gt=geometry_his_fusion_with_gt
        self.geometry_his_fusion_after_sup=geometry_his_fusion_after_sup
        self.depth_conv_head_split=depth_conv_head_split
        self.depth_gt=depth_gt
        
        if geometry_his_fusion:
            self.depth_his_fusion_post_layer=GeometryHistoryFusion(depth_channels,grid_config,input_size=input_size,
                downsample=downsample,forward_post=True,context_channel=mid_channels,depth_his_sup_w_his=depth_his_sup_w_his,
                per_pixel_weight=per_pixel_weight,gate_net_with_his=gate_net_with_his)

            if gate_net_with_his:
                if gate_net_mlp:
                    self.gate_net=nn.Sequential(
                    nn.Conv2d(
                        depth_channels*2, depth_channels, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(
                        depth_channels, 1, kernel_size=1, stride=1, padding=0),
                    nn.Sigmoid()
                    )
                else:
                    self.gate_net=nn.Sequential(
                        nn.Conv2d(
                            depth_channels*2, 1, kernel_size=1, stride=1, padding=0),
                        
                        nn.Sigmoid()
                        )
            else:
                if gate_net_mlp:
                    self.gate_net=nn.Sequential(
                    nn.Conv2d(
                        mid_channels, mid_channels//2, kernel_size=1, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(
                        mid_channels//2, 1, kernel_size=1, stride=1, padding=0),
                    nn.Sigmoid()
                    )
                else:
                    self.gate_net=nn.Sequential(
                            nn.Conv2d(
                                mid_channels, 1, kernel_size=1, stride=1, padding=0),
                            
                            nn.Sigmoid()
                            )

            self.gate_net_with_his=gate_net_with_his
            
        #################
        self.depth2occ_intra=depth2occ_intra
        self.depth2occ_intra_post_norm=depth2occ_intra_post_norm
        self.length=length
        
        self.depth_channels = depth_channels
        self.train_depth_only=train_depth_only

        self.soft_filling_with_offset=soft_filling_with_offset
    
        
        self.occlude_tau=occlude_tau
        self.depth2occ_inter=depth2occ_inter
        if depth2occ_inter:
            self.depth2occ_inter_downsample=depth2occ_inter_downsample
            self.depth2occ_inter_kernal_size=depth2occ_inter_kernal_size
            self.depth2occ_inter_weight_net=nn.Conv2d(
                    mid_channels,
                    depth2occ_inter_kernal_size*depth2occ_inter_kernal_size*3,
                    kernel_size=1,
                    stride=1,
                    padding=0)
            self.depth2occ_inter_bias_net=nn.Conv2d(
                    mid_channels,
                    depth2occ_inter_kernal_size*depth2occ_inter_kernal_size*2,
                    kernel_size=1,
                    stride=1,
                    padding=0)
            self.depth2occ_inter_bias_net.bias.data = torch.tensor(z).view(-1).float()
            zz=torch.zeros(depth2occ_inter_kernal_size,depth2occ_inter_kernal_size)
            # zz[depth2occ_inter_kernal_size//2,depth2occ_inter_kernal_size//2]=1.
            
            
            self.depth2occ_inter_weight_net.bias.data=zz.unsqueeze(-1).repeat(1,1,3).view(-1)
            
            torch.nn.init.kaiming_normal_(self.depth2occ_inter_weight_net.weight)
            torch.nn.init.kaiming_normal_(self.depth2occ_inter_bias_net.weight)
        
        
        #####################
        self.depth_stereo=stereo
        if stereo:
            cost_volumn_channels=depth_channels
            
            if geometry_group:
                cost_volumn_channels=cost_volumn_channels//geometry_group
            
            
            depth_conv_input_channels += cost_volumn_channels
            downsample_net = nn.Conv2d(depth_conv_input_channels,
                                    mid_channels, 1, 1, 0)
            cost_volumn_net = []
            for stage in range(int(2)):
                cost_volumn_net.extend([
                    nn.Conv2d(cost_volumn_channels, cost_volumn_channels, kernel_size=3,
                            stride=2, padding=1),
                    nn.BatchNorm2d(cost_volumn_channels)])
            self.cost_volumn_net = nn.Sequential(*cost_volumn_net)
            
            self.bias = bias
        
            self.cv_frustum = self.create_frustum(grid_config['depth'],
                                                    input_size,
                                                    downsample=self.downsample//4)
            ##############################
      
        
        depth_conv_list = [
           BasicBlock(depth_conv_input_channels, mid_channels,
                                      downsample=downsample_net),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        ]
        if use_aspp:
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
        if not depth_conv_head_split:
            depth_conv_list.append(
                nn.Conv2d(
                    mid_channels,
                    depth_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0))
        else:
            self.depth_head=nn.Conv2d(
                    mid_channels,
                    depth_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0)
            self.depth_head2=nn.Conv2d(
                    mid_channels,
                    depth_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0)
        self.depth_conv = nn.Sequential(*depth_conv_list)

        ##########################################
        if self.soft_filling_with_offset:
            
            offset_channel=depth_channels
            
            if geometry_group:
                offset_channel=offset_channel//geometry_group
            
            
            self.offset_conv = nn.Conv2d(
                    context_channels,
                    offset_channel*3,
                    kernel_size=1,
                    stride=1,
                    padding=0)
     
            constant_init(self.offset_conv, 0)


        if self.depth2occ_intra:
            occ_channels=self.length
            if geometry_group:
                occ_channels=occ_channels*geometry_group
            occ_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                        downsample=downsample_net),
                            BasicBlock(mid_channels, mid_channels),
                            BasicBlock(mid_channels, mid_channels)]
           

            occ_conv_list.append(
                nn.Conv2d(
                    mid_channels,
                    occ_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0))
            self.occ_conv = nn.Sequential(*occ_conv_list)
      
        ######################################
        self.depth_sigmoid=depth_sigmoid
        self.num_cls=num_cls
        
        

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
        # if self.sid:
        #     d_sid = torch.arange(self.D).float()
        #     depth_cfg_t = torch.tensor(depth_cfg).float()
        #     d_sid = torch.exp(torch.log(depth_cfg_t[0]) + d_sid / (self.D-1) *
        #                       torch.log((depth_cfg_t[1]-1) / depth_cfg_t[0]))
        #     d = d_sid.view(-1, 1, 1).expand(-1, H_feat, W_feat)
        x = torch.linspace(0, W_in - 1, W_feat,  dtype=torch.float)\
            .view(1, 1, W_feat).expand(self.D, H_feat, W_feat)
        y = torch.linspace(0, H_in - 1, H_feat,  dtype=torch.float)\
            .view(1, H_feat, 1).expand(self.D, H_feat, W_feat)
        

        # D x H x W x 3
        return torch.stack((x, y, d), -1)
    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        # pass
        frustum =self.cv_frustum.to(metas['post_trans'].device)

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
    # @force_fp32()
    def calculate_cost_volumn(self, metas):
        # t111=time.time()
        prev, curr = metas['cv_feat_list']
        
        group_size = 4
        _, c, hf, wf = curr.shape
        hi, wi = hf * 4, wf * 4
        B, N, _ = metas['post_trans'].shape
        D, H, W, _ = self.cv_frustum.shape
        # self.gen_grid(metas, B, N, D, H, W, hi, wi)
        grid = self.gen_grid(metas, B, N, D, H, W, hi, wi).to(curr.dtype)

        prev = prev.view(B * N, -1, H, W)
        curr = curr.view(B * N, -1, H, W)
        cost_volumn = 0
      
        for fid in range(curr.shape[1] // group_size):
            prev_curr = prev[:, fid * group_size:(fid + 1) * group_size, ...]
            warp_prev = F.grid_sample(prev_curr, grid,
                                      align_corners=True,
                                      padding_mode='zeros')
            curr_tmp = curr[:, fid * group_size:(fid + 1) * group_size, ...]
            cost_volumn_tmp = curr_tmp.unsqueeze(2) - \
                              warp_prev.view(B * N, -1, D, H, W)
            cost_volumn_tmp = cost_volumn_tmp.abs().sum(dim=1)
            cost_volumn += cost_volumn_tmp
       
        if not self.bias == 0:
            invalid = warp_prev[:, 0, ...].view(B * N, D, H, W) == 0
           
            cost_volumn[invalid] = cost_volumn[invalid] + self.bias
           
        cost_volumn = - cost_volumn
        cost_volumn = cost_volumn.softmax(dim=1)
        
        
        return cost_volumn
  
    @force_fp32()
    def forward(self, x, cam_params,stereo_metas=None,img_metas=None,cost_volumn=None,**kwargs):
        output=dict()

        x = x.to(torch.float32) # FIX distill type error
        b,n,c,h,w=x.shape
        
        if self.depth_stereo:
            if stereo_metas['cv_feat_list'][0] is None:
                BN, _, H, W = x.shape
                scale_factor = float(stereo_metas['downsample'])/\
                               stereo_metas['cv_downsample']
                cost_volumn_ = \
                    torch.zeros((BN, self.depth_channels,
                                 int(H*scale_factor),
                                 int(W*scale_factor))).to(x)
            else:
                with torch.no_grad():
                    cost_volumn_ = self.calculate_cost_volumn(stereo_metas)
             
                if cost_volumn is not None:
                    cost_volumn=torch.cat((cost_volumn,cost_volumn_),dim=1)
                else:
                    cost_volumn=cost_volumn_
        if cost_volumn is not None:
            cost_volumn = self.cost_volumn_net(cost_volumn)
        
        mlp_input = self.get_mlp_input(*cam_params)
        mlp_input=mlp_input.float()
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        #############
  
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.reduce_conv, x)
        else:
            x = self.reduce_conv(x)
        if not self.train_depth_only:
            context_se = self.context_mlp(mlp_input)[..., None, None]
            if self.with_cp and x.requires_grad:
                context_ = cp.checkpoint(self.context_se, x, context_se)
            else:
                context_ = self.context_se(x, context_se)
            
            
                
            context = self.context_conv(context_)
        else:
            context=None

        ###########
        if self.soft_filling_with_offset:
            coor_offsets=self.offset_conv(context)
            ###################
     
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        # depth_se=None
        depth_ = self.depth_se(x, depth_se)
    
        if self.depth2occ_inter:
            depth2occ_inter_weight=self.depth2occ_inter_weight_net(depth_)
            depth2occ_inter_bias=self.depth2occ_inter_bias_net(depth_)
            depth2occ_inter_weight=depth2occ_inter_weight.reshape(B*N,self.depth2occ_inter_kernal_size,self.depth2occ_inter_kernal_size,3,H,W).sigmoid()
            
            depth2occ_inter_bias=depth2occ_inter_bias.reshape(B*N,self.depth2occ_inter_kernal_size,self.depth2occ_inter_kernal_size,2,H,W)
   
            depth2occ_inter_bias=depth2occ_inter_bias*self.depth2occ_inter_downsample
         
            depth2occ_inter_bias=torch.clamp(depth2occ_inter_bias,self.depth2occ_inter_downsample*(-2),self.depth2occ_inter_downsample*2)
     
            output.update({'depth2occ_inter_weight':depth2occ_inter_weight,'depth2occ_inter_bias':depth2occ_inter_bias})
                
                
        if cost_volumn is not None:
           
            depth_ = torch.cat([depth_, cost_volumn], dim=1)
       
        
        if self.depth2occ_intra:
    
            depth_feat = self.depth_conv(depth_)
            if self.depth_conv_head_split:
                depth=self.depth_head(depth_feat)
            else:
                depth=depth_feat

            occ_weight = self.occ_conv(depth_.clone())
            if self.geometry_group:

                occ_weight=occ_weight.reshape(occ_weight.shape[0]*self.geometry_group,-1,*occ_weight.shape[2:])

            # occ_weight=occ_weight.view(B, N, self.length, H, W)
            occ_weight=(occ_weight/self.occlude_tau).sigmoid()
      
        else:
            if self.with_cp and depth_.requires_grad:
                depth_feat = cp.checkpoint(self.depth_conv, depth_)
                if self.depth_conv_head_split:
                    depth = cp.checkpoint(self.depth_head, depth_feat)
                else:
                    depth=depth_feat
            else:
                depth_feat = self.depth_conv(depth_)
                if self.depth_conv_head_split:
                    depth=self.depth_head(depth_feat)
                else:
                    depth=depth_feat
        
        if not self.train_depth_only:
            context = context.view(B, N,  self.context_channels, H, W)
        if self.geometry_group:
            N=N*self.geometry_group
        depth = depth.view(B, N, -1, H, W)
        
        
        if self.depth_sigmoid:
            depth=depth.sigmoid()
        else:
            depth = depth.softmax(dim=2)
        output['depth_pred']=depth

        if self.geometry_denoise and self.training:
            depth_copy=depth.clone()
            gt_depth=self.get_downsampled_gt_depth(kwargs['gt_depth'])
                    
            b,n,d,h,w=depth.shape
            gt_depth=gt_depth.reshape(b,n,h,w,d).permute(0,1,4,2,3)
            fg_mask = torch.max(gt_depth, dim=2,keepdim=True).values > 0.0
            fg_mask=fg_mask.repeat(1,1,d,1,1)
            
            depth_=depth.clone()

            depth_[fg_mask]=gt_depth[fg_mask]*self.geometry_denoise_rate+depth[fg_mask]*(1-self.geometry_denoise_rate)
            depth=depth_
            
        if self.depth2occ_intra:
            if self.depth_gt and not self.geometry_his_fusion:
                if not self.training:
                    kwargs['gt_depth']=kwargs['gt_depth'][0]
                gt_depth=self.get_downsampled_gt_depth(kwargs['gt_depth'])
                    
                b,n,d,h,w=depth.shape
                gt_depth=gt_depth.reshape(b,n,h,w,d).permute(0,1,4,2,3)
                fg_mask = torch.max(gt_depth, dim=2,keepdim=True).values > 0.0
                fg_mask=fg_mask.repeat(1,1,d,1,1)
                
                depth_=depth.clone()

                depth_[fg_mask]=gt_depth[fg_mask]
            else:
                depth_=depth
            occ_weight=cal_depth2occ(occ_weight,depth_.reshape(B*N,-1, H, W))
            occ_weight=occ_weight.reshape(B,N,-1, H, W)
            
        if self.geometry_his_fusion:
            if self.depth2occ_intra:
                depth=occ_weight
            depth_for_fuse=self.depth_head2(depth_feat)
            depth_for_fuse = depth_for_fuse.view(B, N, -1, H, W)

            if self.geometry_his_fusion_with_gt:
                if not self.training:
                    kwargs['gt_depth']=kwargs['gt_depth'][0]
                gt_depth=self.get_downsampled_gt_depth(kwargs['gt_depth'])
                mask=gt_depth.max(1)[0]<=0

                gt_depth[mask]=depth_for_fuse.permute(0,1,3,4,2).reshape(-1,depth_for_fuse.shape[2]).contiguous()[mask]
                gt_depth=gt_depth.reshape(depth.shape[0],depth.shape[1],depth.shape[3],depth.shape[4],depth.shape[2]).permute(0,1,4,2,3).contiguous()
                depth_fused=self.depth_his_fusion_post_layer.forward_post(depth_feat,depth_for_fuse,img_metas,stereo_metas,gt_depth)
                gt_depth=gt_depth.permute(0,1,3,4,2).contiguous()
                gt_depth=gt_depth.view(-1,gt_depth.shape[-1])
            else:
                depth_fused=self.depth_his_fusion_post_layer.forward_post(depth_feat,depth_for_fuse,img_metas,stereo_metas)
            
            depth_fused=depth_fused.softmax(2)

            if self.gate_net_with_his:
                gate_input=torch.cat((depth_fused.flatten(0,1),depth.flatten(0,1)),dim=1)
            else:
                gate_input=depth_feat
                
            
            fuse_weight=self.gate_net(gate_input)
            
            fuse_weight=fuse_weight.view(B, N, -1, H, W)

            if self.depth_gt:
                
                gt_depth[mask]=depth.permute(0,1,3,4,2).reshape(-1,depth.shape[2]).contiguous()[mask]
                gt_depth=gt_depth.reshape(depth.shape[0],depth.shape[1],depth.shape[3],depth.shape[4],depth.shape[2]).permute(0,1,4,2,3).contiguous()
                depth_fused=fuse_weight*depth_fused+(1-fuse_weight)*gt_depth
            else:
                depth_fused=fuse_weight*depth_fused+(1-fuse_weight)*depth
            output['geometry']=depth_fused

            if not self.geometry_his_fusion_after_sup:
                output['depth_pred']=depth_fused
        elif self.depth2occ_intra:
            if self.depth2occ_intra_post_norm:
                output['geometry']=occ_weight/occ_weight.sum(2,keepdim=True)
            else:
                output['geometry']=occ_weight
        else:
            output['geometry']=depth
        if self.soft_filling_with_offset:
            output['coor_offsets']=coor_offsets
        if not self.train_depth_only:
            if self.use_context_post_ln:
                context=self.context_post_ln(context.permute(0,1,3,4,2)).permute(0,1,4,2,3)

        output['context']=context
        return output


    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B, N, _, _ = rot.shape
        if bda is None:
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)
        
        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)
        
        if intrin.shape[-1] == 4:
            # for KITTI, the intrin matrix is 3x4
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                intrin[:, :, 0, 3],
                intrin[:, :, 1, 3],
                intrin[:, :, 2, 3],
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
                bda[:, :, 2, 2],
            ], dim=-1)
            
            if bda.shape[-1] == 4:
                mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
        else:
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
                bda[:, :, 2, 2],
            ], dim=-1)
        
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
        return mlp_input
    

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        downsample = self.downsample
        # if self.downsample == 8 and self.se_depth_map:
        #    downsample = 16 
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)
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
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:,
                                                                           1:]
        return gt_depths.float()
    def get_downsampled_gt_depth_(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        downsample = self.downsample

        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)
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
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = gt_depths.long().reshape(-1)-1
        return gt_depths
    def get_downsampled_gt_depth_semantics(self, gt_depths,gt_semantics):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        downsample = self.downsample
        # if self.downsample == 8 and self.se_depth_map:
        #    downsample = 16 
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   downsample, W // downsample,
                                   downsample, 1)
        gt_semantics=gt_semantics.view(B * N, H // self.downsample,
                                      self.downsample, W // self.downsample,
                                      self.downsample, 1)                           
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_semantics=gt_semantics.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, downsample * downsample)
        gt_semantics=gt_semantics.view(-1, self.downsample * self.downsample)
        gt_depths_tmp = torch.where(gt_depths == 0.0,
                                    1e5 * torch.ones_like(gt_depths),
                                    gt_depths)
        gt_depths,idx = torch.min(gt_depths_tmp, dim=-1)
        gt_semantics=gt_semantics[torch.arange(gt_semantics.shape[0]).to(gt_semantics.device).long(),idx]
        gt_depths = gt_depths.view(B * N, H // downsample,
                                   W // downsample)
        gt_semantics=gt_semantics.view(B , N, H // self.downsample,
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
        gt_depths = torch.where((gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
                                gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(
            gt_depths.long(), num_classes=self.depth_channels + 1).view(-1, self.depth_channels + 1)[:,
                                                                           1:]
        return gt_depths.float(),gt_semantics
   
    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4,
                                          2).contiguous().view(-1, self.depth_channels)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        
        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds,
                depth_labels,
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
        return dict(loss_depth=self.loss_depth_weight * depth_loss)
    
    @force_fp32()
    def get_depth_loss_(self, depth_labels, depth_preds):
        depth_labels = self.get_downsampled_gt_depth_(depth_labels)
        depth_preds = depth_preds.permute(0, 1, 3, 4,
                                          2).contiguous().view(-1, self.depth_channels)

        fg_mask = depth_labels > -0.5
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        

        # depth_loss =F.binary_cross_entropy(depth_preds,depth_labels,reduction='none',).sum() / max(1.0, fg_mask.sum())
        with autocast(enabled=False):
            depth_loss = F.nll_loss(
                (depth_preds+1e-8).log(),
                depth_labels,
            )
        return dict(loss_depth= depth_loss)




@HEADS.register_module()
class CM_ContextNet(nn.Module):
    """
        Camera parameters aware depth net
    """
    def __init__(self,
                 in_channels=512,
                 context_channels=64,
                 mid_channels=512,
                 cam_channels=27,
                 with_cp=False,
                 ):
        super(CM_ContextNet, self).__init__()
        self.with_cp = with_cp
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_channels = context_channels
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(cam_channels)
        self.context_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B, N, _, _ = rot.shape
        
        if bda is None:
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)
        
        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)
        
        if intrin.shape[-1] == 4:
            # for KITTI, the intrin matrix is 3x4
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                intrin[:, :, 0, 3],
                intrin[:, :, 1, 3],
                intrin[:, :, 2, 3],
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
                bda[:, :, 2, 2],
            ], dim=-1)
            
            if bda.shape[-1] == 4:
                mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
        else:
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
                bda[:, :, 2, 2],
            ], dim=-1)
        
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
        return mlp_input
    
    @force_fp32()
    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.reduce_conv, x)
        else:
            x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        if self.with_cp and x.requires_grad:
            context = cp.checkpoint(self.context_se, x, context_se)
        else:
            context = self.context_se(x, context_se)
        context = self.context_conv(context)
        context = context.view(B, N,  self.context_channels, H, W)
        return context


@HEADS.register_module()
class CM_ContextNet_View_Trans(nn.Module):
    """
        Camera parameters aware depth net
    """
    def __init__(self,
                 in_channels=512,
                 context_channels=64,
                 mid_channels=512,
                 cam_channels=27,
                 with_cp=False,
                 ):
        super(CM_ContextNet, self).__init__()
        self.with_cp = with_cp
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_channels = context_channels
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm1d(cam_channels)
        self.context_mlp = Mlp(cam_channels, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B, N, _, _ = rot.shape
        
        if bda is None:
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)
        
        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)
        
        if intrin.shape[-1] == 4:
            # for KITTI, the intrin matrix is 3x4
            mlp_input = torch.stack([
                intrin[:, :, 0, 0],
                intrin[:, :, 1, 1],
                intrin[:, :, 0, 2],
                intrin[:, :, 1, 2],
                intrin[:, :, 0, 3],
                intrin[:, :, 1, 3],
                intrin[:, :, 2, 3],
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
                bda[:, :, 2, 2],
            ], dim=-1)
            
            if bda.shape[-1] == 4:
                mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
        else:
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
                bda[:, :, 2, 2],
            ], dim=-1)
        
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
        return mlp_input
    
    @force_fp32()
    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(self.reduce_conv, x)
        else:
            x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        if self.with_cp and x.requires_grad:
            context = cp.checkpoint(self.context_se, x, context_se)
        else:
            context = self.context_se(x, context_se)
        context = self.context_conv(context)
        context = context.view(B, N,  self.context_channels, H, W)
        return context
