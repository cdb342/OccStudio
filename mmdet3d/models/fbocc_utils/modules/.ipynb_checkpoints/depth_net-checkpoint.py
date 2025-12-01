# Copyright (c) 2022-2023, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/FB-BEV/blob/main/LICENSE

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
from mmcv.cnn import xavier_init, constant_init

def convert_color(img_path):
    plt.figure()
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.imsave(img_path, img, cmap=plt.get_cmap('viridis'))
    plt.close()


def save_tensor(tensor, path, pad_value=254.0,normalize=False):
    print('save_tensor', path)
    tensor = tensor.to(torch.float).detach().cpu()
    max_ = tensor.flatten(1).max(-1).values[:, None, None]
    min_ = tensor.flatten(1).min(-1).values[:, None, None]
    tensor = (tensor-min_)/(max_-min_)
    if tensor.type() == 'torch.BoolTensor':
        tensor = tensor*255
    if len(tensor.shape) == 3:
        tensor = tensor.unsqueeze(1)
    tensor = make_grid(tensor, pad_value=pad_value, normalize=normalize).permute(1, 2, 0).numpy().copy()
    torchvision.utils.save_image(torch.tensor(tensor).permute(2, 0, 1), path)
    convert_color(path)


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
        ####################
        # self.zz=nn.Conv3d(24,12,1)
        # self.zz=nn.Conv3d(12,6,1)
        # self.zz=nn.Conv2d(24,12,1)
        ####################
    
    @force_fp32()
    def forward(self, x, x_se):
        # import pdb;pdb.set_trace()
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        # import pdb;pdb.set_trace()
        return x * self.gate(x_se)
        # return x * self.gate(x_se).reshape(2,12,-1,1,1).contiguous().sum(0)
        # return self.zz(x * self.gate(x_se).unsqueeze(0).contiguous()).contiguous().squeeze(0).contiguous()
        # import pdb;pdb.set_trace()
        # return x[:12,...].contiguous() * self.gate(x_se.contiguous())[:12,...]
        # return self.zz((x.contiguous() * self.gate(x_se.contiguous()).contiguous()).permute(1,0,2,3)).contiguous().permute(1,0,2,3).contiguous()
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
                 #############
                 stereo=False,
                 aspp_mid_channels=512,
                 depth2occ=False,
                 length=44,
                 train_depth_only=False,
                 depth2occ_v2=False,
                 depth2occ_v3=False,
                 depth2occ_v3_sup_sem_only=False,
                 num_cls=17,
                 depth2occ_with_prototype=False,
                 prototype_channels=64,
                 sup_sem_only=False,
                 direct_learn_hideen=False,
                 depth2occ_composite=False,
                 deform_lift_with_offset=False,
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

        
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        if not train_depth_only:
            self.context_mlp = Mlp(27, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
            self.context_conv = nn.Conv2d(
                        mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        depth_conv_input_channels = mid_channels
        downsample_net = None
        #################
        self.depth2occ=depth2occ
        self.length=length
        
        self.depth_channels = depth_channels
        self.train_depth_only=train_depth_only
        self.depth2occ_v2=depth2occ_v2
        self.depth2occ_v3=depth2occ_v3
        self.depth2occ_v3_sup_sem_only=depth2occ_v3_sup_sem_only
        self.depth2occ_with_prototype=depth2occ_with_prototype
        self.sup_sem_only=sup_sem_only
        self.direct_learn_hideen=direct_learn_hideen
        self.depth2occ_composite=depth2occ_composite
        if depth2occ_composite:
            self.length_group=torch.tril(torch.ones(length, length)) 
        self.deform_lift_with_offset=deform_lift_with_offset
        
        # import pdb;pdb.set_trace()
        #####################
        if stereo:
            cost_volumn_channels=depth_channels
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
        # if self.depth2occ:
        #     depth_channels+=length
        # import pdb;pdb.set_trace()
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
        depth_conv_list.append(
            nn.Conv2d(
                mid_channels,
                depth_channels,
                kernel_size=1,
                stride=1,
                padding=0))
        self.depth_conv = nn.Sequential(*depth_conv_list)

        ##########################################
        if self.deform_lift_with_offset:
            self.offset_conv = nn.Conv2d(
                context_channels,
                depth_channels*3,
                kernel_size=1,
                stride=1,
                padding=0)
            constant_init(self.offset_conv, 0)
        if self.depth2occ_with_prototype:
            if not self.sup_sem_only:
                self.hidden_length_predictor = nn.Sequential(
                    nn.Linear(prototype_channels, prototype_channels*2),
                    nn.ReLU(inplace=True),
                    nn.Linear(prototype_channels*2, length),
                    nn.Sigmoid(),
                )
            
            self.context_remap = nn.Sequential(
                nn.Conv2d(
                        context_channels, context_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                        context_channels, prototype_channels, kernel_size=1, stride=1, padding=0),
            )


        if self.depth2occ_v3:
            occ_channels=self.length
            self.cls_predictor = nn.Sequential(
                nn.Conv2d(
                        context_channels, context_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                        context_channels, num_cls, kernel_size=1, stride=1, padding=0,bias=False),
            )
            if not self.depth2occ_v3_sup_sem_only:
                self.occ_conv=nn.Sequential(
                    nn.Conv2d(
                        num_cls,occ_channels, kernel_size=1, stride=1, padding=0,bias=False),
                        nn.Sigmoid(),
                )
                constant_init(self.occ_conv, 0)
            
        if self.depth2occ_v2:
            occ_channels=self.length
            self.occ_conv = nn.Sequential(
                nn.Conv2d(
                        mid_channels, mid_channels, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                        mid_channels, occ_channels, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )
        if self.depth2occ:
            occ_channels=self.length
            occ_conv_list = [BasicBlock(depth_conv_input_channels, mid_channels,
                                        downsample=downsample_net),
                            BasicBlock(mid_channels, mid_channels),
                            BasicBlock(mid_channels, mid_channels)]
            # if use_aspp:
            #     occ_conv_list.append(ASPP(mid_channels, aspp_mid_channels))
            # if use_dcn:
            #     occ_conv_list.append(
            #         build_conv_layer(
            #             cfg=dict(
            #                 type='DCN',
            #                 in_channels=mid_channels,
            #                 out_channels=mid_channels,
            #                 kernel_size=3,
            #                 padding=1,
            #                 groups=4,
            #                 im2col_step=128,
            #             )))

            occ_conv_list.append(
                nn.Conv2d(
                    mid_channels,
                    occ_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0))
            self.occ_conv = nn.Sequential(*occ_conv_list)
        # self.occ_conv=nn.Conv2d(
        #     depth_conv_input_channels,
        #     self.length,
        #     kernel_size=1,
        #     stride=1,
        #     padding=0)
        

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
        # import pdb;pdb.set_trace()

        # D x H x W x 3
        return torch.stack((x, y, d), -1)
    def gen_grid(self, metas, B, N, D, H, W, hi, wi):
        # pass
        frustum =self.cv_frustum.to(metas['post_trans'].device)
        # import pdb;pdb.set_trace()

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
        prev, curr = metas['cv_feat_list']
        # import pdb;pdb.set_trace()
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
  
    @force_fp32()
    def forward(self, x, mlp_input,stereo_metas=None,maskformerocc_head=None):

        # if not  x.requires_grad: 
        x = x.to(torch.float32) # FIX distill type error
        # import pdb;pdb.set_trace()
        ###########
        
        #################
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        #############
        
        ##############
        # x=x[:12,...].contiguous()
        # mlp_input=mlp_input[:12,...].contiguous()
        #############################
        # import pdb;pdb.set_trace()
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
        if self.deform_lift_with_offset:
            coor_offsets=self.offset_conv(context)
            ###################
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        # depth_se=None
        depth_ = self.depth_se(x, depth_se)
        # import pdb;pdb.set_trace()
        if not stereo_metas is None:
            # import pdb;pdb.set_trace()
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
                    # self.calculate_cost_volumn(stereo_metas)
            # import pdb;pdb.set_trace()
            cost_volumn = self.cost_volumn_net(cost_volumn)
            # import pdb;pdb.set_trace()
            depth_ = torch.cat([depth_, cost_volumn], dim=1)

        
        
        
        # import pdb;pdb.set_trace()
        if self.depth2occ:
            # if self.with_cp and depth_.requires_grad:
            #     occ = cp.checkpoint(self.occ_conv, depth_)
            # else:
            depth = self.depth_conv(depth_)
            occ = self.occ_conv(depth_.clone())
            # occ=occ.view(B, N, self.length, H, W)
            if self.depth2occ_composite:
                # import pdb;pdb.set_trace()
                length_weight=occ.softmax(dim=1)
                occ=torch.einsum('bchw,cc->bchw',length_weight,self.length_group.to(occ.device))
                occ=[length_weight,occ]
            else:
                occ=occ.sigmoid()
        elif self.depth2occ_v2:
            
            occ = self.occ_conv(context_)
            # occ=occ.view(B, N, self.length, H, W)
            if self.with_cp and depth_.requires_grad:
                depth = cp.checkpoint(self.depth_conv, depth_)
            else:
                depth = self.depth_conv(depth_)
        elif self.depth2occ_v3:
            predict_cls=self.cls_predictor(context)
            if not self.depth2occ_v3_sup_sem_only:
                occ=self.occ_conv(F.softmax(predict_cls,dim=1))
                occ=[predict_cls,occ]
            else:
                occ=[predict_cls,None]
            # import pdb;pdb.set_trace()
            if self.with_cp and depth_.requires_grad:
                depth = cp.checkpoint(self.depth_conv, depth_)
            else:
                depth = self.depth_conv(depth_)
            # import pdb;pdb.set_trace()
        elif self.depth2occ_with_prototype:
            # import pdb;pdb.set_trace()
            if not self.sup_sem_only:
                hidden_length = self.hidden_length_predictor(maskformerocc_head.query_feat.weight)
            else:
                hidden_length=None     
            context_remap = self.context_remap(context)
            context_remap=context_remap.reshape(B,N,*context_remap.shape[1:]).permute(0,2,1,3,4)
            occ=[hidden_length,context_remap]

            

            if self.with_cp and depth_.requires_grad:
                depth = cp.checkpoint(self.depth_conv, depth_)
            else:
                depth = self.depth_conv(depth_)
        else:
            if self.with_cp and depth_.requires_grad:
                depth = cp.checkpoint(self.depth_conv, depth_)
            else:
                depth = self.depth_conv(depth_)
        # import pdb;pdb.set_trace()
        if not self.train_depth_only:
            context = context.view(B, N,  self.context_channels, H, W)
        depth = depth.view(B, N, -1, H, W)
        # if self.depth2occ:
        #     occ=depth[:,:,:self.length,...]
        #     depth=depth[:,:,self.length:,...]
        #     occ=occ.sigmoid()
        
        if self.direct_learn_hideen:
            depth=depth.sigmoid()
        else:
            depth = depth.softmax(dim=2)
        ####################
        # context = context.view(B, N//2,  self.context_channels, H, W).contiguous()
        # depth = depth.view(B, N//2, self.depth_channels, H, W).contiguous()
        if self.deform_lift_with_offset:
            depth=[coor_offsets,depth]
#########################

        if self.depth2occ or self.depth2occ_v2 or self.depth2occ_v3 or self.depth2occ_with_prototype:
            return context, depth, occ
        else:
            return context, depth


    # def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
    #     rot_=rot[:,:6,...].contiguous()
    #     tran_=tran[:,:6,...].contiguous()
    #     intrin_=intrin[:,:6,...].contiguous()
    #     post_rot_=post_rot[:,:6,...].contiguous()
    #     post_tran_=post_tran[:,:6,...].contiguous()
        
    #     B, N, _, _ = rot_.shape
    #     bda = bda.view(B, 1, 3, 3).repeat(1, N, 1, 1)
    #     mlp_input = torch.stack([
    #         intrin_[:, :, 0, 0],
    #         intrin_[:, :, 1, 1],
    #         intrin_[:, :, 0, 2],
    #         intrin_[:, :, 1, 2],
    #         post_rot_[:, :, 0, 0],
    #         post_rot_[:, :, 0, 1],
    #         post_tran_[:, :, 0],
    #         post_rot_[:, :, 1, 0],
    #         post_rot_[:, :, 1, 1],
    #         post_tran_[:, :, 1],
    #         bda[:, :, 0, 0],
    #         bda[:, :, 0, 1],
    #         bda[:, :, 1, 0],
    #         bda[:, :, 1, 1],
    #         bda[:, :, 2, 2],
    #     ],
    #                             dim=-1)
    #     sensor2ego = torch.cat([rot_, tran_.reshape(B, N, 3, 1)],
    #                            dim=-1).reshape(B, N, -1)
    #     mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
    #     # mlp_input=mlp_input.contiguous()[:,:6,...].contiguous()
    #     return mlp_input
    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda):
        # rot=rot[:,:6,...].contiguous()
        # tran=tran[:,:6,...].contiguous()
        # intrin=intrin[:,:6,...].contiguous()
        # post_rot=post_rot[:,:6,...].contiguous()
        # post_tran=post_tran[:,:6,...].contiguous()
        
        B, N, _, _ = rot.shape
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
            bda[:, :, 2, 2],
        ],
                                dim=-1)
        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)],
                               dim=-1).reshape(B, N, -1)
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)
        
        # import pdb;pdb.set_trace()
        # mlp_input_=mlp_input.contiguous()[:,:6,...].contiguous()
        # mlp_input_=torch.split(mlp_input.reshape(2,2,6,27), 1, 1)[0].squeeze(1)
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
        gt_semantics=gt_semantics.view(B * N, H // self.downsample,
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
        # import pdb;pdb.set_trace()

        # depth_loss =F.binary_cross_entropy(depth_preds,depth_labels,reduction='none',).sum() / max(1.0, fg_mask.sum())
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
        # import pdb;pdb.set_trace()

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
        self.bn = nn.BatchNorm1d(27)
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

    
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
