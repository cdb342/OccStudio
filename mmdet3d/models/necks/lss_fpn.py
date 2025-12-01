# Copyright (c) Phigent Robotics. All rights reserved.

import torch
import torch.nn as nn

from mmdet.models import NECKS
import torch.utils.checkpoint as cp
from torch.utils.checkpoint import checkpoint
from mmdet3d.models.backbones.resnet import ConvModule
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
@NECKS.register_module()
class FPN_LSS(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 with_cp=False,
                 use_input_conv=False):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.with_cp = with_cp
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], \
                 feats[self.input_feature_index[1]]
        if self.with_cp:
            if self.lateral:
                x2 = cp.checkpoint(self.lateral_conv, x2)
            x1 = cp.checkpoint(self.up, x1)
            x = torch.cat([x2, x1], dim=1)
            if self.input_conv is not None:
                x = cp.checkpoint(self.input_conv, x)
            x = cp.checkpoint(self.conv, x)
            if self.extra_upsample:
                x = cp.checkpoint(self.up2, x)
        else:
            if self.lateral:
                x2 = self.lateral_conv(x2)
            x1 = self.up(x1)
            x = torch.cat([x2, x1], dim=1)
            if self.input_conv is not None:
                x = self.input_conv(x)
            x = self.conv(x)
            if self.extra_upsample:
                x = self.up2(x)
        return x

@NECKS.register_module()
class FPN_LSS2(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 with_cp=False,
                 use_input_conv=False):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.with_cp = with_cp
        self.up = nn.Upsample(
            scale_factor=8, mode='bilinear', align_corners=True)
        self.up22 = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.up33 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        
        x4,x3,x2, x1 = feats[self.input_feature_index[0]], \
                 feats[self.input_feature_index[1]], \
                 feats[self.input_feature_index[2]], \
                 feats[self.input_feature_index[3]]
        if self.with_cp:
            if self.lateral:
                x2 = cp.checkpoint(self.lateral_conv, x2)
            x1 = cp.checkpoint(self.up, x1)
            x2= cp.checkpoint(self.up22, x2)
            x3= cp.checkpoint(self.up33, x3)
            # x = torch.cat([x2, x1], dim=1)
            x = torch.cat([x4,x3,x2, x1], dim=1)
            if self.input_conv is not None:
                x = cp.checkpoint(self.input_conv, x)
            x = cp.checkpoint(self.conv, x)
            if self.extra_upsample:
                x = cp.checkpoint(self.up2, x)
        else:
            if self.lateral:
                x2 = self.lateral_conv(x2)
            x1 = self.up(x1)
            x2 = self.up22(x2)
            x3 = self.up33(x3)
            # x = torch.cat([x2, x1], dim=1)
            x = torch.cat([x4,x3,x2, x1], dim=1)
            if self.input_conv is not None:
                x = self.input_conv(x)
            x = self.conv(x)
            if self.extra_upsample:
                x = self.up2(x)
        return x

@NECKS.register_module()
class FPN_LSS3(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 with_cp=False,
                 use_input_conv=False):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.with_cp = with_cp
        self.up = nn.Upsample(
            scale_factor=4, mode='bilinear', align_corners=True)
        self.up22 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        # self.up33 = nn.Upsample(
        #     scale_factor=2, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        
        x3,x2, x1 = feats[self.input_feature_index[0]], \
                 feats[self.input_feature_index[1]], \
                 feats[self.input_feature_index[2]]
        if self.with_cp:
            if self.lateral:
                x2 = cp.checkpoint(self.lateral_conv, x2)
            x1 = cp.checkpoint(self.up, x1)
            x2= cp.checkpoint(self.up22, x2)
            # x3= cp.checkpoint(self.up33, x3)
            # x = torch.cat([x2, x1], dim=1)
            x = torch.cat([x3,x2, x1], dim=1)
            if self.input_conv is not None:
                x = cp.checkpoint(self.input_conv, x)
            x = cp.checkpoint(self.conv, x)
            if self.extra_upsample:
                x = cp.checkpoint(self.up2, x)
        else:
            if self.lateral:
                x2 = self.lateral_conv(x2)
            x1 = self.up(x1)
            x2 = self.up22(x2)
            # x3 = self.up33(x3)
            # x = torch.cat([x2, x1], dim=1)
            
            x = torch.cat([x3,x2, x1], dim=1)
            if self.input_conv is not None:
                x = self.input_conv(x)
            x = self.conv(x)
            if self.extra_upsample:
                x = self.up2(x)
        return x
@NECKS.register_module()
class LSSFPN3D(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 use_deblock=False):
        super().__init__()
        if isinstance(in_channels,list):
            in_channels_all=sum(in_channels)
        else:
            in_channels_all=in_channels
        if use_deblock:
            in_channels_all+=in_channels[0]//2
        self.up1 =  nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 =  nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)
        
        self.use_deblock=use_deblock
        if use_deblock:
            upsample_cfg=dict(type='deconv3d', bias=False)
            upsample_layer = build_conv_layer(
                    upsample_cfg,
                    in_channels=in_channels[0],
                    out_channels=in_channels[0]//2,
                    kernel_size=2,
                    stride=2,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(dict(type='BN3d', ), in_channels[0]//2)[1],
                                    nn.ReLU(inplace=True))

        self.conv = ConvModule(
            in_channels_all,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp

    def forward(self, feats):
        if self.use_deblock:
            x_16, x_32 = feats
            if self.with_cp:
                x_8 = checkpoint(self.deblock, x_16)
            else:
                x_8 = self.deblock(x_16)
        else:
            
            x_8, x_16, x_32 = feats
        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)
        x = torch.cat([x_8, x_16, x_32], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x
    
@NECKS.register_module()
class LSSFPN3D2(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 with_cp=False,
                 use_deblock=False):
        super().__init__()
        if isinstance(in_channels,list):
            in_channels_all=sum(in_channels)
        else:
            in_channels_all=in_channels
        if use_deblock:
            in_channels_all+=in_channels[0]//2
        self.up1 =  nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.up2 =  nn.Upsample(
            scale_factor=4, mode='trilinear', align_corners=True)
        
        self.up3 =  nn.Upsample(
            scale_factor=8, mode='trilinear', align_corners=True)

        self.conv = ConvModule(
            in_channels_all,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
            conv_cfg=dict(type='Conv3d'),
            norm_cfg=dict(type='BN3d', ),
            act_cfg=dict(type='ReLU',inplace=True))
        self.with_cp = with_cp
        self.use_deblock=use_deblock
        if use_deblock:
            upsample_cfg=dict(type='deconv3d', bias=False)
            upsample_layer = build_conv_layer(
                    upsample_cfg,
                    in_channels=in_channels[0],
                    out_channels=in_channels[0]//2,
                    kernel_size=2,
                    stride=2,
                    padding=0)

            self.deblock = nn.Sequential(upsample_layer,
                                    build_norm_layer(dict(type='BN3d', ), in_channels[0]//2)[1],
                                    nn.ReLU(inplace=True))

    def forward(self, feats):
        # import pdb;pdb.set_trace()
        if self.use_deblock:
            x_16, x_32,x_64 = feats
            if self.with_cp:
                x_8 = checkpoint(self.deblock, x_16)
            else:
                x_8 = self.deblock(x_16)
        else:
            x_8, x_16, x_32,x_64 = feats
        x_16 = self.up1(x_16)
        x_32 = self.up2(x_32)
        x_64 = self.up3(x_64)
        
        x = torch.cat([x_8, x_16, x_32,x_64], dim=1)
        if self.with_cp:
            x = checkpoint(self.conv, x)
        else:
            x = self.conv(x)
        return x
    
@NECKS.register_module()
class FPN_LSS_BEVDet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None,
                 use_input_conv=False):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(
            scale_factor=scale_factor, mode='bilinear', align_corners=True)
        # assert norm_cfg['type'] in ['BN', 'SyncBN']
        channels_factor = 2 if self.extra_upsample else 1
        self.input_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=1,
                padding=0,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        ) if use_input_conv else None
        if use_input_conv:
            in_channels = out_channels * channels_factor
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels * channels_factor,
                out_channels * channels_factor,
                kernel_size=3,
                padding=1,
                bias=False),
            build_norm_layer(
                norm_cfg, out_channels * channels_factor, postfix=0)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=extra_upsample,
                    mode='bilinear',
                    align_corners=True),
                nn.Conv2d(
                    out_channels * channels_factor,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    out_channels, out_channels, kernel_size=1, padding=0),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(
                    lateral, lateral, kernel_size=1, padding=0, bias=False),
                build_norm_layer(norm_cfg, lateral, postfix=0)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2, x1 = feats[self.input_feature_index[0]], \
                 feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        if self.input_conv is not None:
            x = self.input_conv(x)
        x = self.conv(x)
        if self.extra_upsample:
            x = self.up2(x)
        return x