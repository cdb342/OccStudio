import torch.utils.checkpoint as checkpoint
from torch import nn

from mmcv.cnn.bricks.conv_module import ConvModule
from mmdet.models import BACKBONES
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmcv.cnn import build_conv_layer, build_norm_layer, build_plugin_layer
from mmcv.runner import BaseModule
import torch
import torch.nn.functional as F
import math
# CONV_LAYERS.register_module('ConvTranspose2d', module=nn.ConvTranspose2d)
    
@BACKBONES.register_module()
class NormConvolution(nn.Module):

    def __init__(
            self,
            numC_input,
            num_layer=[2, 2, 2],
            num_channels=None,
            stride=[2, 2, 2],
            backbone_output_ids=None,
            with_cp=False,
            norm_cfg=dict(type='BN3d', ),
            post_act=False,
            groups=8,
            kernel_size=3,
            use_act=False,
            softmax_act=False,
            transpose_conv=False,
            dilations = [1],
            use_tau=False,
            use_channel_conv=False,
            channel_conv_out_norm=False,
            learnable_weight=False,
            learnable_weight_dim=256,
            with_bias=False,
            norm_type=None,
    ):
        super(NormConvolution, self).__init__()
        # build backbone
        assert len(num_layer) == len(stride)
        num_channels = [numC_input*2**(i+1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) \
            if backbone_output_ids is None else backbone_output_ids
        layers = []
        curr_numC = numC_input
        
        Block=NormConvolutionLayer
 
        for i in range(len(num_layer)):
            layer = [
                Block(
                    curr_numC,
                    num_channels[i],
                    stride=stride[i],
                    post_act=post_act,
                    downsample=None,
                    groups=groups,
                    kernel_size=kernel_size,
                    use_act=use_act,
                    softmax_act=softmax_act,
                    transpose_conv=transpose_conv,
                    dilations =dilations,
            use_tau=use_tau,
            use_channel_conv=use_channel_conv,
            channel_conv_out_norm=channel_conv_out_norm,
            learnable_weight=learnable_weight,
            learnable_weight_dim=learnable_weight_dim,
            with_bias=with_bias,
            norm_type=norm_type,
            )
            ]
            curr_numC = num_channels[i]
            layer.extend([
                Block(curr_numC, curr_numC,groups=groups,kernel_size=kernel_size,use_act=use_act,softmax_act=softmax_act,
                      transpose_conv=transpose_conv,dilations =dilations,use_tau=use_tau,use_channel_conv=use_channel_conv,
                      channel_conv_out_norm=channel_conv_out_norm,
                      post_act=post_act,learnable_weight=learnable_weight,
                      learnable_weight_dim=learnable_weight_dim,with_bias=with_bias,norm_type=norm_type)
                for _ in range(num_layer[i] - 1)
            ])
            layers.append(nn.Sequential(*layer))
        self.layers = nn.Sequential(*layers)

        self.with_cp = with_cp
        self.learnable_weight=learnable_weight
    def forward(self, x):
        feats = []
        x_tmp = x
        for lid, layer in enumerate(self.layers):
            if self.with_cp:
                
                x_tmp = checkpoint.checkpoint(layer, x_tmp)
            else:
                x_tmp = layer(x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
            if self.learnable_weight:
                x_tmp=(x_tmp,x[1])
        return feats
    
class NormConvolutionLayer(nn.Module):
    def __init__(self,
                 channels_in, channels_out, stride=1, downsample=None,norm_cfg=dict(type='BN3d', ),post_act=False,groups=1,kernel_size=3,use_act=False,softmax_act=False,
                 transpose_conv=False,dilations = [1],use_tau=False,use_channel_conv=False,channel_conv_out_norm=False,
                 learnable_weight=False,learnable_weight_dim=256,with_bias=False,norm_type=None,):
        super().__init__()
        self.learnable_weight=learnable_weight
        self.softmax_act=softmax_act
        if learnable_weight:
            self.weight_net=nn.ModuleList()
            self.weight_net.append(nn.Sequential(
                    nn.Conv2d(learnable_weight_dim,
                            learnable_weight_dim*2,
                            kernel_size=3,
                            padding=1,
                            stride=1),
                    nn.SyncBatchNorm(learnable_weight_dim*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(learnable_weight_dim*2,
                            learnable_weight_dim*2,
                            kernel_size=3,
                            padding=1,
                            stride=1),
                    nn.SyncBatchNorm(learnable_weight_dim*2),
                    nn.ReLU(inplace=True),
                    
                    )
                )
        if learnable_weight:
            self.weight_head=nn.ModuleList()
            for i in range(len(dilations)):
                weight_head_i=nn.Sequential(
                        nn.Linear(learnable_weight_dim*2,
                                learnable_weight_dim*2),
                        nn.ReLU(inplace=True),
                        nn.Linear(learnable_weight_dim*2,
                                kernel_size**3*groups),
                        )
                
                init_weight=1.
                bias_i=torch.randn(groups,1,kernel_size,kernel_size,kernel_size).abs()*init_weight
                bias_i[:,:,kernel_size//2,kernel_size//2,kernel_size//2]=1.

                weight_head_i[2].bias.data=bias_i.flatten()
                self.weight_head.append( weight_head_i)
        else:
            if self.softmax_act:
                init_weight=1.
                weight=torch.randn(groups,1,kernel_size,kernel_size,kernel_size).abs()*init_weight
                weight[:,:,kernel_size//2,kernel_size//2,kernel_size//2]=1.
                
            else:
                weight=torch.ones(groups,1,kernel_size,kernel_size,kernel_size)*1e-8
                weight[:,:,kernel_size//2,kernel_size//2,kernel_size//2]=1.
            
            self.weight=nn.ParameterList()
            self.weight.append(nn.Parameter(weight))
            for i in range(len(dilations)-1):
                self.weight.append(nn.Parameter(weight.clone()))
        if with_bias:
            self.bias=nn.Parameter(torch.randn(channels_in,1,1,1)*0.02)
        self.tau=1.
        if use_tau:
            self.tau=nn.Parameter(torch.ones(1))    
        self.use_tau=use_tau
        
        self.groups=groups
        self.kernel_size = kernel_size
        self.dilations =dilations
        self.downsample = downsample
        self.use_act=use_act
        if use_act:
            act_type=nn.ELU(inplace=True)
            self.act=nn.ModuleList()
            
            for i in range(len(dilations)):
                self.act.append(act_type)
            if use_channel_conv and post_act:
                self.act.append(act_type)
        if norm_type is not None:
            self.norm=nn.ModuleList()
            for i in range(len(dilations)):
                self.norm.append(build_norm_layer(norm_type,channels_out)[1])

        self.transpose_conv=transpose_conv
        self.use_channel_conv=use_channel_conv
        
        if use_channel_conv:
            self.channel_conv_out_norm=channel_conv_out_norm
            if learnable_weight:
                self.weight_head_channel=nn.Sequential(
                            nn.Linear(learnable_weight_dim*2,
                                    learnable_weight_dim*2),
                            nn.ReLU(inplace=True),
                            nn.Linear(learnable_weight_dim*2,
                                    channels_out*channels_in),
                            )
            else:
                self.channel_weight=nn.Parameter(torch.randn(channels_out,channels_in,1,1,1)*0.02)
            if with_bias:
                self.channel_bias=nn.Parameter(torch.randn(channels_out,1,1,1)*0.02)
                
        self.post_act=post_act     
        self.channels_in=channels_in
        self.channels_out=channels_out
        self.with_bias=with_bias
        self.norm_type=norm_type
    def pred_weight(self,feat):
        b,n,c,h,w=feat.shape
        feat=feat.reshape(b*n,c,h,w)
        feat=self.weight_net[0](feat)
        feat=feat.reshape(b,n,-1,h,w)
        feat=feat.mean((1,3,4))

        return feat
        
    def forward(self, x):
        if self.learnable_weight:
            x,sup_feat=x
        
        if self.downsample is not None:
            
            identity = self.downsample(x)
        else:
            identity = x.clone()
        b, c, h, w, z = x.shape
        
        def prob_conv(x,weight,groups,dilation):
            
            b, c, h, w, z = x.shape
            if self.learnable_weight:

                if  isinstance(self.tau,float):
                    tau=self.tau
                else:
                    tau=F.relu(self.tau)+1e-8
                
                weight=weight/tau
                weight=weight.reshape(b*self.groups,-1).softmax(-1)
                weight=weight.view(b*groups, 1, self.kernel_size,self.kernel_size,self.kernel_size)
                
                x=x.view(b,c//groups,groups, h, w, z)
                x=x.permute(1,0,2,3,4,5)
                x=x.view(c//groups,b*groups,h,w,z)

                if self.transpose_conv:
                    output = F.conv_transpose3d(x, weight, padding=dilation*(self.kernel_size//2), groups=b*groups,dilation=dilation)  
                else:
                    output = F.conv3d(x, weight, padding=dilation*(self.kernel_size//2), groups=b*groups,dilation=dilation) 

                output=output.view(c//groups,b,groups,h,w,z)
                output=output.permute(1,0,2,3,4,5)
                output=output.view(b,c,h,w,z)
            else:
                if self.softmax_act:
                    if  isinstance(self.tau,float):
                        tau=self.tau
                    else:
                        tau=F.relu(self.tau)+1e-8
                    
                    weight=weight/tau
                    weight=weight.reshape(self.groups,-1).softmax(-1)
                    weight=weight.view(groups, 1, self.kernel_size,self.kernel_size,self.kernel_size)       
                else:
                    weight = torch.abs(weight)  
                    weight_copy=weight
                    # weight=weight/(weight.sum(-1,keepdim=True)+1e-8)
                    weight_sum=(weight.sum(dim=(2, 3,4), keepdim=True)).detach()

                    weight=weight / (weight_sum+1e-8)
                    weight=weight.detach()-weight_copy.detach()+weight_copy

                x=x.view(b*c//groups,groups, h, w, z)

                if self.transpose_conv:
                    output = F.conv_transpose3d(x, weight, padding=dilation*(self.kernel_size//2), groups=groups,dilation=dilation)  # 分组卷积
                else:
                    output = F.conv3d(x, weight, padding=dilation*(self.kernel_size//2), groups=groups,dilation=dilation)  # 分组卷积

                output=output.view(b,c, h, w, z)
                
            return output
        
        def prob_conv_channel(x,weight):
            
            b, c, h, w, z = x.shape
            if self.learnable_weight: 
                
                weight=weight.view(b,self.channels_out,self.channels_in)
                
                if  isinstance(self.tau,float):
                    tau=self.tau
                else:
                    tau=F.relu(self.tau)+1e-8
                
                weight=weight/tau
                if self.channel_conv_out_norm:
                    weight=weight.softmax(1)
                else:
                    weight=weight.softmax(2) 

                output=torch.einsum('bchwz,bkc->bkhwz',x,weight).contiguous()
            else:
                if self.softmax_act:
                    if  isinstance(self.tau,float):
                        tau=self.tau
                    else:
                        tau=F.relu(self.tau)+1e-8
                    
                    weight=weight/tau
                    if self.channel_conv_out_norm:
                        weight=weight.softmax(0)
                    else:
                        weight=weight.softmax(1)   
                # b, c, h, w, z = x.shape
                output = F.conv3d(x, weight)
            return output
        
        if self.learnable_weight:
            weight_feat=self.pred_weight(sup_feat)
            weight_channel=self.weight_head_channel(weight_feat)
        if not self.use_channel_conv:
            out=identity
            for i in range(len(self.dilations)):
                if self.learnable_weight:
                    weight=self.weight_head[i](weight_feat)
                else:
                    weight=self.weight[i]
                xi=prob_conv(x,weight,self.groups,self.dilations[i])
                if self.norm_type is not None:
                    xi=self.norm[i](xi)
                if self.use_act:
                    xi=self.act[i](xi)
                out=out+xi
            x=out/(len(self.dilations)+1)
            if self.with_bias:
                x=x+self.bias
            x = x.view(b, c//self.groups,self.groups, h, w, z)
            x=x.permute(0,2,1,3,4,5).contiguous()
            x=x.view(b,c, h, w, z)
        else:
            
            out=0
            for i in range(len(self.dilations)):
                if self.learnable_weight:
                    weight=self.weight_head[i](weight_feat)
                else:
                    weight=self.weight[i]
                xi=prob_conv(x,weight,self.groups,self.dilations[i])
                if self.norm_type is not None:
                    xi=self.norm[i](xi)
                if self.use_act:
                    xi=self.act[i](xi)
                out=out+xi
            x=out/(len(self.dilations))
            if self.with_bias:
                x=x+self.bias
            if self.learnable_weight:
                channel_weight=weight_channel
            else:
                channel_weight=self.channel_weight
            x=prob_conv_channel(x,channel_weight)
            if self.with_bias:
                x=x+self.channel_bias
            x=(x+identity)/2

            if self.post_act:
                if self.use_act:
                    x=self.act[-1](x)
                    
            
        return x
