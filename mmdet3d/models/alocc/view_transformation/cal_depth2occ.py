import torch
import torch.nn.functional as F
import time


def cal_depth2occ(occ_weight,depth):
    # depth: [BN,d,H,W]
    # occ_weight: [BN,len_occ,H,W]
    # return: [BN,d,H,W]
    bn,d,h,w=depth.shape
    len_occ=occ_weight.shape[1]
    occ_weight=occ_weight.permute(0,2,3,1).reshape(bn*h*w,len_occ)

    depth_2_occ=torch.cat([torch.zeros_like(occ_weight[:,:1]).to(depth.device),occ_weight,torch.zeros(occ_weight.shape[0],d-len_occ).to(depth.device)],dim=1).repeat(1,d).reshape(bn*h*w,d+1,d)[:,:d,:]
    depth_2_occ=depth_2_occ.triu()+torch.eye(d,d).to(depth.device).unsqueeze(0)

    occ=torch.matmul(depth.permute(0,2,3,1).reshape(bn*h*w,1,d),depth_2_occ).reshape(bn,h,w,d).permute(0,3,1,2)#bn,d,h,w

    return occ
