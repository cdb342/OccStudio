import torch
import torch.nn.functional as F
from itertools import product
import time

z=list(product([0,1],[0,1], [0,1]))#[8,3]
idx=(torch.tensor([0,1,2]).unsqueeze(0).repeat(len(z),1),torch.tensor(z))
def broadcast_pred(pred,coord):
    
    coord1=torch.floor(coord)
    dist1=coord-coord1
    coord2=coord1+1
    dist2=coord2-coord
    coord_=torch.stack([coord1,coord2],dim=-1)#[92800,3,2]
    dist_=torch.stack([dist1,dist2],dim=-1)
    coord_all=coord_[:,idx[0],idx[1]]#[92800,8,3]
    dist_all=dist_[:,idx[0],idx[1]]#[92800,8,3]

    weight=dist_all.norm(dim=-1)#[92800,8]

    pred_=torch.einsum('qnm,qmc->qnc',weight.unsqueeze(-1),pred.unsqueeze(1))#[92800,8,18]
    pred_voxel=torch.sparse_coo_tensor(coord_all.reshape(-1,3).t(),pred_.reshape(-1,pred_.shape[-1]),(200,200,16,18)).to_dense()
    
    return pred_voxel

def broadcast_pred_linear_interpolation(coord,idx=idx,weight_flip=True):
    """
    For each input coordinate, this function calculates the coordinates of the 8
    surrounding voxels and the corresponding trilinear interpolation weights.

    Args:
        coord (torch.Tensor): A tensor of shape (N, 3) representing N
            3D coordinates.
        idx (tuple, optional): A tuple containing two tensors that define the
            indices for the 8 corners of a voxel. Defaults to a pre-defined idx.
        weight_flip (bool, optional): If True, the calculated weights are
            flipped. Defaults to True.

    Returns:
        tuple:
            - torch.Tensor: The calculated trilinear interpolation weights of
              shape (N, 8).
            - torch.Tensor: The coordinates of the 8 surrounding voxels of
              shape (N, 8, 3).
    """
    with torch.no_grad():
        coord1=torch.floor(coord)
        coord2=coord1+1
    dist1=coord-coord1
    
    dist2=coord2-coord
    coord_=torch.stack([coord1,coord2],dim=-1)#[92800,3,2]
    
    dist_=torch.stack([dist1,dist2],dim=-1)

    idx=(idx[0].to(coord.device),idx[1].to(coord.device))
    coord_all=coord_[:,idx[0],idx[1]]#[92800,8,3]
    dist_all=dist_[:,idx[0],idx[1]]#[92800,8,3]
    
    # weight=torch.prod(dist_all,dim=-1)#[92800,8]
    weight=dist_all[...,0]*dist_all[...,1]*dist_all[...,2]
    if weight_flip:
        
        weight=torch.flip(weight,dims=[-1])

    return weight,coord_all
# if __name__=='__main__':
#     coord=torch.randn(100,3).cuda().abs()
#     pred=torch.randn(100,18).cuda()
#     start=time.time()
#     pred_voxel=broadcast_pred_linear_interpolation(coord)
#     print(pred_voxel.shape)
#     end=time.time()
#     print(f'总耗时:{end-start}秒')
    