import torch
from mmdet3d.models.backbones.convnext import ConvNeXtBlock
def freeze_model(model, freeze_dict, logger=None):
    
    def freeze_submodel(submodel, subfreeze_dict, prefix=''):
        for key in subfreeze_dict:
            if isinstance(subfreeze_dict[key], dict):
                subprefix = f"{prefix}.{key}" if prefix else key
                freeze_submodel(getattr(submodel, key), subfreeze_dict[key], prefix=subprefix)
            elif isinstance(subfreeze_dict[key], list):
                for idx in subfreeze_dict[key]:
                    subprefix = f"{prefix}.{key}.{idx}" if prefix else f"{key}.{idx}"
                    freeze_parameters(getattr(submodel, key)[idx], logger=logger, prefix=subprefix)
            elif subfreeze_dict[key]:
                submodule = getattr(submodel, key)
                
                freeze_parameters(submodule, logger=logger, prefix=f"{prefix}.{key}")

    def freeze_parameters(module, logger=None, prefix=''):
        module.eval()  # Set the module to eval mode
        if isinstance(module, torch.nn.ModuleList):

            for idx, sub_module in enumerate(module):

                freeze_parameters(sub_module, logger=logger, prefix=f"{prefix}.{idx}")
        elif isinstance(module, ConvNeXtBlock):

            for name, sub_module in module.named_children():

                freeze_parameters(sub_module, logger=logger, prefix=f"{prefix}.{name}")
        elif isinstance(module, torch.nn.Module):

            if len(list(module.children())) > 0:
                for name, sub_module in module.named_children():
                    try:
                        if hasattr(sub_module,'gamma') :
                            sub_module.gamma.requires_grad = False
                            if logger:
                                logger.info(f'Freezed {prefix}.{name}.gamma and parameters')
                    except:
                        pass
                    freeze_parameters(sub_module, logger=logger, prefix=f"{prefix}.{name}")
            else:
                for name, param in module.named_parameters():
                
                    param.requires_grad = False
                    if logger:
                        logger.info(f'Freezed {prefix}.{name} parameter')
        elif isinstance(submodule, (torch.nn.modules.batchnorm._BatchNorm,
                                           torch.nn.modules.instancenorm._InstanceNorm,
                                           torch.nn.SyncBatchNorm,
                                           torch.nn.modules.normalization.GroupNorm,
                                           torch.nn.modules.normalization.LayerNorm)):
            submodule.eval()  # Set normalization layers to eval mode
            
            if hasattr(submodule, 'weight') and hasattr(submodule, 'bias'):
                submodule.weight.requires_grad = False  # Freeze weight parameter
                submodule.bias.requires_grad = False    # Freeze bias parameter
                if logger:
                    logger.info(f'Freezed {prefix}.{key} parameters')
        else:

            raise ValueError(f"Unsupported module type: {type(module)}")



    for key in freeze_dict:

        if isinstance(freeze_dict[key], dict):
            freeze_submodel(getattr(model, key), freeze_dict[key], prefix=key)
        elif freeze_dict[key]:
            submodule = getattr(model, key)
            freeze_parameters(submodule, logger=logger, prefix=key)




                
if __name__ == '__main__':
    # model = torch.nn.Sequential(
    #     torch.nn.Linear(1, 1),
    #     torch.nn.Linear(1, 1),
    #     torch.nn.Linear(1, 1)
    # )
    # print(model)
    # freeze_dict = {'0': True, '1': False, '2': True}
    # model=torch.load('ckpts/cascade_mask_rcnn_convnext_xlarge_22k_3x.pth')
    # freeze_dict = dict(
    # img_backbone=dict(stages=[0,1,2],downsample_layers=[0,1,2],norm2=True),
    # )
    # import pdb; pdb.set_trace()
    freeze_model(model, freeze_dict)
    import pdb; pdb.set_trace()