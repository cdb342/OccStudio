# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

import mmdet
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.apis import multi_gpu_test, set_random_seed
from mmdet3d.apis.test import custom_multi_gpu_test
from mmdet.datasets import replace_ImageToTensor
import os.path as osp
import time
if mmdet.__version__ > '2.23.0':
    # If mmdet version > 2.23.0, setup_multi_processes would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import setup_multi_processes
else:
    from mmdet3d.utils import setup_multi_processes

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument(
        '--save',
        action='store_true',
        help='save occupancy_data')
    parser.add_argument(
        '--save_dir',
        type=str,
        default='cam_vis/',
        help='save occupancy_data')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where results will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--no-aavt',
        action='store_true',
        help='Do not align after view transformer.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi','cpu'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both specified, '
            '--options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    # assert args.out or args.eval or args.format_only or args.show \
    #     or args.show_dir, \
    #     ('Please specify at least one operation (save/eval/format/show the '
    #      'results / save the results) with the argument "--out", "--eval"'
    #      ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    cfg = compat_cfg(cfg)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    
    # each process may have different time
    out_dir =  osp.join('test', args.config.split('/')[-1][:-3], time.ctime().replace(' ','_').replace(':','_'))[:-8]

    if args.save:
        cfg.model.occupancy_save_path = out_dir
        mmcv.mkdir_or_exist(out_dir)
        mmcv.mkdir_or_exist(os.path.join(out_dir, 'occupancy_pred'))

    cfg.model.pretrained = None

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    elif args.launcher == 'cpu':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)


    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=1, dist=distributed, shuffle=False)

    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }

    # set random seeds
    if args.seed is not None:
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataloader

    dataset = build_dataset(cfg.data.test)
    
    
    
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    if not args.no_aavt:
        if '4D' in cfg.model.type:
            cfg.model.align_after_view_transfromation=True
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', revise_keys=[(r'^module\.', ''), (r'^teacher\.', '')])
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility


    sync_bn = cfg.get('sync_bn', False)
    if distributed and sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        print('Convert to SyncBatchNorm')

    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE
    from grad_cam import GradCAM,LayerCAM
    import cv2
    import numpy as np
    def imdenormalize(img, mean, std, to_bgr=True):
        assert img.dtype != np.uint8
        mean = mean.reshape(1, -1).astype(np.float64)
        std = std.reshape(1, -1).astype(np.float64)
        
        img = cv2.multiply(img, std)  # make a copy
        cv2.add(img, mean, img)  # inplace
        img_uint8 = np.rint(img).astype(np.uint8)
        if to_bgr:
            
            # img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR, img_uint8)  # inplace
        return img

    
    if args.launcher =='cpu':
        # 
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        model.eval()
        
        
        prog_bar = mmcv.ProgressBar(len(dataset))

        normalize_cfg=dict(
             mean=np.array([123.675, 116.28, 103.53]), std=np.array([58.395, 57.12, 57.375]), to_bgr=True)
        for i, data in enumerate(data_loader):
            # with torch.no_grad():
            
            # result = model(return_loss=False, rescale=True, **data)
            # target_layer = [model.module.img_backbone.layer4[2].conv3]
            target_layer = [model.module.depth_net.context_conv]

            # Initialize the Grad-CAM class
            grad_cam = LayerCAM(model, target_layer,use_cuda=True)

            # # Load your input tensor
            # input_tensor = torch.randn(1, 1, 32, 32, 32)

            # Get the Grad-CAM result

            vis_class=8
            if  (data['gt_occupancy'][0][0]==vis_class).sum()==0:
                
                continue
            result = grad_cam(targets=np.array([vis_class]),input_tensor=data['img_inputs'][0][0],img_args=dict(return_loss=False, rescale=True,return_raw_occ=True,vis_class=vis_class, **data))
            
            for i in range(result.shape[0]):
                
                result_i = np.uint8(result[i]*255)
                color_map = cv2.applyColorMap(result_i, cv2.COLORMAP_JET)
                
                save_dir=osp.join('cam_vis8',args.save_dir,str(vis_class), data['img_metas'][0].data[0][0]['scene_name'],data['img_metas'][0].data[0][0]['sample_idx'])
                if not osp.exists(save_dir):
                    mmcv.mkdir_or_exist(save_dir)
                # cv2.imwrite(osp.join(save_dir,f'grad_cam_{i}.jpg'), color_map)

                # print(data[img_metas][0].keys())
                
                # img=data['img_inputs'][0][0][0].reshape(6,3,*data['img_inputs'][0][0][0].shape[1:])[:6,0][i]#[3,256,704]
                img=data['img_inputs'][0][0][0].reshape(6,2,*data['img_inputs'][0][0][0].shape[1:])[:6,0][i]#[3,256,704]
                img = img.cpu().numpy().astype(np.float64).transpose(1, 2, 0)
                img=imdenormalize(img,**normalize_cfg)
                # cv2.imwrite(osp.join(save_dir,f'img_{i}.jpg'), img)

                alpha = 0.6  # 透明度，可以根据需要调整
                overlay = cv2.addWeighted(img.astype(np.uint8), alpha, color_map, 1 - alpha, 0)
                cv2.imwrite(osp.join(save_dir,f'grad_cam_{i}.jpg'), overlay)
            print(f'grad_cam_{i}.jpg saved')
            prog_bar.update()
            # except:
            #     prog_bar.update()
            # Save the Grad-CAM result
            # torch.save(result, "cam/grad_cam.jpg")
            
            
            
    if not distributed:
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        

        if cfg.get('use_custom_gpu_test', True):
            outputs = custom_multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect)
        else:
            outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                             args.gpu_collect)              

    rank, _ = get_dist_info()

    
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        kwargs['jsonfile_prefix'] = out_dir
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if True:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # kwargs['save'] =  args.save
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric=args.eval, **kwargs))
            print(dataset.evaluate(outputs, **eval_kwargs))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('fork')
    main()
