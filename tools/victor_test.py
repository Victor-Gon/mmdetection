# Author: victorg

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

# Set CUDA_VISIBLE_DEVICES to choose the device
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

# Import torch after setting CUDA_VISIBLE_DEVICES
import torch
print("Visible GPUs:", torch.cuda.device_count())
print("torch.cuda.current_device() ➜", torch.cuda.current_device())
print("Using GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Clear the GPU memory cache
torch.cuda.empty_cache()

# Get GPU memory information
free, total = torch.cuda.mem_get_info()
print(f"GPU free memory: {free/1024**2:.1f} MB / {total/1024**2:.1f} MB")

# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.evaluation import DumpDetResults
from mmdet.registry import RUNNERS
from mmdet.utils import setup_cache_size_limit_of_dynamo


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--work-dir', help='directory to save eval metrics')
    parser.add_argument('--out', type=str, help='dump predictions to a pkl file')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--show-dir', help='dir to save visualized results')
    parser.add_argument('--wait-time', type=float, default=2, help='interval of show (s)')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config options')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)

    # ADD THIS: nchannels param for non-RGB input
    parser.add_argument('--nchannels', type=int, default=None,
                        help='(Required for non-RGB) Number of input channels (e.g. 3 or 7)')

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Warn if nchannels not provided
    if args.nchannels is None:
        warnings.warn(
            'You did not specify --nchannels. Using the config\'s default in_channels; '
            'if your pipeline produces non-3-channel inputs (e.g. 7), please pass --nchannels accordingly.',
            UserWarning
        )

    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # override nchannels in the model backbone if provided
    if args.nchannels is not None:
        if 'model' in cfg and 'backbone' in cfg.model:
            cfg.model.backbone.in_channels = args.nchannels
        elif 'model' in cfg and 'detector' in cfg.model and 'backbone' in cfg.model.detector:
            cfg.model.detector.backbone.in_channels = args.nchannels
        else:
            warnings.warn('Could not find model.backbone or model.detector.backbone to set in_channels.')

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, setting it to default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, setting it to default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [dict(type='RandomFlip', prob=1.), dict(type='RandomFlip', prob=0.)],
                    [dict(
                        type='PackDetInputs',
                        meta_keys=('img_id', 'img_path', 'ori_shape',
                                   'img_shape', 'scale_factor', 'flip',
                                   'flip_direction'))],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline

    # build runner
    runner = Runner.from_cfg(cfg) if 'runner_type' not in cfg else RUNNERS.build(cfg)

    # dump results
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), 'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(DumpDetResults(out_file_path=args.out))

    # start testing
    runner.test()


if __name__ == '__main__':
    main()