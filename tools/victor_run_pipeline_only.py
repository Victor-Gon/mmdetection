# Author: victorg

import sys
import os
import warnings
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(__file__, '..', '..')))

os.environ['CUDA_VISIBLE_DEVICES'] = '2'

import torch
print("Visible GPUs:", torch.cuda.device_count())
print("torch.cuda.current_device() ➜", torch.cuda.current_device())
print("Using GPU:", torch.cuda.get_device_name(0))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.cuda.empty_cache()
free, total = torch.cuda.mem_get_info()
print(f"GPU free memory: {free/1024**2:.1f} MB / {total/1024**2:.1f} MB")

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mmdet.registry import DATASETS
from mmengine.registry import init_default_scope
from mmdet.utils import setup_cache_size_limit_of_dynamo


def parse_args():
    parser = argparse.ArgumentParser(description='Run pipeline only (no model)')
    parser.add_argument('config', help='Config file path')
    parser.add_argument('--n', type=int, default=10, help='Number of images to process')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    return parser.parse_args()


def main():
    args = parse_args()
    setup_cache_size_limit_of_dynamo()

    # Load config
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Remove model-related configs
    for key in ['model', 'train_cfg', 'val_cfg', 'test_cfg',
                'optim_wrapper', 'param_scheduler', 'custom_hooks',
                'default_hooks', 'log_processor', 'val_evaluator',
                'test_evaluator', 'train_dataloader']:
        cfg.pop(key, None)

    # Use test_dataloader to drive the pipeline
    if 'test_dataloader' not in cfg:
        raise ValueError('Config must contain test_dataloader to run this script.')

    init_default_scope('mmdet')
    dataset = DATASETS.build(cfg.test_dataloader['dataset'])
    dataloader = Runner.build_dataloader(cfg.test_dataloader)

    print(f'[INFO] Starting to iterate and transform {args.n} samples...')

    for i, data in enumerate(dataloader):
        if i >= args.n:
            print(f'[INFO] Processed {args.n} images. You can now stop the script.')
            break

    print('[INFO] Done.')


if __name__ == '__main__':
    main()