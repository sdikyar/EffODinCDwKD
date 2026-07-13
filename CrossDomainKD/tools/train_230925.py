# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
import time
import warnings
import glob

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
import torch.distributed as dist

import sys
sys.path.append("/root/datasets/byf/code/MGD-I/")
from mmdet import __version__
from mmdet.apis import init_random_seed, set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.distillation import build_distiller


def main(args):
    bbox_mAP = 0

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
   
   # --- START OF THE UPDATE ---
    # Create a unique subdirectory for each parallel trial to avoid conflicts.
    # This uses the Slurm process ID (rank) to make a unique folder name.
    if os.environ.get('SLURM_PROCID'):
        rank = os.environ.get('SLURM_PROCID')
        cfg.work_dir = osp.join(cfg.work_dir, f'trial_rank_{rank}')
    # --- END OF THE UPDATE ---


    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
 
    # --- SAFER OPTION FOR OPTUNA ---
    # Even if launched with Slurm, we want each Optuna trial to run as a
    # completely independent single-GPU job. So we disable distributed init.
    if 'SLURM_PROCID' in os.environ:
        distributed = False
        # pick local rank if provided, otherwise default to 0
        cfg.gpu_ids = [int(os.environ.get('LOCAL_RANK', 0))]
    else:
        distributed = False
        cfg.gpu_ids = [0]

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    seed = init_random_seed(args.seed)
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    distiller_cfg = cfg.get('distiller',None)
    if distiller_cfg is None:
        model = build_detector(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()
    else:
        teacher_cfg = Config.fromfile(cfg.teacher_cfg)
        student_cfg = Config.fromfile(cfg.student_cfg)
        
        model = build_distiller(cfg.distiller,teacher_cfg,student_cfg,
         train_cfg=student_cfg.get('train_cfg'), 
         test_cfg=student_cfg.get('test_cfg'))

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    
    # --- START OF THE FIX ---
    # If we are NOT in a true distributed job, we must ensure the single-GPU
    # evaluation hook is used to prevent errors.
    # If we are NOT in a true distributed job (which is the case for each
    # independent Optuna trial), we must ensure the single-GPU evaluation hook is used.
    if not distributed:
        # Check the main evaluation config
        if cfg.get('evaluation') and cfg.evaluation.get('type') is None:
            cfg.evaluation.type = 'EvalHook'
        # Also check for custom hooks that might be forcing distributed evaluation
        if 'custom_hooks' in cfg:
            for i, hook in enumerate(cfg.custom_hooks):
                if hook.get('type') == 'DistEvalHook':
                    cfg.custom_hooks[i]['type'] = 'EvalHook'
    # --- END OF THE FIX ---


    # This call runs the training and validation
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)

   # --- START OF THE MODIFICATION ---
    # Each of the 8 processes is an independent trial and should not communicate.
    # This code finds the log file in the process's own unique work_dir and parses it.

    try:
        # The unique work_dir (e.g., '.../trial_rank_0') was set earlier in main().
        # Find the latest log file in that unique directory.
        log_files = glob.glob(osp.join(cfg.work_dir, '*.log'))
        if not log_files:
            print(f"Warning: No log file found in {cfg.work_dir}")
            return 0.0

        latest_log = max(log_files, key=os.path.getctime)
        
        bbox_mAP = 0.0
        with open(latest_log, 'r') as file:
            # Read all lines and search from the end to find the result quickly.
            for line in reversed(file.readlines()):
                if 'bbox_mAP: ' in line:
                    start = line.find('bbox_mAP: ') + len('bbox_mAP: ')
                    end = line.find(",", start) # Find the comma after the mAP value
                    bbox_mAP = float(line[start:end])
                    break # Stop after finding the first valid mAP line

        # Return the value found in this process's own log file.
        return bbox_mAP

    except Exception as e:
        # If any error occurs (e.g., parsing error), log it and return 0.0
        # so this single failed trial doesn't stop the entire Optuna study.
        print(f"Error parsing log file in {cfg.work_dir}: {e}")
        return 0.0
    
    # --- END OF THE MODIFICATION ---


if __name__ == '__main__':
    from opt import parse_args
    main(parse_args())
