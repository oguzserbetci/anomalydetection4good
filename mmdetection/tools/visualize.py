import argparse
import os.path as osp

import mmcv
import torch
import numpy as np
from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel
from mmcv.parallel import DataContainer as DC

from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('img_dir', help='path to directory with images to process')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out_dir', default='visualizations/', help='output directory for images')
    parser.add_argument(
        '--corruption',
        type=str,
        default=None,
        choices=[None, 'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                 'speckle_noise', 'gaussian_blur', 'spatter', 'saturate'],
        help='corruption')
    parser.add_argument(
        '--severity',
        type=int,
        default=5,
        help='corruption severity level')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    print('load config ...')
    # load config
    cfg = mmcv.Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    print('load data ...')
    
    if cfg.data.test.type == 'CocoDataset':
        dataset_type = 'coco'
    elif cfg.data.test.type == 'VOCDataset':
        dataset_type = 'voc'
    elif cfg.data.test.type == 'CityscapesDataset':
        dataset_type = 'cityscapes'
    elif cfg.data.test.type == 'PipistrelDataset':
        dataset_type = 'pipistrel'
    else:
        print("unknown dataset type")
        
        
    
    # get data
    data_cfg = cfg.data.test
    data_cfg.type = 'ImageFolder'
    data_cfg.img_prefix = args.img_dir
    data_cfg.corruption = args.corruption
    data_cfg.corruption_severity = args.severity
    dataset = build_dataset(data_cfg)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    print('load model ...')
    
    # load model
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[0])

    # define image preprocessing 
    # img_transform = ImageTransform(size_divisor=cfg.data.test.size_divisor, **cfg.img_norm_cfg)

    # load image
    # img = mmcv.imread(image)
    # corrupt image
    # if cfg.data.test.corruption is not None:
    #     img = corrupt(img, severity=cfg.data.test.corruption_severity, corruption_name=cfg.data.test.corruption)
    # prepare image for model
    # imgs = []
    # img_metas = []
    # _img, _img_meta = prepare_single(img, cfg.data.test.img_scale, False, None)
    # imgs.append(_img)
    # img_metas.append(DC(_img_meta, cpu_only=True))
    # data = dict(img=imgs, img_meta=img_metas)

    print('evaluate ...')
    # evaluate model on images
    model.eval()
    results = []
    #for i, data in enumerate(data_loader):
    #    print(i)
        
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not True, **data)
        results.append(result)
    
        out_file = osp.join(args.out_dir,'{:02d}.jpg'.format(i))

        # use model.show_result to visualize
        model.module.show_result(data, result, cfg.img_norm_cfg, out_file=out_file, dataset=dataset_type)
    
if __name__ == '__main__':
    main()