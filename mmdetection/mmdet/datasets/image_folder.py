import os
import os.path as osp

import mmcv
import numpy as np
from mmcv.parallel import DataContainer as DC

from .registry import DATASETS
from .custom import CustomDataset
from imagecorruptions import corrupt
from .transforms import ImageTransform
from .utils import to_tensor

@DATASETS.register_module
class ImageFolder(CustomDataset):

    CLASSES = None

    def __init__(self,
                 img_prefix,
                 img_scale,
                 img_norm_cfg,
                 size_divisor=None,
                 resize_keep_ratio=True,
                 test_mode=False, 
                 corruption=None, 
                 corruption_severity=1,
                 **kwargs):
        # prefix of images path
        self.img_prefix = img_prefix
        
        # list images
        self.img_infos = [{'filename': file_name} for file_name in os.listdir(self.img_prefix)]

        # (long_edge, short_edge) or [(long1, short1), (long2, short2), ...]
        self.img_scales = img_scale if isinstance(img_scale,
                                                  list) else [img_scale]
        assert mmcv.is_list_of(self.img_scales, tuple)
        # normalization configs
        self.img_norm_cfg = img_norm_cfg

        # padding border to ensure the image size can be divided by
        # size_divisor (used for FPN)
        self.size_divisor = size_divisor

        # in test mode or not
        self.test_mode = test_mode

        # transforms
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor, **self.img_norm_cfg)
        
        # image rescale if keep ratio
        self.resize_keep_ratio = resize_keep_ratio
        
        self.corruption = corruption
        self.corruption_severity = corruption_severity

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        # load image
        img = mmcv.imread(osp.join(self.img_prefix, img_info['filename']))
        # corrupt image
        if self.corruption is not None:
            img = corrupt(img, severity=self.corruption_severity, corruption_name=self.corruption)

        def prepare_single(img, scale, flip, proposal=None):
            _img, img_shape, pad_shape, scale_factor = self.img_transform(
                img, scale, flip, keep_ratio=True)
            _img = to_tensor(_img)
            _img_meta = dict(
                ori_shape=(img.shape[0], img.shape[1], 3),
                img_shape=img_shape,
                pad_shape=pad_shape,
                scale_factor=scale_factor,
                flip=flip)
            return _img, _img_meta

        imgs = []
        img_metas = []
        for scale in self.img_scales:
            _img, _img_meta = prepare_single(img, scale, False, None)
            imgs.append(_img)
            img_metas.append(DC(_img_meta, cpu_only=True))
        data = dict(img=imgs, img_meta=img_metas)
        return data