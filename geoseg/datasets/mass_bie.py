import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random

CLASSES = ('Building', 'Background')
PALETTE = [[255, 255, 255],  [0, 0, 0]]

ORIGIN_IMG_SIZE = (512, 512)
INPUT_IMG_SIZE = (512, 512)
TEST_IMG_SIZE = (512, 512)


class MassBuildDataset(Dataset):
    def __init__(self, data_root='/root/autodl-tmp/Massa_512', mode='train', 
                 img_dir='img', img_aug_dir='img_aug', mask_dir='mask',
                 img_suffix='.png', img_aug_suffix='.png', mask_suffix='.png', 
                 transform=None, mosaic_ratio=0.25, img_size=ORIGIN_IMG_SIZE):
        self.data_root = data_root
        self.img_dir = img_dir  # 原图目录
        self.img_aug_dir = img_aug_dir  # 增强图目录
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.img_aug_suffix = img_aug_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio if mode == 'train' else 0.0  # 只有训练模式使用mosaic
        self.img_size = img_size
        
        # 只有训练模式需要检查增强图目录
        self.need_aug = (mode == 'train')
        self.img_ids = self.get_img_ids()
        
        # 训练模式下验证增强图目录是否存在
        if self.need_aug and not osp.exists(osp.join(data_root, img_aug_dir)):
            raise ValueError(f"增强图目录不存在: {osp.join(data_root, img_aug_dir)}")

    def __getitem__(self, index):
        img, mask = None, None
        img_aug = None  # 默认为None
        
        # 非mosaic模式或验证/测试模式
        if (random.random() > self.mosaic_ratio) or self.mode in ['val', 'test']:
            if self.need_aug:  # 训练模式：加载原图、增强图和mask
                img, img_aug, mask = self.load_img_and_mask(index)
            else:  # 验证/测试模式：只加载原图和mask
                img, mask = self.load_img_and_mask(index)
        else:  # mosaic模式（只在训练时生效）
            img, img_aug, mask = self.load_mosaic_img_and_mask(index)
        
        # 应用变换
        if self.transform:
            # 对原图和mask应用相同变换
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            # 只有训练模式才对增强图应用变换
            if self.need_aug and img_aug is not None:
                augmented_aug = self.transform(image=img_aug)
                img_aug = augmented_aug['image']

        # 转换为Tensor并调整维度
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        
        # 构建结果字典
        results = {
            'img_id': self.img_ids[index], 
            'img': img,  # 原图，用于分割
            'gt_semantic_seg': mask  # 分割标签
        }
        
        # 只有训练模式才添加增强图
        if self.need_aug and img_aug is not None:
            img_aug = torch.from_numpy(img_aug).permute(2, 0, 1).float()
            results['img_aug'] = img_aug  # 增强图，用于重建
        
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self):
        """获取图片ID列表，根据模式决定是否需要匹配增强图"""
        img_path = osp.join(self.data_root, self.img_dir)
        mask_path = osp.join(self.data_root, self.mask_dir)
        
        # 获取原图和mask的文件名
        img_filenames = {f.split('.')[0] for f in os.listdir(img_path) 
                        if f.endswith(self.img_suffix)}
        mask_filenames = {f.split('.')[0] for f in os.listdir(mask_path) 
                         if f.endswith(self.mask_suffix)}
        
        # 训练模式需要同时匹配增强图，验证/测试模式只需要原图和mask匹配
        if self.need_aug:
            img_aug_path = osp.join(self.data_root, self.img_aug_dir)
            img_aug_filenames = {f.split('.')[0] for f in os.listdir(img_aug_path) 
                               if f.endswith(self.img_aug_suffix)}
            common_ids = img_filenames & img_aug_filenames & mask_filenames
            
            # 检查增强图缺失
            missing_in_aug = img_filenames - img_aug_filenames
            if missing_in_aug:
                print(f"警告: 增强图中缺少{len(missing_in_aug)}个文件: {list(missing_in_aug)[:5]}...")
        else:
            common_ids = img_filenames & mask_filenames
        
        # 检查原图和mask的匹配情况
        if not common_ids:
            if self.need_aug:
                raise ValueError("原图、增强图和mask之间没有匹配的文件ID")
            else:
                raise ValueError("原图和mask之间没有匹配的文件ID")
                
        missing_in_mask = img_filenames - mask_filenames
        if missing_in_mask:
            print(f"警告: mask中缺少{len(missing_in_mask)}个文件: {list(missing_in_mask)[:5]}...")
            
        return sorted(list(common_ids))

    def load_img_and_mask(self, index):
        """
        加载数据，根据模式决定是否返回增强图
        训练模式返回 (img, img_aug, mask)
        验证/测试模式返回 (img, mask)
        """
        img_id = self.img_ids[index]
        
        # 原图路径
        img_name = osp.join(self.data_root, self.img_dir, 
                           f"{img_id}{self.img_suffix}")
        # mask路径
        mask_name = osp.join(self.data_root, self.mask_dir, 
                           f"{img_id}{self.mask_suffix}")
        
        # 加载原图
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.uint8)
        
        # 加载mask
        mask = cv2.imread(mask_name, cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.float32)
        
        # 训练模式加载增强图
        if self.need_aug:
            img_aug_name = osp.join(self.data_root, self.img_aug_dir, 
                                  f"{img_id}{self.img_aug_suffix}")
            img_aug = cv2.imread(img_aug_name, cv2.IMREAD_COLOR)
            img_aug = cv2.cvtColor(img_aug, cv2.COLOR_BGR2RGB)
            img_aug = img_aug.astype(np.uint8)
            return img, img_aug, mask
        else:
            return img, mask

    def load_mosaic_img_and_mask(self, index):
        """加载mosaic模式的原图、增强图和对应的mask（只在训练时调用）"""
        # 随机选择4张图片组成mosaic
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        
        # 加载4组原图、增强图和mask（训练模式下）
        img_a, img_aug_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, img_aug_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, img_aug_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, img_aug_d, mask_d = self.load_img_and_mask(indexes[3])

        w = self.img_size[1]
        h = self.img_size[0]

        start_x = w // 4
        start_y = h // 4
        # 随机选择拼接中心坐标
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(start_y, (h - start_y))

        # 计算每个部分的裁剪尺寸
        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        # 创建随机裁剪器
        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        # 裁剪原图
        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        # 拼接原图和mask
        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)

        # 裁剪增强图（使用相同的裁剪参数）
        croped_a_aug = random_crop_a(image=img_aug_a.copy())
        croped_b_aug = random_crop_b(image=img_aug_b.copy())
        croped_c_aug = random_crop_c(image=img_aug_c.copy())
        croped_d_aug = random_crop_d(image=img_aug_d.copy())

        img_aug_crop_a = croped_a_aug['image']
        img_aug_crop_b = croped_b_aug['image']
        img_aug_crop_c = croped_c_aug['image']
        img_aug_crop_d = croped_d_aug['image']

        # 拼接增强图
        top_aug = np.concatenate((img_aug_crop_a, img_aug_crop_b), axis=1)
        bottom_aug = np.concatenate((img_aug_crop_c, img_aug_crop_d), axis=1)
        img_aug = np.concatenate((top_aug, bottom_aug), axis=0)

        # 确保数组内存连续
        img = np.ascontiguousarray(img)
        img_aug = np.ascontiguousarray(img_aug)
        mask = np.ascontiguousarray(mask)

        return img, img_aug, mask


def get_training_transform():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.RandomRotate90(p=0.5),
        albu.RandomBrightnessContrast(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)


def get_validation_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def get_test_transform():
    test_transform = [
        albu.Normalize()
    ]
    return albu.Compose(test_transform)
