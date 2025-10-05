import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision_bie import *  # 注意导入的是修改后的训练文件
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def label_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]  # Background
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]        # Building
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if not mask_id.endswith('.png'):
        mask_id += '.png'
    if rgb:
        mask_tif = label_to_rgb(mask)
        cv2.imwrite(mask_id, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        cv2.imwrite(mask_id, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to config")
    arg("-o", "--output_path", type=Path, help="Path to save masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default="lr", choices=[None, "d4", "lr"])
    arg("--rgb", help="Output RGB images", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    # 加载模型
    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    model.cuda()
    model.eval()
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    # 处理TTA（测试时增强）
    # 在处理TTA的部分，修改模型包装逻辑
    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        # 自定义TTA包装逻辑，只处理分割logits
        class TTASegWrapper(tta.SegmentationTTAWrapper):
            def forward(self, x):
                # 对每个增强变换，只取分割logits并合并
                outputs = []
                for transformer in self.transforms:
                    augmented_x = transformer.augment_image(x)
                    # 模型返回(seg_logits, rec_logits)，只取seg_logits
                    seg_logits, _ = self.model(augmented_x)
                    deaugmented_logits = transformer.deaugment_mask(seg_logits)
                    outputs.append(deaugmented_logits)
                # 合并所有增强结果
                return torch.mean(torch.stack(outputs), dim=0)
        model_tta = TTASegWrapper(model, transforms)

    elif args.tta == "d4":
        transforms = tta.Compose([
            tta.HorizontalFlip(), 
            tta.VerticalFlip(), 
            tta.Rotate90(angles=[0, 90, 180, 270])
        ])
        # 同样使用自定义包装器
        class TTASegWrapper(tta.SegmentationTTAWrapper):
            def forward(self, x):
                outputs = []
                for transformer in self.transforms:
                    augmented_x = transformer.augment_image(x)
                    seg_logits, _ = self.model(augmented_x)  # 只取分割输出
                    deaugmented_logits = transformer.deaugment_mask(seg_logits)
                    outputs.append(deaugmented_logits)
                return torch.mean(torch.stack(outputs), dim=0)
        model_tta = TTASegWrapper(model, transforms)

    else:
        model_tta = model

    test_dataset = config.test_dataset
    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            num_workers=4,
            pin_memory=True,
            drop_last=False,
        )
        results = []
        for input in tqdm(test_loader):
            imgs = input['img'].cuda()  # 输入图像
            image_ids = input["img_id"]
            
            # 模型预测：提取分割logits（忽略重建logits）
            if args.tta in ["lr", "d4"]:
                # TTA模式：包装后的模型返回的是合并后的seg_logits
                seg_logits = model_tta(imgs)
            else:
                # 非TTA模式：从模型输出中提取seg_logits
                seg_logits, _ = model(imgs)  # 关键：只取分割输出
            
            # 上采样到输入图像的尺寸（与训练/验证阶段一致）
            seg_logits = F.interpolate(
                seg_logits,
                size=imgs.shape[2:],  # 输入图像的H和W
                mode='bilinear',
                align_corners=model.net.align_corners  # 与模型保持一致
            )
            
            # 计算预测掩码
            seg_probs = nn.Softmax(dim=1)(seg_logits)
            predictions = seg_probs.argmax(dim=1)  # 形状：(B, H, W)

            # 处理每个样本
            for i in range(predictions.shape[0]):
                mask = predictions[i].cpu().numpy()  # 预测掩码
                
                # 计算指标（如果有标签）
                if 'gt_semantic_seg' in input:
                    gt_mask = input['gt_semantic_seg'][i].cpu().numpy()
                    evaluator.add_batch(pre_image=mask, gt_image=gt_mask)
                
                # 保存结果
                mask_name = image_ids[i]
                results.append((mask, str(args.output_path / mask_name), args.rgb))

    # 保存预测结果
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    print(f'Image writing time: {t1 - t0:.2f}s')

    # 打印指标
    if 'gt_semantic_seg' in test_dataset[0]:  # 确保测试集有标签
        iou_per_class = evaluator.Intersection_over_Union()
        f1_per_class = evaluator.F1()
        OA = evaluator.OA()
        precision = evaluator.Precision()
        recall = evaluator.Recall()
        
        for class_name, iou, f1 in zip(config.CLASSES, iou_per_class, f1_per_class):
            print(f'F1_{class_name}: {f1:.6f}, IOU_{class_name}: {iou:.6f}')
        print(f'\nMean F1: {np.nanmean(f1_per_class[:-1]):.6f}, '
              f'mIOU: {np.nanmean(iou_per_class[:-1]):.6f}, '
              f'OA: {OA:.6f}, '
              f'Precision: {np.nanmean(precision[:-1]):.6f}, '
              f'Recall: {np.nanmean(recall[:-1]):.6f}')


if __name__ == "__main__":
    main()
# python building_seg_test.py -c ./config/mass/tdfnet.py -o ./results/mass_vmamba_sdi2/tdfnet --rgb -t 'lr'
# python building_seg_test.py -c ./config/mass/tdfnet.py -o ./results/mass_pvt/tdfnet --rgb -t 'lr'
# python building_seg_test.py -c ./config/mass/afaMamba.py -o ./results/mass_pvt/afaMamba --rgb -t 'lr'

# python building_seg_test.py -c ./config/mass/afeNet.py -o ./results/mass_afe_vmamba/tdfnet --rgb -t 'lr'
# python building_seg_test.py -c ./config/mass/tdfnet.py -o ./results/mass_pvt2/tdfnet --rgb -t 'lr'
# python building_seg_test.py -c ./config/mass/afaMamba.py -o ./results/mass_pvt/afaMamba --rgb -t 'lr'

# python building_seg_test.py -c ./config/mass/bienet.py -o ./results/mass_vmamba_bie/tdfnet --rgb -t 'lr'
