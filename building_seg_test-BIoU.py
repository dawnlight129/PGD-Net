import ttach as tta
import multiprocessing.pool as mpp
import multiprocessing as mp
import time
from train_supervision import *
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    来自官方 LVIS / Boundary IoU 仓库
    把二值 mask 转换成边界 mask
    mask: 必须是 0/1 二值图，形状为 (H, W)
    返回: 边界区域的 0/1 二值图
    """
    mask = mask.astype(np.uint8)
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    # 加一个像素的边界，防止腐蚀时边界被吃掉
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]

    # 原图 - 腐蚀图 = 边界区域（0/1 二值图）
    boundary = mask - mask_erode
    return boundary

def compute_boundary_iou_official(pred_mask, gt_mask):
    """
    计算官方标准 Boundary IoU
    pred_mask: 预测掩码，(H,W) 0/1 二值图
    gt_mask: 真值掩码，(H,W) 0/1 二值图
    """
    # 生成边界掩码
    gt_bound = mask_to_boundary(gt_mask)
    pred_bound = mask_to_boundary(pred_mask)

    # 计算边界区域的交集和并集
    intersection = (gt_bound & pred_bound).sum()
    union = gt_bound.sum() + pred_bound.sum() - intersection

    if union == 0:
        return np.nan  # 无边界时返回 NaN，后续取均值会自动忽略
    return intersection / union

def label_to_rgb(mask):
    h, w = mask.shape[0], mask.shape[1]
    mask_rgb = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [255, 255, 255]
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [0, 0, 0]
    return mask_rgb


def img_writer(inp):
    (mask, mask_id, rgb) = inp
    if rgb:
        mask_name_tif = mask_id + '.png'
        mask_tif = label_to_rgb(mask)
        cv2.imwrite(mask_name_tif, mask_tif)
    else:
        mask_png = mask.astype(np.uint8)
        mask_name_png = mask_id + '.png'
        cv2.imwrite(mask_name_png, mask_png)


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("-c", "--config_path", type=Path, required=True, help="Path to  config")
    arg("-o", "--output_path", type=Path, help="Path where to save resulting masks.", required=True)
    arg("-t", "--tta", help="Test time augmentation.", default="lr", choices=[None, "d4", "lr"])
    arg("--rgb", help="whether output rgb images", action='store_true')
    return parser.parse_args()


def main():
    args = get_args()
    config = py2cfg(args.config_path)
    args.output_path.mkdir(exist_ok=True, parents=True)

    model = Supervision_Train.load_from_checkpoint(
        os.path.join(config.weights_path, config.test_weights_name + '.ckpt'), config=config)
    model.cuda()
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = Evaluator(num_class=config.num_classes)
    evaluator.reset()

    boundary_iou_list = []  # <-- 存储 Boundary IoU

    if args.tta == "lr":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip()])
        model = tta.SegmentationTTAWrapper(model, transforms)
    elif args.tta == "d4":
        transforms = tta.Compose([tta.HorizontalFlip(), tta.VerticalFlip(), tta.Rotate90(angles=[0,90,180,270])])
        model = tta.SegmentationTTAWrapper(model, transforms)

    test_dataset = config.test_dataset

    with torch.no_grad():
        test_loader = DataLoader(
            test_dataset, batch_size=1, num_workers=4,
            pin_memory=True, drop_last=False)
        
        results = []
        for input in tqdm(test_loader):
            raw_predictions = model(input['img'].cuda())
            image_ids = input["img_id"]
            has_gt = 'gt_semantic_seg' in input.keys()
            if has_gt:
                masks_true = input['gt_semantic_seg']

            raw_predictions = nn.Softmax(dim=1)(raw_predictions)
            predictions = raw_predictions.argmax(dim=1)

            for i in range(raw_predictions.shape[0]):
                pred_mask = predictions[i].cpu().numpy()
                mask_name = image_ids[i]
                results.append((pred_mask, str(args.output_path / mask_name), args.rgb))

                if has_gt:
                    gt_mask = masks_true[i].cpu().numpy()
                    evaluator.add_batch(pre_image=pred_mask, gt_image=gt_mask)
                    # add
                    biou = compute_boundary_iou_official(pred_mask, gt_mask)
                    if not np.isnan(biou):
                        boundary_iou_list.append(biou)
                    

    # 保存图片
    t0 = time.time()
    mpp.Pool(processes=mp.cpu_count()).map(img_writer, results)
    t1 = time.time()
    print(f'images writing spends: {t1-t0:.2f} s')

    # 输出原有指标
    iou_per_class = evaluator.Intersection_over_Union()
    f1_per_class = evaluator.F1()
    OA = evaluator.OA()
    precision = evaluator.Precision()
    recall = evaluator.Recall()

    for class_name, class_iou, class_f1 in zip(config.CLASSES, iou_per_class, f1_per_class):
        print(f'F1_{class_name}:{class_f1:.4f}, IOU_{class_name}:{class_iou:.4f}')

    mean_f1 = np.nanmean(f1_per_class[:-1])
    mean_iou = np.nanmean(iou_per_class[:-1])
    mean_p = np.nanmean(precision[:-1])
    mean_r = np.nanmean(recall[:-1])
    mean_biou = np.nanmean(boundary_iou_list)

    # ========== 输出 Boundary IoU ==========
    print('-' * 60)
    print(f'F1:{mean_f1:.4f}, mIOU:{mean_iou:.4f}, OA:{OA:.4f}, P:{mean_p:.4f}, R:{mean_r:.4f}')
    print(f'Boundary-IoU: {mean_biou:.4f}')
    print('-' * 60)

if __name__ == "__main__":
    main()
    
# python building_seg_test-BIoU.py -c ./config/mass/afeNet.py -o ./results/PGD-Net/PGD-net-BIoU --rgb -t 'lr'