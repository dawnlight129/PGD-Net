# -*- coding: utf-8 -*-
import os
import numpy as np
import cv2
import copy


def enhancement_strategy_building_integrity(input_path, input_label_path, output_aug_path):
    # 创建输出目录（关键修复）
    if not os.path.exists(output_aug_path):
        os.makedirs(output_aug_path, exist_ok=True)
        print(f"已创建输出目录: {output_aug_path}")

    # 过滤只处理图像文件（关键修复）
    image_extensions = ('.png', '.jpg', '.jpeg')
    files = [f for f in os.listdir(input_path) if f.lower().endswith(image_extensions)]
    number = len(files)
    
    if number == 0:
        print(f"警告：输入目录 {input_path} 中没有找到图像文件")
        return

    # 增强参数
    alpha_1 = 1.2  # 前景增强对比度
    alpha_2 = 0.8  # 背景减弱对比度
    beta = 0       # 亮度因子

    for index in range(number):
        name = files[index]
        print(f"处理 {index+1}/{number}：{name}")

        # 构建文件路径
        full_image_path = os.path.join(input_path, name)  # 使用os.path.join避免路径拼接错误
        full_label_path = os.path.join(input_label_path, name)
        full_output_aug_path = os.path.join(output_aug_path, name)

        # 检查输入文件是否存在（关键修复）
        if not os.path.exists(full_image_path):
            print(f"警告：图像文件不存在 {full_image_path}，跳过")
            continue
        if not os.path.exists(full_label_path):
            print(f"警告：标签文件不存在 {full_label_path}，跳过")
            continue

        # 读取图像（带错误处理）
        img = cv2.imread(full_image_path)
        if img is None:
            print(f"警告：无法读取图像 {full_image_path}，跳过")
            continue

        # 读取掩码
        gray_image = cv2.imread(full_label_path, cv2.IMREAD_GRAYSCALE)
        if gray_image is None:
            print(f"警告：无法读取掩码 {full_label_path}，跳过")
            continue

        # 图像处理逻辑
        background = 255 - gray_image
        _, thresholded = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_foreground = np.zeros_like(img)
        for contour in contours:
            mask = np.zeros_like(gray_image)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            masked_image = cv2.bitwise_and(img, img, mask=mask)
            masked_image_new = copy.deepcopy(masked_image)

            # 计算前景平均像素（避免空轮廓导致的NaN）
            non_zero_mask = mask != 0
            if np.sum(non_zero_mask) == 0:
                continue  # 跳过空轮廓
            avg = [
                np.mean(masked_image[:, :, 0][non_zero_mask]),
                np.mean(masked_image[:, :, 1][non_zero_mask]),
                np.mean(masked_image[:, :, 2][non_zero_mask])
            ]
            masked_image[non_zero_mask] = avg
            masked_image_out = (masked_image / 2 + masked_image_new / 2).astype(np.uint8)
            image_foreground = (image_foreground + masked_image_out).astype(np.uint8)

        # 背景和增强处理
        color_background_image = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_background = img * ((color_background_image / 255).astype(np.uint8))
        f_enhanced = cv2.convertScaleAbs(image_foreground, alpha=alpha_1, beta=beta)
        b_diminished = cv2.convertScaleAbs(img_background, alpha=alpha_2, beta=beta)
        img_enhanced = f_enhanced + b_diminished
        img_enhanced = cv2.cvtColor(img_enhanced, cv2.COLOR_RGB2BGR)

        # 保存增强图像（带检查）
        write_success = cv2.imwrite(full_output_aug_path, img_enhanced)
        if not write_success:
            print(f"警告：无法保存增强图像 {full_output_aug_path}，请检查权限")
        else:
            print(f"已保存：{full_output_aug_path}")


if __name__ == '__main__':
    # 建议使用相对路径或确保绝对路径正确
    input_path = "/root/autodl-tmp/Massa_512/train/img"
    input_label_path = "/root/autodl-tmp/Massa_512/train/mask"
    output_aug_path = "/root/autodl-tmp/Massa_512/train/aug"

    # 检查输入目录是否存在
    if not os.path.exists(input_path):
        print(f"错误：输入图像目录不存在 {input_path}")
    elif not os.path.exists(input_label_path):
        print(f"错误：输入标签目录不存在 {input_label_path}")
    else:
        enhancement_strategy_building_integrity(input_path, input_label_path, output_aug_path)
        print("处理完成")
    