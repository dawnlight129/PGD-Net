# 漏标区域和多标区域的提取与可视化
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 读取图像（灰度模式）
img_model = cv2.imread('/root/EB-TDFNet/results/mass_afe_groupmamba_attn2_4/tdfnet/22828930_15_05.png', 0)  # 模型分割结果
img_label = cv2.imread('/root/autodl-tmp/Massa_512/test/label/22828930_155.png', 0)  # 标签

# 确保图像尺寸一致
assert img_model.shape == img_label.shape, "两张图像尺寸不一致，请检查！"

# 2. 二值化（若图像已为纯二值图，可适当调整阈值）
_, img_model_bin = cv2.threshold(img_model, 127, 255, cv2.THRESH_BINARY)
_, img_label_bin = cv2.threshold(img_label, 127, 255, cv2.THRESH_BINARY)

# 3. 计算漏标和多标区域
## 漏标：标签有但模型没有 → 标签为255且模型为0
false_negative = np.logical_and(img_label_bin == 255, img_model_bin == 0)
## 多标：模型有但标签没有 → 模型为255且标签为0
false_positive = np.logical_and(img_model_bin == 255, img_label_bin == 0)

# 4. 将分割结果转换为彩色图像以便标记
img_model_color = cv2.cvtColor(img_model_bin, cv2.COLOR_GRAY2BGR)

# 5. 用不同颜色标记漏标和多标区域
# 漏标区域用红色标记 (B, G, R)
img_model_color[false_negative] = [0, 0, 255]
# 多标区域用蓝色标记
img_model_color[false_positive] = [255, 0, 0]

# 6. 计算统计信息
total_pixels = img_model.size
漏标像素数 = np.sum(false_negative)
多标像素数 = np.sum(false_positive)
漏标比例 = (漏标像素数 / total_pixels) * 100
多标比例 = (多标像素数 / total_pixels) * 100

print(f"漏标像素数: {漏标像素数} ({漏标比例:.2f}%)")
print(f"多标像素数: {多标像素数} ({多标比例:.2f}%)")

# 7. 可视化结果
plt.figure(figsize=(18, 6))

# 模型分割结果
plt.subplot(1, 3, 1)
plt.imshow(img_model_bin, cmap='gray')
plt.title('模型分割结果')
plt.axis('off')

# 标签
plt.subplot(1, 3, 2)
plt.imshow(img_label_bin, cmap='gray')
plt.title('标签')
plt.axis('off')

# 标记了差异的分割结果
plt.subplot(1, 3, 3)
# 转换BGR为RGB以便matplotlib正确显示颜色
plt.imshow(cv2.cvtColor(img_model_color, cv2.COLOR_BGR2RGB))
plt.title('分割结果（红色：漏标，蓝色：多标）')
plt.axis('off')

plt.tight_layout()
plt.show()

# 8. 保存结果（可选）
cv2.imwrite('Net_prompt_groupmamba_attnV2的分割结果1.png', img_model_color)