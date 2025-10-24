# import torch

# # 加载权重文件
# ckpt_path ="log/mass/mass-VMamba-DSAFormer-DGCM-3module-ARB-2/mass-VMamba-DSAFormer-DGCM-3module-ARB-2-v1.ckpt"  # 替换为你的权重文件路径
# checkpoint = torch.load(ckpt_path, map_location="cpu")  # map_location="cpu" 避免依赖GPU

# # 1. 查看checkpoint的所有顶级键名（了解整体结构）
# print("Checkpoint包含的顶级键：", checkpoint.keys())

# # 2. 查看模型参数（state_dict）的结构
# if "state_dict" in checkpoint:
#     model_params = checkpoint["state_dict"]
#     # 打印state_dict中的前10个参数名（避免输出过长）
#     print("\n模型参数（state_dict）中的全部键名：")
#     for i, key in enumerate(model_params.keys()):
#         # if i < 10:
#         print(f"  - {key}")
#         # else:
#         #     print("  - ...（更多参数）")
#         #     break
#     # 查看某个具体参数的形状（如第一个参数）
#     first_key = next(iter(model_params.keys()))
#     print(f"\n第一个参数的形状：{model_params[first_key].shape}")

# # 3. 查看其他元数据（如训练轮次、超参数等）
# if "epoch" in checkpoint:
#     print(f"\n保存时的训练轮次：{checkpoint['epoch']}")
# if "hparams" in checkpoint:
#     print(f"超参数示例：{checkpoint['hparams']}")  # 可能是字典或Namespace对象


import torch


# ckpt_path ="log/mass/mass-VMamba-DSAFormer-DGCM-3module-ARB-2/mass-VMamba-DSAFormer-DGCM-3module-ARB-2-v1.ckpt" 
checkpoint = torch.load("log/mass/mass-new-prompt-groupmamba/mass-new-prompt-groupmamba.ckpt", map_location="cpu")
state_dict = checkpoint["state_dict"]  # 假设权重存在state_dict键下

# 保存所有键到文本文件
with open("groupmamba_prompt_all_keys.txt", "w", encoding="utf-8") as f:
    for key in state_dict.keys():
        f.write(key + "\n")
print("所有键已保存到 all_keys.txt，可打开文件查看完整内容")