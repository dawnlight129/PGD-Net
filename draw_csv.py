import pandas as pd
import matplotlib.pyplot as plt

# ======================== 1. 定义安全转换函数 ========================
def safe_convert(x):
    """处理空值、nan、非数字字符串，返回float或None"""
    if pd.isna(x):  # 处理Pandas内置NaN
        return None
    if not isinstance(x, str):
        x = str(x)  # 强制转字符串（兼容数值类型输入）
    x_stripped = x.strip()
    # 处理空字符串或"nan"字符串
    if not x_stripped or x_stripped.lower() == "nan":
        return None
    # 尝试转浮点数，捕获异常
    try:
        return float(x_stripped)
    except ValueError:
        return None  # 非数字字符串直接返回None


# ======================== 2. 读取并合并数据 ========================
# 修改为你的CSV路径
file_path = "EB-TDFNet/log/mass/mass-new-afamamba-rsnet50/version_11/metrics.csv"  
df = pd.read_csv(file_path, dtype=str)  # 强制按字符串读取，避免自动类型转换

# 检查列名是否匹配
expected_columns = [
    'val_loss', 'val_mIoU', 'val_F1', 'val_OA', 
    'epoch', 'step', 
    'train_loss', 'train_mIoU', 'train_F1', 'train_OA'
]
assert list(df.columns) == expected_columns, \
    f"列名不匹配！预期：{expected_columns}，实际：{list(df.columns)}"

# 合并交替行（验证行+训练行）
merged_data = []
for i in range(len(df)):
    if i % 2 == 0:  # 偶数行：验证集数据
        val_row = df.iloc[i].to_dict()
        # 奇数行：训练集数据（若存在）
        train_row = df.iloc[i+1].to_dict() if (i+1) < len(df) else {}
        
        # 提取验证集指标（必选）
        val_metrics = {
            'val_loss': safe_convert(val_row.get('val_loss', '')),
            'val_mIoU': safe_convert(val_row.get('val_mIoU', '')),
            'val_F1': safe_convert(val_row.get('val_F1', '')),
            'val_OA': safe_convert(val_row.get('val_OA', '')),
            'epoch': safe_convert(val_row.get('epoch', '')),
            'step': safe_convert(val_row.get('step', ''))
        }
        
        # 提取训练集指标（可选，不存在则为None）
        train_metrics = {
            'train_loss': safe_convert(train_row.get('train_loss', '')),
            'train_mIoU': safe_convert(train_row.get('train_mIoU', '')),
            'train_F1': safe_convert(train_row.get('train_F1', '')),
            'train_OA': safe_convert(train_row.get('train_OA', ''))
        }
        
        # 合并为一行
        merged = {**val_metrics, **train_metrics}
        merged_data.append(merged)

# 构建DataFrame并清洗（仅删除关键列缺失的行）
merged_df = pd.DataFrame(merged_data)
merged_df.dropna(subset=['epoch', 'val_loss', 'train_loss'], inplace=True)
merged_df['epoch'] = merged_df['epoch'].astype(int)  # 确保epoch为整数


# ======================== 3. 调试信息（关键！） ========================
print("\n================ 数据调试报告 ================")
print(f"原始CSV行数：{len(df)}")
print(f"合并后行数：{len(merged_data)}")
print(f"清洗后有效行数：{len(merged_df)}")

if not merged_df.empty:
    print("\n### 前5行数据预览 ###")
    print(merged_df.head())
    
    print("\n### 指标统计描述 ###")
    print(merged_df.describe())
else:
    print("警告：清洗后无有效数据！可能原因：")
    print(" 1. CSV为空或列名错误；")
    print(" 2. 所有行的epoch/val_loss/train_loss均缺失；")
    print(" 3. 数据格式错误（如非数字字符串）。")


# ======================== 4. 绘制图像（仅当数据有效时） ========================
if not merged_df.empty:
    # -------------------- ① 损失曲线 --------------------
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['epoch'], merged_df['train_loss'], 
             label='Train Loss', marker='o', linestyle='-', color='#FF5722')
    plt.plot(merged_df['epoch'], merged_df['val_loss'], 
             label='Val Loss', marker='s', linestyle='--', color='#1976D2')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # -------------------- ② mIoU曲线 --------------------
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['epoch'], merged_df['train_mIoU'], 
             label='Train mIoU', marker='o', linestyle='-', color='#4CAF50')
    plt.plot(merged_df['epoch'], merged_df['val_mIoU'], 
             label='Val mIoU', marker='s', linestyle='--', color='#9C27B0')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('Training vs Validation mIoU')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # -------------------- ③ F1曲线 --------------------
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['epoch'], merged_df['train_F1'], 
             label='Train F1', marker='o', linestyle='-', color='#FFC107')
    plt.plot(merged_df['epoch'], merged_df['val_F1'], 
             label='Val F1', marker='s', linestyle='--', color='#00BCD4')
    plt.xlabel('Epoch')
    plt.ylabel('F1')
    plt.title('Training vs Validation F1')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # -------------------- ④ OA曲线 --------------------
    plt.figure(figsize=(10, 6))
    plt.plot(merged_df['epoch'], merged_df['train_OA'], 
             label='Train OA', marker='o', linestyle='-', color='#795548')
    plt.plot(merged_df['epoch'], merged_df['val_OA'], 
             label='Val OA', marker='s', linestyle='--', color='#607D8B')
    plt.xlabel('Epoch')
    plt.ylabel('OA')
    plt.title('Training vs Validation OA')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()