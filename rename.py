import os
from pathlib import Path

# 指定主文件夹路径
main_folder = Path('/root/autodl-tmp/JMregion/train')

# 子文件夹名称
subfolders = ['image', 'label', 'mask']

# 遍历每个子文件夹
for subfolder in subfolders:
    # 获取子文件夹的完整路径
    folder_path = main_folder / subfolder
    
    # 确保子文件夹存在
    if folder_path.exists():
        # 遍历子文件夹中的所有文件
        for file in folder_path.iterdir():
            # 如果是文件（不是文件夹）
            if file.is_file():
                # 获取文件名
                filename = file.name
                
                # 去除文件名中的'image'和'label'字段
                new_filename = filename.replace('image', '').replace('label', '')
                
                # 构造新的文件路径
                new_file_path = file.with_name(new_filename)
                
                # 重命名文件
                file.rename(new_file_path)
                print(f'Renamed "{file}" to "{new_file_path}"')
    else:
        print(f'The folder "{folder_path}" does not exist.')

print('All files have been renamed.')
