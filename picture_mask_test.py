import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt
import io
import pyarrow.parquet as pq
import argparse
import os
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 对图像应用mask
def apply_mask(image, mask):
    if image.size != mask.size:
        mask = mask.resize(image.size)  # 大小不一致则改变mask大小
    image_np = np.array(image)          # 转化为numpy
    mask_np = np.array(mask)
    result = np.ones_like(image_np) * 255   # 创建白色背景
    mask_binary = (mask_np > 127)           # 获取mask遮盖的位置
    result[mask_binary] = image_np[mask_binary]      # 将原图中对应mask未遮盖的像素拷贝到背景中
    return Image.fromarray(result.astype(np.uint8))  # 转化为 Image


def main(args):
    parquet_dir = args.dir    # 存储数据集文件夹路径
    if not os.path.exists(parquet_dir):     # 检查文件夹是否存在
        raise ValueError(f'该文件夹不存在: {os.path.abspath(parquet_dir)}')
    parquet_files = []
    for file in glob.glob(os.path.join(parquet_dir, '*.parquet')):
        filename = os.path.basename(file)    # 获取文件名
        if filename.lower().startswith('cn') and '-processed' not in filename.lower():
            parquet_files.append(file)       # 存入符合要求的文件路径
    print(f'找到{len(parquet_files)}个符合要求的文件:')
    if len(parquet_files) == 0:
        print('不存在parquet文件')  # 没有符合要求的文件则结束
        return
    elif len(parquet_files) == 1:
        input_file = parquet_files[0]  # 一个符合要求的文件则直接使用
    else:
        input_file = random.choice(parquet_files)  # 多个符合要求的文件则随机使用
    print(f'使用{input_file}')

    table = pq.read_table(input_file)  # 读取parquet文件
    df = table.to_pandas()             # 转化为DataFrame
    print(f'总图片数: {len(df)}')
    print(type(table))

    num_samples = min(10, len(df))     # 设定样本数量
    sample_indices = random.sample(range(len(df)), num_samples)  # 抽取10张图片或者选取所有图片
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:  # 处理matplotlib设计缺陷
        axes = axes.reshape(1, -1)
    for i, idx in enumerate(sample_indices):  # mask遮盖效果可视化
        row = df.iloc[idx]
        img_bytes = row['image']['bytes']    # 提取数据
        mask_bytes = row['mask']['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # 转换模式
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
        fuse_picture = apply_mask(image, mask)    # 应用mask
        axes[i, 0].imshow(image)
        axes[i, 0].set_title(f'图片 {idx}')
        axes[i, 0].axis('off')
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f'掩码 {idx}')
        axes[i, 1].axis('off')
        axes[i, 2].imshow(fuse_picture)
        axes[i, 2].set_title(f'融合结果 {idx}')
        axes[i, 2].set_xticks([])
        axes[i, 2].set_yticks([])

    plt.tight_layout()
    output_path = 'picture_mask_test_output.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'保存为{output_path}')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="图像mask处理测试")
    parser.add_argument("-d", "--dir", type=str, default='ChaHu', help="数据集所在文件夹")
    args_main = parser.parse_args()
    main(args_main)




