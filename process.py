import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from tqdm import tqdm
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import glob
import os
from PIL import Image
import io
import argparse


# 对图像应用mask
def apply_mask(image, mask):
    if image.size != mask.size:
        mask = mask.resize(image.size)  # 大小不一致则改变mask大小
    image_np = np.array(image)          # 转化为numpy
    mask_np = np.array(mask)
    result = np.ones_like(image_np) * 255  # 创建白色背景
    mask_binary = (mask_np > 127)          # 获取mask遮盖的位置
    result[mask_binary] = image_np[mask_binary]      # 将原图中对应mask未遮盖的像素拷贝到背景中
    return Image.fromarray(result.astype(np.uint8))  # 转化为 Image


# 生成处理后数据集
def process_dataset(input_file, output_file):
    table = pq.read_table(input_file)    # 转化为Table
    print("字段数量:", len(table.schema))
    print("字段名称:", table.schema.names)
    df = table.to_pandas()              # 转化为DataFrame
    processed_images = []               # 存放图像信息
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        img_bytes = row['image']['bytes']   # 取出图像数据
        mask_bytes = row['mask']['bytes']   # 取出mask数据
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # 转换为RGB模式
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')    # 转化为单通道模式
        fuse_picture = apply_mask(image, mask)    # 对原图应用mask

        buffer = io.BytesIO()
        fuse_picture.save(buffer, format='JPEG')  # 转化为JPEG格式
        fuse_bytes = buffer.getvalue()            # 获取字节流

        processed_images.append({                 # 保存mask处理的图像信息
            'bytes': fuse_bytes,
            'path': row['image']['path']
        })

    new_df = pd.DataFrame({    # 存储数据集信息
        'id': df['id'],
        'image': processed_images,
        'caption': df['caption'],
        'time': df['time'],
        'geometric shape type': df['geometric shape type'],
        'natural shape type': df['natural shape type'],
        'flower type': df['flower type'],
        'handle type': df['handle type'],
        'innovative': df['innovative']
    })

    table = pa.Table.from_pandas(new_df)  # 转化为Table
    pq.write_table(table, output_file)    # 写入成parquet文件
    print(f'保存为{output_file}')
    return len(new_df)


def main(args):
    parquet_dir = args.dir
    if not os.path.exists(parquet_dir):  # 检查文件夹是否存在
        raise ValueError(f'该文件夹不存在: {os.path.abspath(parquet_dir)}')
    parquet_files = []
    for file in glob.glob(os.path.join(parquet_dir, '*.parquet')):
        filename = os.path.basename(file)   # 获取文件名
        if filename.lower().startswith('cn') and '-processed' not in filename.lower():
            parquet_files.append(file)  # 满足if的条件则存储路径

    print(f'存在{len(parquet_files)}个符合要求的文件:')
    for f in parquet_files:
        print(f'  - {f}')   # 输出符合要求的parquet路径

    total_processed = 0     # 处理的数量计数
    for input_file in parquet_files:
        output_file = input_file.replace('.parquet', '-processed.parquet')  # 设定输出的文件名
        # print(output_file)
        count = process_dataset(input_file, output_file)    # 生成处理后的数据集
        total_processed += count       # 计数

    print(f'处理{total_processed}张图片')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="数据预处理")
    parser.add_argument("-d", "--dir", type=str, default='ChaHu', help="数据集所在文件夹")
    args_main = parser.parse_args()
    main(args_main)

