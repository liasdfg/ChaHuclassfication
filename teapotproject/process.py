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



def apply_mask(image, mask):
    if image.size != mask.size:
        mask = mask.resize(image.size)
    image_np = np.array(image)
    mask_np = np.array(mask)
##################### 裁剪原图,增加细节 借助大模型实现 ##################
    nonzero = np.argwhere(mask_np > 0)
    if len(nonzero) > 0:
        y_min, x_min = nonzero.min(axis=0)
        y_max, x_max = nonzero.max(axis=0)
        h, w = mask_np.shape
        y_min = max(0, y_min - 15)
        x_min = max(0, x_min - 15)
        y_max = min(h, y_max + 15)
        x_max = min(w, x_max + 15)
        image_np = image_np[y_min:y_max, x_min:x_max]
        mask_np = mask_np[y_min:y_max, x_min:x_max]
######################################################################
    result = np.ones_like(image_np) * 255
    mask_binary = (mask_np > 127)
    result[mask_binary] = image_np[mask_binary]
    return Image.fromarray(result.astype(np.uint8))



def process_dataset(input_file, output_file):
    table = pq.read_table(input_file)
    print("字段数量:", len(table.schema))
    print("字段名称:", table.schema.names)
    df = table.to_pandas()
    processed_images = []
    for idx, row in tqdm(df.iterrows(),total=len(df)):
        img_bytes = row['image']['bytes']
        mask_bytes = row['mask']['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
        fuse_picture = apply_mask(image, mask)

        buffer = io.BytesIO()
        fuse_picture.save(buffer, format='JPEG')
        fuse_bytes = buffer.getvalue()

        processed_images.append({
            'bytes': fuse_bytes,
            'path': row['image']['path']
        })

    new_df = pd.DataFrame({
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

    table = pa.Table.from_pandas(new_df)
    pq.write_table(table, output_file)
    print(f'保存为{output_file}')
    return len(new_df)


def main(path='.'):
    parquet_files = []
    for filename in glob.glob(os.path.join(path, '*.parquet')):
        file = os.path.basename(filename)
        if file.lower().startswith('cn') and '-processed' not in file.lower():
            parquet_files.append(filename)

    print(f'存在{len(parquet_files)}个符合要求的文件:')
    for f in parquet_files:
        print(f'  - {f}')

    total_processed = 0
    for input_file in parquet_files:
        output_file = input_file.replace('.parquet', '-processed.parquet')
        print(output_file)
        count = process_dataset(input_file, output_file)
        total_processed += count

    print(f'处理{total_processed}张图片')

if __name__ == '__main__':
    main(r'E:\teapotproject\ChaHu')
