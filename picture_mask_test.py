import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from PIL import Image
import glob
import random
import matplotlib.pyplot as plt
import io
import pyarrow.parquet as pq
plt.rcParams['font.sans-serif'] = ['SimHei']      # 黑体
plt.rcParams['axes.unicode_minus'] = False
def apply_mask(image, mask):
    if image.size != mask.size:
        mask = mask.resize(image.size)
    image_np = np.array(image)
    mask_np = np.array(mask)
  ################ 裁剪原图,增加细节 借助大模型实现#######################
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
#############################################################################
    result = np.ones_like(image_np) * 255
    mask_binary = (mask_np > 127)
    result[mask_binary] = image_np[mask_binary]
    return Image.fromarray(result.astype(np.uint8))

def main():
    parquet_files = glob.glob('*.parquet')
    print(parquet_files)
    if len(parquet_files) == 0:
        print('不存在parquet文件')
        return
    elif len(parquet_files) == 1:
        input_file = parquet_files[0]
    else:
        input_file = random.choice(parquet_files)
    print(f'使用{input_file}')

    table = pq.read_table(input_file)
    df = table.to_pandas()
    print(f'总图片数: {len(df)}')
    print(type(table))

    num_samples = min(10, len(df))
    sample_indices = random.sample(range(len(df)), num_samples)
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:  # 处理matplotlib设计缺陷
        axes = axes.reshape(1, -1)
    for i, idx in enumerate(sample_indices):
        row = df.iloc[idx]
        img_bytes = row['image']['bytes']
        mask_bytes = row['mask']['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        mask = Image.open(io.BytesIO(mask_bytes)).convert('L')
        fuse_picture = apply_mask(image, mask)
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
        for spine in axes[i, 2].spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(5)

    plt.tight_layout()
    output_path = 'picture_mask_test_output.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'保存为{output_path}')
    plt.show()

if __name__ == '__main__':
    main()


