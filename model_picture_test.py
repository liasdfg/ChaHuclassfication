import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import glob
import pyarrow.parquet as pq
import pandas as pd
import torch
import random
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torchvision import transforms
from datetime import datetime
import torch.nn.functional as F
from model import ResNeXt_CBAM
import argparse
# ------------------------
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TASK_NAME_LIST = ['geometric shape type', 'natural shape type']
# IMAGE_SIZE = 224
# ------------------------------


# 加载数据集
def load_test_datasets(test_dir):
    if not os.path.exists(test_dir):  # 检查文件夹是否存在
        raise ValueError(f'该文件夹不存在: {os.path.abspath(test_dir)}')
    parquet_files = []
    for file in glob.glob(os.path.join(test_dir, '*.parquet')):
        filename = os.path.basename(file)   # 获取文件名
        if filename.lower().startswith('test_dataset'):
            parquet_files.append(file)      # 符合要求则保存路径
    print(f'找到{len(parquet_files)}个符合要求的文件:')
    df_list = []
    for f in parquet_files:
        df = pq.read_table(f).to_pandas()   # 读取并转化为DataFrame
        df_list.append(df)
        print(f'已加载 {os.path.basename(f)}: {len(df)}条记录')

    combined_df = pd.concat(df_list, ignore_index=True)  # 拼接DataFrame
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)  # 打乱DataFrame
    # print(combined_df)
    print(f'总记录数: {len(combined_df)}')
    return combined_df


#  图片预测结果生成器
def pictures_probs_loader(model,dataframe,transform,task_list,num_pictures,device):
    sample_indices = random.sample(range(len(dataframe)), num_pictures)    # 随机抽取索引
    for idx in sample_indices:
        output_pred = {}
        row = dataframe.iloc[idx]   # 根据索引获得相应记录
        # print(row)
        img_bytes = row['image']['bytes']   # 读取图像数据
        image_org = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # 转换模式
        image = transform(image_org)   # 变换处理
        model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(device)  # 扩充维度
            output = model(image)                  # 获取预测结果
            for index,type_name in enumerate(task_list):  # 根据任务列表按不同任务进行预测
                probabilities = torch.nn.functional.softmax(output[index], dim=1)[0]  # softmax计算得到概率
                output_pred[type_name] = probabilities   # 存储概率
        yield image_org, output_pred   # 返回生成器


# 解析映射文件
def analyze_mapping(pkl_name, dir_path):
    if not os.path.exists(dir_path):  # 检查文件夹是否存在
        raise ValueError(f'该文件夹不存在: {os.path.abspath(dir_path)}')
    pkl_path = os.path.join(dir_path, pkl_name)   # 获取pkl文件路径
    with open(pkl_path,'rb') as f:
        pkl_data = pickle.load(f)  # 加载pkl文件
    class_dict = {}  # 存储索引对应的类别
    type_list = []   # 获取分类任务名称
    for type_name, value in pkl_data.items():
        if type_name != "num_classes":  # 过滤pkl文件里num_classes键
            class_dict[type_name] = {v: k for k, v in value.items()}  # 反转索引和类别的关系
            type_list.append(type_name)  # 记录任务名称
        else:                           # 针对num_classes获取任务的类别数
            num_dict = {v: k for v, k in zip(type_list, value)}  # 获取任务对应的类别数
    return class_dict, num_dict


# 预测结果可视化
def plot_picture_pred(loader, type_idx_class, class_num_dict, task_list, save_dir="save_picture"):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for img_org, output_pred in loader:  # 通过生成器获取图片和概率
        for type_name in task_list:      # 根据任务列表进行可视化
            prob = output_pred[type_name]   # 通过任务名称获取概率
            idx_class_dict = type_idx_class[type_name]  # 获取任务对应的类别索引
            num_classes = class_num_dict[type_name]    # 获取任务的类别数

            predicted_idx = torch.argmax(prob).item()   # 获取概率最大的类别索引
            predicted_class = idx_class_dict[predicted_idx]  # 根据索引获取类别
            confidence = prob[predicted_idx].item()      # 获取置信度

            class_perf = [(idx_class_dict[i], prob[i].item()) for i in range(num_classes)]  # 将所有索引转化为对应名称和概率
            class_perf.sort(key=lambda x: x[1])     # 按概率值排序

            categories = [x[0] for x in class_perf]  # 获取类别名，用于绘图
            probs = [x[1] for x in class_perf]       # 获取概率值，用于绘图

            plt.figure(figsize=(12, 6))
            # 原图
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(img_org))
            plt.title(f"Predicted: {predicted_class} ({confidence:.2%})", fontsize=14)
            plt.axis('off')
            # 概率条形图
            plt.subplot(1, 2, 2)
            bars = plt.barh(categories, probs, color='skyblue')
            plt.xlabel('Probability')
            plt.title('Class Probabilities', fontsize=14)
            plt.xlim(0, 1.1)
            plt.grid(axis='x', linestyle='--', alpha=0.7)
            for bar, cat in zip(bars, categories):
                if cat == predicted_class:
                    bar.set_color('orange')   # 预测的类别标记为橙色
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.2%}',
                         va='center', ha='left')  # 数值标记
            plt.suptitle(type_name, fontweight='bold')
            plt.tight_layout()

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)   # 生成存储图片的文件夹
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间
            picture_name = f"picture_pred_{type_name}_{current_time}"   # 按任务和时间生成图片名
            save_path = os.path.join(save_dir, picture_name)
            plt.savefig(save_path)
            plt.show()


def main(args):
    image_size = args.image_size    # 图片大小
    device = torch.device(args.device)  # 使用设备
    # ------------------------------
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # ------------------------------
    df = load_test_datasets(args.dir)  # 加载数据集
    pkl_class, pkl_num = analyze_mapping(args.pkl_name, args.dir_path)  # 解析pkl文件
    print(pkl_class)
    print(pkl_num)
    model = ResNeXt_CBAM(pkl_num).to(device)  # 获取模型
    state_dict = torch.load("model_save/model_best.pth",map_location=device)  # 获取权重
    model_state = model.load_state_dict(state_dict)  # 加载权重
    print(model_state)
    loader = pictures_probs_loader(model, df, test_transform, args.task_name_list, args.image_num, device)  # 创建生成器
    plot_picture_pred(loader, pkl_class, pkl_num, args.task_name_list)  # 可视化并保存结果

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="模型测试")
    parser.add_argument("-i", "--image_num", type=int, default=10, help="测试样本数量")
    parser.add_argument("-d", "--dir", type=str, default='.', help="测试数据集所在文件夹")
    parser.add_argument("-n", "--pkl_name", type=str, default='label_mapping.pkl', help="pkl文件名称")
    parser.add_argument("-p", "--dir_path", type=str, default='.', help="pkl文件所在文件夹")
    parser.add_argument("--image_size", type=int, default=224, help="图像尺寸")
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="使用设备")
    parser.add_argument("-t", "--task_name_list", nargs='+',
                        metavar='',
                        default=['geometric shape type', 'natural shape type'],
                        choices=['geometric shape type', 'natural shape type', 'flower type', 'handle type'],
                        help='任务列表')
    args_main = parser.parse_args()
    main(args_main)
