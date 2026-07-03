import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import glob
import pyarrow.parquet as pq
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import torch
import torch.nn as nn
import time
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
from model import ResNeXt_CBAM
import argparse
torch.backends.cudnn.benchmark = True
# ---------------------------------------
# IMAGE_SIZE = 224
# BATCH_SIZE = 32
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# TASK_NAME_LIST = ['geometric shape type', 'natural shape type']
# LEARNING_RATE = 0.1
# MOMENTUM = 0.9
# WEIGHT_DECAY = 0.0001
# NUM_EPOCHS = 150
# TEST_SIZE = 0.1
# ---------------------------------


# 数据集
class ChaHuDataset(Dataset):
    def __init__(self, dataframe, label_dict, transform=None):
        self.dataframe = dataframe
        self.label_dict = label_dict  # 任务类别名索引字典
        self.transform = transform    # 变换

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        # print(row)
        img_bytes = row['image']['bytes']   # 获取图像数据
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')  # 模式转换
        output_list = []
        for task_name in self.label_dict:
            label = self.label_dict[task_name].get(row[task_name])  # 获取对应类别名称的索引
            output_list.append(label)   # 添加索引到列表
        if self.transform:
            image = self.transform(image)  # 应用变换
        output_tuple = tuple(output_list)  # 转化为元组
        # print(output_tuple)
        return image, output_tuple


# 数据读取
def load_processed_datasets(processed_dir='ChaHu'):
    if not os.path.exists(processed_dir):  # 检查文件夹存在
        raise ValueError(f'该文件夹不存在: {os.path.abspath(processed_dir)}')
    parquet_files = []
    for file in glob.glob(os.path.join(processed_dir, '*.parquet')):
        filename = os.path.basename(file)   # 获取文件名
        if filename.lower().startswith('cn') and '-processed' in filename.lower():
            parquet_files.append(file)      # 存储符合要求的路径
    print(f'找到{len(parquet_files)}个符合要求的文件:')
    df_list = []
    for f in parquet_files:
        df = pq.read_table(f).to_pandas()   # 读取parquet并转化为DataFrame
        df_list.append(df)
        print(f'已加载 {os.path.basename(f)}: {len(df)}条记录')

    combined_df = pd.concat(df_list, ignore_index=True)  # 合并DataFrame
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)  # 打乱数据并重置索引
    # print(combined_df)
    print(f'总记录数: {len(combined_df)}')
    return combined_df


# 数据清洗，构建类别映射字典、类别数量字典
def struct_type_dict(dataframe, task_list, min_num=10):
    output_dict = {}  # 存储类别到索引的映射
    output_len = {}   # 存储任务的类别数量
    for task_name in task_list:
        dataframe.loc[:, task_name] = dataframe[task_name].str.rstrip('\t')   # 处理不合规的类别名
        class_dist = dataframe[task_name].value_counts()                      # 统计各类别样本数量
        rare_class = class_dist[class_dist <= min_num].index.tolist()         # 少于一定数量的视为罕见类别
        if rare_class:
            dataframe = dataframe[~dataframe[task_name].isin(rare_class)]    # 去除罕见类别行
            class_dist = dataframe[task_name].value_counts()                 # 统计各类别样本数量
        class_names = class_dist.index.tolist()        # 获取类别名称列表
        class_num = len(class_names)                   # 获取类别数量
        class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}  # 构建名称索引字典
        output_dict[task_name] = class_name_to_idx     # 按任务存入字典
        output_len[task_name] = class_num
    dataframe = dataframe.reset_index(drop=True)   # 重置索引
    print(f'剩余记录数: {len(dataframe)}')
    return dataframe, output_dict, output_len


# 划分数据集
def split_dataset(dataframe, test_size, stratify_type):
    # 分层抽样划分训练集测试集
    try:
        train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=42,
                                             stratify=dataframe[stratify_type])
        print('分层抽样训练集和测试集')
    # 随机抽样划分训练集测试集
    except ValueError:
        train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=42)
        print('随机划分训练集和测试集')
    # 分层抽样产生验证集
    try:
        train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df[stratify_type])
        print('分层抽样产生验证集')
    # 随机划分产生验证集
    except ValueError:
        train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)
        print('随机划分产生验证集')
    print(f'训练集: {len(train_df)} 条记录')
    print(f'验证集: {len(val_df)} 条记录')
    print(f'测试集: {len(test_df)} 条记录')
    test_df.to_parquet('test_dataset.parquet', index=False)  # 测试集保存为Parquet
    print("已保存测试集为test_dataset.parquet")
    return train_df, val_df


# 保存类别映射与数量字典
def label_mapping(name_idx_dist, len_dist):
    label_dist = {}
    len_list = []
    for type_name in name_idx_dist:
        label_dist[type_name] = name_idx_dist[type_name]  # 保存名称索引映射字典
        len_list.append(len_dist[type_name])     # 保存类别数量
    label_dist['num_classes'] = tuple(len_list)  # 存为num_classes的键
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_dist, f)               # 保存为pkl文件
    print("标签映射已保存至label_mapping.pkl")
    return label_dist


# 用于动态调整任务权重
def dynamic_task_weight(val_accs, base_weights=None):
    if base_weights is None:                                 # 初始化任务权重
        base_weights = [1.0 / len(val_accs)] * len(val_accs)
    val_err = [1 - acc for acc in val_accs]                  # 计算错误率
    weight_err = [w / sum(val_err) for w in val_err]         # 计算错误率占总错误比例
    weights = [0.7 * base + 0.3 * err for base, err in zip(base_weights, weight_err)]  # 计算权重
    return weights


# 模型训练
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, tasks,
                device, class_weights=None, base_task_weights=None, weight_adjust_method='hybrid',
                use_dynamic_weights=True):
    # 存储损失
    train_losses = []
    val_losses = []
    # 存储准确率
    train_accs = {task: [] for task in tasks}
    val_accs = {task: [] for task in tasks}
    best_acc = 0.0
    best_model_path = 'model_save/model_best.pth'  # 模型保存位置
    os.makedirs('model_save', exist_ok=True)       # 创建保存模型的文件夹

    num_tasks = len(tasks)                          # 计算任务数量
    task_weights = [1.0 / num_tasks] * num_tasks    # 计算初始的任务权重

    criterions = {}
    for task in tasks:
        if class_weights is not None and task in class_weights:  # 存在类别权重则构建加权CrossEntropyLoss
            weight_tensor = torch.tensor(class_weights[task], dtype=torch.float32).to(device)
            criterions[task] = nn.CrossEntropyLoss(weight=weight_tensor)
        else:  # 无类别权重则使用普通CrossEntropyLoss
            criterions[task] = criterion

    for epoch in range(num_epochs):
        start_time = time.time()   # 记录开始时间
        # ----------------------------------------------
        model.train()       # 设置为训练模式
        total_loss = 0.0
        correct = {task: 0 for task in tasks}
        total = {task: 0 for task in tasks}
        for images, labels in train_loader:
            images = images.to(device)
            labels = [l.to(device) for l in labels]  # 获取标签
            optimizer.zero_grad()
            outputs = model(images)

            # print(outputs)

            train_loss = 0.0
            for i, task in enumerate(tasks):
                task_loss = criterions[task](outputs[i], labels[i])  # 计算每个任务的损失
                train_loss += task_weights[i] * task_loss            # 基于任务权重计算训练损失

            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item() * images.size(0)     # 计算总损失
            for name, out, lbl in zip(tasks, outputs, labels):   # 统计正确预测的数量
                _, pred = torch.max(out.data, 1)                 # 获取预测类别
                total[name] += lbl.size(0)                       # 获取样本数
                correct[name] += (pred == lbl).sum().item()      # 获取正确预测样本数

        avg_loss = total_loss / len(train_loader.dataset)        # 计算平均损失

        for task in tasks:
            train_accs[task].append(correct[task] / total[task])  # 计算每个任务的准确率

        scheduler.step()    # 更新学习率
        # ----------------------------------------------
        model.eval()        # 调整为评估模式
        val_total_loss = 0.0
        val_correct = {task: 0 for task in tasks}
        val_total = {task: 0 for task in tasks}

        with torch.no_grad():   # 关闭梯度计算
            for images, labels in val_loader:
                images = images.to(device)
                labels = [l.to(device) for l in labels]
                outputs = model(images)

                val_loss = 0.0
                for i, task in enumerate(tasks):
                    task_loss = criterions[task](outputs[i], labels[i])  # 计算每个任务的损失
                    val_loss += task_weights[i] * task_loss              # 基于任务权重计算训练损失

                val_total_loss += val_loss.item() * images.size(0)     # 计算总损失

                for name, out, lbl in zip(tasks, outputs, labels):   # 统计正确预测的数量
                    _, pred = torch.max(out.data, 1)                 # 获取预测类别
                    val_total[name] += lbl.size(0)                   # 获取样本数
                    val_correct[name] += (pred == lbl).sum().item()  # 获取正确预测样本数

        avg_val_loss = val_total_loss / len(val_loader.dataset)

        current_val_accs = []
        for task in tasks:
            acc = val_correct[task] / val_total[task]  # 计算每个任务的准确率
            val_accs[task].append(acc)
            current_val_accs.append(acc)

        if epoch > 0 and use_dynamic_weights:  # 动态权重调整
            if weight_adjust_method == 'accuracy':  # 直接使用函数计算的权重
                task_weights = dynamic_task_weight(current_val_accs, base_task_weights)
            elif weight_adjust_method == 'hybrid':  # 结合上一次的权重来计算新权重
                acc_weights = dynamic_task_weight(current_val_accs, base_task_weights)
                task_weights = [0.9 * w + 0.1 * s for w, s in zip(task_weights, acc_weights)]
        # -------------------------------------------------------------------
        epoch_time = time.time() - start_time   # 计算一轮耗时
        print(f'Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f}s')
        print(f'  Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        weight_str = ", ".join([f"{task}={task_weights[i]:.4f}" for i, task in enumerate(tasks)])
        print(f'  Task Weights: {weight_str}')
        for task in tasks:
            print(f'  {task}: Train Acc={train_accs[task][-1]:.4f}, Val Acc={val_accs[task][-1]:.4f}')

        avg_val_acc = sum(current_val_accs) / len(tasks)

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), best_model_path)  # 保存模型参数
            print(f'最佳模型平均准确率: {best_acc:.4f}')

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

    return model, train_losses, val_losses, train_accs, val_accs


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, task_list):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")  # 获取当前时间
    save_name = f'multitask_training_curves_{current_time}.png'  # 设置保存的名称
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.subplot(1, 2, 2)
    for name in task_list:
        plt.plot(train_accs[name], label=f'{name} Train')
        plt.plot(val_accs[name], label=f'{name} Val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name)
    print(f"已保存为{save_name}")
    plt.show()


def main(args):
    image_size = args.image_size         # 图片大小
    device = torch.device(args.device)   # 使用类型
    # 训练集变换
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # 验证集变换
    val_test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # -------------------------------------------
    df = load_processed_datasets(args.processed_dir)   # 读取数据集
    df, dict_type_name, dict_name_num = struct_type_dict(df, args.task_name_list, 20)  # 清洗数据并构建映射
    print(dict_type_name)
    print(dict_name_num)
    train_df, val_df = split_dataset(df, args.test_size, args.task_name_list[0])   # 划分数据集
    train_dataset = ChaHuDataset(train_df, dict_type_name, train_transform)
    val_dataset = ChaHuDataset(val_df, dict_type_name, val_test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    map_label = label_mapping(dict_type_name, dict_name_num)  # 保存映射pkl
    print(map_label)
    model = ResNeXt_CBAM(dict_name_num).to(device)    # 获取模型

    criterion = nn.CrossEntropyLoss()     # 构建损失函数
    optimizer = optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum,
                          weight_decay=args.weight_decay)   # 定义优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 定义调度器

    _, t_losses, v_losses, t_accs, v_accs = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                        scheduler, args.num_epochs, args.task_name_list, device,
                                                        base_task_weights=args.base_task_weights,
                                                        weight_adjust_method=args.weight_adjust_method)  # 训练
    plot_training_curves(t_losses, v_losses, t_accs, v_accs, args.task_name_list)  # 可视化并保存训练曲线


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="模型训练")
    parser.add_argument("-p", "--processed_dir", type=str, default="ChaHu", help="处理后数据集所在文件夹")
    parser.add_argument("--num_workers", type=int, default=min(4, os.cpu_count()), help="工作进程数")
    parser.add_argument("--image_size", type=int, default=224, help="图像尺寸")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="批次大小")
    parser.add_argument("-d", "--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="使用设备")
    parser.add_argument("-t", "--task_name_list", nargs='+',
                        metavar='',
                        default=['geometric shape type', 'natural shape type'],
                        choices=['geometric shape type', 'natural shape type', 'flower type', 'handle type'],
                        help='任务列表')
    parser.add_argument("-l", "--learning_rate", type=float, default=0.1, help="学习率")
    parser.add_argument("-m", "--momentum", type=float, default=0.9, help="动量")
    parser.add_argument("-w", "--weight_decay", type=float, default=0.0001, help="权重衰减")
    parser.add_argument("-n", "--num_epochs", type=int, default=150, help="训练轮数")
    parser.add_argument("--test_size", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--base_task_weights", nargs='+', type=float, default=None, help="任务权重")
    parser.add_argument("--weight_adjust_method", type=str, default='hybrid', choices=['accuracy', 'hybrid'],
                        help='权重调整方法')
    args_main = parser.parse_args()
    main(args_main)


