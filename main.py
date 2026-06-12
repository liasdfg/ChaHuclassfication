import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import glob
import pyarrow.parquet as pq
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import pickle
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import torch
import torch.nn as nn
import time
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F
# ---------------------------------------
IMAGE_SIZE = 224
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TASK_NAME_LIST = ['geometric shape type', 'natural shape type']
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 30
TEST_SIZE = 0.1
# ---------------------------------


class ChaHuDataset(Dataset):
    def __init__(self, dataframe, label_dict, transform=None):  # 使用不为label_mapping产生的字典
        self.dataframe = dataframe
        self.label_dict = label_dict
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        # print(row)
        img_bytes = row['image']['bytes']
        image = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        output_list = []
        for type_name in self.label_dict:
            label = self.label_dict[type_name].get(row[type_name], 0)
            output_list.append(label)
        if self.transform:
            image = self.transform(image)
        output_tuple = tuple(output_list)
        # print(output_tuple)
        return image, output_tuple


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

############ 这段代码借助大模型实现 #####################
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            kernel_size=(x.size(-2), x.size(-1))).pow(1.0 / self.p)
##############################################################

class MultiTaskResNet34(nn.Module):
    def __init__(self, type_len_list, pretrained=True):
        super().__init__()
        self.type_len_list = type_len_list
        resnet = models.resnet34(pretrained=pretrained)
        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
            )
        self.layer2 = nn.Sequential(
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        self.layer3 = resnet.layer4
        self.se1 = SELayer(256)
        self.se2 = SELayer(512)
        self.gem = GeM()
        self.embedding = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.heads = nn.ModuleDict()
        for type_name, num in type_len_list.items():
            self.heads[type_name] = nn.Linear(512, num)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.se1(x)
        x = self.layer3(x)
        x = self.se2(x)
        x = self.gem(x)
        x = torch.flatten(x, 1)
        x = self.embedding(x)
        outputs = []
        for type_name in self.type_len_list.keys():
            outputs.append(
                self.heads[type_name](x)
            )
        return tuple(outputs)

def load_processed_datasets(processed_dir='ChaHu'):
    if not os.path.exists(processed_dir):
        raise ValueError(f'该文件夹不存在: {os.path.abspath(processed_dir)}')
    parquet_files = []
    for file in glob.glob(os.path.join(processed_dir, '*.parquet')):
        filename = os.path.basename(file)
        if filename.lower().startswith('cn') and '-processed' in filename.lower():
            parquet_files.append(file)
    print(f'找到{len(parquet_files)}个符合要求的文件:')
    df_list = []
    for f in parquet_files:
        df = pq.read_table(f).to_pandas()
        df_list.append(df)
        print(f'已加载 {os.path.basename(f)}: {len(df)}条记录')

    combined_df = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    # print(combined_df)
    print(f'总记录数: {len(combined_df)}')
    return combined_df


def filter_by_id_range(df, start_id, end_id):  # 输入JN开头的id
    def extract_id_number(id_str):
        match = re.match(r'JN(\d+)', str(id_str))
        return int(match.group(1)) if match else None

    start_num = extract_id_number(start_id)
    end_num = extract_id_number(end_id)
    print(f'筛选ID范围: {start_id} ({start_num}) - {end_id} ({end_num})')
    df['id_num'] = df['id'].apply(extract_id_number)
    filtered_df = df[(df['id_num'] >= start_num) & (df['id_num'] <= end_num)]
    filtered_df = filtered_df.drop('id_num', axis=1)
    print(f'筛选前记录数: {len(df)}')
    print(f'筛选后记录数: {len(filtered_df)}')
    return filtered_df


def struct_type_dict(dataframe, type_list, min_num=100):
    output_dict = {}
    output_len = {}
    for type_name in type_list:
        dataframe.loc[:, type_name] = dataframe[type_name].str.rstrip('\t')
        class_dist = dataframe[type_name].value_counts()
        rare_class = class_dist[class_dist <= min_num].index.tolist()
        if rare_class:
            dataframe = dataframe[~dataframe[type_name].isin(rare_class)]
            class_dist = dataframe[type_name].value_counts()
        class_names = class_dist.index.tolist()
        class_num = len(class_names)
        class_name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        output_dict[type_name] = class_name_to_idx
        output_len[type_name] = class_num
    dataframe = dataframe.reset_index(drop=True)
    print(f'剩余记录数: {len(dataframe)}')
    return dataframe, output_dict, output_len


def split_dataset(dataframe, test_size, stratify_type):
    try:
        train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=42,
                                             stratify=dataframe[stratify_type])
        print('分层抽样训练集和测试集')
    except ValueError:
        train_df, test_df = train_test_split(dataframe, test_size=test_size, random_state=42)
        print('随机划分训练集和测试集')
    try:
        train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42, stratify=train_df[stratify_type])
        print('分层抽样产生验证集')
    except ValueError:
        train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)
        print('随机划分产生验证集')
    print(f'训练集: {len(train_df)} 条记录')
    print(f'验证集: {len(val_df)} 条记录')
    print(f'测试集: {len(test_df)} 条记录')
    test_df.to_parquet('test_dataset.parquet', index=False)
    print("已保存测试集为test_dataset.parquet")
    return train_df, val_df


def label_mapping(name_idx_dist, len_dist):
    label_dist = {}
    len_list = []
    for type_name in name_idx_dist:
        label_dist[type_name] = name_idx_dist[type_name]
        len_list.append(len_dist[type_name])
    label_dist['num_classes'] = tuple(len_list)
    with open('label_mapping.pkl', 'wb') as f:
        pickle.dump(label_dist, f)
    print("标签映射已保存至label_mapping.pkl")
    return label_dist


def dynamic_task_weight(val_accs, base_weights=[0.5, 0.5]):
    inv_accs = [1 - acc for acc in val_accs]
    inv_accs = [w / sum(inv_accs) for w in inv_accs]
    weights = [0.7 * base + 0.3 * inv for base, inv in zip(base_weights, inv_accs)]
    return weights


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, tasks,
                class_weights=None, use_dynamic_weights=True, weight_adjust_method='hybrid'):
    train_losses = []
    val_losses = []
    train_accs = {task: [] for task in tasks}
    val_accs = {task: [] for task in tasks}
    best_acc = 0.0
    best_model_path = 'model_save/model_best.pth'
    os.makedirs('model_save', exist_ok=True)

    num_tasks = len(tasks)
    task_weights = [1.0 / num_tasks] * num_tasks

############# 转化为字典的思路由大模型提供 #######################
    criterions = {}
    for task in tasks:
        if class_weights is not None and task in class_weights:
            weight_tensor = torch.tensor(class_weights[task], dtype=torch.float32).to(DEVICE)
            criterions[task] = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterions[task] = criterion
#################################################################

    for epoch in range(num_epochs):
        start_time = time.time()
        # ----------------------------------------------
        model.train()
        total_loss = 0.0
        correct = {task: 0 for task in tasks}
        total = {task: 0 for task in tasks}
        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = [l.to(DEVICE) for l in labels]
            optimizer.zero_grad()
            outputs = model(images)

            # print(outputs)

            train_loss = 0.0
            for i, task in enumerate(tasks):
                task_loss = criterions[task](outputs[i], labels[i])
                train_loss += task_weights[i] * task_loss

            train_loss.backward()
            optimizer.step()

            total_loss += train_loss.item() * images.size(0)
            for name, out, lbl in zip(tasks, outputs, labels):
                _, pred = torch.max(out.data, 1)
                total[name] += lbl.size(0)
                correct[name] += (pred == lbl).sum().item()

        avg_loss = total_loss / len(train_loader.dataset)

        for task in tasks:
            train_accs[task].append(correct[task] / total[task])

        scheduler.step()
        # ----------------------------------------------
        model.eval()
        val_total_loss = 0.0
        val_correct = {task: 0 for task in tasks}
        val_total = {task: 0 for task in tasks}

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = [l.to(DEVICE) for l in labels]
                outputs = model(images)

                val_loss = 0.0
                for i, task in enumerate(tasks):
                    task_loss = criterions[task](outputs[i], labels[i])
                    val_loss += task_weights[i] * task_loss

                val_total_loss += val_loss.item() * images.size(0)

                for name, out, lbl in zip(tasks, outputs, labels):
                    _, pred = torch.max(out.data, 1)
                    val_total[name] += lbl.size(0)
                    val_correct[name] += (pred == lbl).sum().item()

        avg_val_loss = val_total_loss / len(val_loader.dataset)

        current_val_accs = []
        for task in tasks:
            acc = val_correct[task] / val_total[task]
            val_accs[task].append(acc)
            current_val_accs.append(acc)

        if epoch > 0 and use_dynamic_weights:
            if weight_adjust_method == 'accuracy':
                task_weights = dynamic_task_weight(current_val_accs)
            elif weight_adjust_method == 'hybrid':
                acc_weights = dynamic_task_weight(current_val_accs)
                task_weights = [0.9 * w + 0.1 * s for w, s in zip(task_weights, acc_weights)]
        # -------------------------------------------------------------------
        epoch_time = time.time() - start_time
        print(f'Epoch [{epoch + 1}/{num_epochs}], Time: {epoch_time:.2f}s')
        print(f'  Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        weight_str = ", ".join([f"{task}={task_weights[i]:.4f}" for i, task in enumerate(tasks)])
        print(f'  Task Weights: {weight_str}')
        for task in tasks:
            print(f'  {task}: Train Acc={train_accs[task][-1]:.4f}, Val Acc={val_accs[task][-1]:.4f}')

        avg_val_acc = sum(current_val_accs) / len(tasks)

        if avg_val_acc > best_acc:
            best_acc = avg_val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f'Best model saved with avg val acc: {best_acc:.4f}')

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)

    return model, train_losses, val_losses, train_accs, val_accs


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, task_list):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f'multitask_training_curves_{current_time}.png'
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
    plt.savefig(save_path)
    print(f"已保存为{save_path}")
    plt.show()


def main():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),
        transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # -------------------------------------------
    df = load_processed_datasets()
    df, dict1, len1 = struct_type_dict(df, TASK_NAME_LIST, 20)
    print(dict1)
    print(len1)
    train_df, val_df = split_dataset(df, TEST_SIZE, "geometric shape type")
    train_dataset = ChaHuDataset(train_df, dict1, train_transform)
    val_dataset = ChaHuDataset(val_df, dict1, val_test_transform)

    num_workers = min(4, os.cpu_count())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)
    map_label = label_mapping(dict1, len1)
    print(map_label)
    model = MultiTaskResNet34(len1).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    _, t_losses, v_losses, t_accs, v_accs = train_model(model, train_loader, val_loader, criterion, optimizer,
                                                        scheduler, NUM_EPOCHS, TASK_NAME_LIST)
    plot_training_curves(t_losses, v_losses, t_accs, v_accs, TASK_NAME_LIST)


if __name__ == '__main__':
    main()