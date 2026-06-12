import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import glob
import pyarrow.parquet as pq
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import random
from PIL import Image
import io
import matplotlib.pyplot as plt
import numpy as np
import pickle
from torchvision import transforms
from datetime import datetime
import torch.nn.functional as F
# ------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TASK_NAME_LIST = ['geometric shape type', 'natural shape type']
IMAGE_SIZE = 224
# ------------------------------

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

###################### 借助大模型实现 ###############################
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            kernel_size=(x.size(-2), x.size(-1))).pow(1.0 / self.p)
#############################################################

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

def load_test_datasets(test_dir='.'):
    if not os.path.exists(test_dir):
        raise ValueError(f'该文件夹不存在: {os.path.abspath(test_dir)}')
    parquet_files = []
    for file in glob.glob(os.path.join(test_dir, '*.parquet')):
        filename = os.path.basename(file)
        if filename.lower().startswith('test_dataset'):
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

def pictures_probs_loader(model,dataframe,transform,task_list,num_pictures):
    sample_indices = random.sample(range(len(dataframe)), num_pictures)
    for idx in sample_indices:
        output_pred = {}
        row = dataframe.iloc[idx]
        # print(row)
        img_bytes = row['image']['bytes']
        image_org = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        image = transform(image_org)
        model.eval()
        with torch.no_grad():
            image = image.unsqueeze(0).to(DEVICE)
            output = model(image)
            for index,type_name in enumerate(task_list):
                probabilities = torch.nn.functional.softmax(output[index], dim=1)[0]
                output_pred[type_name] = probabilities
        yield image_org, output_pred

def analyze_mapping(pkl_name='label_mapping.pkl', dir_path='.'):
    if not os.path.exists(dir_path):
        raise ValueError(f'该文件夹不存在: {os.path.abspath(dir_path)}')
    pkl_path = os.path.join(dir_path, pkl_name)
    with open(pkl_path,'rb') as f:
        pkl_data = pickle.load(f)
    class_dict = {}
    type_list = []
    for type_name, value in pkl_data.items():
        if type_name != "num_classes":
            class_dict[type_name] = {v: k for k, v in value.items()}
            type_list.append(type_name)
        else:
            num_dict = {v: k for v, k in zip(type_list, value)}
    return class_dict, num_dict

def plot_picture_pred(loader, type_idx_class, class_num_dict, task_list, save_dir="save_picture"):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    for img_org, output_pred in loader:
        for type_name in task_list:
            prob = output_pred[type_name]
            idx_class_dict = type_idx_class[type_name]
            num_classes = class_num_dict[type_name]

            predicted_idx = torch.argmax(prob).item()
            predicted_class = idx_class_dict[predicted_idx]
            confidence = prob[predicted_idx].item()

            class_perf = [(idx_class_dict[i], prob[i].item()) for i in range(num_classes)]
            class_perf.sort(key=lambda x: x[1])

            categories = [x[0] for x in class_perf]
            probs = [x[1] for x in class_perf]

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.imshow(np.array(img_org))
            plt.title(f"Predicted: {predicted_class} ({confidence:.2%})", fontsize=14)
            plt.axis('off')

            plt.subplot(1, 2, 2)
            bars = plt.barh(categories, probs, color='skyblue')
            plt.xlabel('Probability')
            plt.title('Class Probabilities', fontsize=14)
            plt.xlim(0, 1.1)
            plt.grid(axis='x', linestyle='--', alpha=0.7)

            for bar, cat in zip(bars, categories):
                if cat == predicted_class:
                    bar.set_color('orange')
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                         f'{width:.2%}',
                         va='center', ha='left')
            plt.suptitle(type_name, fontweight='bold')
            plt.tight_layout()

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            picture_name = f"picture_pred_{type_name}_{current_time}"
            save_path = os.path.join(save_dir, picture_name)
            plt.savefig(save_path)
            plt.show()

def main():
    # ------------------------------
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # ------------------------------
    df = load_test_datasets()
    pkl_class, pkl_num = analyze_mapping()
    print(pkl_class)
    print(pkl_num)
    model = MultiTaskResNet34(pkl_num,pretrained=False).to(DEVICE)
    state_dict = torch.load("model_save/model_best.pth",map_location=DEVICE)
    msg = model.load_state_dict(state_dict)
    print(msg)
    loader = pictures_probs_loader(model,df,test_transform,TASK_NAME_LIST,10)
    plot_picture_pred(loader,pkl_class,pkl_num,TASK_NAME_LIST)

if __name__ == "__main__":
    main()
