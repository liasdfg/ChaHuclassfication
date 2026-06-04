import os.path
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from datasets import load_dataset
import sys
import json
import psutil
import platform
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm.auto import tqdm
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader, Dataset
class ChaHuDataSet(Dataset):
    def __init__(self,dataset,label_class="geometric shape type",transform=None):
        self.dataset = dataset
        self.transform = transform
        self.label_class = label_class
        self.classes_list = sorted(set([shape.strip() for shape in self.dataset.unique(self.label_class) if shape]))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes_list)}
        self.idx_to_class = {i: cls_name for i, cls_name in enumerate(self.classes_list)}
        self.normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        # print(self.class_to_idx)
        print(self.idx_to_class)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        sample = self.dataset[item]
        image = sample['image'].convert('RGB')
        mask = sample['mask'].convert('L')
        label = sample[self.label_class].strip()
        image_np = np.array(image)
        mask_np = np.array(mask) / 255.0
        if self.transform:
            transformed = self.transform(image=image_np,mask=mask_np)
            image_np = transformed['image']
            mask_np = transformed['mask']
        # 使用mask去除背景
        res_image = (image_np * mask_np[..., np.newaxis]).astype(np.uint8)
        res_image = self.normalize(image=res_image)['image']
        res_image = np.transpose(res_image, (2, 0, 1)) #维度转化,转化成深度学习需要的维度
        label_idx = self.class_to_idx[label]
        image_tensor = torch.tensor(res_image,dtype=torch.float32)
        label_tensor = torch.tensor(label_idx,dtype=torch.long)
        return image_tensor,label_tensor

def unify_label(train_data_set,val_data_set,label_class="geometric shape type"):
    train_label = set(train_data_set[label_class])
    val_label = set(val_data_set[label_class])
    common_label = train_label & val_label
    def replace_label(sample):
        if sample[label_class] not in common_label:
            sample[label_class] = "其他"
        return sample
    train_data_set = train_data_set.map(replace_label)
    val_data_set = val_data_set.map(replace_label)
    return train_data_set,val_data_set

def get_model(model_name,class_num): #获取所需模型
    if model_name=="resnet34":
        model = models.resnet34(pretrained=True)
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in,class_num)
    elif model_name=="googlenet":
        model = models.googlenet(pretrained=True)
        num_fc_in = model.fc.in_features
        model.fc = nn.Linear(num_fc_in,class_num)
    elif model_name=="vgg19":
        model = models.vgg19(pretrained=True)
        num_in = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_in,class_num)
    else:
        model = None
        print("不支持", {model_name})
    return model

def find_batch_size(model,device,start_size=1):
    if device=="cpu":
        available_memory = psutil.virtual_memory().available
        if available_memory > 16 * 1024 * 1024 * 1024:  # 16GB+
            batch_size = 32
        elif available_memory > 8 * 1024 * 1024 * 1024:  # 8GB+
            batch_size = 16
        elif available_memory > 4 * 1024 * 1024 * 1024:  # 4GB+
            batch_size = 8
        else:
            batch_size = 4
        return batch_size
    else:
        batch_size = start_size
        while True:
            try:
                x = torch.randn(batch_size,3,224,224,device=device)
                model.train()
                y = model(x)
                if isinstance(y,tuple): #googlenet会输出元组
                    y = y[0]
                loss = y.mean()
                loss.backward()
                model.zero_grad()
                del x,y,loss
                batch_size *= 2
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    return batch_size//2
                break

def get_num_workers(): #得到核心数
    core_num = psutil.cpu_count(logical=False)  # 获取物理核心数
    if platform.system() == 'Windows':
        return min(4,core_num//2)
    else:
        return min(8,core_num)

def train_model(model, train_dataloader, criterion, optimizer, device):
    model.train()
    current_loss = 0.0
    correct_num = 0
    total_num = 0
    progress_bar = tqdm(train_dataloader,desc="模型训练",file=sys.stdout) #进度条对象
    for input_data,label in progress_bar:
        input_data,label = input_data.to(device),label.to(device)
        optimizer.zero_grad()
        output = model(input_data)
        if isinstance(output, tuple): #处理googlenet输出的元组
            loss = criterion(output[0],label)+0.3*criterion(output[1],label)+0.3*criterion(output[2],label)
            output = output[0]
        else:
            loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        current_loss += loss.cpu().item()
        _, pred = output.max(1)
        total_num += label.size(0)
        correct_num += pred.eq(label).sum().cpu().item()
        progress_bar.set_postfix({
            'train_loss': f'{current_loss / (progress_bar.n + 1):.4f}',
            'train_acc': f'{100. * correct_num / total_num:.2f}%'
        })
    return current_loss/len(train_dataloader),correct_num/total_num

def validate_model(model,val_dataloader,criterion,device):
    model.eval()
    current_loss = 0.0
    correct_num = 0
    total_num = 0
    with torch.no_grad():
        progress_bar = tqdm(val_dataloader,desc="模型验证",file=sys.stdout) #进度条对象
        for input_data,label in progress_bar:
            input_data,label = input_data.to(device),label.to(device)
            output = model(input_data)
            loss = criterion(output,label)
            current_loss += loss.cpu().item()
            _, pred = output.max(1)
            total_num += label.size(0)
            correct_num += pred.eq(label).sum().cpu().item()
            progress_bar.set_postfix({
                'val_loss': f'{current_loss / (progress_bar.n + 1):.4f}',
                'val_acc': f'{100. * correct_num / total_num:.2f}%'
            })
    return current_loss/len(val_dataloader),correct_num/total_num

if __name__ == "__main__":
    # albumentations 不对mask产生颜色转变
    train_transform = A.Compose([
        A.Resize(height=224, width=224),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Perspective(scale=(0.05, 0.1), p=0.4),
        A.RandomBrightnessContrast(p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.5),
        A.RGBShift(p=0.5)
    ])
    val_transform = A.Compose(
        [
            A.Resize(height=224, width=224)
        ]
    )
    class_type = "natural shape type"
    pot_cn_dataset = load_dataset("./ChaHu", split="CN")
    train_test_data = pot_cn_dataset.train_test_split(test_size=0.2, seed=18)
    train_dataset, val_dataset = unify_label(train_test_data['train'],train_test_data['test'],class_type)
    print(len(train_dataset))
    print(len(val_dataset))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("使用设备为",device)

    train_data = ChaHuDataSet(train_dataset,class_type,train_transform)
    val_data = ChaHuDataSet(val_dataset,class_type,val_transform)
    if train_data.class_to_idx != val_data.class_to_idx:
        sys.exit("标签字典错误")
    label_map = train_data.idx_to_class
    map_path = "./label_json/label_map(" + class_type + ").json"
    print(map_path)
    with open(map_path, 'w') as f:
        json.dump(label_map, f)
    model = get_model("resnet34",len(train_data.classes_list))
    model = model.to(device)
    batch_size = 100
    num_worker = get_num_workers()
    print("batch_size:",batch_size)
    print("num_worker:",num_worker)
    train_loader = DataLoader(train_data,batch_size=batch_size,num_workers=num_worker,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=batch_size,num_workers=num_worker,shuffle=False)

    lr = 0.00001
    epochs = 50
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    best_val_acc = 0.0
    for epoch in range(epochs):
        print("%2d/%2d"%(epoch,epochs))
        print("*"*20)
        train_loss,train_acc = train_model(model,train_loader,criterion,optimizer,device)
        val_loss,val_acc = validate_model(model,val_loader,criterion,device)
        print("train_loss:%4f,train_acc:%.2f%%"%(train_loss,train_acc * 100))
        print("val_loss:%4f,val_acc:%.2f%%"%(val_loss, val_acc * 100))
        scheduler.step(val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),f"model_save/best_model({class_type}).pt")
            print("最佳模型更新")





