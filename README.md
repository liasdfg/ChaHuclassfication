# ChaHuclassfication

## 项目简介

本项目是一个用于紫砂壶图像分类深度学习项目。该项目包含**多任务学习模型**：基于**ResNeXt-50 (32×4d)**，结合 **Convolutional Block Attention Module(CBAM)** 和 **full pre-activation**，实现多个不同角度（几何形状、自然形状、花卉类型、把手类型）的紫砂壶分类

## 数据集说明

- 数据集托管于 Hugging Face Datasets：[AGI-FBHC/ChaHu](https://huggingface.co/datasets/AGI-FBHC/ChaHu)

- 数据集准备工作：将下载的数据集Cn-00000-of-00001.parquet，CN-00000-of-00003.parquet，CN-00001-of-00003.parquet，CN-00002-of-00003.parquet四个文件复制到ChaHu目录下。


## 数据集结构

| 字段 | 类型 | 描述 |
|------|------|------|
| `id` | string | 图像唯一标识符（如 JN000001） |
| `image` | image | 紫砂壶图像 |
| `mask` | image | 图像遮罩，用于提取壶体区域 |
| `caption` | string | 描述文字 |
| `time` | string | 时间信息 |
| `geometric shape type` | string | 几何形状类型 |
| `natural shape type` | string | 自然形状类型 |
| `flower type` | string | 花卉类型 |
| `handle type` | string | 把手类型 |
| `innovative` | string | 是否创新 |


## **项目结构**

```bash
ChaHu/
├── ChaHu/
|	├──Cn-00000-of-00001.parquet
|	├──CN-00000-of-00003.parquet
|	├── CN-00001-of-00003.parquet
|	├── CN-00002-of-00003.parquet
|	├── Cn-00000-of-00001-processed.parquet	# process.py脚本通过mask处理后的文件
|	├── CN-00000-of-00003-processed.parquet 
|	├── CN-00001-of-00003-processed.parquet
|	├── CN-00002-of-00003-processed.parquet
├── model_save/
|	├── model_best.pth
├── save_picture/ 			# 保存model_picture_test.py生成的预测图
├── label_mapping.pkl 		# 标签映射文件
├── process.py   			# 数据预处理脚本
├── main.py      			# 主训练脚本
├── model.py                # 模型文件
├── resnext50_cbam.onnx     # 模型onnx
├── model_picture_test.py 	# 模型测试脚本
├── picture_mask_test.py    # 图像mask处理图示测试脚本
├── multitask_training_curves_20260701_220152.png   # 训练曲线图
├── picture_mask_test_output.png 					# mask处理后图示
├── test_dataset.parquet							# main.py生成的测试集文件
└── README.md
```
## 模型结构

本项目采用多任务图像分类网络，基于**ResNeXt-50 (32×4d)**，结合 **Convolutional Block Attention Module(CBAM)** 和 **full pre-activation**，适用于紫砂壶精细分类任务。
* **模型结构如下**
<img width="1221" height="466" alt="1" src="https://github.com/user-attachments/assets/9206ee5a-f073-474b-aa4b-3abfd7a558c7" />
<img width="1199" height="763" alt="2" src="https://github.com/user-attachments/assets/e2f43d73-004b-4f12-9399-d113c0f80932" />
<img width="1297" height="711" alt="3" src="https://github.com/user-attachments/assets/edcfad28-a81c-48a8-9a91-b81e43a14dd8" />
### 1. 基础网络
* 使用 `torchvision.models.resnet34(pretrained=True)` 作为骨干网络。
* ResNet 的卷积层划分：
  - **layer1**：卷积 + BN + ReLU + MaxPool，提取浅层特征，主要负责边缘、纹理和基础形状信息提取，为后续特征聚合提供基础。
  - **layer2**：ResNet 原 layer1-layer3，提取中层特征，其包含更多语义信息，能够捕捉紫砂壶的细微形态差异，如流把角度、口盖比例。
  - **layer3**：ResNet 原 layer4，提取深层特征，其具备较大感受野，捕捉整体轮廓和器型信息，增强分类判别力。

### 2. SE注意力模块（Squeeze-and-Excitation）
* 考虑到紫砂壶类间差异微小，主要靠轮廓比例、口盖线条、流把弧度区分，且紫砂泥料质感、窑变色泽、包浆光泽等细节信息分散在不同通道，SE 模块能增强重要通道权重，提高判别能力。
* 在 `layer2` 和 `layer3` 后分别加入 SE 模块（`SELayer`），用于自动学习通道权重，突出纹理、颜色和轮廓等重要特征，压制背景噪声、反光或划痕等无用特征。
* 轻量化设计，直接嵌入 ResNet34，不改变主干结构，避免增加过多计算量。

### 3. 全局特征聚合（GeM池化）
* 相比普通平均池化，GeM 可灵活调节不同空间位置的权重，紫砂壶特征分布不均（纹理、色泽、光泽），GeM 有助于捕捉这些重要区域，提升特征判别力。
* 使用 **GeM 池化** (`GeM`) 将卷积特征聚合为全局向量。


### 4. 特征嵌入层
* 提取的全局特征可以通过嵌入层进一步增强判别力，并加上 BatchNorm 和 Dropout，提升训练稳定性和抗过拟合能力。
* 全连接嵌入层： Linear → BatchNorm → ReLU → Dropout
* 输出 512 维特征向量，用于多任务分类


### 5. 多任务分类头
* 紫砂壶分类任务包含多种属性（如几何形状与自然形态），多任务头共享特征提取层，节省训练资源，同时提升特征泛化能力。
* 使用 `nn.ModuleDict` 为每个任务生成独立分类头，每个任务的输出维度对应类别数量。
* 支持同时预测多个任务，如几何形状类型和自然形状类型。


### 6. 模型创新点

1. **SE模块增强判别力**：自动放大纹理、色泽、轮廓等重要通道，抑制无效信息。
2. **GeM池化提高特征聚合能力**：灵活调整空间权重，突出紫砂壶关键特征。
3. **多任务支持**：共享特征提取层，可按需求灵活预测多个任务，提升训练效率和泛化能力。
4. **轻量化与高效性兼顾**：在保持 ResNet34 主干的同时加入轻量模块，兼顾轻量化与高效性。
5. **嵌入层稳定训练**：BatchNorm + ReLU + Dropout 提高训练稳定性，降低小数据集过拟合风险。

## 训练步骤

1. 运行 `process.py`，通过掩码提取紫砂壶图像有效区域，处理后在 `ChaHu` 目录下生成四个新文件：`Cn-00000-of-00001-processed.parquet`、`CN-00000-of-00003-processed.parquet`、`CN-00001-of-00003-processed.parquet`、`CN-00002-of-00003-processed.parquet`。
### 参数配置：
| 参数          | 默认值 | 描述           |
| ------------- | ------ | -------------- |
| dir    | 'ChaHu'    | 数据集所在文件夹       |

   * 提取紫砂壶有效区域效果图如下所示

<img width="3542" height="3627" alt="4" src="https://github.com/user-attachments/assets/8a608f53-9b79-44de-95d8-2b09ffbddb34" />


2. 运行 `main.py`，完成**数据集划分、模型构建、多任务训练**全部流程：

- 对处理后的数据集按照 **76% 训练集、14% 验证集、10% 测试集** 进行划分，以几何形状类型geometric shape为依据执行**分层抽样**，确保各子集类别分布与原始数据集保持一致；

- 基于**ResNeXt-50 (32×4d)**，结合 **Convolutional Block Attention Module(CBAM)** 和 **full pre-activation**，构建紫砂壶多任务分类模型，**可依据任务列表同时完成多个分类任务**：几何形状、自然形状、花卉类型、把手类型；

  项目支持四个并行分类任务：

| 任务         | 描述             | 示例类别                         |
| ------------ | ---------------- | -------------------------------- |
| **几何形状** | 壶的整体几何形态 | 石瓢壶，仿古壶，汉铎壶等         |
| **自然形状** | 模仿自然形态     | 南瓜壶，竹节壶，莲子壶等         |
| **花卉类型** | 花卉装饰图案     | 梅桩壶、供春壶、佛手壶等         |
| **把手类型** | 壶把手的样式     | 三叉提梁壶，单式提梁壶，软提梁壶 |

* 采用**动态任务权重策略**，根据各任务在验证集上的准确率自动调整训练优先级，实现多任务协同优化。

### 训练参数配置：

| 参数          | 默认值 | 描述           |
| ------------- | ------ | -------------- |
| processed_dir    | "ChaHu"  | 处理后数据集所在文件夹 |
|  num_workers   | min(4, os.cpu_count())  |  工作进程数   |
|  image_size   | 224  |   图像尺寸  |
|  batch_size   | 32  |   批次大小  |
|  device   | 'cuda' if torch.cuda.is_available() else 'cpu'  |   使用设备  |
|  task_name_list   |  ['geometric shape type', 'natural shape type'] |  任务列表   |
|   learning_rate  |  0.1 |  学习率   |
|   momentum  |  0.9 |  动量   |
|   weight_decay  |  0.0001 |  权重衰减   |
| num_epochs    | 150  |   训练轮数  |
|  test_size   | 0.1  |  测试集比例   |
|  base_task_weights   |  None |  任务权重   |
|  weight_adjust_method   | 'hybrid'  |  权重调整方法   |


​	3. 运行model_picture_test.py，测试模型效果，输出分类概率柱状图。

### 参数配置：
| 参数          | 默认值 | 描述           |
| ------------- | ------ | -------------- |
| image_num | 10   |  测试样本数量   |
| dir |  '.'  |   测试数据集所在文件夹  |
| pkl_name |  label_mapping.pkl  | pkl文件名称    |
| dir_path |  '.'  |  pkl文件所在文件夹   |
| image_size |  224  |   图像尺寸  |
| device |  'cuda' if torch.cuda.is_available() else 'cpu'  |  使用设备   |
| task_name_list |  ['geometric shape type', 'natural shape type']  |  任务列表   |

### 模型输出

训练完成后会生成：

- `model_save/model_best.pth` - 最佳验证准确率模型
- `multitask_training_curves_20260701_220152.png` - 训练曲线图

## 实验结果 

* **训练效果如下所示**

<img width="727" height="166" alt="运行截图1" src="https://github.com/user-attachments/assets/12f4fb14-b1b0-475f-b9ef-5c46866594a3" />

<img width="1600" height="800" alt="multitask_training_curves_20260701_220152" src="https://github.com/user-attachments/assets/6c71d2ad-d094-4d6a-a6d3-cb38bc54f2a4" />

* **测试结果**

  | 紫砂壶分类头 | 任务准确率 |
  | ------------ | ---------- |
  | **几何形状** | **0.5754**  |
  | **自然形状** | **0.8590**  |

  由于**几何形状**分类头包含紫砂壶壶型最多，包含有30多个类，且对于一些相似形状茶壶较难分辨，所以任务准确率较低，而**自然形状**只有8个左右，所以准确率高很多。

* 下面是抽取的紫砂壶的各类别概率分布
* **几何形状类：**


<img width="1200" height="600" alt="picture_pred_geometric shape type_20260702_170131" src="https://github.com/user-attachments/assets/89b2b8ab-116e-4818-9406-e33946e8d501" />


<img width="1200" height="600" alt="picture_pred_geometric shape type_20260702_170118" src="https://github.com/user-attachments/assets/a22b4fd6-810b-4fea-b7a9-1cde11119e11" />


<img width="1200" height="600" alt="picture_pred_geometric shape type_20260702_170138" src="https://github.com/user-attachments/assets/8970d5a2-f1c2-4aef-9654-296c655e97ac" />




* **自然形状类：**

<img width="1200" height="600" alt="picture_pred_natural shape type_20260702_170153" src="https://github.com/user-attachments/assets/a36b7be1-c88d-487d-b0cf-fd495a94da37" />
<img width="1200" height="600" alt="picture_pred_natural shape type_20260702_170205" src="https://github.com/user-attachments/assets/5adb2111-3259-48c6-9e43-22fa66b64e4a" />
<img width="1200" height="600" alt="picture_pred_natural shape type_20260702_170142" src="https://github.com/user-attachments/assets/c2a8e6fa-8062-4923-832e-081c160093b7" />


## 核心代码说明

### 动态任务权重机制

为解决多任务学习中任务优化不均衡、收敛速度不一致的问题，本项目设计并实现了**基于验证集准确率的动态任务权重策略**，具体实现如下：

1. **动态权重计算**

   以各任务在验证集上的准确率为依据，对表现较差的任务自动分配更高权重。

   首先通过 `1 - acc` 得到任务难度系数，归一化后作为动态修正项；

   再将基础权重与动态项加权融合，得到最终任务权重：

   task_weights=0.7×base_weights+0.3×weight_err

   其中 `base_weights` 初始化为 `[0.5, 0.5]`，保证训练初期稳定。

2. **训练控制策略**

   - **hybrid 混合模式**：采用历史权重与当前权重平滑融合（`0.9×历史 + 0.1×新计算`），使权重更新更平滑、训练更稳定，避免权重剧烈波动。

该机制能够在训练过程中**自动聚焦困难任务**，使四个分类任务均衡优化，显著提升模型整体收敛稳定性与最终分类精度。核心代码如下：

```python
# 用于动态调整任务权重
def dynamic_task_weight(val_accs, base_weights=None):
    if base_weights is None:                                 # 初始化任务权重
        base_weights = [1.0 / len(val_accs)] * len(val_accs)
    val_err = [1 - acc for acc in val_accs]                  # 计算错误率
    weight_err = [w / sum(val_err) for w in val_err]         # 计算错误率占总错误比例(归一化)
    weights = [0.7 * base + 0.3 * err for base, err in zip(base_weights, weight_err)]  # 计算权重
    return weights


# 根据验证准确率动态调整任务权重（从第二个epoch开始）
      if epoch > 0 and use_dynamic_weights:  # 动态权重调整
            if weight_adjust_method == 'accuracy':  # 直接使用函数计算的权重
                task_weights = dynamic_task_weight(current_val_accs, base_task_weights)
            elif weight_adjust_method == 'hybrid':  # 结合上一次的权重来计算新权重
                acc_weights = dynamic_task_weight(current_val_accs, base_task_weights)
                task_weights = [0.9 * w + 0.1 * s for w, s in zip(task_weights, acc_weights)]
```
### CBAM
。。。。。

```python
# 通道注意力
class ChannelAttention(nn.Module):
    def __init__(self,channels,reduction=16):
        super(ChannelAttention, self).__init__()
        self.channels = channels    # 输入通道数
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels//reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels//reduction,channels)
        )

    def forward(self,x):
        x_avg = self.avg_pool(x)
        x_max = self.max_pool(x)
        avg_pool = x_avg.view(x_avg.size(0), -1)
        max_pool = x_max.view(x_max.size(0), -1)
        channel_att_avg = self.shared_mlp(avg_pool)
        channel_att_max = self.shared_mlp(max_pool)
        channel_att_sum = channel_att_avg + channel_att_max
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


# 空间注意力
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7,
                              stride=1, padding=3, dilation=1, groups=1, bias=False)
        self.bn = nn.BatchNorm2d(1, eps=1e-5, momentum=0.01, affine=True)

    def forward(self,x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.conv(x_compress)
        x_out = self.bn(x_out)
        scale = torch.sigmoid(x_out)
        return x * scale


# CBAM模块
class CBAM(nn.Module):
    def __init__(self,channels,reduction=16):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention()

    def forward(self,x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
```


### 多任务分类结构

为充分利用不同分类任务之间的相关性，本项目采用 **共享特征提取 + 独立分类头** 的多任务学习结构。

本项目使用的两个分类任务：

- geometric shape type（几何形状）
- natural shape type（自然形状）

模型共享同一个主干网络，最终通过多个独立分类头完成预测。其可以提高特征利用率，降低模型参数量，增强模型泛化能力，促进不同任务之间的信息共享。核心代码如下：

```python
        # 多任务分类头
        self.heads = nn.ModuleDict()
        for type_name, num in type_len_list.items():
            self.heads[type_name] = nn.Linear(1000, num)
```

```python
        outputs = []
        # 按任务列表计算每个任务的输出
        for type_name in self.type_len_list.keys():
            outputs.append(
                self.heads[type_name](x)
            )
        return tuple(outputs)

```

### 数据增强

```python
# 数据增强
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE + 32, IMAGE_SIZE + 32)),  # 先放大到目标尺寸+32
    transforms.RandomCrop((IMAGE_SIZE, IMAGE_SIZE)),        # 随机裁剪到目标尺寸
    transforms.RandomHorizontalFlip(p=0.5),                 # 随机水平翻转（50%概率）
    transforms.RandomVerticalFlip(p=0.2),                   # 随机垂直翻转（20%概率）
    transforms.RandomRotation(20),                          # 随机旋转±20度
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
    transforms.ToTensor(),                                  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化（ImageNet均值/标准差）
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),            # 直接resize到目标尺寸
    transforms.ToTensor(),                                  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])
```

### SGD（动量 + L2正则） + StepLR阶梯式学习率衰减

```python
    criterion = nn.CrossEntropyLoss()     # 构建损失函数
    optimizer = optim.SGD(model.parameters(),lr=args.learning_rate,momentum=args.momentum,
                          weight_decay=args.weight_decay)   # 定义优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)  # 定义调度器

```
## 借助大模型实现的代码说明
### picture_mask_test.py
无
### process.py
无
### model.py
无
### main.py
无
### model_picture_test.py
无

