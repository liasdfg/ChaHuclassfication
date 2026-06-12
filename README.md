# ChaHuclassfication

## 项目简介

本项目是一个用于紫砂壶图像分类深度学习项目。该项目包含**多任务学习模型**：基于**ResNet-34**，结合 **SE注意力模块** 和 **GeM池化**，实现多个不同角度（几何形状、自然形状、花卉类型、把手类型）的紫砂壶分类

## 数据集说明

- 数据集托管于 Hugging Face Datasets：[AGI-FBHC/ChaHu](https://huggingface.co/datasets/AGI-FBHC/ChaHu)

- 数据集准备工作：将下载的数据集cn-00000-of-00001.parquet，CN-00000-of-00003.parquet，CN-00001-of-00003.parquet，CN-00002-of-00003.parquet三个文件复制到ChaHu目录下。


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
├── data_split # 分隔训练集和测试集
├── image      # readme使用到的一些图片
├── image_save # 保存的训练结果
├── model_save # 保存的模型 
├── process/
|	├── CN-00000-of-00003-new.parquet # process.py脚本通过mask处理后的文件
|	├── CN-00001-of-00003-new.parquet
|	├── CN-00002-of-00003-new.parquet
├── vis_results   # 保存的测试结果
├── main.py       # 主训练脚本（多任务SE-ResNet）
├── process.py    # 数据预处理脚本
├── README.md
├── test_model.py #数据集测试脚本
└── test_pic.py   #图像mask处理图示测试脚本
```
## 模型结构

本项目采用多任务图像分类网络，基于 **ResNet34**，结合 **SE注意力模块** 和 **GeM池化**，适用于紫砂壶精细分类任务。整体模型结构如下：

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

1. 运行 `process.py`，通过掩码提取紫砂壶图像有效区域，处理后在 `ChaHu` 目录下生成四个新文件：`cn-00000-of-00001-processed.parquet`、`CN-00000-of-00003-processed.parquet`、`CN-00001-of-00003-processed.parquet`、`CN-00002-of-00003-processed.parquet`。

   * 提取紫砂壶有效区域效果图如下所示

 <img width="3569" height="3648" alt="picture_mask" src="https://github.com/user-attachments/assets/02567355-73e0-4e29-a2c3-16acee711ab6" />


2. 运行 `main.py`，完成**数据集划分、模型构建、多任务训练**全部流程：

- 对处理后的数据集按照 **76% 训练集、14% 验证集、10% 测试集** 进行划分，以几何形状类型geometric shape为依据执行**分层抽样**，确保各子集类别分布与原始数据集保持一致；

- 基于 **ResNet34**，结合 **SE注意力模块** 和 **GeM池化**，构建紫砂壶多任务分类模型，**可依据任务列表同时完成多个分类任务**：几何形状、自然形状、花卉类型、把手类型；

  项目支持四个并行分类任务：

| 任务         | 描述             | 示例类别                         |
| ------------ | ---------------- | -------------------------------- |
| **几何形状** | 壶的整体几何形态 | 石瓢壶，仿古壶，汉铎壶等         |
| **自然形状** | 模仿自然形态     | 南瓜壶，竹节壶，莲子壶等         |
| **花卉类型** | 花卉装饰图案     | 梅桩壶、供春壶、佛手壶等         |
| **把手类型** | 壶把手的样式     | 三叉提梁壶，单式提梁壶，软提梁壶 |

* 采用**动态任务权重策略**，根据各任务在验证集上的准确率自动调整训练优先级，实现多任务协同优化。

### 训练参数配置（可在脚本中修改）：

| 参数          | 默认值 | 描述           |
| ------------- | ------ | -------------- |
| IMAGE_SIZE    | 224    | 图像尺寸       |
| BATCH_SIZE    | 64     | 批次大小       |
| TASK_NAME_LIST | ['geometric shape type', 'natural shape type']   | 任务列表     |
| LEARNING_RATE | 3e-4   | 学习率         |
| WEIGHT_DECAY   | 1e-4    | 权重衰减       |
| NUM_EPOCHS    | 50     | 训练轮数       |
| TEST_SIZE     | 0.1   | 测试集比例     |


​	3. 运行model_picture_test.py，测试模型效果，输出分类概率柱状图。

### 模型输出

训练完成后会生成：

- `model_save/model_best.pth` - 最佳验证准确率模型
- `multitask_training_curves_20260611_230726.png` - 训练曲线图

## 实验结果 

* **训练效果如下所示**

<img width="871" height="168" alt="运行截图1" src="https://github.com/user-attachments/assets/aa80afa1-6999-42e6-81d1-fe251ff757ab" />

<img width="1600" height="800" alt="multitask_training_curves_20260611_230726" src="https://github.com/user-attachments/assets/532a9c35-e40c-4424-b536-a3d15344329b" />

* **测试结果**

  | 紫砂壶分类头 | 任务准确率 |
  | ------------ | ---------- |
  | **几何形状** | **0.6774**  |
  | **自然形状** | **0.9077**  |

  由于**几何形状**分类头包含紫砂壶壶型最多，包含有30多个类，且对于一些相似形状茶壶较难分辨，所以任务准确率最低，而**自然形状**只有8个左右，所以准确率高很多。

* 下面是抽取的紫砂壶的各类别概率分布
* **几何形状类：**



<img width="1200" height="600" alt="picture_pred_geometric shape type_20260612_102912" src="https://github.com/user-attachments/assets/4570fd26-fadc-41d3-b8d1-6e5ea7a45ca5" />

<img width="1200" height="600" alt="picture_pred_geometric shape type_20260612_102919" src="https://github.com/user-attachments/assets/e21eb447-fd0e-45ca-b3e9-acdaff838cf5" />

<img width="1200" height="600" alt="picture_pred_geometric shape type_20260612_112359" src="https://github.com/user-attachments/assets/c17cf1cc-c542-4b1d-b36b-2f96b2ad9d79" />



* **自然形状类：**

<img width="1200" height="600" alt="picture_pred_natural shape type_20260612_102924" src="https://github.com/user-attachments/assets/4b7f0cad-7d08-46a1-9970-315816d6c399" />
<img width="1200" height="600" alt="picture_pred_natural shape type_20260612_112359" src="https://github.com/user-attachments/assets/55deed56-3051-4461-be78-ab94ff9a19bc" />
<img width="1200" height="600" alt="picture_pred_natural shape type_20260612_102920" src="https://github.com/user-attachments/assets/89fb31a6-3756-4c06-9e0b-7f73cda02803" />

## 核心代码说明

### 动态任务权重机制

为解决多任务学习中任务优化不均衡、收敛速度不一致的问题，本项目设计并实现了**基于验证集准确率的动态任务权重策略**，具体实现如下：

1. **动态权重计算**

   以各任务在验证集上的准确率为依据，对表现较差的任务自动分配更高权重。

   首先通过 `1 - acc` 得到任务难度系数，归一化后作为动态修正项；

   再将基础权重与动态项加权融合，得到最终任务权重：

   task_weight=0.7×base_weight+0.3×inv_acc

   其中 `base_weight` 初始化为 `[0.25, 0.25, 0.25, 0.25]`，保证训练初期稳定。

2. **训练控制策略**

   - **hybrid 混合模式**：采用历史权重与当前权重平滑融合（`0.9×历史 + 0.1×新计算`），使权重更新更平滑、训练更稳定，避免权重剧烈波动。

该机制能够在训练过程中**自动聚焦困难任务**，使四个分类任务均衡优化，显著提升模型整体收敛稳定性与最终分类精度。核心代码如下：

```python
# 根据验证准确率动态调整任务权重
def dynamic_task_weight(val_accs, base_weights=[0.25, 0.25, 0.25, 0.25]):
    # 表现差的任务分配更高权重
    inv_accs = [1 - acc for acc in val_accs]
    inv_accs = [w / sum(inv_accs) for w in inv_accs]
    # 混合权重
    weights = [0.7 * base + 0.3 * inv for base, inv in zip(base_weights, inv_accs)]
    return weights

# 根据验证准确率动态调整任务权重（从第二个epoch开始）
        if epoch > 0 and use_dynamic_weights:
            if weight_adjust_method == 'accuracy':
                task_weights = dynamic_task_weight(current_val_accs)
            elif weight_adjust_method == 'hybrid':
				# acc_weights为动态任务权重，task_weights为包含acc_weights的历史状态，实现平滑过渡
                acc_weights = dynamic_task_weight(current_val_accs)
                task_weights = [0.9 * a + 0.1 * s for a, s in zip(task_weights, acc_weights)]
```
### SE注意力机制

为增强模型对紫砂壶细粒度特征的感知能力，本项目在 ResNet34 的中深层特征提取阶段引入 **SE（Squeeze-and-Excitation）注意力模块**。
可实现强化关键纹理与轮廓特征、抑制背景和噪声信息、提高模型判别能力。核心代码如下：

```python
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
```

### GeM池化

传统 ResNet 使用全局平均池化（Global Average Pooling）进行特征聚合，但平均池化会平等对待所有空间位置的信息。
因此本项目采用 **GeM池化（Generalized Mean Pooling）**，GeM通过引入可学习参数 p，能够自动学习最适合当前任务的特征聚合方式。核心代码如下：

```python
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()

        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(
            x.clamp(min=self.eps).pow(self.p),
            kernel_size=(x.size(-2), x.size(-1))
        ).pow(1.0 / self.p)
```
### 多任务分类结构

为充分利用不同分类任务之间的相关性，本项目采用 **共享特征提取 + 独立分类头** 的多任务学习结构。

本项目使用的两个分类任务：

- geometric shape type（几何形状）
- natural shape type（自然形状）

模型共享同一个 ResNet34 主干网络，最终通过多个独立分类头完成预测。其可以提高特征利用率，降低模型参数量，增强模型泛化能力，促进不同任务之间的信息共享。核心代码如下：

```python
self.heads = nn.ModuleDict()

for type_name, num in type_len_list.items():
    self.heads[type_name] = nn.Linear(512, num)
```

```python
outputs = []

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

### AdamW优化器+余弦退火

```python
# 优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY) # AdamW优化器
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS) # 余弦退火
```
## 借助大模型实现的代码说明
### picture_mask_test.py
原图裁剪代码
```python
# 裁剪原图,增加细节
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
```
### process.py
原图裁剪代码
```python
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
```
### main.py
GeM
```python
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            kernel_size=(x.size(-2), x.size(-1))).pow(1.0 / self.p)

```
损失函数构建

```python
    criterions = {}
    for task in tasks:
        if class_weights is not None and task in class_weights:
            weight_tensor = torch.tensor(class_weights[task], dtype=torch.float32).to(DEVICE)
            criterions[task] = nn.CrossEntropyLoss(weight=weight_tensor)
        else:
            criterions[task] = criterion

```
### model_picture_test.py
GeM
```python
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                            kernel_size=(x.size(-2), x.size(-1))).pow(1.0 / self.p)
```


