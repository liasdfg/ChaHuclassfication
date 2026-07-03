import torch.nn as nn
import torch
import math


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


# 预激活Bottleneck
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride, cbam_block=False,first_block=False,
                 downsample=None, base_width=4, cardinality=32):
        super(Bottleneck, self).__init__()
        self.stride = stride
        self.first_block = first_block  # 标记是否为第一个块
        self.cbam_block = cbam_block   # 标记是否使用CBAM
        width = int(math.floor(planes*(base_width/64))*cardinality)  # 计算宽度
        out_planes = planes * self.expansion   # 计算输出通道数
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            width,
            width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(width)

        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(
            width,
            out_planes,
            kernel_size=1,
            stride=1,
            bias=False
        )
        self.downsample = downsample
        self.cbam = CBAM(in_planes) if self.cbam_block else None

    def forward(self, x):
        shortcut = x
        if not self.first_block:  # 预激活下对一个块进行特殊处理
            x = self.bn1(x)
            x = self.relu1(x)
        if self.cbam is not None:  # CBAM用于处理上一个stage的输出
            shortcut_cbam = x
            x = self.cbam(x)
            x += shortcut_cbam
        if self.downsample is not None:   # 下采样
            shortcut = self.downsample(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv2(x)

        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv3(x)

        x += shortcut   # 残差连接

        return x


class ResNeXt_CBAM(nn.Module):
    def __init__(self,type_len_list):
        super(ResNeXt_CBAM, self).__init__()
        self.in_planes = 64  # 初始输入通道数
        self.type_len_list = type_len_list  # 任务类别数量列表
        self.conv1 = nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1,ceil_mode=False)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, first_block=True)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2,cbam_block=True)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)
        self.bn2 = nn.BatchNorm2d(self.in_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avg_pool1 = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(in_features=2048,out_features=1000,bias=True)
        # 多任务分类头
        self.heads = nn.ModuleDict()
        for type_name, num in type_len_list.items():
            self.heads[type_name] = nn.Linear(1000, num)

    def _make_layer(self, block, planes, block_num, stride=1, cbam_block=False, first_block=False):
        downsample = None
        out_planes = planes * block.expansion
        if stride != 1 or self.in_planes != out_planes:  # 满足条件则进行downsample
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
            )
        layers = []
        # 每个stage第一个block特殊处理
        layers.append(block(self.in_planes, planes, stride, cbam_block, first_block, downsample))
        self.in_planes = planes * block.expansion  # 更新下一个stage的输入通道数
        for _ in range(1,block_num):
            layers.append(block(self.in_planes,planes,stride=1))  # 添加stage剩余的block
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.max_pool1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.avg_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        outputs = []
        # 按任务列表计算每个任务的输出
        for type_name in self.type_len_list.keys():
            outputs.append(
                self.heads[type_name](x)
            )
        return tuple(outputs)


# if __name__ == '__main__':
#     len_dict = {'a':2}
#     model = ResNeXt_CBAM(len_dict)
#     model.eval()
#
#     x_in = torch.randn(1, 3, 224, 224)
#
#     torch.onnx.export(
#         model,
#         x_in,
#         "resnext50_cbam.onnx")
