import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    expansion=1
    def __init__(self,in_planes,out_planes,stride=1,downsample=None,**kwargs):
        super(BasicBlock,self).__init__()

        self.conv1=nn.Conv2d(in_planes,out_planes,stride,padding=1,kernel_size=3)
        self.batchNorm1=nn.BatchNorm2d(out_planes)
        self.relu1=nn.ReLU(inplace=True)
        self.conv2=nn.Conv2d(out_planes,out_planes,stride,padding=1,kernel_size=3)
        self.batchNorm2=nn.BatchNorm2d(out_planes)
        self.relu2=nn.ReLU(inplace=True)
        self.downsample=downsample
    def forward(self,x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_planes,out_planes,stride,downsample,group=1,width_per_group=64):
        super(BottleNeck,self).__init__()
        width=int(out_planes*(width_per_group/64.))*group
        self.conv1=nn.Conv2d(in_planes,width,kernel_size=1,stride=1,bias=False)
        self.bn1=nn.BatchNorm2d(width)
        self.conv2=nn.Conv2d(width,width,kernel_size=3,stride=stride,groups=group,padding=1,bias=False)
        self.bn2=nn.BatchNorm2d(width)
        self.conv3=nn.Conv2d(width,out_planes*self.expansion,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm2d(out_planes*self.expansion)
        self.relu=nn.ReLU(inplace=True)
        self.downsample=downsample

    def forward(self,x):
        identity=x
        if self.downsample is not  None:
            identity=self.downsample(x)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.conv2(out)
        out=self.bn2(out)
        out=self.conv3(out)
        out=self.bn3(out)
        out+=identity
        out=self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self,block,block_num,num_classes,include_Top=True,group=1,width_per_group=64):
        super(ResNet,self).__init__()
        self.in_planes=64
        self.group=group
        self.width_per_group=width_per_group
        self.include_Top=include_Top
        self.conv1=nn.Conv2d(3,self.in_planes,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm2d(self.in_planes)
        self.relu=nn.ReLU(True)
        self.maxpooling1=nn.MaxPool2d(kernel_size=3,stride=2)
        self.layer1=self.make_layer(block,64,block_num[0])
        self.layer2=self.make_layer(block,128,block_num[1],stride=2)
        self.layer3 = self.make_layer(block, 256, block_num[2], stride=2)
        self.layer4 = self.make_layer(block, 512, block_num[3], stride=2)
        if self.include_Top:
            self.avgpool=nn.AdaptiveAvgPool2d((1,1))
            self.Flatten=nn.Flatten(1)
            self.fc=nn.Linear(512*block.expansion,num_classes)
    def make_layer(self,block,channel,block_num,stride=1):
        downsample=None
        layer=[]
        if stride!=1 or channel!=self.in_planes*block.expansion:
            downsample=nn.Sequential(
                nn.Conv2d(self.in_planes,channel*block.expansion,kernel_size=1,stride=stride,bias=False),
                nn.BatchNorm2d(channel*block.expansion)
            )
        layer.append(block(self.in_planes,channel,downsample=downsample,stride=stride,group=self.group,width_per_group=self.width_per_group))
        self.in_planes=channel*block.expansion
        for i in range(1,block_num):
            layer.append(block(self.in_planes,channel,group=self.group,width_per_group=self.width_per_group))
        return nn.Sequential(*layer)
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.maxpooling1(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        if self.include_Top:
            x=self.avgpool(x)
            x=self.Flatten(x)
            x=self.fc(x)
        return x
def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(BottleNeck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(BottleNeck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(BottleNeck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(BottleNeck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)