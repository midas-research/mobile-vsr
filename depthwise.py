import torch.nn as nn
import torch.nn.functional as F


class LipResBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, out_planes, stride=1, reduction=1):
        
        super(LipResBlock, self).__init__()
        self.expansion = 1
        self.in_planes = in_planes
        self.mid_planes = mid_planes = int(self.expansion * out_planes)
        self.out_planes = out_planes

        self.conv1 = nn.Conv2d(
                                in_planes, 
                                mid_planes, 
                                kernel_size=1, 
                                bias=False
                            )
        self.bn1 = nn.BatchNorm2d(mid_planes)
        
        self.depth = nn.Conv2d(
                                mid_planes,
                                mid_planes, 
                                kernel_size=3, 
                                padding=1,
                                stride=1, 
                                bias=False, 
                                groups=mid_planes
                            )
        self.bn2 = nn.BatchNorm2d(mid_planes)
        
        self.conv3 = nn.Conv2d(
                                mid_planes, 
                                out_planes, 
                                kernel_size=1, 
                                bias=False, 
                                stride=stride
                            )
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        #out = F.relu(self.conv1(x))
        self.int_nchw = out.size()
        out = self.bn2(self.depth(out))
        # out = self.depth(out)
        out = self.bn3(self.conv3(out))
        # out = self.conv3(out)
        self.out_nchw = out.size()
        
        out += self.shortcut(x)
        out = F.relu(out)
        return out