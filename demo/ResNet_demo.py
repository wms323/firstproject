import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels=mid_channels*self.expansion
        
        self.conv1=nn.Conv2d(
        in_channels, 
        mid_channels, 
        kernel_size=3, 
        padding=1, 
        stride=stride, 
        bias=False
        )
        self.bn1=nn.BatchNorm2d(mid_channels)
        
        self.conv2=nn.Conv2d(
        mid_channels, 
        out_channels, 
        kernel_size=3, 
        padding=1, 
        stride=1, 
        bias=False
        )
        self.bn2=nn.BatchNorm2d(out_channels)
        
        self.shortcut=nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu=nn.ReLU(inplace=True)
        
    def forward(self,x):
        identity=self.shortcut(x)
        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)

        out+=identity
        out=self.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, mid_channels, stride=1):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1,stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut=nn.Sequential()
        if stride!=1 or in_channels!=out_channels:
            self.shortcut=nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu=nn.ReLU(inplace=True)

    def forward(self,x):
        identity=self.shortcut(x)

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu(out)

        out=self.conv3(out)
        out=self.bn3(out)

        out+=identity
        out=self.relu(out)
        return out
    
if __name__ == "__main__":
    torch.manual_seed(42)
    x = torch.randn(1, 64, 32, 32)
    # BasicBlock 测试
    basic=BasicBlock(64, 64, stride=1)
    out=basic(x)
    print(f"BasicBlock: {x.shape} -> {out.shape}")
    assert out.shape==(1, 64, 32, 32), f"BasicBlock 形状错误: {out.shape}"
    # Bottleneck 测试 (通道变化)
    bottle=Bottleneck(64, 16, stride=2)
    out=bottle(x)
    print(f"Bottleneck: {x.shape} -> {out.shape}")
    assert out.shape==(1, 64, 16, 16), f"Bottleneck 形状错误: {out.shape}"