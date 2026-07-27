import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d

class DeformConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.kernel_size=kernel_size
        #偏移量通道数
        offset_channels=kernel_size*kernel_size*2
        #偏移量预测分支
        self.offset_conv=nn.Conv2d(
        in_channels, 
        offset_channels, 
        kernel_size=kernel_size, 
        stride=stride, 
        padding=padding, 
        bias=True)
        #零初始化
        nn.init.constant_(self.offset_conv.weight,0)
        nn.init.constant_(self.offset_conv.bias,0)
        #卷积分支
        self.deform_conv=DeformConv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias
        )
        self.relu=nn.ReLU(inplace=True)
        
    def forward(self,x):
        offset=self.offset_conv(x)
        out=self.deform_conv(x,offset)
        out=self.relu(out)
        return out

if __name__ == "__main__":
    torch.manual_seed(42)
    x=torch.randn(1, 64, 32, 32)
    module=DeformConvModule(64, 64, kernel_size=3, stride=1, padding=1)
    out=module(x)
    print(f"DeformConvModule: {x.shape} -> {out.shape}")
    assert out.shape==(1, 64, 32, 32), f"形状错误,期望 (1,64,32,32), 实际 {out.shape}"

    w_zero = (module.offset_conv.weight==0).all()
    b_zero = (module.offset_conv.bias==0).all()
    print(f"offset_conv.weight 全为零: {w_zero}")
    print(f"offset_conv.bias   全为零: {b_zero}")
    assert w_zero and b_zero, "零初始化失败!"

    reg_conv = nn.Conv2d(64, 64, 3, padding=1, bias=False)
    reg_conv.weight.data = module.deform_conv.weight.data.clone()
    with torch.no_grad():
        offset = module.offset_conv(x)
        out_deform = module.deform_conv(x, offset)
        out_reg = reg_conv(x)
    diff = (out_deform - out_reg).abs().max()
    print(f"标准卷积 vs 可变形卷积 最大差异: {diff.item():.8f}")
    assert diff < 1e-5, f"零偏移时应该等价, 但 diff={diff.item():.4f}"
    print("  通过 — 零偏移时行为 = 标准卷积")

    learn = DeformConvModule(1, 1, kernel_size=3, bias=False)
    before=learn.offset_conv.weight.clone()       
    x4 = torch.randn(1, 1, 32, 32)
    target = torch.randn(1, 1, 32, 32)
    opt = torch.optim.SGD(learn.offset_conv.parameters(), lr=0.01)    # 提示: learn.offset_conv.parameters()
    loss_fn = nn.MSELoss()  
    for i in range(20):
        opt.zero_grad()
        out=learn(x4)
        loss=loss_fn(out, target)
        loss.backward()
        opt.step()  
    after = learn.offset_conv.weight.clone()
    change = (after - before).abs().sum()
    print(f"训练前 weight 绝对值之和: {before.abs().sum().item():.4f}")
    print(f"训练后 weight 绝对值之和: {after.abs().sum().item():.4f}")
    print(f"偏移量变化总量: {change.item():.4f}")
    assert change > 0, "offset 没变化, 学习失败!" 
    
      



