import torch
import torch.nn as nn
import torch.nn.functional as F

# 复数卷积基础操作
class ComplexConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv_real = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_imag = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    
    def forward(self, x):  # x: [B, C, H, W, 2] (real and imag)
        real = self.conv_real(x[..., 0]) - self.conv_imag(x[..., 1])
        imag = self.conv_real(x[..., 1]) + self.conv_imag(x[..., 0])
        return torch.stack([real, imag], dim=-1)

# Complex Depthwise Separable Convolution
class DWCConv(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        # Depthwise卷积
        self.depthwise = ComplexConv2d(channels, channels, kernel_size, stride, padding)
        # Pointwise卷积
        self.pointwise = ComplexConv2d(channels, channels, 1)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# Complex Group Normalization
class ComplexGroupNorm(nn.Module):
    def __init__(self, num_groups, num_channels):
        super().__init__()
        self.gn_real = nn.GroupNorm(num_groups, num_channels)
        self.gn_imag = nn.GroupNorm(num_groups, num_channels)
    
    def forward(self, x):
        real = self.gn_real(x[..., 0])
        imag = self.gn_imag(x[..., 1])
        return torch.stack([real, imag], dim=-1)

# Complex ELU激活函数
class ComplexELU(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        real = F.elu(x[..., 0], alpha=self.alpha)
        imag = F.elu(x[..., 1], alpha=self.alpha)
        return torch.stack([real, imag], dim=-1)

# Layer Scale（可学习对角矩阵）
class LayerScale(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1, 1))
    
    def forward(self, x):
        return x * self.scale

# Subblock1：DWCConv + 残差连接
class Subblock1(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.dwconv = DWCConv(channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.layerscale = LayerScale(channels)
    
    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.layerscale(x)
        return x + residual

# Subblock2：ComplexGN + 1x1 Conv
class Subblock2(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        self.norm = ComplexGroupNorm(groups, channels)
        self.conv1 = ComplexConv2d(channels, channels, 1)
        self.conv2 = ComplexConv2d(channels, channels, 1)
        self.dwconv = DWCConv(channels, kernel_size=5, padding=2)
        self.layerscale = LayerScale(channels)
    
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        x = self.layerscale(x)
        return x + residual

# Subblock3：ComplexELU + 3x3 DWCConv
class Subblock3(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.activation = ComplexELU()
        self.conv1 = ComplexConv2d(channels, channels, 1)
        self.dwconv = DWCConv(channels, kernel_size=3, padding=1)
        self.conv2 = ComplexConv2d(channels, channels, 1)
        self.layerscale = LayerScale(channels)
    
    def forward(self, x):
        residual = x
        x = self.activation(x)
        x = self.conv1(x)
        x = self.dwconv(x)
        x = self.conv2(x)
        x = self.layerscale(x)
        return x + residual

# 完整的FPB模块
class FPB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.subblock1 = Subblock1(channels)
        self.subblock2 = Subblock2(channels)
        self.subblock3 = Subblock3(channels)
        self.final_conv = ComplexConv2d(channels, channels, 1)
    
    def forward(self, x):
        x = self.subblock1(x)
        x = self.subblock2(x)
        x = self.subblock3(x)
        x = self.final_conv(x)
        return x

# 测试代码
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size, channels, H, W = 4, 64, 128, 128
    
    # 输入格式：[Batch, Channel, Height, Width, Real/Imag]
    dummy_input = torch.randn(batch_size, channels, H, W, 2).to(device)
    
    fpb = FPB(channels).to(device)
    output = fpb(dummy_input)
    
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")