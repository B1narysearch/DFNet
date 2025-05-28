import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.fft import fft2, ifft2, fftshift, ifftshift
import torch
import torch.nn as nn
from einops import rearrange
from pytorch_wavelets import DWTForward
import torchvision
from thop import profile

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 4, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attn = F.adaptive_avg_pool2d(x, 1)  # 全局平均池化
        attn = self.fc1(attn)
        attn = F.relu(attn)
        attn = self.fc2(attn)
        attn = self.sigmoid(attn)
        return x * attn

class CA_Block(nn.Module):
    def __init__(self, channel, h, w, reduction=16):
        super(CA_Block, self).__init__()
 
        self.h = h
        self.w = w
 
        self.avg_pool_x = nn.AdaptiveAvgPool2d((h, 1))
        self.avg_pool_y = nn.AdaptiveAvgPool2d((1, w))
 
        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel//reduction, kernel_size=1, stride=1, bias=False)
 
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel//reduction)
 
        self.F_h = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
        self.F_w = nn.Conv2d(in_channels=channel//reduction, out_channels=channel, kernel_size=1, stride=1, bias=False)
 
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()
 
    def forward(self, x):
 
        x_h = self.avg_pool_x(x).permute(0, 1, 3, 2)
        x_w = self.avg_pool_y(x)
 
        x_cat_conv_relu = self.relu(self.conv_1x1(torch.cat((x_h, x_w), 3)))
 
        x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([self.h, self.w], 3)
 
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.permute(0, 1, 3, 2)))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w))
 
        out = x * s_h.expand_as(x) * s_w.expand_as(x)
 
        return out

class DFHGB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DFHGB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.Dconv5 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=5, dilation=5), nn.BatchNorm2d(in_channels), nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(in_channels * 2,out_channels, kernel_size=1)
        )

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )

        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True),
            nn.Sigmoid()
        )
        self.attn = ChannelAttention(in_channels * 2)
        self.norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
    def forward(self, F1):
        F1_input = self.conv1(F1)

        F1_s = self.Dconv3(F1_input)
        F1_s = self.Dconv5(F1_s)

        F1_f = torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(F1_s.float()).real) * torch.fft.fft2(F1_s.float())))

        F1 = self.residual(F1)

        F_out = self.out(self.attn(torch.cat((F1_f, F1_s), 1))) + F1
        #F_out = self.feed(F_out)
        #F_out = self.out(F1_f) + F1
        return F_out


class ECA(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((torch.log2(torch.tensor(channel, dtype=torch.float32)) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.relu(y)
        # y = self.sigmoid(y)
        return x * y.expand_as(x)

class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class FResNet(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.weight = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 16, 1),
            nn.BatchNorm2d(in_channels // 16),
            nn.ReLU(True),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Conv2d(out_channels + in_channels, out_channels, kernel_size=1),
            # nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
        if stride != 1 or out_channels != in_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out_f = torch.abs(
            torch.fft.ifft2(self.weight(torch.fft.fft2(x.float()).real) * torch.fft.fft2(x.float())))
        out = self.out(torch.cat((out_f, out), 1))
        out += residual
        out = self.relu(out)
        return out


class UNet(nn.Module):
    def __init__(self, input_channels, block=FResNet):
        super().__init__()
        param_channels = [16, 32, 64, 128, 256]
        param_blocks = [2, 2, 2, 2]
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_init = nn.Conv2d(input_channels, param_channels[0], 1, 1)

        self.fsfm = DFHGB(param_channels[3], param_channels[4])

        self.encoder_0 = self._make_layer(param_channels[0], param_channels[0], block)
        self.encoder_1 = self._make_layer(param_channels[0], param_channels[1], block, param_blocks[0])
        self.encoder_2 = self._make_layer(param_channels[1], param_channels[2], block, param_blocks[1])
        self.encoder_3 = self._make_layer(param_channels[2], param_channels[3], block, param_blocks[2])

        # self.middle_layer = self._make_layer(param_channels[3], param_channels[4], block, param_blocks[3])

        self.mid3 = DFHGB(param_channels[3], param_channels[3])
        self.mid2 = DFHGB(param_channels[2], param_channels[2])
        self.mid1 = DFHGB(param_channels[1], param_channels[1])
        self.mid0 = DFHGB(param_channels[0], param_channels[0])


        self.decoder_3 = self._make_layer(param_channels[3] + param_channels[4], param_channels[3], block,
                                          param_blocks[2])
        self.decoder_2 = self._make_layer(param_channels[2] + param_channels[3], param_channels[2], block,
                                          param_blocks[1])
        self.decoder_1 = self._make_layer(param_channels[1] + param_channels[2], param_channels[1], block,
                                          param_blocks[0])
        self.decoder_0 = self._make_layer(param_channels[0] + param_channels[1], param_channels[0], block)


        self.output_0 = nn.Conv2d(param_channels[0], 1, 1)


    def _make_layer(self, in_channels, out_channels, block, block_num=1):
        layer = []
        layer.append(block(in_channels, out_channels))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)

    def forward(self, x):
        x_e0 = self.encoder_0(self.conv_init(x))  # 16*256*256
        x_e1 = self.encoder_1(self.pool(x_e0))  # 32*128*128
        x_e2 = self.encoder_2(self.pool(x_e1))  # 64*64*64
        x_e3 = self.encoder_3(self.pool(x_e2))  # 128*32*32

        # x_m = self.middle_layer(self.pool(x_e3))
        x_m = self.fsfm(self.pool(x_e3))

        # x_d3 = self.decoder_3(torch.cat([x_e3, self.up(x_m)], 1))
        # x_d2 = self.decoder_2(torch.cat([x_e2, self.up(x_d3)], 1))
        # x_d1 = self.decoder_1(torch.cat([x_e1, self.up(x_d2)], 1))
        # x_d0 = self.decoder_0(torch.cat([x_e0, self.up(x_d1)], 1))

        x_f3 = self.mid3(x_e3)
        x_d3 = self.decoder_3(torch.cat([x_f3, self.up(x_m)], 1))

        x_f2 = self.mid2(x_e2)
        x_d2 = self.decoder_2(torch.cat([x_f2, self.up(x_d3)], 1))

        x_f1 = self.mid1(x_e1)
        x_d1 = self.decoder_1(torch.cat([x_f1, self.up(x_d2)], 1))

        x_f0 = self.mid0(x_e0)
        x_d0 = self.decoder_0(torch.cat([x_f0, self.up(x_d1)], 1))

        output = self.output_0(x_d0)
        return output
    
if __name__ == '__main__':
    model = UNet(3)
    print(model)
    input = torch.randn(1, 3, 256, 256)
    flop, para = profile(model, inputs=(input, ))
    print('Flops:',"%.2fM" % (flop/1e6), 'Params:',"%.2fM" % (para/1e6))
    total = sum([param.nelement() for param in model.parameters()])
    print('Number of parameter: %.2fM' % (total/1e6))
 