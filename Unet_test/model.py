import torch
from torch import nn
class DoubleDownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleDownSample, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.relu(x)
    
class DoubleUpSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DoubleUpSample, self).__init__()
        self.up_conv_1 = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def crop_tensor(self, tensor, target_tensor):
        target_h, target_w = target_tensor.shape[2], target_tensor.shape[3]
        tensor_h, tensor_w = tensor.shape[2], tensor.shape[3]
        delta_h = (tensor_h - target_h) // 2
        delta_w = (tensor_w - target_w) // 2
        return tensor[:, :, delta_h:delta_h + target_h, delta_w:delta_w + target_w]

    def forward(self, x, target_x=None):
        x = self.up_conv_1(x)
        if target_x is not None:
            crop = self.crop_tensor(target_x, x)
            x = torch.cat([crop, x], dim=1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return self.relu(x)

class U_net(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(U_net, self).__init__()
        self.down_sample1 = DoubleDownSample(in_channels, 64)
        self.down_sample2 = DoubleDownSample(64, 128)
        self.down_sample3 = DoubleDownSample(128, 256)
        self.down_sample4 = DoubleDownSample(256, 512)
        self.down_sample5 = DoubleDownSample(512, 1024)
        self.maxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.up_sample1 = DoubleUpSample(1024, 512)
        self.up_sample2 = DoubleUpSample(512, 256)
        self.up_sample3 = DoubleUpSample(256, 128)
        self.up_sample4 = DoubleUpSample(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_sample1(x)
        x = self.maxPool(x1)
        x2 = self.down_sample2(x)
        x = self.maxPool(x2)
        x3 = self.down_sample3(x)
        x = self.maxPool(x3)
        x4 = self.down_sample4(x)
        x = self.maxPool(x4)
        x5 = self.down_sample5(x)

        y1 = self.up_sample1(x5, x4)
        y2 = self.up_sample2(y1, x3)
        y3 = self.up_sample3(y2, x2)
        y4 = self.up_sample4(y3, x1)
        y4 = self.final_conv(y4)
        y4 = self.sigmoid(y4)

        return y4