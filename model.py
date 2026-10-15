from torch import nn
import torch
from neuron import Neuron
from torchsummary import summary



class VGG11(nn.Module):
    def __init__(self, neuron=Neuron, c_in=2, numclass=10):
        super(VGG11, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=c_in, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64))
        self.LIF1 = neuron()
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64))
        self.LIF2 = neuron()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128))
        self.LIF3 = neuron()
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128))
        self.LIF4 = neuron()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(256))
        self.LIF5 = neuron()
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(256))
        self.LIF6 = neuron()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512))
        self.LIF7 = neuron()
        self.conv8 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512))
        self.LIF8 = neuron()

        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.linear = nn.Sequential(nn.Linear(in_features=512, out_features=numclass))

    def forward(self, input):
        x1 = self.conv1(input)
        x1 = self.LIF1(x1)
        x2 = self.conv2(x1)
        x2 = self.LIF2(x2)
        x3 = self.pool1(x2)
        x3 = self.conv3(x3)
        x3 = self.LIF3(x3)
        x4 = self.conv4(x3)
        x4 = self.LIF4(x4)
        x5 = self.pool2(x4)
        x5 = self.conv5(x5)
        x5 = self.LIF5(x5)
        x6 = self.conv6(x5)
        x6 = self.LIF6(x6)
        x7 = self.pool3(x6)
        x7 = self.conv7(x7)
        x7 = self.LIF7(x7)
        x8 = self.conv8(x7)
        x8 = self.LIF8(x8)
        out = self.flatten(x8)
        out = self.linear(out)
        return out


class CGCVGG11(nn.Module):
    def __init__(self, neuron=Neuron, c_in=4, numclass=10):
        super(CGCVGG11, self).__init__()
        self.conv_a = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv_b = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv_c = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv_d = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64))
        self.LIF1 = neuron()
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64))
        self.LIF2 = neuron()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128))
        self.LIF3 = neuron()
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128))
        self.LIF4 = neuron()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(256))
        self.LIF5 = neuron()
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(256))
        self.LIF6 = neuron()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512))
        self.LIF7 = neuron()
        self.conv8 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512))
        self.LIF8 = neuron()

        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.linear = nn.Sequential(nn.Linear(in_features=512, out_features=numclass))

    def CGC(self, x):
        x_0 = torch.unsqueeze(x[:, 0, :], 1)
        x_1 = torch.unsqueeze(x[:, 1, :], 1)
        x_2 = torch.unsqueeze(x[:, 2, :], 1)
        x_3 = torch.unsqueeze(x[:, 3, :], 1)
        a =self.conv_a(torch.cat([x_0, x_1], dim=1))
        b = self.conv_b(torch.cat([x_2, x_3], dim=1))
        c = self.conv_c(torch.cat([x_0, x_2], dim=1))
        d = self.conv_d(torch.cat([x_1, x_3], dim=1))
        out = torch.cat([a, b, c, d], dim=1)
        return out

    def forward(self, input):
        x = self.CGC(input)
        x1 = self.conv1(x)
        x1 = self.LIF1(x1)
        x2 = self.conv2(x1)
        x2 = self.LIF2(x2)
        x3 = self.pool1(x2)
        x3 = self.conv3(x3)
        x3 = self.LIF3(x3)
        x4 = self.conv4(x3)
        x4 = self.LIF4(x4)
        x5 = self.pool2(x4)
        x5 = self.conv5(x5)
        x5 = self.LIF5(x5)
        x6 = self.conv6(x5)
        x6 = self.LIF6(x6)
        x7 = self.pool3(x6)
        x7 = self.conv7(x7)
        x7 = self.LIF7(x7)
        x8 = self.conv8(x7)
        x8 = self.LIF8(x8)
        out = self.flatten(x8)
        out = self.linear(out)
        return out


class Learn_CGCVGG11(nn.Module):
    def __init__(self, neuron=Neuron, c_in=4, numclass=10):
        super(Learn_CGCVGG11, self).__init__()
        self.conv_z = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv_f = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv_a = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv_b = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv_c = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv_d = nn.Sequential(nn.Conv2d(in_channels=2, out_channels=1, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(1))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=4, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64))
        self.LIF1 = neuron()
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64))
        self.LIF2 = neuron()
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128))
        self.LIF3 = neuron()
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(128))
        self.LIF4 = neuron()
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(256))
        self.LIF5 = neuron()
        self.conv6 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(256))
        self.LIF6 = neuron()
        self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv7 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512))
        self.LIF7 = neuron()
        self.conv8 = nn.Sequential(nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(512))
        self.LIF8 = neuron()

        self.flatten = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
        self.linear = nn.Sequential(nn.Linear(in_features=512, out_features=numclass))

    def jw(self, x):
        a = torch.unsqueeze(x[:, 0, :], 1)
        b = torch.unsqueeze(x[:, 1, :], 1)
        x_z = x[:,2:6,: ]
        x_f = x[:,6:10,: ]
        c = self.conv_z(x_z)
        d = self.conv_f(x_f)
        out = torch.cat([a, b, c, d], dim=1)
        return out

    def CGC(self, x):
        x_0 = torch.unsqueeze(x[:, 0, :], 1)
        x_1 = torch.unsqueeze(x[:, 1, :], 1)
        x_2 = torch.unsqueeze(x[:, 2, :], 1)
        x_3 = torch.unsqueeze(x[:, 3, :], 1)
        a =self.conv_a(torch.cat([x_0, x_1], dim=1))
        b = self.conv_b(torch.cat([x_2, x_3], dim=1))
        c = self.conv_c(torch.cat([x_0, x_2], dim=1))
        d = self.conv_d(torch.cat([x_1, x_3], dim=1))
        out = torch.cat([a, b, c, d], dim=1)
        return out

    def forward(self, input):
        input = self.jw(input)
        x = self.CGC(input)
        x1 = self.conv1(x)
        x1 = self.LIF1(x1)
        x2 = self.conv2(x1)
        x2 = self.LIF2(x2)
        x3 = self.pool1(x2)
        x3 = self.conv3(x3)
        x3 = self.LIF3(x3)
        x4 = self.conv4(x3)
        x4 = self.LIF4(x4)
        x5 = self.pool2(x4)
        x5 = self.conv5(x5)
        x5 = self.LIF5(x5)
        x6 = self.conv6(x5)
        x6 = self.LIF6(x6)
        x7 = self.pool3(x6)
        x7 = self.conv7(x7)
        x7 = self.LIF7(x7)
        x8 = self.conv8(x7)
        x8 = self.LIF8(x8)
        out = self.flatten(x8)
        out = self.linear(out)
        return out


if __name__ == '__main__':
    x = torch.rand([16, 10, 64, 64], device='cuda')
    model = Learn_CGCVGG11(c_in=10)
    model.to('cuda')
    y = model(x)
    print(y.shape)
    # summary(model,(4, 64, 64))
