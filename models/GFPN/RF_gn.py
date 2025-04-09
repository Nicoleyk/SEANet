import torch
import torch.nn as nn
import torch.nn.functional as F
    
class BasicConv2d(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(c1, c2,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.gn = nn.GroupNorm(c2//4,c2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.gn(x)
        x = self.relu(x)
        return x   


class RF_D_gn(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RF_D_gn, self).__init__()
        self.relu = nn.ReLU(True)

        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        
        self.branch4 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )

        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 1)
        self.end = BasicConv2d(out_channel, out_channel, kernel_size=(3, 3),stride=2,padding=1)


    def forward(self, x):
        x0 = self.branch0(x)
        x4=self.branch4(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x0)
        x3 = self.branch3(x0)    
        x_cat = self.conv_cat(torch.cat((x1, x2, x3, x4), dim=1))
        x=x0+x_cat
        x = self.end(x)
        return x
    