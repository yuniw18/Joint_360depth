import torch
import torch.nn as nn
import torch.nn.functional as torch_nn_func
import math
import torch.nn.functional as F
from collections import namedtuple

# Code below from https://github.com/nianticlabs/monodepth2
class PoseCNN(nn.Module):
    def __init__(self, num_input_frames):
        super(PoseCNN, self).__init__()

        self.num_input_frames = 2

        self.convs = {}
        self.convs[0] = nn.Conv2d(3 * self.num_input_frames, 16, 7, 2, 3)
        self.convs[1] = nn.Conv2d(16, 32, 5, 2, 2)
        self.convs[2] = nn.Conv2d(32, 64, 3, 2, 1)
        self.convs[3] = nn.Conv2d(64, 128, 3, 2, 1)
        self.convs[4] = nn.Conv2d(128, 256, 3, 2, 1)
        self.convs[5] = nn.Conv2d(256, 256, 3, 2, 1)
        self.convs[6] = nn.Conv2d(256, 256, 3, 2, 1)

        # Delta movements (x,y,z) and rotation (x,y,z)
        self.pose_conv2 = nn.Conv2d(256, 6, 1)

        self.num_convs = len(self.convs)

        self.relu = nn.ReLU(True)
        self.tanh = nn.Tanh()
        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, left,right):
        out = torch.cat((left,right), dim = 1)

        for i in range(self.num_convs):
            out = self.convs[i](out)
            out = self.relu(out)

        out = self.pose_conv2(out)
        out_tanh = self.tanh(out)
        
        out = out.mean(3).mean(2)
        out = 0.01 * out.view(-1, self.num_input_frames - 1, 1, 6)

        ## Only delta 'x' is used for paper, refer to the Technical Appendix
        ## modify according to the video data if needed
        x = out[..., 0]
        y = out_tanh[..., 1] * 90
        return x,y



