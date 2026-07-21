import numpy as np
import torch as tr
import torch.nn as nn

import sys
from pathlib import Path
code_dir = Path(__file__).parent.parent.parent  # ´Ó code/tl/utils/ »Øµ½ code/
sys.path.insert(0, str(code_dir))


from models.models.eegnet import EEGNet_feature

def backbone_net(args, return_type='y'):
    netF = EEGNet_feature(n_classes=args.class_num,
                        Chans=args.chn,
                        Samples=args.time_sample_num,
                        kernLenght=int(args.sample_rate // 2),
                        F1=4,
                        D=2,
                        F2=8,
                        dropoutRate=0.25,
                        norm_rate=0.5)
    if return_type == 'y':
        netC = FC(args.feature_deep_dim, args.class_num)
    elif return_type == 'xy':
        netC = FC_xy(args.feature_deep_dim, args.class_num)
    return netF, netC

class FC(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC, self).__init__()
        self.fc = nn.Linear(nn_in, nn_out)

    def forward(self, x):
        x = self.fc(x)
        return x


class FC_xy(nn.Module):
    def __init__(self, nn_in, nn_out):
        super(FC_xy, self).__init__()
        self.nn_out = nn_out
        self.fc = nn.Linear(nn_in, nn_out)

    def forward(self, x):
        y = self.fc(x)
        return x, y