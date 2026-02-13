import torch
import torch.nn as nn
from models.cunet2d import CUNet2D
from models.convlstm import ConvLSTM

class CUNet_CLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cunet = CUNet2D()

        # ConvLSTM takes bottleneck features (64 channels)
        self.convlstm = ConvLSTM(input_dim=64, hidden_dim=64)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x_seq):
        # x_seq shape:
        # [batch, seq_len, 1, H, W]

        batch_size, seq_len, _, H, W = x_seq.size()

        seg_outputs = []
        bottleneck_seq = []

        for t in range(seq_len):
            seg_out, bottleneck = self.cunet(x_seq[:, t])
            seg_outputs.append(seg_out)
            bottleneck_seq.append(bottleneck)

        seg_outputs = torch.stack(seg_outputs, dim=1)
        bottleneck_seq = torch.stack(bottleneck_seq, dim=1)

        lstm_outputs, last_hidden = self.convlstm(bottleneck_seq)

        cls_out = self.classifier(last_hidden)

        return seg_outputs, cls_out
