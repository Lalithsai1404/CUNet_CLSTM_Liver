import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            input_dim + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding
        )

        self.hidden_dim = hidden_dim

    def forward(self, x, h_prev, c_prev):
        combined = torch.cat([x, h_prev], dim=1)
        conv_out = self.conv(combined)

        i, f, o, g = torch.chunk(conv_out, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        return h, c


class ConvLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_dim, hidden_dim, kernel_size)

    def forward(self, input_seq):
        # input_seq shape:
        # [batch, seq_len, channels, H, W]

        batch_size, seq_len, _, H, W = input_seq.size()

        h = torch.zeros(batch_size, self.hidden_dim, H, W, device=input_seq.device)
        c = torch.zeros(batch_size, self.hidden_dim, H, W, device=input_seq.device)

        outputs = []

        for t in range(seq_len):
            h, c = self.cell(input_seq[:, t], h, c)
            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)

        return outputs, h
