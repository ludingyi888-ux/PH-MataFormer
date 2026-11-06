"""
ConvLSTM 基准模型 (Shi et al., NeurIPS 2015)

编码器-解码器结构:
  Encoder: 4层 Conv2d 逐层下采样压缩
  ConvLSTM: 4层堆叠, hidden_dims=[128,128,128,128]
  Decoder: 4层 ConvTranspose2d 对称上采样重建
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """标准 ConvLSTM 单元"""
    def __init__(self, in_channels, hidden_channels, kernel_size=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels + hidden_channels, 4 * hidden_channels,
            kernel_size, padding=padding, bias=False
        )

    def forward(self, x, state):
        h, c = state
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, g, o = torch.split(gates, self.hidden_channels, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class ConvLSTMStack(nn.Module):
    """堆叠多层 ConvLSTM (BCHW, 逐帧处理)"""
    def __init__(self, in_channels, hidden_dims, kernel_size=5):
        super().__init__()
        self.num_layers = len(hidden_dims)
        self.cells = nn.ModuleList()
        for i, hd in enumerate(hidden_dims):
            ic = in_channels if i == 0 else hidden_dims[i-1]
            self.cells.append(ConvLSTMCell(ic, hd, kernel_size))

    def forward(self, x_seq):
        """x_seq: (T, B, C, H, W) → outputs: (T, B, C_out, H, W)"""
        T, B, C, H, W = x_seq.shape
        states = [None] * self.num_layers
        outputs = []
        for t in range(T):
            h = x_seq[t]
            new_states = []
            for i, cell in enumerate(self.cells):
                if states[i] is None:
                    states[i] = (
                        torch.zeros(B, cell.hidden_channels, H, W, device=h.device),
                        torch.zeros(B, cell.hidden_channels, H, W, device=h.device),
                    )
                h, c = cell(h, states[i])
                new_states.append((h, c))
            states = new_states
            outputs.append(h)
        return torch.stack(outputs)  # (T, B, C, H, W)


class ConvLSTMBaseline(nn.Module):
    """
    ConvLSTM 视频预测基线

    Encoder: 4层 Conv3s2 下采样 128→8
    ConvLSTM: 4层 hidden=[128,128,128,128], kernel=5
    Decoder: 4层 ConvTranspose2d 上采样回 128
    """
    def __init__(self, in_channels=1, T_in=4, T_out=4,
                 H=128, W=320,
                 hidden_dims=(128, 128, 128, 128),
                 kernel_size=5):
        super().__init__()
        self.T_in = T_in
        self.T_out = T_out
        self.H = H
        self.W = W

        # Encoder: 逐步下采样
        enc_dims = [in_channels, 32, 64, 128, 128]
        self.encoder = nn.ModuleList()
        for i in range(4):
            self.encoder.append(nn.Sequential(
                nn.Conv2d(enc_dims[i], enc_dims[i+1], 3, 2, 1, bias=False),
                nn.BatchNorm2d(enc_dims[i+1]),
                nn.LeakyReLU(0.2, inplace=True),
            ))

        # ConvLSTM 在最低分辨率演化
        self.lstm = ConvLSTMStack(enc_dims[-1], hidden_dims, kernel_size)

        # Decoder: 对称上采样
        dec_dims = [hidden_dims[-1], 128, 64, 32, 16]
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(nn.Sequential(
                nn.ConvTranspose2d(dec_dims[i], dec_dims[i+1], 4, 2, 1, bias=False),
                nn.BatchNorm2d(dec_dims[i+1]),
                nn.LeakyReLU(0.2, inplace=True) if i < 3 else nn.Identity(),
            ))
        self.final_conv = nn.Conv2d(dec_dims[-1], in_channels, 3, 1, 1, bias=False)

    def forward(self, x):
        """
        x: (B, T_in, C, H, W)
        returns: dict with 'preds' (B, T_out, C, H, W)
        """
        B, T, C, H_in, W_in = x.shape

        # Encode each input frame
        encoded_frames = []
        for t in range(T):
            h = x[:, t]
            for enc in self.encoder:
                h = enc(h)
            encoded_frames.append(h)  # (B, 128, 8, 20)

        encoded = torch.stack(encoded_frames)  # (T_in, B, 128, 8, 20)

        # ConvLSTM rollout
        lstm_out = self.lstm(encoded)  # (T_in, B, 128, 8, 20)

        # Decode each output frame (use last LSTM state, then auto-regressive)
        preds = []
        state_h = lstm_out[-1]  # Use encoding of last input frame

        for _ in range(self.T_out):
            h = state_h
            for i, dec in enumerate(self.decoder):
                h = dec(h)
            frame = self.final_conv(h)
            preds.append(frame)

            # Simple auto-regressive: re-encode prediction for next step
            h_re = frame
            for enc in self.encoder:
                h_re = enc(h_re)
            # Blend with current state
            state_h = 0.5 * state_h + 0.5 * h_re

        preds = torch.stack(preds, dim=1)  # (B, T_out, C, H, W)
        return {'preds': preds}
