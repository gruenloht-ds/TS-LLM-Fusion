import torch
import torch.nn as nn

import numpy as np
import ptwt
import pywt
from scipy.interpolate import interp1d
import time
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class DFT_series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, top_k = 5):
        super(DFT_series_decomp, self).__init__()
        self.top_k = top_k

    def forward(self, x):
        xf = torch.fft.rfft(x)
        freq = abs(xf)
        freq[0] = 0
        top_k_freq, top_list = torch.topk(freq, 5)
        xf[freq <= top_k_freq.min()] = 0
        x_season = torch.fft.irfft(xf)
        x_trend = x - x_season
        return x_season, x_trend

class WT_series_decomp(nn.Module):
    def __init__(self, wavelet='db4', level=4):
        super(WT_series_decomp, self).__init__()
        self.wavelet = wavelet
        self.level = level

    def forward(self, x):
        dwt = DWT1DForward(wave=self.wavelet, J=self.level).to(x.device)
        y1, yh = dwt(x)
        season_4_trend = [torch.zeros_like(c, device=x.device) for c in yh]
        dwt_inv = DWT1DInverse(wave='db4').to(x.device)
        x_trend = dwt_inv((y1, season_4_trend))
        x_season = x - x_trend
        return x_season, x_trend

class Res2Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, scales=4, kernel_size=3, stride=1, dilation=1,drop=True, sea_and_tre=True):
        """
        Res2Net-like block for Conv1D.
        Args:
            in_channels:
            out_channels:
            scales:
            kernel_size:
            stride:
            dilation:
            drop:
            sea_and_tre:
        """
        super(Res2Block1D, self).__init__()
        assert out_channels % scales == 0, "out_channels must be divisible by scales"
        self.scales = scales
        self.split_channels = out_channels // scales

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.convs = nn.ModuleList([
            nn.Conv1d(self.split_channels, self.split_channels, kernel_size=kernel_size,
                      stride=stride, padding=dilation, dilation=dilation, bias=False)
            for _ in range(scales - 1)
        ])

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1, bias=False)

        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.3)
        self.drop = drop
        self.sea_and_tre = sea_and_tre
        if self.sea_and_tre:
            self.decomposition = WT_series_decomp()

    def forward(self, x):
        residual = x
        x = self.gelu(self.conv1(x))
        if self.drop:
            x = self.dropout(x)

        splits = torch.split(x, self.split_channels, dim=1)
        out = [splits[0]]

        for i in range(1, self.scales):
            if i == 1:
                out.append(self.gelu(self.convs[i - 1](splits[i])))
            else:
                out.append(self.gelu(self.convs[i - 1](splits[i] + out[-1])))

        if self.sea_and_tre == True:
            x_season, x_trend = [], []
            for out_ in out:
                season, trend = self.decomposition(out_)
                x_season.append(season)
                x_trend.append(trend)

            for i in range(1, len(x_season)):
                x_season[i] = x_season[i] + x_season[i - 1]

            for i in range(len(x_trend) - 2, -1, -1):
                x_trend[i] = x_trend[i] + x_trend[i + 1]

            for i in range(len(out)):
                out[i] = x_season[i] + x_trend[i]

        out = torch.cat(out, dim=1)
        if self.drop:
            out = self.dropout(out)
        out = self.conv3(out)
        out += residual
        if self.drop:
            out = self.dropout(out)
        return self.gelu(out)

class Res2Net1D(nn.Module):
    def __init__(self, in_channels, middle_channels, num_blocks=2, scales=4, drop=True):
        super(Res2Net1D, self).__init__()
        self.in_channels = in_channels
        self.scales = scales

        self.conv1 = nn.Conv1d(in_channels, middle_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.gelu = nn.GELU()

        self.layers = nn.Sequential(*[
            Res2Block1D(middle_channels, middle_channels, scales=scales) for _ in range(num_blocks)
        ])

        self.dropout = nn.Dropout(0.3)
        self.drop = drop

    def forward(self, x):
        x = self.gelu(self.conv1(x))
        if self.drop:
            x = self.dropout(x)
        x = self.layers(x)

        return x

if __name__ == "__main__":
    from torchviz import make_dot
    import time
    s = time.time()
    import torch

    model = Res1Net1D(2, 64, 2, 16)
    model = model.cuda()

    x = torch.randn(2, 2, 768)
    x = x.cuda()
    y = model(x)

    print(y.shape)

    make_dot(y, params=dict(model.named_parameters())).render("model_architecture", format="png")
    print(time.time() - s)