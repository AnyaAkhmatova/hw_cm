import numpy as np
import math

import torch
import torch.nn.functional as F
import torch.nn as nn


class SincConv_fast(nn.Module):
    """Sinc-based convolution
    Parameters
    ----------
    in_channels : `int`
        Number of input channels. Must be 1.
    out_channels : `int`
        Number of filters.
    kernel_size : `int`
        Filter length.
    sample_rate : `int`, optional
        Sample rate. Defaults to 16000.
    Usage
    -----
    See `torch.nn.Conv1d`
    Reference
    ---------
    Mirco Ravanelli, Yoshua Bengio,
    "Speaker Recognition from raw waveform with SincNet".
    https://arxiv.org/abs/1808.00158
    """

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, in_channels=1,
                 stride=1, padding=0, dilation=1, bias=False, groups=1, min_low_hz=50, min_band_hz=50):

        super(SincConv_fast,self).__init__()

        if in_channels != 1:
            msg = "SincConv only support one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if kernel_size%2==0:
            self.kernel_size=self.kernel_size+1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        self.sample_rate = sample_rate
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        low_hz = 0
        high_hz = self.sample_rate / 2 - (self.min_low_hz + self.min_band_hz)

        mel = np.linspace(self.to_mel(low_hz),
                          self.to_mel(high_hz),
                          self.out_channels + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1)) 

        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))

        n_lin=torch.linspace(0, (self.kernel_size/2)-1, steps=int((self.kernel_size/2)))
        self.window_=0.54-0.46*torch.cos(2*math.pi*n_lin/self.kernel_size)

        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2*math.pi*torch.arange(-n, 0).view(1, -1) / self.sample_rate 

    def forward(self, waveforms):
        """
        Parameters
        ----------
        waveforms : `torch.Tensor` (batch_size, 1, n_samples)
            Batch of waveforms.
        Returns
        -------
        features : `torch.Tensor` (batch_size, out_channels, n_samples_out)
            Batch of sinc filters activations.
        """

        self.n_ = self.n_.to(waveforms.device)

        self.window_ = self.window_.to(waveforms.device)

        low = self.min_low_hz  + torch.abs(self.low_hz_) 

        high = torch.clamp(low + self.min_band_hz + torch.abs(self.band_hz_),self.min_low_hz,self.sample_rate/2)
        band=(high-low)[:,0] 

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        band_pass_left=((torch.sin(f_times_t_high)-torch.sin(f_times_t_low))/(self.n_/2))*self.window_ 
        band_pass_center = 2*band.view(-1,1) 
        band_pass_right= torch.flip(band_pass_left,dims=[1]) 

        band_pass=torch.cat([band_pass_left,band_pass_center,band_pass_right],dim=1) 

        band_pass = band_pass / (2*band[:,None]) 

        self.filters = (band_pass).view(
            self.out_channels, 1, self.kernel_size)

        return F.conv1d(waveforms, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                         bias=None, groups=1)


class SincConv(nn.Module):

    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, sinc_type='mel'):
        super().__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1
        
        self.sinc_type = sinc_type

        low_hz = 0
        high_hz = self.sample_rate / 2
        
        if sinc_type == 'mel':
            mel = np.linspace(self.to_mel(low_hz),
                              self.to_mel(high_hz),
                              self.out_channels + 1)
            hz = self.to_hz(mel)
        elif sinc_type == 'inv mel':
            mel = np.linspace(self.to_mel(low_hz),
                              self.to_mel(high_hz),
                              self.out_channels + 1)
            hz = self.to_hz(mel)
            hz = np.abs(np.flip(hz) - 1)
        else:
            hz = np.linspace(low_hz, high_hz, self.out_channels + 1)

        self.low_hz_ = hz[:-1]
        self.high_hz_ = hz[1:]

        kernel_range = np.arange(-((self.kernel_size - 1) // 2), ((self.kernel_size - 1) // 2) + 1)
        self.filters = torch.zeros(self.out_channels, self.kernel_size)
        for i in range(self.filters.shape[0]):
            cur_filter = 2 * self.high_hz_[i] * np.sinc(kernel_range * 2 * self.high_hz_[i]) - \
                2 * self.low_hz_[i] * np.sinc(kernel_range * 2 * self.low_hz_[i])
            cur_filter *= np.hamming(self.kernel_size)
            self.filters[i, :] = torch.tensor(cur_filter)
        self.filters = self.filters.unsqueeze(1)

    def forward(self, waveforms):
        self.filters = self.filters.to(waveforms.device)
        return F.conv1d(waveforms, self.filters, stride=1,
                        padding=0, dilation=1,
                        bias=None, groups=1)


class FMS(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=True)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        y, _ = torch.max(x, dim=-1)
        y = self.sigmoid(self.fc(y))
        y = y.unsqueeze(-1)
        y = x * y + y
        return y
    
    
class SincBlock(nn.Module):
    def __init__(self, sinc_out_channels, sinc_kernel_size, pool_kernel_size, 
                 base_sinc_conv=True, sinc_type='mel', abs=False,
                 min_low_hz=50, min_band_hz=50):
        super().__init__()
        self.base_sinc_conv = base_sinc_conv
        if self.base_sinc_conv:
            self.sinc_filters = SincConv(sinc_out_channels, sinc_kernel_size, sinc_type=sinc_type)
        else:
            self.sinc_filters = SincConv_fast(sinc_out_channels, sinc_kernel_size, min_low_hz=min_low_hz, min_band_hz=min_band_hz)
        self.max_pool = nn.MaxPool1d(pool_kernel_size)
        self.bn = nn.BatchNorm1d(sinc_out_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.abs = abs
        
    def forward(self, x):
        y = self.sinc_filters(x)
        if self.abs:
            y = torch.abs(y)
        y = self.max_pool(y)
        y = self.bn(y)
        y = self.leaky_relu(y)
        return y
    
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=(kernel_size-1)//2)
        
        self.downsampling = False
        if in_channels != out_channels:
            self.downsampling = True
            self.down = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        
        self.max_pool = nn.MaxPool1d(pool_kernel_size)
        self.fms = FMS(out_channels, out_channels)
        
    def forward(self, x):
        y = self.conv1(self.leaky_relu(self.bn1(x)))
        y = self.conv2(self.leaky_relu(self.bn2(y)))
        if self.downsampling:
            y += self.down(x)
        else:
            y += x
        y = self.max_pool(y)
        y = self.fms(y)
        return y
    
    
class RawNet2(nn.Module):
    def __init__(self, sinc_out_channels, 
                 sinc_kernel_size, 
                 pool_kernel_size, 
                 resblock1_out_channels,
                 resblock2_out_channels,
                 resblock_kernel_size,
                 gru_hidden_size,
                 fc_hidden_size, 
                 sinc_type='mel',
                 sinc_abs=False,
                 base_sinc_conv=True,
                 min_low_hz=50, 
                 min_band_hz=50,
                 gru_bn_lr=True,
                 gru_num_layers=3, 
                 gru_dropout=0.1, 
                 gru_bid=True):
        super().__init__()
        self.sinc_block = SincBlock(sinc_out_channels, sinc_kernel_size, pool_kernel_size, 
                                    base_sinc_conv=base_sinc_conv, sinc_type=sinc_type, abs=sinc_abs,
                                    min_low_hz=min_low_hz, min_band_hz=min_band_hz)
        self.resblocks1 = nn.Sequential(
            ResBlock(sinc_out_channels, resblock1_out_channels, 
                     resblock_kernel_size, pool_kernel_size),
            ResBlock(resblock1_out_channels, resblock1_out_channels, 
                     resblock_kernel_size, pool_kernel_size)
        )
        self.resblocks2 = nn.Sequential(
            ResBlock(resblock1_out_channels, resblock2_out_channels, 
                     resblock_kernel_size, pool_kernel_size),
            ResBlock(resblock2_out_channels, resblock2_out_channels, 
                     resblock_kernel_size, pool_kernel_size),
            ResBlock(resblock2_out_channels, resblock2_out_channels, 
                     resblock_kernel_size, pool_kernel_size),
            ResBlock(resblock2_out_channels, resblock2_out_channels, 
                     resblock_kernel_size, pool_kernel_size)
        )
        self.gru_bn_lr = gru_bn_lr
        if self.gru_bn_lr:
            self.bn = nn.BatchNorm1d(resblock2_out_channels)
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        if gru_bid:
            self.gru = nn.GRU(resblock2_out_channels, gru_hidden_size//2, 
                              num_layers=gru_num_layers, bias=True, 
                              batch_first=True, dropout=gru_dropout, 
                              bidirectional=gru_bid)
        else:
            self.gru = nn.GRU(resblock2_out_channels, gru_hidden_size, 
                              num_layers=gru_num_layers, bias=True, 
                              batch_first=True, dropout=gru_dropout, 
                              bidirectional=gru_bid)
        self.fc1 = nn.Linear(gru_hidden_size, fc_hidden_size)     
        self.fc2 = nn.Linear(fc_hidden_size, 2)     
        
    def forward(self, x):
        y = x.unsqueeze(1)
        y = self.sinc_block(y)
        y = self.resblocks1(y)
        y = self.resblocks2(y)
        if self.gru_bn_lr:
            y = self.leaky_relu(self.bn(y))
        y, _ = self.gru(y.transpose(1, 2))
        y = self.fc1(y[:, -1, :])
        y = self.fc2(y)
        return y
    
