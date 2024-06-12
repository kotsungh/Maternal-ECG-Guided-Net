import numpy as np
import torch
import torch.nn as nn
from typing import Optional

class DownConv(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: Optional[bool] = True):
        super(DownConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool = pool
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(num_features=out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(num_features=out_channels),
            nn.GELU() 
        )
        
        self.avg_pool = nn.AvgPool1d(kernel_size=2)
        
    def forward(self, x):
        
        if self.pool:
            x = self.conv(x)
            return x, self.avg_pool(x)
        else:
            return self.conv(x)

class SubtractionConvBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool: Optional[bool] = True):
        super(SubtractionConvBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pool = pool
        
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(num_features=out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(num_features=out_channels),
            nn.GELU()  
        )
        
        self.avg_pool = nn.AvgPool1d(kernel_size=2)
        
    def forward(self, abecg, mecg):
        
        x = torch.tanh(abecg - mecg)
        x = torch.cat((abecg, x), dim=1)
        x = self.conv(x)
        if self.pool:
            return x, self.avg_pool(x)
        else:
            return x

class UpConv(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(UpConv, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv1d(out_channels * 2, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(num_features=out_channels),
            nn.GELU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding="same"),
            nn.BatchNorm1d(num_features=out_channels),
            nn.GELU()  
        )
           
    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        x = torch.cat((x1, x2), dim=1)
        return self.conv(x)     
    
class MaternalEcgGuidedNet(nn.Module):
    
    def __init__(
            self, 
            n_depths: int, 
            signal_dim: int, 
            abecg_kernel_size: Optional[int] = 7, 
            mecg_kernel_size: Optional[int] = 5,
            base_channels: Optional[int] = 16
        ):
        super(MaternalEcgGuidedNet, self).__init__()
        
        self.n_depths = n_depths
        self.signal_dim = signal_dim
        self.abecg_kernel_size = abecg_kernel_size
        self.mecg_kernel_size = mecg_kernel_size
        self.base_channels = base_channels        # Number of feature maps in first depth
        
        
        self.accept_n_depths = int(np.log2(self.signal_dim // (2 ** 3))) + 1
        if self.n_depths > self.accept_n_depths:
            raise AssertionError(f"Number of depths should be less than or equal to {self.accept_n_depths}")
        
        
        self.down_abecg = nn.ModuleList([
            DownConv(1, self.base_channels, self.abecg_kernel_size) if i == 0 else
            SubtractionConvBlock(self.base_channels * (2 ** i), self.base_channels * (2 ** i), self.abecg_kernel_size, pool=False) if i == self.n_depths - 1 else
            SubtractionConvBlock(self.base_channels * (2 ** i), self.base_channels * (2 ** i), self.abecg_kernel_size) for i in range(self.n_depths)]
        )
        
        self.down_mecg = nn.ModuleList([
            DownConv(1, self.base_channels, self.mecg_kernel_size) if i == 0 else
            DownConv(self.base_channels * (2 ** (i - 1)), self.base_channels * (2 ** i), self.mecg_kernel_size) for i in range(self.n_depths - 1)]
        )
        
        self.up = nn.ModuleList([
            UpConv(self.base_channels * (2 ** i), self.base_channels * (2 ** (i - 1)), self.abecg_kernel_size) for i in range(self.n_depths - 1, 0, -1)]
        )
        
        self.out = nn.Conv1d(in_channels=self.base_channels, out_channels=1, kernel_size=1)
        
    def forward(self, abecg, mecg):
        
        ###### Encoder
        encoder_abecg = {}
        encoder_mecg = {}
        
        for i in range(self.n_depths):
            if i == 0:
                encoder_abecg[f"abecg_{i}"]= self.down_abecg[i](abecg)
                encoder_mecg[f"mecg_{i}"] = self.down_mecg[i](mecg) 
            elif i == (self.n_depths - 1):
                bottom = self.down_abecg[i](encoder_abecg[f"abecg_{i - 1}"][1], encoder_mecg[f"mecg_{i - 1}"][1])
            else:
                encoder_abecg[f"abecg_{i}"] = \
                    self.down_abecg[i](encoder_abecg[f"abecg_{i - 1}"][1], encoder_mecg[f"mecg_{i - 1}"][1])     
                encoder_mecg[f"mecg_{i}"] = self.down_mecg[i](encoder_mecg[f"mecg_{i - 1}"][1]) 
        
        ###### Decoder
        n_temp = len(encoder_abecg)
        for i in range(self.n_depths - 1):
            if i == 0:
                fecg = self.up[i](bottom, encoder_abecg[f"abecg_{n_temp - 1 - i}"][0])
            else:
                fecg = self.up[i](fecg, encoder_abecg[f"abecg_{n_temp - 1 - i}"][0])
        
        fecg = self.out(fecg)
        
        return fecg
        
