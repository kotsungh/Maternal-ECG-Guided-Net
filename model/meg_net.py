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
        
    def forward(self, ab_ecg, m_ecg):
        
        x = torch.tanh(ab_ecg - m_ecg)
        x = torch.cat((ab_ecg, x), dim=1)
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
    


class YNet(nn.Module):
    
    def __init__(self, ab_kernel_size: int = 7, m_kernel_size: int = 5):
        super(YNet, self).__init__()
        
        self.ab_kernel_size = ab_kernel_size
        self.m_kernel_size = m_kernel_size
        
        
        self.down_ab1 = DownConv(1, 16, self.ab_kernel_size)
        self.down_ab2 = SubtractionConvBlock(32, 32, self.ab_kernel_size)
        self.down_ab3 = SubtractionConvBlock(64, 64, self.ab_kernel_size)
        self.down_ab4 = SubtractionConvBlock(128, 128, self.ab_kernel_size)
        self.down_ab5 = SubtractionConvBlock(256, 256, self.ab_kernel_size)
        self.down_ab6 = SubtractionConvBlock(512, 512, self.ab_kernel_size, pool=False)
        
        
        self.down_m1 = DownConv(1, 16, self.m_kernel_size)
        self.down_m2 = DownConv(16, 32, self.m_kernel_size)
        self.down_m3 = DownConv(32, 64, self.m_kernel_size)
        self.down_m4 = DownConv(64, 128, self.m_kernel_size)
        self.down_m5 = DownConv(128, 256, self.m_kernel_size)
        
        
        self.up1 = UpConv(512, 256, self.ab_kernel_size)
        self.up2 = UpConv(256, 128, self.ab_kernel_size)
        self.up3 = UpConv(128, 64, self.ab_kernel_size)
        self.up4 = UpConv(64, 32, self.ab_kernel_size)
        self.up5 = UpConv(32, 16, self.ab_kernel_size)
        self.out = nn.Conv1d(in_channels=16, out_channels=1, kernel_size=1)
        
    def forward(self, ab_ecg, m_ecg):
        
        # Encoder
        ab1_before, ab1_after = self.down_ab1(ab_ecg)                  # [1, 16, 1024], [1, 16, 512]
        m1_before, m1_after = self.down_m1(m_ecg)                      # [1, 16, 1024], [1, 16, 512]
        
        ab2_before, ab2_after = self.down_ab2(ab1_after, m1_after)     # [1, 32, 512], [1, 32, 256]
        m2_before, m2_after = self.down_m2(m1_after)                   # [1, 32, 512], [1, 32, 256]
        
        ab3_before, ab3_after = self.down_ab3(ab2_after, m2_after)     # [1, 64, 256], [1, 64, 128]
        m3_before, m3_after = self.down_m3(m2_after)                   # [1, 64, 256], [1, 64, 128]
        
        ab4_before, ab4_after = self.down_ab4(ab3_after, m3_after)     # [1, 128, 128], [1, 128, 64]
        m4_before, m4_after = self.down_m4(m3_after)                   # [1, 128, 128], [1, 128, 64]
        
        ab5_before, ab5_after = self.down_ab5(ab4_after, m4_after)     # [1, 256, 64], [1, 256, 32]
        m5_before, m5_after = self.down_m5(m4_after)                   # [1, 256, 64], [1, 256, 32]
        
        bottom = self.down_ab6(ab5_after, m5_after)        # [1, 512, 32]
        
        # Decoder
        up1_ = self.up1(bottom, ab5_before)           # [1, 256, 64]
        up2_ = self.up2(up1_, ab4_before)             # [1, 128, 128]
        up3_ = self.up3(up2_, ab3_before)             # [1, 64, 256]
        up4_ = self.up4(up3_, ab2_before)             # [1, 32, 512]
        up5_ = self.up5(up4_, ab1_before)             # [1, 16, 1024]
        out_ = self.out(up5_)                          # [1, 1, 1024]
        
        return out_
        
