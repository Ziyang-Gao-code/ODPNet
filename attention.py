import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Squeeze-and-Excitation Block
    
    Args:
        inchannel (int): Number of input channels
        reduction_ratio (int): Channel reduction ratio for bottleneck
    """
    
    def __init__(self, inchannel, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        reduced_channels = inchannel // reduction_ratio
        
        self.se = nn.Sequential(
            nn.Linear(inchannel, reduced_channels, bias=False),
            nn.ReLU(),
            nn.Linear(reduced_channels, inchannel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, h, w = x.size()
        
        # Global average pooling
        pooled = self.gap(x).view(b, c)
        
        # SE operation
        channel_attention = self.se(pooled).view(b, c, 1, 1)
        
        # Apply channel attention
        return x * channel_attention.expand_as(x)


class TripletAttention(nn.Module):
    """Triplet Attention Module
    
    Captures cross-dimension interactions by computing attention across three branches:
    1. Channel attention along Height dimension (C-H interaction)
    2. Channel attention along Width dimension (C-W interaction)  
    3. Spatial attention (H-W interaction)
    
    The three branches are averaged to produce the final output.
    
    Args:
        gate_channels (int): Number of input channels
        reduction_ratio (int): Channel reduction ratio for attention computation
        
    Shape:
        - Input: (B, C, H, W)
        - Output: (B, C, H, W)
        
    Examples:
        >>> attention = TripletAttention(gate_channels=64, reduction_ratio=16)
        >>> x = torch.randn(2, 64, 32, 32)
        >>> out = attention(x)
        >>> print(out.shape)
        torch.Size([2, 64, 32, 32])
    """
    
    def __init__(self, gate_channels, reduction_ratio=16):
        super(TripletAttention, self).__init__()
        
        # H dimension channel attention
        self.ChannelGateH = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels, bias=False),
            nn.Sigmoid()
        )
        
        # W dimension channel attention
        self.ChannelGateW = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels, bias=False),
            nn.Sigmoid()
        )
        
        # Spatial attention using SE block
        self.SpatialGate = ChannelAttention(gate_channels, reduction_ratio=16)
    
    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Output tensor of shape (B, C, H, W)
        """
        # Original shape: (batch, C, H, W)
        
        # Branch 1: H dimension attention
        # Permute to (batch, H, C, W) to compute attention along H
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        h_attention = self.ChannelGateH(x)
        h_attention = h_attention.view(h_attention.size(0), h_attention.size(1), 1, 1)
        # Apply attention and permute back
        x_out1 = x_perm1 * h_attention.permute(0, 2, 1, 3).expand_as(x_perm1)
        x_out1 = x_out1.permute(0, 2, 1, 3).contiguous()
        
        # Branch 2: W dimension attention
        # Permute to (batch, W, H, C) to compute attention along W
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        w_attention = self.ChannelGateW(x)
        w_attention = w_attention.view(w_attention.size(0), w_attention.size(1), 1, 1)
        # Apply attention and permute back
        x_out2 = x_perm2 * w_attention.permute(0, 3, 2, 1).expand_as(x_perm2)
        x_out2 = x_out2.permute(0, 3, 2, 1).contiguous()
        
        # Branch 3: Spatial attention
        spatial = self.SpatialGate(x)
        
        # Average the three branches
        x_out = (1 / 3) * (spatial + x_out1 + x_out2)
        
        return x_out

