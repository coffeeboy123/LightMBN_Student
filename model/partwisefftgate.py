import torch
import torch.nn as nn
import torch.nn.functional as F

class PartWiseFFTGate(nn.Module):
    def __init__(self, in_channels=512, fft_type='amplitude', reduction=4):
        super().__init__()
        self.fft_type = fft_type

        self.head_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1)
        )
        self.upper_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1)
        )
        self.lower_fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, kernel_size=1)
        )

    def forward(self, par):
        """
        Input:
            p_par: shape (B, C, 12, 1)  # part-wise feature map
        Output:
            p_head, p_upper, p_lower: weighted feature maps
            weights: part importance weights (B, 3, 1, 1)
        """

        p_head = par[:, :, 0:4, :]
        p_upper = par[:, :, 4:14, :]
        p_lower = par[:, :, 14:24, :]

        p_head_avg = F.adaptive_avg_pool2d(p_head, (1, 1))
        p_upper_avg = F.adaptive_avg_pool2d(p_upper, (1, 1))
        p_lower_avg = F.adaptive_avg_pool2d(p_lower, (1, 1))

        # Apply 2D FFT
        def fft_feature(x):
            f = torch.fft.fft2(x, norm='ortho')
            if self.fft_type == 'amplitude':
                return torch.abs(f)
            elif self.fft_type == 'real':
                return f.real
            elif self.fft_type == 'imag':
                return f.imag
            else:
                raise ValueError(f"Invalid fft_type: {self.fft_type}")

        head_fft = fft_feature(p_head)
        upper_fft = fft_feature(p_upper)
        lower_fft = fft_feature(p_lower)

        # Extract importance scores
        w_head = self.head_fc(head_fft)
        w_upper = self.upper_fc(upper_fft)
        w_lower = self.lower_fc(lower_fft)

        w_head = F.adaptive_avg_pool2d(w_head, (1, 1))
        w_upper = F.adaptive_avg_pool2d(w_upper, (1, 1))
        w_lower = F.adaptive_avg_pool2d(w_lower, (1, 1))


        weights = torch.cat([w_head, w_upper, w_lower], dim=1)  # (B, 3, 1, 1)
        weights = F.softmax(weights, dim=1)

        

        # Apply weights
        p_head_out = p_head_avg * weights[:, 0:1, :, :]
        p_upper_out = p_upper_avg * weights[:, 1:2, :, :]
        p_lower_out = p_lower_avg * weights[:, 2:3, :, :]

        return p_head_out, p_upper_out, p_lower_out, weights
