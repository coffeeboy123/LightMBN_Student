import torch
import math
import random
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch import nn

torch_ver = torch.__version__[:3]

__all__ = ['BatchDrop', 'BatchFeatureErase_Top', 'BatchRandomErasing',
           'PAM_Module', 'CAM_Module', 'Dual_Module', 'SE_Module']


class BatchRandomErasing(nn.Module):
    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[0.4914, 0.4822, 0.4465]):
        super(BatchRandomErasing, self).__init__()

        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def forward(self, img):
        if self.training:

            if random.uniform(0, 1) > self.probability:
                return img

            for attempt in range(100):

                area = img.size()[2] * img.size()[3]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.size()[3] and h < img.size()[2]:
                    x1 = random.randint(0, img.size()[2] - h)
                    y1 = random.randint(0, img.size()[3] - w)
                    if img.size()[1] == 3:
                        img[:, 0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[:, 1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[:, 2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    else:
                        img[:, 0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    return img

        return img


class BatchDrop(nn.Module):
    """
    Ref: Batch DropBlock Network for Person Re-identification and Beyond
    https://github.com/daizuozhuo/batch-dropblock-network/blob/master/models/networks.py
    Created by: daizuozhuo
    """

    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x

class BatchElementDropout(nn.Module):
    """Feature map에 element-wise로 dropout 적용하는 모듈"""
    def __init__(self, drop_prob=0.2):
        super(BatchElementDropout, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        mask = torch.bernoulli((1 - self.drop_prob) * torch.ones_like(x)).to(x.device)
        return x * mask


class BatchDropTop(nn.Module):
    """
    Ref: Top-DB-Net: Top DropBlock for Activation Enhancement in Person Re-Identification
    https://github.com/RQuispeC/top-dropblock/blob/master/torchreid/models/bdnet.py
    Created by: RQuispeC

    """

    def __init__(self, h_ratio):
        super(BatchDropTop, self).__init__()
        self.h_ratio = h_ratio

    def forward(self, x, visdrop=False):
        if self.training or visdrop:
            b, c, h, w = x.size()
            rh = round(self.h_ratio * h)
            act = (x**2).sum(1)
            act = act.view(b, h * w)
            act = F.normalize(act, p=2, dim=1)
            act = act.view(b, h, w)
            max_act, _ = act.max(2)
            ind = torch.argsort(max_act, 1)
            ind = ind[:, -rh:]
            mask = []
            for i in range(b):
                rmask = torch.ones(h)
                rmask[ind[i]] = 0
                mask.append(rmask.unsqueeze(0))
            mask = torch.cat(mask)
            mask = torch.repeat_interleave(mask, w, 1).view(b, h, w)
            mask = torch.repeat_interleave(mask, c, 0).view(b, c, h, w)
            if x.is_cuda:
                mask = mask.cuda()
            if visdrop:
                return mask
            x = x * mask
        return x

class BatchDropMiddle(nn.Module):
    """
    Middle DropBlock: activation이 중간인 행(row)을 제거하는 모듈
    """
    def __init__(self, h_ratio):
        super(BatchDropMiddle, self).__init__()
        self.h_ratio = h_ratio

    def forward(self, x, visdrop=False):
        # 학습 중이거나 visdrop=True 일 때만 동작
        if self.training or visdrop:
            b, c, h, w = x.size()
            rh = round(self.h_ratio * h)

            # 각 위치별 activation strength 계산
            act = (x ** 2).sum(1)               # (B, H, W)
            act = act.view(b, h * w)            # (B, H*W)
            act = F.normalize(act, p=2, dim=1)  # L2 정규화
            act = act.view(b, h, w)             # (B, H, W)

            # 각 행(row)마다 최대 activation 값 추출
            max_act, _ = act.max(2)             # (B, H)

            # activation 기준으로 행 인덱스 정렬
            ind = torch.argsort(max_act, dim=1) # (B, H)

            # 중간 rh개 인덱스 선택
            start = (h - rh) // 2
            mid_inds = ind[:, start:start + rh] # (B, rh)

            # mask 생성
            mask = []
            for i in range(b):
                rmask = torch.ones(h, device=x.device)
                rmask[mid_inds[i]] = 0
                mask.append(rmask.unsqueeze(0))
            mask = torch.cat(mask, dim=0)       # (B, H)
            mask = mask.unsqueeze(2).repeat(1, 1, w)     # (B, H, W)
            mask = mask.unsqueeze(1).repeat(1, c, 1, 1)  # (B, C, H, W)

            if visdrop:
                return mask  # mask 반환

            return x * mask

        return x

class BatchDropBottom(nn.Module):
    """Bottom DropBlock: 가장 약한 행(row)을 제거하는 방식"""
    def __init__(self, h_ratio):
        super(BatchDropBottom, self).__init__()
        self.h_ratio = h_ratio

    def forward(self, x, visdrop=False):
        if self.training or visdrop:
            b, c, h, w = x.size()
            rh = round(self.h_ratio * h)
            act = (x**2).sum(1)
            act = act.view(b, h * w)
            act = F.normalize(act, p=2, dim=1)
            act = act.view(b, h, w)
            max_act, _ = act.max(2)
            ind = torch.argsort(max_act, 1)       # ascending order
            ind = ind[:, :rh]                     # pick bottom rows
            mask = []
            for i in range(b):
                rmask = torch.ones(h)
                rmask[ind[i]] = 0
                mask.append(rmask.unsqueeze(0))
            mask = torch.cat(mask)
            mask = torch.repeat_interleave(mask, w, 1).view(b, h, w)
            mask = torch.repeat_interleave(mask, c, 0).view(b, c, h, w)
            if x.is_cuda:
                mask = mask.cuda()
            if visdrop:
                return mask
            x = x * mask
        return x



class BatchFeatureErase_Top(nn.Module):
    """
    Ref: Top-DB-Net: Top DropBlock for Activation Enhancement in Person Re-Identification
    https://github.com/RQuispeC/top-dropblock/blob/master/torchreid/models/bdnet.py
    Created by: RQuispeC

    """

    def __init__(self, channels, bottleneck_type, h_ratio=0.33, w_ratio=1., double_bottleneck=False):
        super(BatchFeatureErase_Top, self).__init__()

        self.drop_batch_bottleneck = bottleneck_type(channels, 512)

        self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
        self.drop_batch_drop_top = BatchDropTop(h_ratio)

    def forward(self, x, drop_top=True, bottleneck_features=True, visdrop=False):
        features = self.drop_batch_bottleneck(x)

        if drop_top:
            x = self.drop_batch_drop_top(features, visdrop=visdrop)
        else:
            x = self.drop_batch_drop_basic(features, visdrop=visdrop)
        if visdrop:
            return x  # x is dropmask
        if bottleneck_features:
            return x, features
        else:
            return x

class BatchFeatureErase_Top_student(nn.Module):
    """
    Ref: Top-DB-Net: Top DropBlock for Activation Enhancement in Person Re-Identification
    https://github.com/RQuispeC/top-dropblock/blob/master/torchreid/models/bdnet.py
    Created by: RQuispeC

    """

    def __init__(self, channels, bottleneck_type, h_ratio=0.33, w_ratio=1., double_bottleneck=False):
        super(BatchFeatureErase_Top_student, self).__init__()

        self.drop_batch_bottleneck = bottleneck_type(channels, 128)

        self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
        self.drop_batch_drop_top = BatchDropTop(h_ratio)

    def forward(self, x, drop_top=True, bottleneck_features=True, visdrop=False):
        features = self.drop_batch_bottleneck(x)

        if drop_top:
            x = self.drop_batch_drop_top(features, visdrop=visdrop)
        else:
            x = self.drop_batch_drop_basic(features, visdrop=visdrop)
        if visdrop:
            return x  # x is dropmask
        if bottleneck_features:
            return x, features
        else:
            return x

class BatchFeatureErase_Top_osnet_x0_5(nn.Module):
    """
    Ref: Top-DB-Net: Top DropBlock for Activation Enhancement in Person Re-Identification
    https://github.com/RQuispeC/top-dropblock/blob/master/torchreid/models/bdnet.py
    Created by: RQuispeC

    """

    def __init__(self, channels, bottleneck_type, h_ratio=0.33, w_ratio=1., double_bottleneck=False):
        super(BatchFeatureErase_Top_osnet_x0_5, self).__init__()

        self.drop_batch_bottleneck = bottleneck_type(channels, 256)

        self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
        self.drop_batch_drop_top = BatchDropTop(h_ratio)

    def forward(self, x, drop_top=True, bottleneck_features=True, visdrop=False):
        features = self.drop_batch_bottleneck(x)

        if drop_top:
            x = self.drop_batch_drop_top(features, visdrop=visdrop)
        else:
            x = self.drop_batch_drop_basic(features, visdrop=visdrop)
        if visdrop:
            return x  # x is dropmask
        if bottleneck_features:
            return x, features
        else:
            return x
        
class BatchFeatureErase_Top_osnet_x0_25(nn.Module):
    """
    Ref: Top-DB-Net: Top DropBlock for Activation Enhancement in Person Re-Identification
    https://github.com/RQuispeC/top-dropblock/blob/master/torchreid/models/bdnet.py
    Created by: RQuispeC

    """

    def __init__(self, channels, bottleneck_type, h_ratio=0.33, w_ratio=1., double_bottleneck=False):
        super(BatchFeatureErase_Top_osnet_x0_25, self).__init__()

        self.drop_batch_bottleneck = bottleneck_type(channels, 128)

        self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
        self.drop_batch_drop_top = BatchDropTop(h_ratio)

    def forward(self, x, drop_top=True, bottleneck_features=True, visdrop=False):
        features = self.drop_batch_bottleneck(x)

        if drop_top:
            x = self.drop_batch_drop_top(features, visdrop=visdrop)
        else:
            x = self.drop_batch_drop_basic(features, visdrop=visdrop)
        if visdrop:
            return x  # x is dropmask
        if bottleneck_features:
            return x, features
        else:
            return x

class BatchFeatureErase_Top_Bottom(nn.Module):
    """
    Top DropBlock + Bottom DropBlock 조합 모듈
    """

    def __init__(self, channels, bottleneck_type, h_ratio=0.33):
        super(BatchFeatureErase_Top_Bottom, self).__init__()

        self.drop_batch_bottleneck = bottleneck_type(channels, 512)

        self.drop_batch_drop_top = BatchDropTop(h_ratio)
        self.drop_batch_drop_bottom = BatchDropBottom(h_ratio)

    def forward(self, x, visdrop=False):
        features = self.drop_batch_bottleneck(x)

        x_top = self.drop_batch_drop_top(features, visdrop=visdrop)
        x_bottom = self.drop_batch_drop_bottom(features, visdrop=visdrop)

        if visdrop:
            return x_top, x_bottom  # these are masks in this case

        return x_top, x_bottom, features


class BatchFeatureErase_Top_Element(nn.Module):
    """
    확장된 Top DropBlock + Element-wise Dropout 조합 모듈.
    """

    def __init__(self, channels, bottleneck_type, h_ratio=0.33, w_ratio=1., element_drop_prob=0.33):
        super(BatchFeatureErase_Top_Element, self).__init__()

        self.drop_batch_bottleneck = bottleneck_type(channels, 512)

        self.drop_batch_drop_basic = BatchDrop(h_ratio, w_ratio)
        self.drop_batch_drop_top = BatchDropTop(h_ratio)
        self.element_dropout = BatchElementDropout(drop_prob=element_drop_prob)

    def forward(self, x, drop_top=True, bottleneck_features=True, visdrop=False):
        features = self.drop_batch_bottleneck(x)

        if drop_top:
            x = self.drop_batch_drop_top(features, visdrop=visdrop)
        else:
            x = self.drop_batch_drop_basic(features, visdrop=visdrop)

        if visdrop:
            return x  # drop mask만 리턴

        element_features = self.element_dropout(features)

        if bottleneck_features:
            return x, features, element_features
        else:
            return x, element_features

class BatchFeatureErase_Top_Bottom_Element(nn.Module):
    """
    Top DropBlock + Bottom DropBlock + Element-wise Dropout 조합 모듈.
    """

    def __init__(self, channels, bottleneck_type, h_ratio=0.33, element_drop_prob=0.33):
        super(BatchFeatureErase_Top_Bottom_Element, self).__init__()

        self.drop_batch_bottleneck = bottleneck_type(channels, 512)

        self.drop_batch_drop_top = BatchDropTop(h_ratio)
        self.drop_batch_drop_bottom = BatchDropBottom(h_ratio)
        self.element_dropout = BatchElementDropout(drop_prob=element_drop_prob)

    def forward(self, x, visdrop=False):
        features = self.drop_batch_bottleneck(x)

        x_top = self.drop_batch_drop_top(features, visdrop=visdrop)
        x_bottom = self.drop_batch_drop_bottom(features, visdrop=visdrop)
        x_element = self.element_dropout(features)

        if visdrop:
            return x_top, x_bottom, x_element  # 모두 mask일 수 있음

        return x_top, x_bottom, x_element, features
    

class BatchFeatureErase_Top_Bottom_Element_student(nn.Module):
    """
    Top DropBlock + Bottom DropBlock + Element-wise Dropout 조합 모듈.
    """

    def __init__(self, channels, bottleneck_type, h_ratio=0.33, element_drop_prob=0.33):
        super(BatchFeatureErase_Top_Bottom_Element_student, self).__init__()

        self.drop_batch_bottleneck = bottleneck_type(channels, 128)

        self.drop_batch_drop_top = BatchDropTop(h_ratio)
        self.drop_batch_drop_bottom = BatchDropBottom(h_ratio)
        self.element_dropout = BatchElementDropout(drop_prob=element_drop_prob)

    def forward(self, x, visdrop=False):
        features = self.drop_batch_bottleneck(x)

        x_top = self.drop_batch_drop_top(features, visdrop=visdrop)
        x_bottom = self.drop_batch_drop_bottom(features, visdrop=visdrop)
        x_element = self.element_dropout(features)

        if visdrop:
            return x_top, x_bottom, x_element  # 모두 mask일 수 있음

        return x_top, x_bottom, x_element, features

class BatchFeatureErase_Top_Mid_Bottom_Element(nn.Module):
    """
    Top / Middle / Bottom DropBlock + Element-wise Dropout 조합 모듈.
    """
    def __init__(self, channels, bottleneck_type, h_ratio=0.33, element_drop_prob=0.33):
        super(BatchFeatureErase_Top_Mid_Bottom_Element, self).__init__()
        # shared bottleneck
        self.drop_batch_bottleneck = bottleneck_type(channels, 512)
        # Top / Mid / Bottom DropBlock
        self.drop_top = BatchDropTop(h_ratio)
        self.drop_mid = BatchDropMiddle(h_ratio)
        self.drop_bottom = BatchDropBottom(h_ratio)
        # element-wise dropout
        self.element_dropout = BatchElementDropout(drop_prob=element_drop_prob)

    def forward(self, x, visdrop=False):
        # 1) Bottleneck features
        features = self.drop_batch_bottleneck(x)
        # 2) Top / Mid / Bottom masked
        x_top    = self.drop_top   (features, visdrop=visdrop)
        x_mid    = self.drop_mid   (features, visdrop=visdrop)
        x_bottom = self.drop_bottom(features, visdrop=visdrop)
        # 3) Element-wise dropout
        x_element = self.element_dropout(features)
        # 4) If visdrop, return masks only
        if visdrop:
            return x_top, x_mid, x_bottom, x_element
        # 5) Otherwise return masked features + bottleneck features
        return x_top, x_mid, x_bottom, x_element, features





class SE_Module(Module):

    def __init__(self, channels, reduction=4):
        super(SE_Module, self).__init__()
        self.fc1 = Conv2d(channels, channels // reduction,
                          kernel_size=1, padding=0)
        self.relu = ReLU(inplace=True)
        self.fc2 = Conv2d(channels // reduction, channels,
                          kernel_size=1, padding=0)
        self.sigmoid = Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x


class PAM_Module(Module):
    """ Position attention module"""
    # Ref from SAGAN

    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(
            in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(
            m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous()
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(
            energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out


class Dual_Module(Module):
    """
    # Created by: CASIA IVA
    # Email: jliu@nlpr.ia.ac.cn
    # Copyright (c) 2018

    # Reference: Dual Attention Network for Scene Segmentation
    # https://arxiv.org/pdf/1809.02983.pdf
    # https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py
    """

    def __init__(self, in_dim):
        super(Dual_Module).__init__()
        self.indim = in_dim
        self.pam = PAM_Module(in_dim)
        self.cam = CAM_Module(in_dim)

    def forward(self, x):
        out1 = self.pam(x)
        out2 = self.cam(x)
        return out1 + out2
