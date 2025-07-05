import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock
from .attention import BatchDrop, BatchFeatureErase_Top, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F

class CrossBranchAttention(nn.Module):
    def __init__(self, dim, heads=1):
        super(CrossBranchAttention, self).__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5

        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.out_proj = nn.Conv2d(dim, dim, 1, bias=False)

    def forward(self, query, context):
        B, C, H, W = query.size()
        q = self.q_proj(query).reshape(B, self.heads, C // self.heads, H * W)
        k = self.k_proj(context).reshape(B, self.heads, C // self.heads, H * W)
        v = self.v_proj(context).reshape(B, self.heads, C // self.heads, H * W)

        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * self.scale
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.out_proj(out)
        return out

class LMBN_n_teacher_6_cba(nn.Module):
    def __init__(self, args):
        super(LMBN_n_teacher_6_cba, self).__init__()

        self.n_ch = 2
        self.chs = 512 // self.n_ch

        osnet = osnet_x1_0(pretrained=True)

        self.global_branch = nn.Sequential(copy.deepcopy(osnet.conv1), copy.deepcopy(osnet.maxpool),
                                           copy.deepcopy(osnet.conv2), copy.deepcopy(osnet.conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))
        self.partial_branch = nn.Sequential(copy.deepcopy(osnet.conv1), copy.deepcopy(osnet.maxpool),
                                           copy.deepcopy(osnet.conv2), copy.deepcopy(osnet.conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))
        self.channel_branch = nn.Sequential(copy.deepcopy(osnet.conv1), copy.deepcopy(osnet.maxpool),
                                           copy.deepcopy(osnet.conv2), copy.deepcopy(osnet.conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        # CBA modules
        self.cba_glo_from_par = CrossBranchAttention(512)
        self.cba_glo_from_cha = CrossBranchAttention(512)
        self.cba_par_from_glo = CrossBranchAttention(512)
        self.cba_par_from_cha = CrossBranchAttention(512)
        self.cba_cha_from_glo = CrossBranchAttention(512)
        self.cba_cha_from_par = CrossBranchAttention(512)

        self.reduce_glo = nn.Conv2d(512 * 2, 512, 1, bias=False)
        self.reduce_par = nn.Conv2d(512 * 2, 512, 1, bias=False)
        self.reduce_cha = nn.Conv2d(512 * 2, 512, 1, bias=False)

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.partial_pooling = nn.AdaptiveAvgPool2d((2, 1))
        self.channel_pooling = nn.AdaptiveAvgPool2d((1, 1))

        reduction = BNNeck3(512, args.num_classes, args.feats, return_f=True)
        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)

        self.shared = nn.Sequential(
            nn.Conv2d(self.chs, args.feats, 1, bias=False),
            nn.BatchNorm2d(args.feats),
            nn.ReLU(True)
        )
        self.weights_init_kaiming(self.shared)

        self.reduction_ch_0 = BNNeck(args.feats, args.num_classes, return_f=True)
        self.reduction_ch_1 = BNNeck(args.feats, args.num_classes, return_f=True)

        self.batch_drop_block = BatchFeatureErase_Top(512, OSBlock)
        self.activation_map = args.activation_map

    def forward(self, x):
        glo = self.global_branch(x)
        par = self.partial_branch(x)
        cha = self.channel_branch(x)

        # 🔥 CBA 적용
        glo_from_par = self.cba_glo_from_par(glo, par)
        glo_from_cha = self.cba_glo_from_cha(glo, cha)
        par_from_glo = self.cba_par_from_glo(par, glo)
        par_from_cha = self.cba_par_from_cha(par, cha)
        cha_from_glo = self.cba_cha_from_glo(cha, glo)
        cha_from_par = self.cba_cha_from_par(cha, par)

        glo = self.reduce_glo(torch.cat([glo_from_par, glo_from_cha], dim=1))
        par = self.reduce_par(torch.cat([par_from_glo, par_from_cha], dim=1))
        cha = self.reduce_cha(torch.cat([cha_from_glo, cha_from_par], dim=1))

        if self.activation_map:
            glo_ = glo

        if self.batch_drop_block is not None:
            glo_drop, glo = self.batch_drop_block(glo)

        if self.activation_map:
            _, _, h_par, _ = par.size()
            fmap_p0 = par[:, :, :h_par // 2, :]
            fmap_p1 = par[:, :, h_par // 2:, :]
            fmap_c0 = cha[:, :self.chs, :, :]
            fmap_c1 = cha[:, self.chs:, :, :]
            print('Generating activation maps...')
            return glo, glo_, fmap_c0, fmap_c1, fmap_p0, fmap_p1

        glo_drop = self.global_pooling(glo_drop)
        glo = self.channel_pooling(glo)
        g_par = self.global_pooling(par)
        p_par = self.partial_pooling(par)
        cha = self.channel_pooling(cha)

        p0 = p_par[:, :, 0:1, :]
        p1 = p_par[:, :, 1:2, :]

        f_glo = self.reduction_0(glo)
        f_p0 = self.reduction_1(g_par)
        f_p1 = self.reduction_2(p0)
        f_p2 = self.reduction_3(p1)
        f_glo_drop = self.reduction_4(glo_drop)

        c0 = cha[:, :self.chs, :, :]
        c1 = cha[:, self.chs:, :, :]
        c0 = self.shared(c0)
        c1 = self.shared(c1)
        f_c0 = self.reduction_ch_0(c0)
        f_c1 = self.reduction_ch_1(c1)

        fea = [f_glo[-1], f_glo_drop[-1], f_p0[-1]]

        if not self.training:
            return torch.stack([f_glo[0], f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_c0[0], f_c1[0]], dim=2)

        return [f_glo[1], f_glo_drop[1], f_p0[1], f_p1[1], f_p2[1], f_c0[1], f_c1[1]], fea

    def weights_init_kaiming(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
            nn.init.constant_(m.bias, 0.0)
        elif classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm') != -1:
            if m.affine:
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)


if __name__ == '__main__':
    # Here I left a simple forward function.
    # Test the model, before you train it.
    import argparse

    parser = argparse.ArgumentParser(description='MGN')
    parser.add_argument('--num_classes', type=int, default=751, help='')
    parser.add_argument('--bnneck', type=bool, default=True)
    parser.add_argument('--pool', type=str, default='max')
    parser.add_argument('--feats', type=int, default=512)
    parser.add_argument('--drop_block', type=bool, default=True)
    parser.add_argument('--w_ratio', type=float, default=1.0, help='')

    args = parser.parse_args()
    net = MCMP_n(args)
    # net.classifier = nn.Sequential()
    # print([p for p in net.parameters()])
    # a=filter(lambda p: p.requires_grad, net.parameters())
    # print(a)

    print(net)
    input = Variable(torch.FloatTensor(8, 3, 384, 128))
    net.eval()
    output = net(input)
    print(output.shape)
    print('net output size:')
    # print(len(output))
    # for k in output[0]:
    #     print(k.shape)
    # for k in output[1]:
    #     print(k.shape)