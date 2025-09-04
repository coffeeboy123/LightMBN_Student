import copy
import torch
from torch import nn
from .osnet import osnet_x0_25, OSBlock
from .attention import BatchDrop, PAM_Module, CAM_Module, SE_Module, Dual_Module, BatchFeatureErase_Top_osnet_x0_25
from .bnneck import BNNeck, BNNeck3
from torch.nn import functional as F

from torch.autograd import Variable
from .partweightgate import PartWeightGate_256

class LMBN_n_osnet_x0_25_student(nn.Module):
    def __init__(self, args):
        super(LMBN_n_osnet_x0_25_student, self).__init__()

        osnet = osnet_x0_25(pretrained=True)

        self.backone = nn.Sequential(
            osnet.conv1,
            osnet.maxpool,
            osnet.conv2,
            osnet.conv3[0]
        )

        conv3 = osnet.conv3[1:]

        self.global_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.partial_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.channel_branch = nn.Sequential(copy.deepcopy(
            conv3), copy.deepcopy(osnet.conv4), copy.deepcopy(osnet.conv5))

        self.global_pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.partial_pooling = nn.AdaptiveAvgPool2d((12, 1))
        self.channel_pooling = nn.AdaptiveAvgPool2d((1, 2))
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        reduction = BNNeck3(args.feats, args.num_classes,
                            args.feats, return_f=True)

        self.reduction_0 = copy.deepcopy(reduction)
        self.reduction_1 = copy.deepcopy(reduction)
        self.reduction_2 = copy.deepcopy(reduction)
        self.reduction_3 = copy.deepcopy(reduction)
        self.reduction_4 = copy.deepcopy(reduction)
        self.reduction_5 = copy.deepcopy(reduction)


        self.no_shared_1 = nn.Sequential(nn.Conv2d(
            args.feats, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU(True))
        self.weights_init_kaiming(self.no_shared_1)

        self.no_shared_2 = nn.Sequential(nn.Conv2d(
            args.feats, args.feats, 1, bias=False), nn.BatchNorm2d(args.feats), nn.ReLU(True))
        self.weights_init_kaiming(self.no_shared_2)

        self.reduction_ch_0 = BNNeck(
            args.feats, args.num_classes, return_f=True)
        self.reduction_ch_1 = BNNeck(
            args.feats, args.num_classes, return_f=True)

        # if args.drop_block:
        #     print('Using batch random erasing block.')
        #     self.batch_drop_block = BatchRandomErasing()
        # print('Using batch drop block.')
        # self.batch_drop_block = BatchDrop(
        #     h_ratio=args.h_ratio, w_ratio=args.w_ratio)
        self.batch_drop_block = BatchFeatureErase_Top_osnet_x0_25(args.feats, OSBlock)

        self.activation_map = args.activation_map

    def forward(self, x):
        # if self.batch_drop_block is not None:
        #     x = self.batch_drop_block(x)

        x = self.backone(x)

        glo = self.global_branch(x)
        par = self.partial_branch(x)
        cha = self.channel_branch(x)

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
        glo = self.average_pooling(glo)  # shape:(batchsize, 512,1,1)
        g_par = self.global_pooling(par)  # shape:(batchsize, 512,1,1)
        p_par = self.partial_pooling(par)  # shape:(batchsize, 512,2,1)
        cha = self.channel_pooling(cha)  # shape:(batchsize, 256,1,1)

        p_head = p_par[:, :, 0:2, :]
        p_upper = p_par[:, :, 2:7, :]
        p_lower = p_par[:, :, 7:12, :]

        p_head = F.adaptive_avg_pool2d(p_head, (1, 1))
        p_upper = F.adaptive_avg_pool2d(p_upper, (1, 1))
        p_lower = F.adaptive_avg_pool2d(p_lower, (1, 1))

        f_glo = self.reduction_0(glo)
        f_p0 = self.reduction_1(g_par)
        f_p1 = self.reduction_2(p_head)
        f_p2 = self.reduction_3(p_upper)
        f_p3 = self.reduction_4(p_lower)
        f_glo_drop = self.reduction_5(glo_drop)
        ################

        c0 = cha[:, :, :, 0:1]
        c1 = cha[:, :, :, 1:2]
        c0 = self.no_shared_1(c0)
        c1 = self.no_shared_2(c1)
        f_c0 = self.reduction_ch_0(c0)
        f_c1 = self.reduction_ch_1(c1)

        ################

        fea = [f_glo[-1], f_glo_drop[-1], f_p0[-1]]


        return [f_glo[1], f_glo_drop[1], f_p0[1], f_p1[1], f_p2[1], f_p3[1], f_c0[1], f_c1[1]], fea, torch.stack([f_glo[0], f_glo_drop[0], f_p0[0], f_p1[0], f_p2[0], f_p3[0], f_c0[0], f_c1[0]], dim=2)

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
