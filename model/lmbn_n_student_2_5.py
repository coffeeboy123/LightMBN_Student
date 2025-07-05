import copy
import torch
from torch import nn
from .osnet import osnet_x1_0, OSBlock, osnet_x0_25, LightConv3x3
from .attention import BatchDrop, BatchFeatureErase_Top, PAM_Module, CAM_Module, SE_Module, Dual_Module
from .bnneck import BNNeck, BNNeck3, BNNeck3_depthwise, BNNeck3_neck, BNNeck3_classifier, BNNeck_neck, BNNeck_classifier
from torch.nn import functional as F

from torch.autograd import Variable


class LMBN_n_student_2_5(nn.Module):
    def __init__(self, args):
        super(LMBN_n_student_2_5, self).__init__()

        self.n_ch = 2
        self.chs = 512 // self.n_ch

        osnet = osnet_x0_25(pretrained=True)

        self.backone = nn.Sequential(
            osnet.conv1,
            osnet.conv2,
            osnet.maxpool
        )



        self.global_branch = nn.Sequential(copy.deepcopy(osnet.conv3),copy.deepcopy(osnet.conv4), LightConv3x3(128, 512))


        self.partial_branch = nn.Sequential(copy.deepcopy(osnet.conv3),copy.deepcopy(osnet.conv4), LightConv3x3(128, 512))

        self.channel_branch = nn.Sequential(copy.deepcopy(osnet.conv3),copy.deepcopy(osnet.conv4), LightConv3x3(128, 512))

        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.partial_pooling = nn.AdaptiveAvgPool2d((2, 1))
        self.channel_pooling = nn.AdaptiveAvgPool2d((1, 1))

        reduction_neck = BNNeck3_neck(512, args.feats, return_f=True)

        self.reduction_0_neck = copy.deepcopy(reduction_neck)
        self.reduction_1_neck = copy.deepcopy(reduction_neck)
        self.reduction_2_neck = copy.deepcopy(reduction_neck)
        self.reduction_3_neck = copy.deepcopy(reduction_neck)
        self.reduction_4_neck = copy.deepcopy(reduction_neck)



        self.reduction_0_classifier = BNNeck3_classifier(512*3, args.num_classes,
                                                  feat_dim=512*3, return_f=True)
        self.reduction_1_classifier = BNNeck3_classifier(512*2, args.num_classes,
                                                  feat_dim=512*2, return_f=True)

        self.ch0_pointwise = nn.Sequential(nn.Conv2d(
            self.chs, args.feats, 1, bias=False, groups=256), nn.BatchNorm2d(args.feats), nn.ReLU(True))
        self.weights_init_kaiming(self.ch0_pointwise)

        self.ch1_pointwise = nn.Sequential(nn.Conv2d(
            self.chs, args.feats, 1, bias=False, groups=256), nn.BatchNorm2d(args.feats), nn.ReLU(True))
        self.weights_init_kaiming(self.ch1_pointwise)

        self.reduction_ch_0_neck = BNNeck_neck(
            args.feats, return_f=True)

        self.reduction_ch_1_neck = BNNeck_neck(
            args.feats, return_f=True)

        self.reduction_ch_0_classifier = BNNeck_classifier(
            512*2, args.num_classes, return_f=True)

        # if args.drop_block:
        #     print('Using batch random erasing block.')
        #     self.batch_drop_block = BatchRandomErasing()
        # print('Using batch drop block.')
        # self.batch_drop_block = BatchDrop(
        #     h_ratio=args.h_ratio, w_ratio=args.w_ratio)
        self.batch_drop_block = BatchFeatureErase_Top(512, OSBlock)

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
        glo = self.channel_pooling(glo)  # shape:(batchsize, 512,1,1)
        g_par = self.global_pooling(par)  # shape:(batchsize, 512,1,1)
        p_par = self.partial_pooling(par)  # shape:(batchsize, 512,2,1)
        cha = self.channel_pooling(cha)  # shape:(batchsize, 256,1,1)

        p0 = p_par[:, :, 0:1, :]
        p1 = p_par[:, :, 1:2, :]

        f_glo_after, f_glo_before = self.reduction_0_neck(glo)
        f_p0_after, f_p0_before = self.reduction_1_neck(g_par)
        f_p1_after, f_p1_before = self.reduction_2_neck(p0)
        f_p2_after, f_p2_before = self.reduction_3_neck(p1)
        f_glo_drop_after, f_glo_drop_before = self.reduction_4_neck(glo_drop)

        p_cat = torch.cat([f_p0_after, f_p1_after, f_p2_after], dim=1)  # [B, 512*3]
        glo_cat = torch.cat([f_glo_after, f_glo_drop_after], dim=1)

        p_cat_score = self.reduction_0_classifier(p_cat)
        glo_cat_score = self.reduction_1_classifier(glo_cat)

        ################

        c0 = cha[:, :self.chs, :, :]
        c1 = cha[:, self.chs:, :, :]
        c0 = self.ch0_pointwise(c0)
        c1 = self.ch1_pointwise(c1)
        f_c0_after, f_c0_before = self.reduction_ch_0_neck(c0)
        f_c1_after, f_c1_before = self.reduction_ch_1_neck(c1)

        c_cat = torch.cat([f_c0_after, f_c1_after], dim=1)
        c_cat_score = self.reduction_ch_0_classifier(c_cat)

        fea = [f_glo_before, f_glo_drop_before, f_p0_before, f_p1_before, f_p2_before, f_c0_before, f_c1_before]

        if not self.training:
            return torch.stack([f_glo_after, f_glo_drop_after, f_p0_after, f_p1_after, f_p2_after, f_c0_after, f_c1_after], dim=2)

        return [p_cat_score, glo_cat_score, c_cat_score], fea

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