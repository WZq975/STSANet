import torch
from torch import nn
import numpy as np
import torch.nn.functional as F

class STSANet(nn.Module):
    def __init__(self):
        super(STSANet, self).__init__()
        self.base1 = nn.Sequential(
            SepConv3d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1)),
            BasicConv3d(64, 64, kernel_size=1, stride=1),
            SepConv3d(64, 192, kernel_size=3, stride=1, padding=1),
        )
        self.maxpooling2 = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
        self.base2 = nn.Sequential(
            Mixed_3b(),
            Mixed_3c(),
        )
        self.maxpooling3 = nn.MaxPool3d(kernel_size=(3,3,3), stride=(2,2,2), padding=(1,1,1))
        self.base3 = nn.Sequential(
            Mixed_4b(),
            Mixed_4c(),
            Mixed_4d(),
            Mixed_4e(),
            Mixed_4f(),
        )
        self.maxpooling4 = nn.MaxPool3d(kernel_size=(2,2,2), stride=(2,2,2), padding=(0,0,0))
        self.base4 = nn.Sequential(
            Mixed_5b(),
            Mixed_5c(),
        )

        self.avg_pooling_4 = nn.AdaptiveAvgPool3d((4, 1, 1))
        self.pool_2 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), return_indices=True)
        self.unpool_2 = nn.MaxUnpool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.relu_in = nn.ReLU(inplace=True)
        self.upsample = nn.Upsample(scale_factor=(1, 2, 2), mode='trilinear')
        self.softmax_1 = nn.Softmax(dim=-1)

        self.ln_0_0 = nn.LayerNorm([192, 4, 56, 96])
        self.ln_0_1 = nn.LayerNorm([192, 4, 56, 96])
        self.ln_1_0 = nn.LayerNorm([480, 4, 28, 48])
        self.ln_1_1 = nn.LayerNorm([240, 4, 28, 48])
        self.ln_2_0 = nn.LayerNorm([832, 4, 14, 24])
        self.ln_2_1 = nn.LayerNorm([416, 4, 14, 24])
        self.ln_3_0 = nn.LayerNorm([1024, 4, 7, 12])
        self.ln_3_1 = nn.LayerNorm([512, 4, 7, 12])

        self.branch0_pre = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=(0, 0, 0)),
            nn.ReLU(inplace=True),
        )

        self.branch0_mid = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

        self.branch1_pre = nn.Sequential(
            nn.Conv3d(480, 480, kernel_size=(4, 1, 1), stride=(4, 1, 1), padding=(0, 0, 0)),
            nn.ReLU(inplace=True),
        )

        self.branch1_mid = nn.Sequential(
            nn.Conv3d(480, 240, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

        self.branch2_pre = nn.Sequential(
            nn.Conv3d(832, 832, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)),
            nn.ReLU(inplace=True),
        )

        self.branch2_mid = nn.Sequential(
            nn.Conv3d(832, 416, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

        self.branch3_mid = nn.Sequential(
            nn.Conv3d(1024, 512, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
        )

        self.branch3_beh_0 = nn.Sequential(
            nn.Conv3d(512, 416, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            self.upsample,
        )

        sc_nums = [[[192, 96], [192, 96]], [[480, 224], [240, 128]], [[832, 416], [416, 208]],
                   [[1024, 512], [512, 256]]]
        for i in range(4):
            for j in range(3):
                for k in ['a', 'b']:
                    if j != 2:
                        emb_init, emb_com = sc_nums[i][0][0], sc_nums[i][0][1]
                    else:
                        emb_init, emb_com = sc_nums[i][1][0], sc_nums[i][1][1]

                    self.add_module('branch' + str(i) + '_' + str(j) + '_' + k + '_q', nn.Sequential(
                        nn.Conv3d(emb_init, emb_com, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(emb_com, emb_com, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                        nn.ReLU(inplace=True),
                    ))
                    self.add_module('branch' + str(i) + '_' + str(j) + '_' + k + '_k', nn.Sequential(
                        nn.Conv3d(emb_init, emb_com, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(emb_com, emb_com, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
                        nn.ReLU(inplace=True),
                    ))
                    self.add_module('branch' + str(i) + '_' + str(j) + '_' + k + '_v', nn.Sequential(
                        nn.Conv3d(emb_init, emb_init, kernel_size=(1, 3, 1), stride=(1, 1, 1), padding=(0, 1, 0)),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(emb_init, emb_init, kernel_size=(1, 1, 3), stride=(1, 1, 1), padding=(0, 0, 1)),
                        nn.ReLU(inplace=True),
                    ))

        self.SA_0 = nn.Sequential(
            nn.Conv3d(416 + 416, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Sigmoid()
        )
        self.SA_1 = nn.Sequential(
            nn.Conv3d(240 + 240, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Sigmoid()
        )
        self.SA_2 = nn.Sequential(
            nn.Conv3d(384, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Sigmoid()
        )

        self.CA_0 = nn.Sequential(
            self.avg_pooling_4,
            nn.Conv3d(416+416, 208, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.LayerNorm([208, 4, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(208, 416+416, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Sigmoid()
        )
        self.CA_1 = nn.Sequential(
            self.avg_pooling_4,
            nn.Conv3d(240 + 240, 120, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.LayerNorm([120, 4, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(120, 240 + 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Sigmoid()
        )
        self.CA_2 = nn.Sequential(
            self.avg_pooling_4,
            nn.Conv3d(192+192, 96, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.LayerNorm([96, 4, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv3d(96, 384, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Sigmoid()
        )

        self.inc_0_0 = nn.Sequential(
            nn.Conv3d(416, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        )
        self.inc_0_1 = nn.Sequential(
            nn.Conv3d(416, 240, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(240, 240, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        )
        self.inc_0_2 = nn.Sequential(
            nn.Conv3d(416, 240, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(240, 240, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
        )
        self.inc_0_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=(1,1,1), padding=(1,1,1)),
            nn.Conv3d(416, 240, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        )

        self.inc_1_0 = nn.Sequential(
            nn.Conv3d(240, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        )
        self.inc_1_1 = nn.Sequential(
            nn.Conv3d(240, 192, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        )
        self.inc_1_2 = nn.Sequential(
            nn.Conv3d(240, 192, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
        )
        self.inc_1_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(240, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        )

        self.inc_2_0 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        )
        self.inc_2_1 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
        )
        self.inc_2_2 = nn.Sequential(
            nn.Conv3d(192, 192, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(192, 192, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 2, 2)),
        )
        self.inc_2_3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.Conv3d(192, 192, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
        )

        self.readout_1 = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            self.upsample,
        )

        self.readout_2 = nn.Sequential(

            nn.Conv3d(64, 32, kernel_size=(2, 3, 3), stride=(2, 1, 1), padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.Sigmoid(),
        )

    def da(self, q, k, v):

        batch_size, C, T, h_init, w_init = q.size()
        q = q.view(batch_size, C, T * h_init * w_init)
        batch_size, C, T, h, w = k.size()
        k = k.view(batch_size, C, T * h * w)
        att = torch.bmm(q.permute(0, 2, 1), k)
        att = self.softmax_1(att)
        batch_size, C, T, h, w = v.size()
        v = v.view(batch_size, C, T * h * w)
        add = torch.bmm(att, v.permute(0, 2, 1)).permute(0, 2, 1)
        add = add.view(batch_size, C, T, h_init, w_init)

        return add

    def stsa(self, input_a, input_b, emb_a_q, emb_a_k, emb_a_v, emb_b_q, emb_b_k, emb_b_v):

        out_a_q = emb_a_q(input_a)
        out_a_k = emb_a_k(input_a)
        out_a_v = emb_a_v(input_a)
        out_b_q = emb_b_q(input_b)
        out_b_k = emb_b_k(input_b)
        out_b_v = emb_b_v(input_b)
        add_a = self.da(out_b_q, out_a_k, out_a_v)
        add_b = self.da(out_a_q, out_b_k, out_b_v)

        return add_a, add_b

    def forward(self, x):
        base1 = self.base1(x)  # /2
        out_base1 = self.maxpooling2(base1)
        base2 = self.base2(out_base1)
        out = self.maxpooling3(base2)  # /4
        base3 = self.base3(out)
        out = self.maxpooling4(base3)  # /8
        base4 = self.base4(out)

        base1 = self.branch0_pre(base1)
        base2 = self.branch1_pre(base2)
        base3 = self.branch2_pre(base3)


        # ----------base1-----------
        base1_down, indices = self.pool_2(base1)
        base1_0_down, base1_1_down, base1_2_down, base1_3_down = base1_down.split(1, dim=2)
        out1_0, out1_1 = self.stsa(base1_0_down, base1_1_down, self.branch0_0_a_q, self.branch0_0_a_k,
                                   self.branch0_0_a_v, self.branch0_0_b_q, self.branch0_0_b_k, self.branch0_0_b_v)
        out1_2, out1_3 = self.stsa(base1_2_down, base1_3_down, self.branch0_1_a_q, self.branch0_1_a_k,
                                   self.branch0_1_a_v, self.branch0_1_b_q, self.branch0_1_b_k, self.branch0_1_b_v)
        add = self.unpool_2(torch.cat((out1_0, out1_1, out1_2, out1_3), dim=2), indices=indices)
        base1 = self.ln_0_0(add) + base1

        out_mid = self.branch0_mid(base1)
        out_mid_down, indices = self.pool_2(out_mid)
        base1_one_down, base1_three_down = out_mid_down.split(2, dim=2)
        out1_01, out1_23 = self.stsa(base1_one_down, base1_three_down, self.branch0_2_a_q, self.branch0_2_a_k,
                                   self.branch0_2_a_v, self.branch0_2_b_q, self.branch0_2_b_k, self.branch0_2_b_v)
        add = self.unpool_2(torch.cat((out1_01, out1_23), dim=2), indices=indices)
        out0 = out_mid + self.ln_0_1(add)


        # ---------base2------------------
        base2_0, base2_1, base2_2, base2_3 = base2.split(1, dim=2)

        out1_0, out1_1 = self.stsa(base2_0, base2_1, self.branch1_0_a_q, self.branch1_0_a_k,
                                   self.branch1_0_a_v, self.branch1_0_b_q, self.branch1_0_b_k, self.branch1_0_b_v)

        out1_2, out1_3 = self.stsa(base2_2, base2_3, self.branch1_1_a_q, self.branch1_1_a_k,
                                   self.branch1_1_a_v, self.branch1_1_b_q, self.branch1_1_b_k, self.branch1_1_b_v)
        base2 = self.ln_1_0(torch.cat((out1_0, out1_1, out1_2, out1_3), dim=2)) + base2

        out_mid = self.branch1_mid(base2)
        base2_one, base2_three = out_mid.split(2, dim=2)
        out1_01, out1_23 = self.stsa(base2_one, base2_three, self.branch1_2_a_q, self.branch1_2_a_k,
                                   self.branch1_2_a_v, self.branch1_2_b_q, self.branch1_2_b_k, self.branch1_2_b_v)

        add = torch.cat((out1_01, out1_23), dim=2)
        out1 = self.ln_1_1(add) + out_mid

        #  -------base3-----------
        base3_0, base3_1, base3_2, base3_3 = base3.split(1, dim=2)

        out2_0, out2_1 = self.stsa(base3_0, base3_1, self.branch2_0_a_q, self.branch2_0_a_k,
                                   self.branch2_0_a_v, self.branch2_0_b_q, self.branch2_0_b_k, self.branch2_0_b_v)

        out2_2, out2_3 = self.stsa(base3_2, base3_3, self.branch2_1_a_q, self.branch2_1_a_k,
                                   self.branch2_1_a_v, self.branch2_1_b_q, self.branch2_1_b_k, self.branch2_1_b_v)

        add = self.ln_2_0(torch.cat((out2_0, out2_1, out2_2, out2_3), dim=2))
        base3 = base3 + add

        out_mid = self.branch2_mid(base3)
        # out2 = out_mid
        base3_one, base3_three = out_mid.split(2, dim=2)

        out2_01, out2_23 = self.stsa(base3_one, base3_three, self.branch2_2_a_q, self.branch2_2_a_k,
                                   self.branch2_2_a_v, self.branch2_2_b_q, self.branch2_2_b_k, self.branch2_2_b_v)
        add = self.ln_2_1(torch.cat((out2_01, out2_23), dim=2))
        out2 = out_mid + add

        # -----base4--------
        base4_0, base4_1, base4_2, base4_3 = base4.split(1, dim=2)

        out3_0, out3_1 = self.stsa(base4_0, base4_1, self.branch3_0_a_q, self.branch3_0_a_k,
                                   self.branch3_0_a_v, self.branch3_0_b_q, self.branch3_0_b_k, self.branch3_0_b_v)

        out3_2, out3_3 = self.stsa(base4_2, base4_3, self.branch3_1_a_q, self.branch3_1_a_k,
                                   self.branch3_1_a_v, self.branch3_1_b_q, self.branch3_1_b_k, self.branch3_1_b_v)
        add = self.ln_3_0(torch.cat((out3_0, out3_1, out3_2, out3_3), dim=2))
        base4 = base4 + add

        out_mid = self.branch3_mid(base4)
        base4_one, base4_three = out_mid.split(2, dim=2)
        out3_01, out3_23 = self.stsa(base4_one, base4_three, self.branch3_2_a_q, self.branch3_2_a_k,
                                   self.branch3_2_a_v, self.branch3_2_b_q, self.branch3_2_b_k, self.branch3_2_b_v)
        add = self.ln_3_1(torch.cat((out3_01, out3_23), dim=2))
        out3 = out_mid + add

        # inception
        out3 = self.branch3_beh_0(out3)
        out32 = torch.cat((out3, out2), dim=1)
        att_s = self.SA_0(out32)
        att_c = self.CA_0(out32 * att_s)
        att_c_0, att_c_1 = att_c.split(416, dim=1)
        out32 = out3 * att_c_0 + out2 * att_c_1

        out32_0 = self.inc_0_0(out32)
        out32_1 = self.inc_0_1(out32)
        out32_2 = self.inc_0_2(out32)
        out32_3 = self.inc_0_3(out32)
        out32 = self.upsample(self.relu_in(out32_0 + out32_1 + out32_2 + out32_3))
        out321 = torch.cat((out32, out1), dim=1)
        att_s = self.SA_1(out321)
        att_c = self.CA_1(out321 * att_s)
        att_c_0, att_c_1 = att_c.split(240, dim=1)
        out321 = out32 * att_c_0 + out1 * att_c_1

        out321_0 = self.inc_1_0(out321)
        out321_1 = self.inc_1_1(out321)
        out321_2 = self.inc_1_2(out321)
        out321_3 = self.inc_1_3(out321)
        out321 = self.upsample(self.relu_in(out321_0 + out321_1 + out321_2 + out321_3))
        out = torch.cat((out321, out0), dim=1)
        att_s = self.SA_2(out)
        att_c = self.CA_2(out * att_s)
        att_c_0, att_c_1 = att_c.split(192, dim=1)
        out = out321 * att_c_0 + out0 * att_c_1

        out_0 = self.inc_2_0(out)
        out_1 = self.inc_2_1(out)
        out_2 = self.inc_2_2(out)
        out_3 = self.inc_2_3(out)
        out = self.upsample(self.relu_in(out_0 + out_1 + out_2 + out_3))

        out = self.readout_1(out)
        out = self.readout_2(out)
        out = out.view(out.size(0), out.size(3), out.size(4))

        return out


class BasicConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(BasicConv3d, self).__init__()
        self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class SepConv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super(SepConv3d, self).__init__()
        self.conv_s = nn.Conv3d(in_planes, out_planes, kernel_size=(1,kernel_size,kernel_size), stride=(1,stride,stride), padding=(0,padding,padding), bias=False)
        self.bn_s = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_s = nn.ReLU()

        self.conv_t = nn.Conv3d(out_planes, out_planes, kernel_size=(kernel_size,1,1), stride=(stride,1,1), padding=(padding,0,0), bias=False)
        self.bn_t = nn.BatchNorm3d(out_planes, eps=1e-3, momentum=0.001, affine=True)
        self.relu_t = nn.ReLU()

    def forward(self, x):
        x = self.conv_s(x)
        x = self.bn_s(x)
        x = self.relu_s(x)

        x = self.conv_t(x)
        x = self.bn_t(x)
        x = self.relu_t(x)
        return x

class Mixed_3b(nn.Module):
    def __init__(self):
        super(Mixed_3b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(192, 64, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(192, 96, kernel_size=1, stride=1),
            SepConv3d(96, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(192, 16, kernel_size=1, stride=1),
            SepConv3d(16, 32, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(192, 32, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)

        return out


class Mixed_3c(nn.Module):
    def __init__(self):
        super(Mixed_3c, self).__init__()
        self.branch0 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(256, 128, kernel_size=1, stride=1),
            SepConv3d(128, 192, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(256, 32, kernel_size=1, stride=1),
            SepConv3d(32, 96, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(256, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4b(nn.Module):
    def __init__(self):
        super(Mixed_4b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(480, 192, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(480, 96, kernel_size=1, stride=1),
            SepConv3d(96, 208, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(480, 16, kernel_size=1, stride=1),
            SepConv3d(16, 48, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(480, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4c(nn.Module):
    def __init__(self):
        super(Mixed_4c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 160, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
            SepConv3d(112, 224, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4d(nn.Module):
    def __init__(self):
        super(Mixed_4d, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 128, kernel_size=1, stride=1),
            SepConv3d(128, 256, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 24, kernel_size=1, stride=1),
            SepConv3d(24, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4e(nn.Module):
    def __init__(self):
        super(Mixed_4e, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(512, 112, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(512, 144, kernel_size=1, stride=1),
            SepConv3d(144, 288, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(512, 32, kernel_size=1, stride=1),
            SepConv3d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(512, 64, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_4f(nn.Module):
    def __init__(self):
        super(Mixed_4f, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(528, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(528, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(528, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(528, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5b(nn.Module):
    def __init__(self):
        super(Mixed_5b, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 256, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 160, kernel_size=1, stride=1),
            SepConv3d(160, 320, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 32, kernel_size=1, stride=1),
            SepConv3d(32, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class Mixed_5c(nn.Module):
    def __init__(self):
        super(Mixed_5c, self).__init__()

        self.branch0 = nn.Sequential(
            BasicConv3d(832, 384, kernel_size=1, stride=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv3d(832, 192, kernel_size=1, stride=1),
            SepConv3d(192, 384, kernel_size=3, stride=1, padding=1),
        )
        self.branch2 = nn.Sequential(
            BasicConv3d(832, 48, kernel_size=1, stride=1),
            SepConv3d(48, 128, kernel_size=3, stride=1, padding=1),
        )
        self.branch3 = nn.Sequential(
            nn.MaxPool3d(kernel_size=(3,3,3), stride=1, padding=1),
            BasicConv3d(832, 128, kernel_size=1, stride=1),
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


# model = STSANet()
# print(model)
# print(model.branch_0)

# # model = VSalED()
# # model_dict = model.state_dict()
# pretrained_dict = torch.load('/home/wzq/VSalED-master/weights/s/STSANet_DHF1K.pth')
# pretrained_dict = {k.split(k.split('.')[0] + '.')[1]: v for k, v in pretrained_dict.items()}
#
# # pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# # model_dict.update(pretrained_dict)
# model.load_state_dict(pretrained_dict)
# test = np.zeros((1, 3, 32, 224, 384))
# print(model(torch.FloatTensor(test)).size())
