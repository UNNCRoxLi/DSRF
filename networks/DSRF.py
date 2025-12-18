from functools import partial
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from .network_utils import (
    Classifier, 
    ResBlock, 
    ConvNormAct,
    convert_to_rpm_matrix_v9,
    convert_to_rpm_matrix_v6,
    LinearNormAct
)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:, :x.size(1), :]


class PredictionAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, token_len=25, channel_num=4):
        super(PredictionAttention, self).__init__()
        self.ds = ConvNormAct(d_model,d_model,(2,1))
        self.lp = nn.Linear(d_model,d_model)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, d_model * 4)
        self.drop = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.num_heads=nhead
        self.head_dim=d_model//nhead

        self.lp1 = nn.Linear(nhead, nhead)
        self.lp2 = nn.Linear(nhead, nhead)
        self.lp3 = nn.Linear(nhead, nhead)
        self.lp4 = nn.Linear(nhead, nhead)
        self.m1 = nn.Sequential(nn.Linear(nhead, nhead*4), nn.GELU(), nn.Linear(nhead*4, nhead))
        self.m2 = nn.Sequential(nn.Linear(nhead, nhead*4),nn.GELU(), nn.Linear(nhead*4, nhead))
        self.m3 = nn.Sequential(nn.Linear(nhead, nhead*4),nn.GELU(), nn.Linear(nhead*4, nhead))
        self.m4 = nn.Linear(d_model,d_model)
        # self.m5 = nn.Linear(d_model,d_model)
        self.learnable_mask1 = nn.Parameter(torch.ones(1, self.num_heads, token_len, token_len))
        self.learnable_mask2 = nn.Parameter(torch.ones(1, self.num_heads, token_len, token_len))
        # self.learnable_tokens = nn.Parameter(
        #     nn.init.trunc_normal_(torch.empty(self.num_heads, self.head_dim, token_len), mean=0, std=0.02))
        # self.learnable_bias = nn.Parameter(torch.zeros(self.num_heads, 1, token_len))

        self.learnable_tokens1 = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, self.head_dim, token_len), mean=0, std=0.02))
        self.learnable_bias1 = nn.Parameter(torch.zeros(self.num_heads, 1, token_len))

        self.learnable_tokens2 = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, self.head_dim, token_len), mean=0, std=0.02))
        self.learnable_bias2 = nn.Parameter(torch.zeros(self.num_heads, 1, token_len))


    def forward(self, c1, c2, t):
        b, l, c = c1.shape
        p = self.ds(torch.stack([c1, c2], dim=1).permute(0,3,1,2)).reshape(b,c,l).permute(0,2,1)

        # t = t.reshape(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # t_ = (t @ self.learnable_tokens) + self.learnable_bias
        # t_ = F.softmax(t_ / math.sqrt(self.head_dim), dim=-1)
        # t = (t_ @ t).transpose(1, 2).reshape(b,l,c)
        # t = self.m5(t)
        
        tq = F.normalize(self.q(t).reshape(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1)
        t_sc = torch.matmul(tq, tq.transpose(-2, -1)) * F.sigmoid(self.learnable_mask1)
        t_sc1 = (tq @ self.learnable_tokens1) + self.learnable_bias1

        pk, pv, pv1, pv2 = self.kv(p).reshape(b, l, 4 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).chunk(4, dim=1)
        pk, pv, pv1, pv2 = F.normalize(pk, dim=-1), F.normalize(pv, dim=-1), F.normalize(pv1, dim=-1), F.normalize(pv2, dim=-1)
        pk_sc = torch.matmul(pk, pk.transpose(-2, -1)) * F.sigmoid(self.learnable_mask2)
        pk_sc1 = (pk @ self.learnable_tokens2) + self.learnable_bias2

        sc = self.drop(t_sc - F.gelu(pk_sc))

        t_h = F.softmax(self.lp1(t_sc1.permute(0,2,3,1)) / math.sqrt(self.head_dim), dim=-1)
        sc1 = self.lp3(t_h*sc.permute(0,2,3,1)).permute(0,3,1,2)
        pk_h = F.softmax(self.lp2(pk_sc1.permute(0,2,3,1)) / math.sqrt(self.head_dim), dim=-1)
        sc2 = self.lp4(pk_h*sc.permute(0,2,3,1)).permute(0,3,1,2)
        sc, sc1, sc2 = self.m1(sc.permute(0,2,3,1)).permute(0,3,1,2), self.m2(sc1.permute(0,2,3,1)).permute(0,3,1,2), self.m3(sc2.permute(0,2,3,1)).permute(0,3,1,2)
        
        sc = F.softmax(sc / math.sqrt(self.head_dim), dim=-1)
        x1 = torch.matmul(sc, pv)
        sc1 = F.softmax(sc1 / math.sqrt(self.head_dim), dim=-1)
        x2 = torch.matmul(sc1, pv1)
        sc2 = F.softmax(sc2 / math.sqrt(self.head_dim), dim=-1)
        x3 = torch.matmul(sc2, pv2)
        x = self.m4((x1+x2+x3).transpose(1, 2).reshape(b,l,c))
        # t = self.m5(t)
        x, t = F.normalize(x, dim=-1), F.normalize(t, dim=-1)
        
        x = self.lp(F.gelu(t)-x)
        return x

class GatedPredictionAttentionBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        dropout = 0.0, 
        num_heads = 8,
        token_len=25,
        channel_num=4,
    ):
        super().__init__()
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.lp1 = nn.Linear(in_planes, in_planes*2)
        self.lp2 = nn.Linear(in_planes, in_planes)
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
        
        self.pre_att = PredictionAttention(in_planes, num_heads, token_len=token_len, channel_num=channel_num)
        self.conv1 = nn.Sequential(ConvNormAct(9, 9*4, 3, 1, activate=True), ConvNormAct(9*4, 9, 3, 1, activate=True))
        self.conv2 = nn.Sequential(ConvNormAct(in_planes, in_planes*4, 3, 1, activate=True), ConvNormAct(in_planes*4, in_planes, 3, 1, activate=True))

        self.position1 = PositionalEncoding(in_planes)
        self.position2 = PositionalEncoding(in_planes)
        self.position3 = PositionalEncoding(in_planes)
        self.position4 = PositionalEncoding(in_planes)
        self.position5 = PositionalEncoding(in_planes)
        self.position6 = PositionalEncoding(in_planes)
        self.position7 = PositionalEncoding(in_planes)
        self.position8 = PositionalEncoding(in_planes)
        self.position9 = PositionalEncoding(in_planes)
    
    def forward(self, x):
        shortcut = self.downsample(x)
        # (B,C,T,L) -> (B,T,L,C)
        x = F.normalize(x.permute(0,2,3,1), dim=-1)
        g, x = self.lp1(x).chunk(2, dim=-1)
        g = self.m(self.conv1(g))
        c1, c2, c3 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2])
        c4, c5, c6 = self.position4(x[:,3]), self.position5(x[:,4]), self.position6(x[:,5])
        c7, c8, c9 = self.position7(x[:,6]), self.position8(x[:,7]), self.position9(x[:,8])
        # c1, c2, c3 = x[:,0], x[:,1], x[:,2]
        # c4, c5, c6 = x[:,3], x[:,4], x[:,5]
        # c7, c8, c9 = x[:,6], x[:,7], x[:,8]
        
        e1 = self.pre_att(c1, c2, c3)
        e2 = self.pre_att(c4, c5, c6)
        e3 = self.pre_att(c7, c8, c9)
        x = torch.stack([c1, c2, e1, c4, c5, e2, c7, c8, e3], dim=1)
        x = self.lp2(F.gelu(g)*x)
        # ((B,T,L,C) -> (B,C,T,L)
        x = self.conv2(x.permute(0,3,1,2))
        x = self.drop(x) + shortcut

        
        e = (e1+e2+e3).mean(2).mean(1).mean(0)
        return x, e


class SelfAttention(nn.Module):
    def __init__(
        self,
        in_planes,
        dropout = 0.1,
        num_heads = 8
    ): 
        super().__init__()
        self.q = nn.Linear(in_planes, in_planes)
        self.kv = nn.Linear(in_planes, in_planes*2)
        self.num_heads=num_heads
        self.head_dim=in_planes//num_heads
        self.m = nn.Linear(in_planes, in_planes)
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        b,t,l,c = x.shape
        shortcut = x
        q = F.normalize(self.q(x).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1)
        k, v = self.kv(x).reshape(b, t, l, self.num_heads*2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1), F.normalize(v, dim=-1)

        atten = self.drop(q @ k.transpose(-2, -1))
        atten = F.softmax(atten / math.sqrt(self.head_dim), dim=-1)
        x = (atten @ v)

        x = self.m(x.permute(0,1,3,2,4).reshape(b,t,l,c))
        return x


class Alignment(nn.Module):

    def __init__(
        self,
        in_planes,
        dropout = 0.1,
        num_heads = 8
    ): 
        super().__init__()
        self.selfatten = SelfAttention(in_planes)
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.m = nn.Sequential(nn.Linear(in_planes, in_planes*4), nn.LayerNorm(in_planes*4), nn.GELU())
        # self.lp = nn.Sequential(nn.Linear(in_planes, in_planes//2), nn.LayerNorm(in_planes//2), nn.GELU())
        self.position1 = PositionalEncoding(in_planes)
        self.position2 = PositionalEncoding(in_planes)
        self.position3 = PositionalEncoding(in_planes)
        self.position4 = PositionalEncoding(in_planes)
        self.position5 = PositionalEncoding(in_planes)
        self.position6 = PositionalEncoding(in_planes)
        self.position7 = PositionalEncoding(in_planes)
        self.position8 = PositionalEncoding(in_planes)
        self.position9 = PositionalEncoding(in_planes)
        # self.conv = nn.Sequential(ConvNormAct(in_planes, in_planes*4, 3, 1, activate=True), ConvNormAct(in_planes*4, in_planes, 3, 1, activate=True))
        self.drop = nn.Dropout(dropout)
        # self.conv1 = nn.Sequential(ConvNormAct(9, 9*4, 3, 1, activate=True), ConvNormAct(9*4, 9, 3, 1, activate=True))
        # self.lp1 = nn.Linear(in_planes, in_planes*2)
        # self.lp2 = nn.Linear(in_planes, in_planes)
    
    def forward(self, x):
        b,c,t,l = x.shape
        shortcut = self.downsample(x)
        x = x.permute(0,2,3,1)
        # g, x = self.lp1(x).chunk(2, dim=-1)
        # g = self.m(self.conv1(g))
        c1, c2, c3 = self.position1(x[:,0]), self.position2(x[:,1]), self.position3(x[:,2])
        c4, c5, c6 = self.position4(x[:,3]), self.position5(x[:,4]), self.position6(x[:,5])
        c7, c8, c9 = self.position7(x[:,6]), self.position8(x[:,7]), self.position9(x[:,8])
        x = torch.stack([c1, c2, c3, c4, c5, c6, c7, c8, c9], dim=1)
        x = self.selfatten(x).permute(0,3,1,2)
        # x = self.m(x).permute(0,3,1,2)
        # x = self.lp2(F.gelu(g)*x).permute(0,3,1,2)
        out = self.drop(x)+shortcut
        out = self.m(out.permute(0,2,3,1)).permute(0,3,1,2)
        return out

class PredictiveReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes,
        number_gp = 1, 
        dropout = 0.1, 
        num_heads = 8
    ):

        super().__init__()
        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        for l in range(number_gp):
            setattr(
                self, "gp"+str(l), 
                GatedPredictionAttentionBlock(in_planes, num_heads=num_heads)
            )
        self.number_gp = number_gp

        self.heads = num_heads
        self.head_dim = in_planes // num_heads
        self.align = Alignment(in_planes)

        self.mp1 = nn.Linear(in_planes, in_planes)
        self.mp2 = nn.Linear(3, 3)
        self.mp3 = nn.Linear(9, 9)
        
        self.conv = nn.Sequential(ConvNormAct(in_planes, in_planes*4, 3, 1, activate=True), ConvNormAct(in_planes*4, in_planes, 3, 1, activate=True))
        self.channel_mix = nn.Sequential(nn.Linear(self.head_dim, self.head_dim*4), nn.GELU(), nn.Linear(self.head_dim*4, self.head_dim))
        self.token_mix = nn.Sequential(nn.Linear(25, 25*4), nn.GELU(), nn.Linear(25*4, 25))
        self.drop = nn.Dropout(dropout)


    def forward(self, x, train=False):
        b, c, t, l = x.size()
        for l in range(0, self.number_gp):
            # e.g. [b,9,c,l] -> [b,c,9,l] (l=h*w)
            x, errors = getattr(self, "gp"+str(l))(x)
        # x = self.align(x)
        shortcut = self.downsample(x)
        x = F.normalize(x.permute(0, 2, 3, 1), dim=-1)
        r1, r2, r3 = x.chunk(3, dim=1)
        x = torch.stack([r1,r2,r3], dim=1)
        b, r, t, l, c = x.shape
        x = x.reshape(b, r, t, l, self.heads, self.head_dim).permute(0,1,2,4,3,5)
        x = self.channel_mix(x) + self.token_mix(x.permute(0,1,2,3,5,4)).permute(0,1,2,3,5,4)
        x = self.mp1(x.permute(0,1,2,4,3,5).reshape(b,r,t,l,c))
        x = self.mp2(x.permute(0,2,3,4,1))
        x = x.permute(0,4,1,2,3)
        x = torch.cat([x[:,0], x[:,1], x[:,2]], dim=1)
        x = self.mp3(x.permute(0,2,3,1))
        x = self.conv(x.permute(0,2,3,1))

        out = self.drop(x) + shortcut
        # out = x
        
        return out, errors
    

class DSRF(nn.Module):

    def __init__(self, num_filters=32, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8, 
                #  num_extra_stages=1, reasoning_block=PredictiveReasoningBlock,
                 dsrf_pyramid=(8,4,1), dsrf_per_view=32,
                 num_contexts=8):

        super().__init__()

        channels = [num_filters, num_filters*2, num_filters*3, num_filters*4]
        strides = [2, 2, 2, 2]
        self.dsrf_pyramid = list(dsrf_pyramid)
        self.dsrf_per_view = dsrf_per_view
        depth = len(self.dsrf_pyramid)

        # -------------------------------------------------------------------
        # frame encoder 

        self.in_planes = in_channels

        for l in range(len(strides)):
            setattr(
                self, "res"+str(l), 
                self._make_layer(
                    channels[l], stride=strides[l], 
                    block=ResBlock, dropout=block_drop,
                )
            )
        # -------------------------------------------------------------------

        

        # -------------------------------------------------------------------
        # predictive coding 
        num_extra_stages_l = [1, 1, 1, 1]
        self.num_extra_stages_l = num_extra_stages_l
        self.num_contexts = num_contexts
        # self.in_planes = 128
        new_channels = [16, 16+8, 16+8*2, 32]
        # self.mlp = nn.Linear(512*4, 1024)
        # self.atten1 = Alignment(64)
        # self.atten2 = Alignment(32)
        self.atten3 = Alignment(16)
        self.channel_reducer = ConvNormAct(128, 32, 1, 0, activate=False)
        # self.channel_reducer1 = ConvNormAct(128, 64, 1, 0, activate=False)
        # self.channel_reducer2 = ConvNormAct(128, 32, 1, 0, activate=False)

        # for l in range(8):
        #     setattr(
        #         self, "GetC1"+str(l),
        #         ConvNormAct(32, 32, 1, 0, activate=False)
        #     )
        
        # for l in range(4):
        #     setattr(
        #         self, "GetC2"+str(l),
        #         ConvNormAct(32, 32, 1, 0, activate=False)
        #     )
        
        # for l in range(1):
        #     setattr(
        #         self, "GetC3"+str(l),
        #         ConvNormAct(32, 32, 1, 0, activate=False)
        #     )
        

        self.reduce1 = ConvNormAct(32, 16, 1, 0, activate=False)
        self.reduce2 = ConvNormAct(16, 16, 1, 0, activate=False)
        # self.reduce3 = ConvNormAct(32, 32, 1, 0, activate=False)
        # self.reduce4 = ConvNormAct(64, 64, 1, 0, activate=False)

        for l in range(2):
            setattr(
                self, "GRB1"+str(l), 
                PredictiveReasoningBlock(16, 16, num_heads=8)
            )
        
        for l in range(1):
            setattr(
                self, "GRB2"+str(l), 
                PredictiveReasoningBlock(16, 16, num_heads=8)
            )
        
        # for l in range(1):
        #     setattr(
        #         self, "GRB3"+str(l), 
        #         PredictiveReasoningBlock(16, 16, num_heads=8)
        #     )
        
        # -------------------------------------------------------------------

        self.featr_dims = 1024

        self.classifier = Classifier(
            self.featr_dims, 1, 
            norm_layer = nn.BatchNorm1d, 
            dropout = classifier_drop, 
            hidreduce = classifier_hidreduce
        )

        self.in_channels = in_channels
        self.ou_channels = num_classes


    def _make_layer(self, planes, stride, dropout, block, downsample=True):
        if downsample and block == ResBlock:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size = 2, stride = stride) if stride != 1 else nn.Identity(),
                ConvNormAct(self.in_planes, planes, 1, 0, activate = False, stride=1),
            )
        else:
            downsample = nn.Identity()

        if block == ResBlock:
            stage = block(self.in_planes, planes, downsample, stride = stride, dropout = dropout)

        self.in_planes = planes

        return stage

    def forward(self, x, train=False):
        # print(x.shape)
        # print(self.in_channels)
        if self.in_channels == 1:
            b, n, h, w = x.size()
            x = x.reshape(b*n, 1, h, w)
        elif self.in_channels == 3:
            b, n, _, h, w = x.size()
            x = x.reshape(b*n, 3, h, w)

        for l in range(4):
            x = getattr(self, "res"+str(l))(x)

        if self.num_contexts == 8:
            _, c, h, w = x.size()
            x = convert_to_rpm_matrix_v9(x, b, h, w)
        else:
            x = convert_to_rpm_matrix_v6(x, b, h, w)
        
        x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h*w)
        x = x.permute(0,2,1,3)
        x = self.channel_reducer(x)
        
        x_l1 = []
        _,c,t,l = x.shape
        x = x.reshape(-1,2,c//2,t,l)
        for l in range(2):
            # x_ = getattr(self, "GetC1"+str(l))()
            x_, e = getattr(self, "GRB1"+str(l))(x[:,l])
            x_l1.append(x_)

        x1r = self.reduce1(torch.cat(x_l1, dim=1))
        # x1r = self.atten1(x1r)
        # x1r = self.mix1(x1r)

        x_l2 = []
        _,c,t,l = x1r.shape
        x1 = x1r.reshape(-1,1,c//1,t,l)
        for l in range(1):
            # x_ = getattr(self, "GetC2"+str(l))()
            x_, e = getattr(self, "GRB2"+str(l))(x1[:,l])
            x_l2.append(x_)

        x2r = self.reduce2(torch.cat(x_l2, dim=1))
        # x2r = self.atten2(x2r)
        # x2r = self.mix2(x2r)

        # x_l3 = []
        # _,c,t,l = x2r.shape
        # x2 = x2r.reshape(-1,1,c//1,t,l)
        # for l in range(1):
        #     # x_ = getattr(self, "GetC3"+str(l))()
        #     x_, e = getattr(self, "GRB3"+str(l))(x2[:,l])
        #     x_l3.append(x_)

        # x3r = torch.cat(x_l3, dim=1)
        x2r = self.atten3(x2r)
        # x3r = self.mix3(x3r)
        x = x2r
        # print(x.shape)
        # x = self.reduce4(x)
        # x = getattr(self, "GRB4"+str(0))(x)
        # x = self.mix4(x)

        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, 1024)

        # x1r = x1r.reshape(b, self.ou_channels, -1)
        # x1r = F.adaptive_avg_pool1d(x1r, 512)

        # x2r = x2r.reshape(b, self.ou_channels, -1)
        # x2r = F.adaptive_avg_pool1d(x2r, 512)

        # x3r = x3r.reshape(b, self.ou_channels, -1)
        # x3r = F.adaptive_avg_pool1d(x3r, 512)

        # x = self.mlp(torch.cat([x1r,x2r,x3r,x], dim=-1))
        
        x = x.reshape(b * self.ou_channels, self.featr_dims)

        out = self.classifier(x)

        errors = e

        return out.view(b, self.ou_channels), errors
    

def DSRF_raven(**kwargs):
    return DSRF(**kwargs, num_contexts=8)


def DSRF_analogy(**kwargs):
    return DSRF(**kwargs, num_contexts=5, num_classes=4)

