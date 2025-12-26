import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .network_utils import (
    Classifier, 
    ResBlock, 
    ConvNormAct,
    convert_to_rpm_matrix_v9,
    convert_to_rpm_matrix_v6
)

def safe_softmax(logits, dim=-1):
    orig = logits.dtype
    x = logits.float()
    x = x - x.amax(dim=dim, keepdim=True)
    x = torch.softmax(x, dim=dim)
    return x.to(orig)

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


class GatedAttentionReasoningBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, token_len=25, channel_num=4):
        super(GatedAttentionReasoningBlock, self).__init__()
        self.ds = ConvNormAct(d_model,d_model,(2,1))
        self.lp = nn.Linear(d_model,d_model)
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, d_model * 4)
        self.drop = nn.Dropout(dropout)
        self.drop1 = nn.Dropout(dropout)
        self.num_heads=nhead
        self.head_dim=d_model//nhead

        for i in range(1, 5):
            setattr(self, f"lp{i}", nn.Linear(nhead, nhead))
        for i in range(1, 4):
            setattr(self, f"m{i}", nn.Sequential(nn.Linear(nhead, nhead*4), nn.GELU(), nn.Linear(nhead*4, nhead)))
        self.m4 = nn.Linear(d_model,d_model)
        self.learnable_mask1 = nn.Parameter(torch.ones(1, self.num_heads, token_len, token_len))
        self.learnable_mask2 = nn.Parameter(torch.ones(1, self.num_heads, token_len, token_len))

        self.learnable_tokens1 = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, self.head_dim, token_len), mean=0, std=0.02))
        self.learnable_bias1 = nn.Parameter(torch.zeros(self.num_heads, 1, token_len))

        self.learnable_tokens2 = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(self.num_heads, self.head_dim, token_len), mean=0, std=0.02))
        self.learnable_bias2 = nn.Parameter(torch.zeros(self.num_heads, 1, token_len))


    def forward(self, c1, c2, t):
        b, l, c = c1.shape
        p = self.ds(torch.stack([c1, c2], dim=1).permute(0,3,1,2).contiguous()).reshape(b,c,l).permute(0,2,1)
        
        tq = F.normalize(self.q(t).reshape(b, l, self.num_heads, self.head_dim).permute(0, 2, 1, 3), dim=-1, eps=1e-6)
        t_sc = torch.matmul(tq, tq.transpose(-2, -1)) * F.sigmoid(self.learnable_mask1)
        t_sc1 = (tq @ self.learnable_tokens1) + self.learnable_bias1

        pk, pv, pv1, pv2 = self.kv(p).reshape(b, l, 4 * self.num_heads, self.head_dim).permute(0, 2, 1, 3).chunk(4, dim=1)
        pk, pv, pv1, pv2 = F.normalize(pk, dim=-1, eps=1e-6), F.normalize(pv, dim=-1, eps=1e-6), F.normalize(pv1, dim=-1, eps=1e-6), F.normalize(pv2, dim=-1, eps=1e-6)
        pk_sc = torch.matmul(pk, pk.transpose(-2, -1)) * F.sigmoid(self.learnable_mask2)
        pk_sc1 = (pk @ self.learnable_tokens2) + self.learnable_bias2

        sc = self.drop(t_sc - F.gelu(pk_sc))

        t_h = safe_softmax((self.lp1(t_sc1.permute(0,2,3,1)) / math.sqrt(self.head_dim)).clamp(-50, 50), dim=-1)
        sc1 = self.lp3(t_h*sc.permute(0,2,3,1)).permute(0,3,1,2)
        pk_h = safe_softmax((self.lp2(pk_sc1.permute(0,2,3,1)) / math.sqrt(self.head_dim)).clamp(-50, 50), dim=-1)
        sc2 = self.lp4(pk_h*sc.permute(0,2,3,1)).permute(0,3,1,2)
        sc, sc1, sc2 = self.m1(sc.permute(0,2,3,1)).permute(0,3,1,2), self.m2(sc1.permute(0,2,3,1)).permute(0,3,1,2), self.m3(sc2.permute(0,2,3,1)).permute(0,3,1,2)
        
        sc = safe_softmax((sc / math.sqrt(self.head_dim)).clamp(-50, 50), dim=-1)
        x1 = torch.matmul(sc, pv)
        sc1 = safe_softmax((sc1 / math.sqrt(self.head_dim)).clamp(-50, 50), dim=-1)
        x2 = torch.matmul(sc1, pv1)
        sc2 = safe_softmax((sc2 / math.sqrt(self.head_dim)).clamp(-50, 50), dim=-1)
        x3 = torch.matmul(sc2, pv2)
        x = self.m4((x1+x2+x3).transpose(1, 2).contiguous().reshape(b,l,c))
        
        x, t = F.normalize(x, dim=-1, eps=1e-6), F.normalize(t, dim=-1, eps=1e-6)
        
        x = self.lp(F.gelu(t)-x)
        return x

class DynamicDomainContrastPredictionBlock(nn.Module):

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
        
        self.pre_att = GatedAttentionReasoningBlock(in_planes, num_heads, token_len=token_len, channel_num=channel_num)
        self.conv1 = nn.Sequential(ConvNormAct(9, 9*4, 3, 1, activate=True), ConvNormAct(9*4, 9, 3, 1, activate=True))
        self.conv2 = nn.Sequential(ConvNormAct(in_planes, in_planes*4, 3, 1, activate=True), ConvNormAct(in_planes*4, in_planes, 3, 1, activate=True))
        
        for i in range(1, 10):
            setattr(self, f"position{i}", PositionalEncoding(in_planes))

    
    def forward(self, x):
        shortcut = self.downsample(x)
        # (B,C,T,L) -> (B,T,L,C)
        x = F.normalize(x.permute(0,2,3,1), dim=-1, eps=1e-6)
        g, x = self.lp1(x).chunk(2, dim=-1)
        g = self.m(self.conv1(g))

        pos = [getattr(self, f"position{i}") for i in range(1, 10)]
        c = [pos[i](x[:, i]) for i in range(9)]

        out = []
        for base in range(0, 9, 3):
            a, b, d = c[base], c[base + 1], c[base + 2]
            out.extend([a, b, self.pre_att(a, b, d)])

        x = torch.stack(out, dim=1)

        x = self.lp2(F.gelu(g)*x)
        # ((B,T,L,C) -> (B,C,T,L)
        x = self.conv2(x.permute(0,3,1,2).contiguous())
        x = self.drop(x) + shortcut

        return x


class MultiGranularityRuleMixer(nn.Module):

    def __init__(self, in_planes: int, num_heads = 8, token_len = 25,
                  panel_len = 9, dropout = 0.1,):
        super().__init__()
        assert in_planes % num_heads == 0, f"in_planes ({in_planes}) must be divisible by num_heads ({num_heads})"

        self.downsample = ConvNormAct(in_planes, in_planes, 1, 0, activate=False)
        self.heads = num_heads
        self.head_dim = in_planes // num_heads
        self.token_len = token_len
        self.panel_len = panel_len
        self.num_rows = 3

        # channel-wise mixing (per head)
        self.channel_mix = nn.Sequential(nn.Linear(self.head_dim, self.head_dim*4), nn.GELU(), nn.Linear(self.head_dim*4, self.head_dim))

        # token-wise mixing (along L, default 25)
        self.token_mix = nn.Sequential(nn.Linear(25, 25*4), nn.GELU(), nn.Linear(25*4, 25))
        
        self.mp1 = nn.Linear(in_planes, in_planes)
        self.mp2 = nn.Linear(3, 3)
        self.mp3 = nn.Linear(panel_len, panel_len)

        self.conv = nn.Sequential(
            ConvNormAct(in_planes, in_planes * 4, 3, 1, activate=True),
            ConvNormAct(in_planes * 4, in_planes, 3, 1, activate=True),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):

        shortcut = self.downsample(x)
        x = F.normalize(x.permute(0, 2, 3, 1), dim=-1, eps=1e-6)
        r1, r2, r3 = x.chunk(3, dim=1)
        x = torch.stack([r1,r2,r3], dim=1)
        b, r, t, l, c = x.shape
        x = x.reshape(b, r, t, l, self.heads, self.head_dim).permute(0,1,2,4,3,5)
        x = self.channel_mix(x) + self.token_mix(x.permute(0,1,2,3,5,4)).permute(0,1,2,3,5,4)
        x = self.mp1(x.permute(0,1,2,4,3,5).contiguous().reshape(b,r,t,l,c))
        x = self.mp2(x.permute(0,2,3,4,1))
        x = x.permute(0,4,1,2,3)
        x = torch.cat([x[:,0], x[:,1], x[:,2]], dim=1)
        x = self.mp3(x.permute(0,2,3,1))
        x = self.conv(x.permute(0,2,3,1).contiguous())
        out = self.drop(x) + shortcut
        return out
    

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

        q = F.normalize(self.q(x).reshape(b, t, l, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4), dim=-1, eps=1e-6)
        k, v = self.kv(x).reshape(b, t, l, self.num_heads*2, self.head_dim).permute(0, 1, 3, 2, 4).chunk(2, dim=2)
        k, v = F.normalize(k, dim=-1, eps=1e-6), F.normalize(v, dim=-1, eps=1e-6)

        atten = q @ k.transpose(-2, -1)
        atten = self.drop(safe_softmax((atten / math.sqrt(self.head_dim)).clamp(-50, 50), dim=-1))
        x = (atten @ v)

        x = self.m(x.permute(0,1,3,2,4).contiguous().reshape(b,t,l,c))
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
        for i in range(1, 10):
            setattr(self, f"position{i}", PositionalEncoding(in_planes))
        self.drop = nn.Dropout(dropout)
    
    def forward(self, x):
        b,c,t,l = x.shape
        shortcut = self.downsample(x)
        x = x.permute(0,2,3,1)
        x = torch.stack([getattr(self, f"position{i+1}")(x[:, i]) for i in range(9)], dim=1)
        x = self.selfatten(x).permute(0,3,1,2)
        out = self.drop(x)+shortcut
        out = self.m(out.permute(0,2,3,1)).permute(0,3,1,2).contiguous()
        return out

class ReasoningPyramid(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes,
        number_gp = 1, 
        dropout = 0.1, 
        num_heads = 8
    ):

        super().__init__()
        # DDCP
        for l in range(number_gp):
            setattr(
                self, "ddcp"+str(l), 
                DynamicDomainContrastPredictionBlock(in_planes, num_heads=num_heads)
            )
        self.number_gp = number_gp
        # MRM
        self.mrm = MultiGranularityRuleMixer(in_planes=in_planes, num_heads=num_heads)


    def forward(self, x, train=False):
        # DDCP
        for l in range(0, self.number_gp):
            # e.g. [b,9,c,l] -> [b,c,9,l] (l=h*w)
            x = getattr(self, "ddcp"+str(l))(x)
        # MRM
        out = self.mrm(x)
        
        return out
    

class DSRF(nn.Module):

    def __init__(self, num_filters=32, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8, 
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

        first_stage_channels = self.dsrf_pyramid[0] * self.dsrf_per_view
        self.channel_reducer = ConvNormAct(128, first_stage_channels, 1, 0, activate=False)

        self.reducers = nn.ModuleList()

        for i in range(depth - 1):
            channel_in = self.dsrf_pyramid[i]   * self.dsrf_per_view
            channel_out = self.dsrf_pyramid[i+1] * self.dsrf_per_view
            self.reducers.append(ConvNormAct(channel_in, channel_out, 1, 0, activate=False))

        for stage_idx, views in enumerate(self.dsrf_pyramid, start=1):
            for v in range(views):
                setattr(self, f"RP{stage_idx}_{v}", ReasoningPyramid(self.dsrf_per_view, self.dsrf_per_view, num_heads=8))

        channel_last = self.dsrf_pyramid[-1] * self.dsrf_per_view
        self.atten_last = Alignment(channel_last)

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
        x = x.permute(0,2,1,3).contiguous()
        x = self.channel_reducer(x)

        for stage_idx in range(len(self.dsrf_pyramid)):
            views = self.dsrf_pyramid[stage_idx]
            _, c, t, l = x.shape
            assert c % views == 0
            c_view = c // views
            assert c_view == self.dsrf_per_view

            x = x.reshape(-1, views, c_view, t, l)

            xs = []
            for v in range(views):
                x_v = getattr(self, f"RP{stage_idx+1}_{v}")(x[:, v].contiguous())
                xs.append(x_v)

            x = torch.cat(xs, dim=1)

            if stage_idx < len(self.dsrf_pyramid) - 1:
                x = self.reducers[stage_idx](x)

        x = self.atten_last(x)

        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, 1024)
        
        x = x.reshape(b * self.ou_channels, self.featr_dims)

        out = self.classifier(x)

        return out.view(b, self.ou_channels)
    

def DSRF_raven(**kwargs):
    return DSRF(**kwargs, num_contexts=8)


