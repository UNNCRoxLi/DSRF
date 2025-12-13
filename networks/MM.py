""" Swin Transformer
A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`
    - https://arxiv.org/pdf/2103.14030

Code/weights from https://github.com/microsoft/Swin-Transformer

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
from functools import partial


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)


def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C]
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        _, _, H, W = x.shape

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # to pad the last 3 dimensions,
            # (W_left, W_right, H_top,H_bottom, C_front, C_back)
            x = F.pad(x, (0, self.patch_size[1] - W % self.patch_size[1],
                          0, self.patch_size[0] - H % self.patch_size[0],
                          0, 0))

        # 下采样patch_size倍
        x = self.proj(x)
        _, _, H, W = x.shape
        # flatten: [B, C, H, W] -> [B, C, HW]
        # transpose: [B, C, HW] -> [B, HW, C]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """
        x: B, H*W, C
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # to pad the last 3 dimensions, starting from the last dimension and moving forward.
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C]
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C]
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C]
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C]
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C]
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C]

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1]
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2

        return x, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                                depth=depths[i_layer],
                                num_heads=num_heads[i_layer],
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # x: [B, L, C]
        x = x.view(-1, 80, 80).unsqueeze(1)
        x, H, W = self.patch_embed(x)
        x = self.pos_drop(x)
        out = [x]
        for layer in self.layers:
            x, H, W = layer(x, H, W)
            out.append(x)
        x = self.norm(x)  # [B, L, C]
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1)
        x = self.head(x)
        return out


def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swin_small_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model


def swinB(num_classes: int = 1000, **kwargs):
    # trained ImageNet-1K
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224.pth
    model = SwinTransformer(in_chans=1,
                            patch_size=4,
                            window_size=7,
                            embed_dim=64,
                            depths=(2, 2, 2),
                            num_heads=(4, 8, 16),
                            num_classes=num_classes,
                            drop_path_rate=0.5,
                            **kwargs)
    return model

class ResNet(nn.Module):

    def __init__(
        self, 
        num_filters=32, 
        block_drop=0.0, 
        classifier_drop=0.0, 
        classifier_hidreduce=1.0,
        in_channels=1,
        num_classes=1,
        num_extra_stages=1
    ):
        super().__init__()

        channels = [num_filters, num_filters*2, num_filters*3, num_filters*4]
        strides = [2, 2, 2, 2]

        # -------------------------------------------------------------------
        # frame encoder 

        self.inplanes = in_channels
        self.num_stages = len(strides)
        self.num_extra_stages = num_extra_stages

        for l in range(self.num_stages):
            setattr(
                    self, "res"+str(l), 
                    self._make_layer(
                        channels[l], 
                        stride = strides[l], 
                        block = ResBlock, 
                        dropout = block_drop
                    )
                )
        # -------------------------------------------------------------------

        for l in range(self.num_extra_stages):            
            setattr(
                self, "res"+str(4+l), 
                self._make_layer(
                    self.inplanes, 
                    stride=1,
                    block=ResBlock, 
                    dropout=block_drop
                )
            )

        self.featr_dims = 1024

        self.classifier = Classifier(
            self.featr_dims, num_classes, 
            norm_layer=nn.BatchNorm1d, 
            dropout=classifier_drop, 
            hidreduce=classifier_hidreduce
        )

        self.in_channels = in_channels
        self.ou_channels = 8


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_layer(self, planes, stride, block, dropout):
        downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2) if stride == 2 else nn.Identity(),
            ConvNormAct(self.inplanes, planes, 1, 0, activate=False),
        )

        stage = block(self.inplanes, planes, downsample, stride=stride, dropout=dropout)

        self.inplanes = planes

        return stage


    def forward(self, x):

        b, n, h, w = x.size()
        if self.in_channels == 1:
            x = x.reshape(b*n, 1, h, w)

        for l in range(self.num_stages+self.num_extra_stages):
            x = getattr(self, "res"+str(l))(x)

        _, _, h, w = x.size()
        # for raven
        if self.ou_channels == 8:
            x = convert_to_rpm_matrix_v9(x, b, h, w)

        x = x.reshape(b, self.ou_channels, -1)
        x = F.adaptive_avg_pool1d(x, self.featr_dims)    
        x = x.reshape(-1, self.featr_dims)
        
        out = self.classifier(x)

        if self.ou_channels == 1:
            return out
        else:
            return out.view(b, self.ou_channels)

def convert_to_rpm_matrix_v9(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 16, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:8], output[:,i].unsqueeze(1)), dim=1) for i in range(8, 16)], 
        dim=1
    )

    return output


def convert_to_rpm_matrix_v6(input, b, h, w):
    # b: batch
    # h: height
    # w: width
    output = input.reshape(b, 9, -1, h, w)
    output = torch.stack(
        [torch.cat((output[:,:5], output[:,i].unsqueeze(1)), dim=1) for i in range(5, 9)], 
        dim=1
    )

    return output


def ConvNormAct(
        inplanes, ouplanes, kernel_size=3, 
        padding=0, stride=1, activate=True
    ):

    block = [nn.Conv2d(inplanes, ouplanes, kernel_size, padding=padding, bias=False, stride=stride)]
    block += [nn.BatchNorm2d(ouplanes)]
    if activate:
        block += [nn.ReLU()]
    
    return nn.Sequential(*block)


class ResBlock(nn.Module):

    def __init__(self, inplanes, ouplanes, downsample, stride=1, dropout=0.0):
        super().__init__()

        mdplanes = ouplanes

        self.conv1 = ConvNormAct(inplanes, mdplanes, 3, 1, stride=stride)
        self.conv2 = ConvNormAct(mdplanes, mdplanes, 3, 1)
        self.conv3 = ConvNormAct(mdplanes, ouplanes, 3, 1)

        self.downsample = downsample
        self.drop = nn.Dropout(p=dropout) if dropout > 0. else nn.Identity()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.drop(out)
        identity = self.downsample(x)
        out = out + identity
        return out

class Classifier(nn.Module):

    def __init__(self, inplanes, ouplanes, norm_layer=nn.BatchNorm2d, dropout=0.0, hidreduce=1.0):
        super().__init__()

        midplanes = inplanes // hidreduce

        self.mlp = nn.Sequential(
            nn.Linear(inplanes, midplanes, bias=False),
            norm_layer(midplanes),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(midplanes, ouplanes)
        )

    def forward(self, x):
        return self.mlp(x)

class PredictiveReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes, 
        downsample, 
        stride = 1, 
        dropout = 0.0, 
        num_contexts = 8
    ):

        super().__init__()

        self.stride = stride

        md_planes = ou_planes*4
        # self.pconv = ConvNormAct(in_planes, in_planes, (num_contexts, 1))
        self.pconv = ConvNormAct(in_planes, in_planes, (2, 1))
        self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()

        self.downsample = downsample

        self.se_block1=nn.Sequential(
            nn.Linear(32, 32 // 16, False),
            nn.ReLU(),
            nn.Linear(32 // 16, 32, False),
            nn.Sigmoid()
        )
        self.se_block2=nn.Sequential(
            nn.Linear(32, 32 // 16, False),
            nn.ReLU(),
            nn.Linear(32 // 16, 32, False),
            nn.Sigmoid()
        )
        self.se_block3=nn.Sequential(
            nn.Linear(32, 32 // 16, False),
            nn.ReLU(),
            nn.Linear(32 // 16, 32, False),
            nn.Sigmoid()
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        
        b, c, t, l = x.size()
        contexts, choices = x[:,:,:t-1], x[:,:,t-1:]
        first_row_q, first_row_ans = contexts[:,:,0:2], contexts[:,:,2,:].unsqueeze(2)
        second_row_q, second_row_ans = contexts[:,:,3:5], contexts[:,:,5,:].unsqueeze(2)
        third_row_q = contexts[:,:,6:8]
        

        first_row_pred = self.pconv(first_row_q)
        second_row_pred = self.pconv(second_row_q)
        third_row_pred = self.pconv(third_row_q)

        b_1, c_1, _, _ = first_row_pred.size()
        y_1 = self.avg_pool(first_row_pred).view(b_1,c_1)
        y_1 = self.se_block1(y_1).view(b_1,c_1,1,1)

        b_2, c_2, _, _ = second_row_pred.size()
        y_2 = self.avg_pool(second_row_pred).view(b_2,c_2)
        y_2 = self.se_block2(y_2).view(b_2,c_2,1,1)

        b_3, c_3, _, _ = third_row_pred.size()
        y_3 = self.avg_pool(third_row_pred).view(b_3,c_3)
        y_3 = self.se_block3(y_3).view(b_3,c_3,1,1)

        first_pred_err = F.relu(first_row_ans) - first_row_pred
        second_pred_err = F.relu(second_row_ans) - second_row_pred
        third_pred_err = F.relu(choices) - third_row_pred

        out = torch.cat((first_row_q + first_row_q * y_1, first_pred_err, second_row_q + second_row_q * y_2, second_pred_err, third_row_q + third_row_q * y_3, third_pred_err), dim=2)

        # predictions = self.pconv(contexts)
        # prediction_errors = F.relu(choices) - predictions
        
        # out = torch.cat((contexts, prediction_errors), dim=2)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        identity = self.downsample(x)
        # out = out + identity
        
        return out

class Version1ReasoningBlock(nn.Module):

    def __init__(
        self, 
        in_planes, 
        ou_planes, 
        downsample, 
        stride = 1, 
        dropout = 0.0, 
        num_contexts = 8
    ):

        super().__init__()

        self.stride = stride

        md_planes = ou_planes*4
        # self.pconv = ConvNormAct(in_planes, in_planes, (num_contexts, 1))
        self.pconv = ConvNormAct(in_planes, in_planes, (2, 1))
        self.conv1 = ConvNormAct(in_planes, md_planes, 3, 1)
        self.conv2 = ConvNormAct(md_planes, ou_planes, 3, 1)
        self.drop = nn.Dropout(dropout) if dropout > .0 else nn.Identity()

        self.downsample = downsample

    def forward(self, x):
        
        b, c, t, l = x.size()
        contexts, choices = x[:,:,:t-1], x[:,:,t-1:]
        first_row_q, first_row_ans = contexts[:,:,0:2], contexts[:,:,2,:].unsqueeze(2)
        second_row_q, second_row_ans = contexts[:,:,3:5], contexts[:,:,5,:].unsqueeze(2)
        third_row_q = contexts[:,:,6:8]

        first_row_pred = self.pconv(first_row_q)
        second_row_pred = self.pconv(second_row_q)
        third_row_pred = self.pconv(third_row_q)

        first_pred_err = F.relu(first_row_ans) - first_row_pred
        second_pred_err = F.relu(second_row_ans) - second_row_pred
        third_pred_err = F.relu(choices) - third_row_pred

        out = torch.cat((first_row_q, first_pred_err, second_row_q, second_pred_err, third_row_q, third_pred_err), dim=2)

        # predictions = self.pconv(contexts)
        # prediction_errors = F.relu(choices) - predictions
        
        # out = torch.cat((contexts, prediction_errors), dim=2)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.drop(out)
        identity = self.downsample(x)
        out = out + identity
        
        return out

class MM(nn.Module):

    def __init__(self, num_filters=32, block_drop=0.0, classifier_drop=0.0, 
                 classifier_hidreduce=1.0, in_channels=1, num_classes=8, 
                 num_extra_stages=1, reasoning_block=PredictiveReasoningBlock,
                 num_contexts=8):

        super().__init__()

        channels = [num_filters, num_filters*2, num_filters*3, num_filters*4]
        strides = [2, 2, 2, 2]

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

        self.swin_model = swinB()
        self.dim_reducer1 = ConvNormAct(128, 96, 1, 0, activate=False)
        self.dim_reducer2 = ConvNormAct(256, 128, 1, 0, activate=False)

        # -------------------------------------------------------------------
        # predictive coding 
        self.num_extra_stages = num_extra_stages
        self.num_contexts = num_contexts
        self.in_planes = 32
        self.channel_reducer1 = ConvNormAct(channels[1], self.in_planes, 1, 0, activate=False)    
        self.channel_reducer2 = ConvNormAct(channels[2], self.in_planes, 1, 0, activate=False)    
        self.channel_reducer = ConvNormAct(channels[3], self.in_planes, 1, 0, activate=False)      

        for l in range(num_extra_stages*3):
            setattr(
                self, "prb"+str(l), 
                self._make_layer(
                    self.in_planes, stride = 1, 
                    block = reasoning_block, 
                    dropout = block_drop
                )
            )

        # setattr(self, "prb0", self._make_layer(64, stride = 1, block = reasoning_block, dropout = block_drop))
        # setattr(self, "prb1", self._make_layer(128, stride = 1, block = reasoning_block, dropout = block_drop))
        # setattr(self, "prb2", self._make_layer(256, stride = 1, block = reasoning_block, dropout = block_drop))
        # -------------------------------------------------------------------

        self.featr_dims = 1024

        # self.classifier = Classifier(
        #     self.featr_dims, 1, 
        #     norm_layer = nn.BatchNorm1d, 
        #     dropout = classifier_drop, 
        #     hidreduce = classifier_hidreduce
        # )

        self.classifier = Classifier(
            self.featr_dims*3, 1, 
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
        elif downsample and (block == PredictiveReasoningBlock or type(block) == partial):
            downsample = ConvNormAct(self.in_planes, planes, 1, 0, activate = False)
            # downsample = ConvNormAct(planes, 128, 1, 0, activate = False)
        else:
            downsample = nn.Identity()

        if block == PredictiveReasoningBlock or type(block) == partial:
            stage = block(self.in_planes, planes, downsample, stride = stride, 
                          dropout = dropout, num_contexts = self.num_contexts)
            # stage = block(planes, 128, downsample, stride = stride, 
                        #   dropout = dropout, num_contexts = self.num_contexts)
        elif block == ResBlock:
            stage = block(self.in_planes, planes, downsample, stride = stride, dropout = dropout)

        self.in_planes = planes

        return stage

    def forward(self, x, train=False):

        if self.in_channels == 1:
            b, n, h, w = x.size()
            x = x.reshape(b*n, 1, h, w)
        elif self.in_channels == 3:
            b, n, h, w, _ = x.size()
            x = x.reshape(b*n, 3, h, w)

        swin_output = self.swin_model(x)
        swin_output_l = swin_output[0].view(-1, 64,20,20)
        swin_output_m = self.dim_reducer1(swin_output[1].view(-1, 128,10,10))
        swin_output_h = self.dim_reducer2(swin_output[2].view(-1, 256,5,5))

        hie_vis_out = []
        for l in range(4): 
            x = getattr(self, "res"+str(l))(x)
            ###################################
            hie_vis_out.append(x) # [torch.Size([16, 32, 40, 40]), 
                                  #  torch.Size([16, 64, 20, 20]), 
                                  #  torch.Size([16, 128, 10, 10]), 
                                  #  torch.Size([16, 256, 5, 5])]

        hie_vis_mix_out=[hie_vis_out[0], hie_vis_out[1]+swin_output_l, hie_vis_out[2]+swin_output_m, hie_vis_out[3]+swin_output_h]
        # hie_vis_mix_out=[hie_vis_out[0], torch.cat((hie_vis_out[1],swin_output_l),dim=1), torch.cat((hie_vis_out[2],swin_output_m),dim=1), torch.cat((hie_vis_out[3],swin_output_h),dim=1)]

        # x = self.channel_reducer(x) # c ==> 32
        x1 = self.channel_reducer1(hie_vis_mix_out[1])
        x2 = self.channel_reducer2(hie_vis_mix_out[2])
        x3 = self.channel_reducer(hie_vis_mix_out[3])
        
        xs = [x1,x2,x3]

        # _, c, h, w = x.size()

        # if self.num_contexts == 8:
        #     x = convert_to_rpm_matrix_v9(x, b, h, w)
        # else:
        #     x = convert_to_rpm_matrix_v6(x, b, h, w)

        hie_reform_out = []
        # for featr in hie_vis_out[1:]:
        for featr in xs:
            _, c, h, w = featr.size()
            hie_reform_out.append(convert_to_rpm_matrix_v9(featr, b, h, w))
                                  # [torch.Size([1, 8, 9, 64, 20, 20]), 
                                  #  torch.Size([1, 8, 9, 128, 10, 10]), 
                                  #  torch.Size([1, 8, 9, 256, 5, 5])]

        # x = x.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
        # # e.g. [b,9,c,l] -> [b,c,9,l] (l=h*w)
        # x = x.permute(0,2,1,3)

        hie_reshape_out=[]
        for featr in hie_reform_out:
            _, _, _, c, h, w = featr.size()
            reshaped_featr = featr.reshape(b * self.ou_channels, self.num_contexts + 1, -1, h * w)
            reshaped_featr = reshaped_featr.permute(0,2,1,3)
            hie_reshape_out.append(reshaped_featr)
                                  # [torch.Size([8, 64, 9, 400]), 
                                  #  torch.Size([8, 128, 9, 100]), 
                                  #  torch.Size([8, 256, 9, 25])]

        # for l in range(0, self.num_extra_stages): 
        #     x = getattr(self, "prb"+str(l))(x)

        hie_prb_out=[]
        for l in range(len(hie_reshape_out[0:3])): 
            # hie_prb_out.append(getattr(self, "prb"+str(l))(hie_reshape_out[l]))
                                  # [torch.Size([8, 128, 9, 400]), 
                                  #  torch.Size([8, 128, 9, 100]), 
                                  #  torch.Size([8, 128, 9, 25])]
            x = hie_reshape_out[l]
            for m in range(l*self.num_extra_stages, l*self.num_extra_stages+self.num_extra_stages): 
                x = getattr(self, "prb"+str(m))(x)
            hie_prb_out.append(x)
        
        # x = x.reshape(b, self.ou_channels, -1)
        # x = F.adaptive_avg_pool1d(x, self.featr_dims)    
        # x = x.reshape(b * self.ou_channels, self.featr_dims)

        final_featr = []
        for featr in hie_prb_out:
            tmp_featr = featr.reshape(b, self.ou_channels, -1)
            tmp_featr = F.adaptive_avg_pool1d(tmp_featr, self.featr_dims)  
            tmp_featr = tmp_featr.reshape(b * self.ou_channels, self.featr_dims) 
            final_featr.append(tmp_featr)

        final = torch.cat(final_featr, dim=1)


        # out = self.classifier(x)
        out = self.classifier(final)

        # return out.view(b, self.ou_channels)
        errors = torch.zeros([1,10]).cuda()
        return out.view(b, self.ou_channels), errors
        # return out.view(b, self.ou_channels), err_total.tolist()
        # return final
    

def predrnet_raven(**kwargs):
    return PredRNet(**kwargs, num_contexts=8)


def predrnet_analogy(**kwargs):
    return PredRNet(**kwargs, num_contexts=5, num_classes=4)