from typing import Callable, Optional

import torch
from einops.layers.torch import Rearrange
from torch import nn
import math
from functools import partial

class WildRelationNetwork(nn.Module):
    def __init__(
        self,
        num_channels: int = 32,
        embedding_size: int = 128,
        image_size: int = 160,
        use_layer_norm: bool = False,
        g_depth: int = 3,
        f_depth: int = 2,
        f_dropout_probability: float = 0.0,
    ):
        super(WildRelationNetwork, self).__init__()
        self.embedding_size = embedding_size
        self.group_objects = GroupObjectsIntoPairs()
        self.group_objects_with = GroupObjectsIntoPairsWith()

        self.cnn = nn.Sequential(
            ConvBnRelu(1, num_channels, kernel_size=3, stride=2, padding=1),
            ConvBnRelu(num_channels, num_channels, kernel_size=3, padding=1),
            ConvBnRelu(num_channels, num_channels, kernel_size=3, padding=1),
            ConvBnRelu(num_channels, num_channels, kernel_size=3, padding=1),
            nn.Flatten(start_dim=-3, end_dim=-1),
        )
        conv_dimension = num_channels * ((40 * (image_size // 80)) ** 2)
        self.object_tuple_size = 2 * (conv_dimension + 9)
        self.tag_panel_embeddings = TagPanelEmbeddings()
        self.g = nn.Sequential(
            DeepLinearBNReLU(
                g_depth, self.object_tuple_size, embedding_size, change_dim_first=True
            ),
            Sum(dim=1),
        )
        self.norm = nn.LayerNorm(embedding_size) if use_layer_norm else Identity()
        self.f = nn.Sequential(
            DeepLinearBNReLU(f_depth, embedding_size, embedding_size),
            nn.Dropout(f_dropout_probability),
            nn.Linear(embedding_size, embedding_size),
        )

    def forward(
        self, context: torch.Tensor, answers: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        num_context_panels = context.size(1)
        num_answer_panels = answers.size(1)
        x = torch.cat([context, answers], dim=1)
        batch_size, num_panels, num_channels, height, width = x.shape
        x = x.view(batch_size * num_panels, num_channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, -1)
        x = self.tag_panel_embeddings(x, num_context_panels)
        context_objects = x[:, :num_context_panels, :]
        choice_objects = x[:, num_context_panels:, :]
        context_pairs = self.group_objects(context_objects)
        context_g_out = self.g(context_pairs)
        f_out = torch.zeros(
            (batch_size, num_answer_panels, self.embedding_size), device=x.device
        ).type_as(x)
        for i in range(num_answer_panels):
            context_choice_pairs = self.group_objects_with(
                context_objects, choice_objects[:, i, :]
            )
            context_choice_g_out = self.g(context_choice_pairs)
            relations = context_g_out + context_choice_g_out
            relations = self.norm(relations)
            f_out[:, i, :] = self.f(relations)
        return f_out


class OddOneOutWildRelationNetwork(WildRelationNetwork):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_panels, num_channels, height, width = x.shape
        x = x.view(batch_size * num_panels, num_channels, height, width)
        x = self.cnn(x)
        x = x.view(batch_size, num_panels, -1)
        x = self.tag_panel_embeddings(x, num_panels)

        embedding_dim = x.shape[-1]
        mask = (
            ~torch.eye(num_panels, device=x.device, dtype=torch.bool)
            .unsqueeze(-1)
            .repeat(1, 1, embedding_dim)
        )
        x = torch.stack(
            [
                x.masked_select(m.repeat(batch_size, 1, 1)).view(
                    batch_size, num_panels - 1, embedding_dim
                )
                for m in mask
            ],
            dim=1,
        )  # b p p-1 d
        x = x.view((batch_size * num_panels), (num_panels - 1), -1)

        x = self.group_objects(x)
        x = self.g(x)
        x = self.norm(x)
        x = self.f(x)
        x = x.view(batch_size, num_panels, -1)
        return x

class StructureAwareLayer(nn.Module):
    def __init__(
        self,
        out_channels: int,
        kernel_size: int = 1,
        num_rows: int = 6,
        num_cols: int = 420,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = kernel_size
        self.weights = torch.nn.Parameter(
            torch.randn(num_rows, num_cols, out_channels, kernel_size)
        )
        self.biases = torch.nn.Parameter(torch.randn(out_channels))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.biases, -bound, bound)

    def forward(
        self, x: torch.Tensor, num_rows: int = 3, num_cols: int = 3
    ) -> torch.Tensor:
        """
        :param x: a tensor of shape (batch_size, in_channels, dim)
        :return: a tensor of shape (batch_size, out_channels, dim / kernel_size)
        """
        w = self.weights.unfold(
            0, num_rows, num_rows
        )  # (self.num_rows / num_rows, num_cols, out_dim, in_dim, num_rows)
        w = w.unfold(
            1, num_cols, num_cols
        )  # (self.num_rows / num_rows, self.num_cols / num_cols, out_dim, in_dim, num_rows, num_cols)
        w = w.mean((0, 1))  # (out_dim, in_dim, num_rows, num_cols)
        w = w.flatten(start_dim=1)  # (out_dim, in_dim * num_rows * num_cols)
        w = w.transpose(0, 1)  # (in_dim * num_rows * num_cols, out_dim)

        batch_size, in_channels, in_dim = x.shape
        num_groups = in_dim // self.kernel_size
        x = x.view(
            batch_size, in_channels, num_groups, self.kernel_size
        )  # n in_c ng gs
        x = x.transpose(1, 2).contiguous()  # b ng in_c gs
        x = x.flatten(0, 1)  # (b ng) in_c gs
        x = x.flatten(1, 2)  # (b ng) (in_c gs)

        x = torch.einsum("bd,dc->bc", x, w) + self.biases  # (b ng) out_c
        x = x.view(batch_size, num_groups, self.out_channels)  # b ng out_c
        return x.transpose(1, 2)  # b out_c ng

class LinearBNReLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super(LinearBNReLU, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        shape = x.shape
        x = x.flatten(0, -2)
        x = self.bn(x)
        x = x.view(shape)
        x = self.relu(x)
        return x


class NonLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, norm: str = "bn"):
        assert norm in ["bn", "ln", "none"]
        super(NonLinear, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        if norm == "bn":
            self.norm = nn.BatchNorm1d(out_dim)
        elif norm == "ln":
            self.norm = nn.LayerNorm(out_dim)
        else:
            self.norm = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        shape = x.shape
        x = x.flatten(0, -2)
        x = self.norm(x)
        x = x.view(shape)
        x = self.relu(x)
        return x


class DeepLinearBNReLU(nn.Module):
    def __init__(
        self, depth: int, in_dim: int, out_dim: int, change_dim_first: bool = True
    ):
        super(DeepLinearBNReLU, self).__init__()
        layers = []
        if change_dim_first:
            layers += [LinearBNReLU(in_dim, out_dim)]
            for _ in range(depth - 1):
                layers += [LinearBNReLU(out_dim, out_dim)]
        else:
            for _ in range(depth - 1):
                layers += [LinearBNReLU(in_dim, in_dim)]
            layers += [LinearBNReLU(in_dim, out_dim)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class MLP(nn.Module):
    def __init__(
        self,
        depth: int,
        in_dim: int,
        out_dim: int,
        change_dim_first: bool = True,
        norm: str = "bn",
    ):
        assert norm in ["bn", "ln", "none"]
        super(MLP, self).__init__()
        layers = []
        if change_dim_first:
            layers += [NonLinear(in_dim, out_dim, norm)]
            for _ in range(depth - 1):
                layers += [NonLinear(out_dim, out_dim, norm)]
        else:
            for _ in range(depth - 1):
                layers += [NonLinear(in_dim, in_dim, norm)]
            layers += [NonLinear(in_dim, out_dim, norm)]
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBnRelu(nn.Module):
    def __init__(self, num_input_channels: int, num_output_channels: int, **kwargs):
        super(ConvBnRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(num_input_channels, num_output_channels, **kwargs),
            nn.BatchNorm2d(num_output_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class FeedForwardResidualBlock(nn.Module):
    def __init__(self, dim: int, expansion_multiplier: int = 1):
        super(FeedForwardResidualBlock, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim * expansion_multiplier),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim * expansion_multiplier),
            nn.Linear(dim * expansion_multiplier, dim),
        )

    def forward(self, x: torch.Tensor):
        return x + self.layers(x)


def FeedForward(
    dim: int,
    expansion_factor: int = 4,
    dropout: float = 0.0,
    dense: Callable[..., nn.Module] = nn.Linear,
    activation: Callable[..., nn.Module] = partial(nn.ReLU, inplace=True),
    output_dim: Optional[int] = None,
):
    output_dim = output_dim if output_dim else dim
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        activation(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, output_dim),
        nn.Dropout(dropout),
    )


class ResidualPreNormFeedForward(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, **kwargs)

    def forward(self, x):
        return self.ff(self.norm(x)) + x


class ResidualFeedForward(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.ff = FeedForward(dim, **kwargs)

    def forward(self, x):
        return self.ff(x) + x


class Scattering(nn.Module):
    def __init__(self, num_groups: int):
        super().__init__()
        self.num_groups = num_groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Equivalent to Rearrange('b c (ng gs) -> b ng (c gs)', ng=num_groups, gs=group_size)
        :param x: a Tensor with rank >= 3 and last dimension divisible by number of groups
        :param num_groups: number of groups
        """
        shape_1 = x.shape[:-1] + (self.num_groups,) + (x.shape[-1] // self.num_groups,)
        x = x.view(shape_1)
        x = x.transpose(-3, -2).contiguous()
        return x.flatten(start_dim=-2)


class Identity(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class GroupObjectsIntoPairs(nn.Module):
    def forward(self, objects: torch.Tensor) -> torch.Tensor:
        batch_size, num_objects, object_size = objects.size()
        return torch.cat(
            [
                objects.unsqueeze(1).repeat(1, num_objects, 1, 1),
                objects.unsqueeze(2).repeat(1, 1, num_objects, 1),
            ],
            dim=3,
        ).view(batch_size, num_objects**2, 2 * object_size)


class Sum(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.sum(dim=self.dim)


class TagPanelEmbeddings(nn.Module):
    """Tags panel embeddings with their absolute coordinates."""

    def forward(
        self, panel_embeddings: torch.Tensor, num_context_panels: int
    ) -> torch.Tensor:
        """
        Concatenates a one-hot encoded vector to each panel.
        The concatenated vector indicates panel absolute position in the RPM.
        :param panel_embeddings: a tensor of shape (batch_size, num_panels, embedding_size)
        :return: a tensor of shape (batch_size, num_panels, embedding_size + 9)
        """
        batch_size, num_panels, _ = panel_embeddings.shape
        tags = torch.zeros((num_panels, 9), device=panel_embeddings.device).type_as(
            panel_embeddings
        )
        tags[:num_context_panels, :num_context_panels] = torch.eye(
            num_context_panels, device=panel_embeddings.device
        ).type_as(panel_embeddings)
        if num_panels > num_context_panels:
            tags[num_context_panels:, num_context_panels] = torch.ones(
                num_panels - num_context_panels, device=panel_embeddings.device
            ).type_as(panel_embeddings)
        tags = tags.expand((batch_size, -1, -1))
        return torch.cat([panel_embeddings, tags], dim=2)


def arrange_for_ravens_matrix(
    x: torch.Tensor, num_context_panels: int, num_answer_panels: int
) -> torch.Tensor:
    batch_size, num_panels, embedding_dim = x.shape
    x = torch.stack(
        [
            torch.cat((x[:, :num_context_panels], x[:, i].unsqueeze(1)), dim=1)
            for i in range(num_context_panels, num_panels)
        ],
        dim=1,
    )
    x = x.view(batch_size * num_answer_panels, num_context_panels + 1, embedding_dim)
    return x


def arrange_for_odd_one_out(x: torch.Tensor) -> torch.Tensor:
    batch_size, num_panels, embedding_dim = x.shape
    mask = (
        ~torch.eye(num_panels, device=x.device, dtype=torch.bool)
        .unsqueeze(-1)
        .repeat(1, 1, embedding_dim)
    )
    x = torch.stack(
        [
            x.masked_select(m.repeat(batch_size, 1, 1)).view(
                batch_size, num_panels - 1, embedding_dim
            )
            for m in mask
        ],
        dim=1,
    )  # b p p-1 d
    x = x.view((batch_size * num_panels), (num_panels - 1), embedding_dim)
    return x

class Classifier(nn.Module):

    def __init__(self, inplanes, ouplanes, norm_layer=nn.BatchNorm2d, dropout=0.0, hidreduce=1.0):
        super().__init__()

        midplanes = inplanes // hidreduce

        self.mlp = nn.Sequential(
            nn.Linear(inplanes, midplanes, bias=False),
            norm_layer(midplanes),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(midplanes, ouplanes)
        )

    def forward(self, x):
        return self.mlp(x)

class SCAR(nn.Module):
    def __init__(
        self,
        num_filters=48, block_drop=0.0, classifier_drop=0.0, 
        classifier_hidreduce=1.0, in_channels=1, num_classes=4, 
        num_extra_stages=1, reasoning_block=None,
        num_contexts=8,
        num_hidden_channels = 32,
        embedding_size = 128,
        ff_dim = 80,
        image_size = 80,
        local_kernel_size = 10,
        global_kernel_size = 10,
        sal_num_rows = 6,
        sal_num_cols = 420,
        ffblock = "pre-norm-residual",
    ):
        super(SCAR, self).__init__()
        assert ff_dim % local_kernel_size == 0
        assert ff_dim % global_kernel_size == 0
        local_group_size = ff_dim // local_kernel_size
        global_group_size = (local_kernel_size * 8) // global_kernel_size
        c = num_hidden_channels
        conv_dimension = (40 * (image_size // 80)) ** 2

        if ffblock == "pre-norm-residual":
            FeedForward = ResidualPreNormFeedForward
        elif ffblock == "residual-without-norm":
            FeedForward = ResidualFeedForward
        elif ffblock == "residual-with-norm":
            FeedForward = FeedForwardResidualBlock
        else:
            raise ValueError(f"Incorrect value for ffblock: {ffblock}")

        self.global_kernel_size = global_kernel_size
        self.global_group_size = global_group_size

        self.model_local = nn.Sequential(
            ConvBnRelu(1, c // 2, kernel_size=3, stride=2, padding=1),
            ConvBnRelu(c // 2, c // 2, kernel_size=3, padding=1),
            ConvBnRelu(c // 2, c, kernel_size=3, padding=1),
            ConvBnRelu(c, c, kernel_size=3, padding=1),
            nn.Flatten(start_dim=-2, end_dim=-1),
            nn.Linear(conv_dimension, ff_dim),
            nn.ReLU(inplace=True),
            FeedForward(ff_dim, activation=nn.GELU),
            nn.Conv1d(
                c, 128, kernel_size=(local_group_size,), stride=(local_group_size,)
            ),
            nn.GELU(),
            nn.Conv1d(128, 8, kernel_size=(1,), stride=(1,)),
            nn.Flatten(start_dim=-2, end_dim=-1),
            FeedForward(local_kernel_size * 8, activation=nn.GELU),
        )

        self.sal = StructureAwareLayer(
            out_channels=64,
            kernel_size=global_group_size,
            num_rows=sal_num_rows,
            num_cols=sal_num_cols,
        )
        self.model_global = nn.Sequential(
            nn.GELU(),
            nn.Conv1d(64, 32, kernel_size=(1,)),
            nn.GELU(),
            nn.Conv1d(32, 5, kernel_size=(1,)),
            nn.Flatten(start_dim=-2, end_dim=-1),
            FeedForward(5 * global_kernel_size, activation=nn.GELU),
            nn.Linear(5 * global_kernel_size, embedding_size),
        )

        self.classifier = Classifier(
            128, 1, 
            norm_layer = nn.BatchNorm1d, 
            dropout = classifier_drop, 
            hidreduce = classifier_hidreduce
        )

    def forward(
        self,
        x,
        train=False,
        num_rows = -1,
        num_cols = 3,
    ):
        # x = torch.cat([context, answers], dim=1)
        context, answers = x[:,:8], x[:,8:]
        num_context_panels = context.size(1)
        num_answer_panels = answers.size(1)
        batch_size, num_panels, height, width = x.shape
        num_rows = (num_context_panels + 1) // 3 if num_rows == -1 else num_rows

        x = x.view((batch_size * num_panels), 1, height, width)
        x = self.model_local(x)
        x = x.view(batch_size, num_panels, -1)

        x = torch.cat(
            [
                x[:, :num_context_panels, :]
                .unsqueeze(dim=1)
                .repeat(1, num_answer_panels, 1, 1),
                x[:, num_context_panels:, :].unsqueeze(dim=2),
            ],
            dim=2,
        )
        x = x.view((batch_size * num_answer_panels), (num_context_panels + 1), -1)

        x = self.sal.forward(x, num_rows=num_rows, num_cols=num_cols)
        x = self.model_global(x)
        x = x.view(batch_size*num_answer_panels, -1)
        x = self.classifier(x).reshape(batch_size,-1)
        errors = torch.zeros([1,10]).cuda()
        return x, errors

class RelationNetworkSAL(nn.Module):
    def __init__(
        self, in_dim, out_channels, out_dim, embedding_size = 128
    ):
        super().__init__()
        self.tag_panel_embeddings = TagPanelEmbeddings()
        self.relation_network = nn.Sequential(
            GroupObjectsIntoPairs(),
            DeepLinearBNReLU(
                2, 2 * (in_dim + 9), embedding_size, change_dim_first=True
            ),
            Sum(dim=1),
            nn.LayerNorm(embedding_size),
            nn.Linear(embedding_size, out_channels * out_dim),
            Rearrange("b (c d) -> b c d", c=out_channels, d=out_dim),
        )

    def forward(
        self, x, num_rows = 3, num_cols = 3
    ):
        num_context_panels = num_rows * num_cols
        x = self.tag_panel_embeddings(x, num_context_panels)
        x = self.relation_network(x)
        return x

class RelationNetworkSCAR(SCAR):
    def __init__(
        self,
        num_filters=48, block_drop=0.0, classifier_drop=0.1, 
        classifier_hidreduce=1.0, in_channels=1, num_classes=4, 
        num_extra_stages=1, reasoning_block=None,
        num_contexts=8,
        num_hidden_channels= 32,
        embedding_size = 128,
        ff_dim = 80,
        image_size = 80,
        local_kernel_size = 10,
        global_kernel_size = 10,
        sal_num_rows = 6,
        sal_num_cols  = 420,
        ffblock = "pre-norm-residual",
    ):
        super().__init__(
            num_filters,
            block_drop,
            classifier_drop,
            classifier_hidreduce,
            in_channels,
            num_classes,
            num_extra_stages,
            reasoning_block,
            num_contexts,
            num_hidden_channels,
            embedding_size,
            ff_dim,
            image_size,
            local_kernel_size,
            global_kernel_size,
            sal_num_rows,
            sal_num_cols,
            ffblock,
        )
        in_dim = local_kernel_size * 8
        self.sal = RelationNetworkSAL(
            in_dim=in_dim, out_channels=64, out_dim=self.global_kernel_size
        )