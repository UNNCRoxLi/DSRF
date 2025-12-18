from .resnet4b import resnet4b
from .DSRF import DSRF_raven, DSRF_analogy

DSRF_PYRAMID = {
    "S": [2, 1],
    "M": [4, 2, 1],
    "L": [8, 4, 1],
}

model_dict = {
    "resnet4b": resnet4b,
    "DSRF_raven": DSRF_raven,
    "DSRF_analogy": DSRF_analogy,
}


def create_net(args):
    net = None

    kwargs = {}
    kwargs["block_drop"] = args.block_drop
    kwargs["classifier_drop"] = args.classifier_drop
    kwargs["classifier_hidreduce"] = args.classifier_hidreduce
    kwargs["num_filters"] = args.num_filters
    kwargs["num_extra_stages"] = args.num_extra_stages
    kwargs["in_channels"] = args.in_channels
    kwargs["dsrf_pyramid"] = DSRF_PYRAMID[args.dsrf_scale]
    kwargs["dsrf_per_view"] = 16 if args.dsrf_light else 32  # light=16, full=32

    net = model_dict[args.arch.lower()](**kwargs)

    return net

