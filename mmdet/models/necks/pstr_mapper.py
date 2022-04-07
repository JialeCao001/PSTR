# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule
from mmcv.ops import DeformConv2dPack

from ..builder import NECKS


@NECKS.register_module()
class PSTRMapper(BaseModule):
    r"""Channel Mapper to reduce/increase channels of backbone features.

    This is used to reduce/increase channels of backbone features.

    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        kernel_size (int, optional): kernel_size for reducing channels (used
            at each scale). Default: 3.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        act_cfg (dict, optional): Config dict for activation layer in
            ConvModule. Default: dict(type='ReLU').
        num_outs (int, optional): Number of output feature maps. There
            would be extra_convs when num_outs larger than the length
            of in_channels.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = ChannelMapper(in_channels, 11, 3).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(PSTRMapper, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.extra_convs = None
        self.convs = nn.ModuleList()
        self.lateral_convs = nn.ModuleList()
        print(len(in_channels))
        for i in range(len(in_channels)):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                kernel_size=1 if i == 2 else kernel_size,
                padding=0 if i == 2 else (kernel_size-1)//2,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            conv = DeformConv2dPack(
                out_channels,
                out_channels,
                3,
                padding=1,
                stride=2 if i == 0 else 1,
            )
            self.lateral_convs.append(l_conv)
            self.convs.append(conv)


    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.convs)
        # outs = [self.convs[i](inputs[i]) for i in range(len(inputs))]
        outs = []
        prev_shape = inputs[-2].shape[2:]
        for i in range(len(self.convs) - 1, -1, -1):
            lateral = self.lateral_convs[i](inputs[i])
            if i == len(self.convs) - 1:
                out = F.interpolate(lateral, size=prev_shape, mode='nearest')
            else:
                out = lateral
            out = self.convs[i](out)
            outs.append(out)
        return tuple(outs)
