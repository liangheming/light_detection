import warnings
from model.module.convs import InvertedResidual
from copy import deepcopy
from torch import nn
from torchvision.models.utils import load_state_dict_from_url

warnings.filterwarnings("ignore", category=Warning)
__all__ = [
    'ShuffleNetV2', 'build_backbone'
]

model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None,
}


class ShuffleNetV2(nn.Module):
    def __init__(
            self,
            stages_repeats,
            stages_out_channels,
            inverted_residual=InvertedResidual,
            act_func=None,
    ) -> None:
        super(ShuffleNetV2, self).__init__()

        if len(stages_repeats) != 3:
            raise ValueError('expected stages_repeats as list of 3 positive ints')
        if len(stages_out_channels) != 4:
            raise ValueError('expected stages_out_channels as list of 5 positive ints')
        self._stage_out_channels = stages_out_channels

        input_channels = 3
        output_channels = self._stage_out_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(negative_slope=0.1, inplace=True) if act_func is None else deepcopy(act_func),
        )
        input_channels = output_channels

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Static annotations for mypy
        self.stage2: nn.Sequential
        self.stage3: nn.Sequential
        self.stage4: nn.Sequential
        stage_names = ['stage{}'.format(i) for i in [2, 3, 4]]
        for name, repeats, output_channels in zip(
                stage_names, stages_repeats, self._stage_out_channels[1:]):
            seq = [inverted_residual(input_channels, output_channels, 2, act_func)]
            for i in range(repeats - 1):
                seq.append(inverted_residual(output_channels, output_channels, 1, act_func))
            setattr(self, name, nn.Sequential(*seq))
            input_channels = output_channels
        self.out_channels = self._stage_out_channels[1:]

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)
        return c3, c4, c5

    def forward(self, x):
        return self._forward_impl(x)


def _shufflenetv2(arch: str, pretrained: bool, progress: bool, *args, **kwargs) -> ShuffleNetV2:
    model = ShuffleNetV2(*args, **kwargs)

    if pretrained:
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError('pretrained {} is not supported as of now'.format(arch))
        else:
            state_dict = load_state_dict_from_url(model_url, progress=progress)
            model.load_state_dict(state_dict, strict=False)

    return model


def shufflenet_v2_x0_5(pretrained: bool = False, progress: bool = True, **kwargs) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 0.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x0.5', pretrained, progress,
                         [4, 8, 4], [24, 48, 96, 192], **kwargs)


def shufflenet_v2_x1_0(pretrained: bool = False, progress: bool = True, **kwargs) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.0', pretrained, progress,
                         [4, 8, 4], [24, 116, 232, 464], **kwargs)


def shufflenet_v2_x1_5(pretrained: bool = False, progress: bool = True, **kwargs) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 1.5x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x1.5', pretrained, progress,
                         [4, 8, 4], [24, 176, 352, 704], **kwargs)


def shufflenet_v2_x2_0(pretrained: bool = False, progress: bool = True, **kwargs) -> ShuffleNetV2:
    """
    Constructs a ShuffleNetV2 with 2.0x output channels, as described in
    `"ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design"
    <https://arxiv.org/abs/1807.11164>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shufflenetv2('shufflenetv2_x2.0', pretrained, progress,
                         [4, 8, 4], [24, 244, 488, 976], **kwargs)


def build_backbone(name, **kwargs) -> ShuffleNetV2:
    name_func_dict = {
        "shufflenet_v2_x0_5": shufflenet_v2_x0_5,
        "shufflenet_v2_x1_0": shufflenet_v2_x1_0,
        "shufflenet_v2_x1_5": shufflenet_v2_x1_5,
        "shufflenet_v2_x2_0": shufflenet_v2_x2_0
    }
    func = name_func_dict.get(name, None)
    assert func is not None
    return func(**kwargs)


if __name__ == '__main__':
    model = build_backbone("shufflenet_v2_x0_5", pretrained=True, act_func=nn.SiLU(inplace=True))
    print(model)
