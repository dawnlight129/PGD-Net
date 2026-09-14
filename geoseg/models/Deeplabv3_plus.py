from itertools import chain
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

# from base import BaseModel
# from utils.helpers import initialize_weights

try:
    from geoseg.models.resnet import resnet50
except ImportError:
    try:
        from .ResNet50 import resnet50
    except ImportError:
        from ResNet50 import resnet50

def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode='fan_out',
                    nonlinearity='relu'
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        

def set_trainable(modules, requires_grad):
    for module in modules:
        for param in module.parameters():
            param.requires_grad = requires_grad


def aspp_branch(in_channels, out_channels, kernel_size, dilation):
    padding = 0 if kernel_size == 1 else dilation
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
        #return summary(self, input_shape=(2, 3, 224, 224))


class ASPP(nn.Module):
    def __init__(self, in_channels=2048, output_stride=16):
        super(ASPP, self).__init__()

        assert output_stride in [8, 16], "DeepLabV3+ only supports output_stride 8 or 16."

        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        else:
            dilations = [1, 12, 24, 36]

        self.aspp1 = aspp_branch(in_channels, 256, 1, dilations[0])
        self.aspp2 = aspp_branch(in_channels, 256, 3, dilations[1])
        self.aspp3 = aspp_branch(in_channels, 256, 3, dilations[2])
        self.aspp4 = aspp_branch(in_channels, 256, 3, dilations[3])

        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.project = nn.Sequential(
            nn.Conv2d(256 * 5, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

        initialize_weights(self)

    def forward(self, x):
        size = x.shape[-2:]

        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_pool(x)
        x5 = F.interpolate(x5, size=size, mode="bilinear", align_corners=True)

        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.project(x)
        return x


class Decoder(nn.Module):
    def __init__(self, low_level_channels=256, num_classes=2):
        super(Decoder, self).__init__()

        self.low_level_project = nn.Sequential(
            nn.Conv2d(low_level_channels, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(48 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

        initialize_weights(self)

    def forward(self, x, low_level_features):
        low_level_features = self.low_level_project(low_level_features)

        x = F.interpolate(
            x,
            size=low_level_features.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )

        x = torch.cat([x, low_level_features], dim=1)
        x = self.classifier(x)
        return x


class DeepLabV3Plus(BaseModel):
    """
    DeepLabV3+ with the project ResNet50 backbone.

    The provided resnet50 backbone returns:
        layer0, layer1, layer2, layer3, layer4

    In this implementation:
        layer1 is used as low-level feature for decoder.
        layer4 is used as high-level feature for ASPP.
    """

    def __init__(
        self,
        num_classes=2,
        backbone_path=None,
        pretrained=True,
        output_stride=16,
        freeze_bn=False,
        freeze_backbone=False,
        **kwargs,
    ):
        super(DeepLabV3Plus, self).__init__()

        assert output_stride in [8, 16], "output_stride should be 8 or 16."

        if output_stride == 16:
            replace_stride_with_dilation = [False, False, True]
        else:
            replace_stride_with_dilation = [False, True, True]

        if pretrained and backbone_path is None:
            raise ValueError(
                "When pretrained=True, backbone_path must be provided, "
                "because your resnet50(backbone_path, pretrained=True) loads weights from a local file."
            )

        self.backbone = resnet50(
            backbone_path=backbone_path,
            pretrained=pretrained,
            replace_stride_with_dilation=replace_stride_with_dilation,
        )

        self.aspp = ASPP(in_channels=2048, output_stride=output_stride)
        self.decoder = Decoder(low_level_channels=256, num_classes=num_classes)

        if freeze_bn:
            self.freeze_bn()

        if freeze_backbone:
            set_trainable([self.backbone], False)

    def forward(self, x):
        input_size = x.shape[-2:]

        layer0, layer1, layer2, layer3, layer4 = self.backbone(x)

        x = self.aspp(layer4)
        x = self.decoder(x, layer1)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)

        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.aspp.parameters(), self.decoder.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()


DeepLab = DeepLabV3Plus


# if __name__ == "__main__":
#     model = DeepLabV3Plus(
#         num_classes=2,
#         backbone_path=None,
#         pretrained=False,
#         output_stride=16,
#     )
#     model.eval()
#     x = torch.randn(1, 3, 512, 512)
#     with torch.no_grad():
#         y = model(x)
#     print(y.shape)

    
    
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLabV3Plus(
        num_classes=2,
        backbone_path="/root/EB-TDFNet/pre_trained_weights/resnet50-19c8e357.pth",
        pretrained=True,
        output_stride=16,
    ).to(device)

    model.eval()

    img = torch.randn(1, 3, 512, 512).to(device)

    with torch.no_grad():
        output = model(img)

    from fvcore.nn import FlopCountAnalysis

    flops = FlopCountAnalysis(model, img)
    print("FLOPs: %.4f G" % (flops.total() / 1e9))

    total_parameters = sum(p.numel() for p in model.parameters())
    print("Params: %.4f M" % (total_parameters / 1e6))