import torch
import torch.nn as nn


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# 간단한 U-Net 모델 정의
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.dconv_down1 = double_conv(3, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, 256)
        self.dconv_down4 = double_conv(256, 512)

        self.bridge = double_conv(512, 1024)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.dconv_up4 = double_conv(512 + 1024, 512)
        self.dconv_up3 = double_conv(256 + 512, 256)
        self.dconv_up2 = double_conv(128 + 256, 128)
        self.dconv_up1 = double_conv(128 + 64, 64)

        self.conv_last = nn.Conv2d(64, 13, 1)  # 12개 class + 1 background

    def forward(self, x):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)

        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)

        x = self.bridge(x)

        x = self.upsample(x)
        x = torch.cat([x, conv4], dim=1)

        x = self.dconv_up4(x)
        x = self.upsample(x)
        x = torch.cat([x, conv3], dim=1)

        x = self.dconv_up3(x)
        x = self.upsample(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        out = self.conv_last(x)

        return out


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DANN(nn.Module):
    def __init__(self, num_classes=13):
        super(DANN, self).__init__()

        # Here we assume the input feature map is of size (num_classes*height*width).
        domain_classifier_in_features = num_classes * 256 * 256

        domain_classifier_out_features = 1000

        self.domain_classifier_layers = nn.Sequential(
            nn.Linear(domain_classifier_in_features, domain_classifier_out_features),
            nn.ReLU(),
            nn.Linear(domain_classifier_out_features, domain_classifier_out_features // 10),
            nn.ReLU(),
            nn.Linear(domain_classifier_out_features // 10, 2),  # Assuming binary classification for the domain
        )

    def forward(self, x, alpha=None):
        reversed_feature_map = GradReverse.apply(x, alpha)
        reversed_feature_map = reversed_feature_map.view(reversed_feature_map.size(0), -1)

        for layer in self.domain_classifier_layers:
            reversed_feature_map = layer(reversed_feature_map)

        return reversed_feature_map



class DANN_UNet(nn.Module):
    def __init__(self, num_classes=13):
        super(DANN_UNet, self).__init__()

        # U-Net and Domain Adversarial Neural Network (DANN) components
        self.unet_part = UNet()

        # the output of the unet is given as input to the domain classifier
        self.domain_classifier = DANN(num_classes)

    def forward(self, x, alpha=None):
        x = self.unet_part(x)
        domain_output = self.domain_classifier(x, alpha)

        return x, domain_output
