import torchvision
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.video import r3d_18

def resnet34(weights, num_classes=1000):
    model = torchvision.models.resnet34(weights=weights)
    print("Model used: ResNet34")
    if num_classes != 1000:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    return model

def resnet50(weights, num_classes=1000):
    model = torchvision.models.resnet50(weights=weights)
    print("Model used: ResNet50")
    if num_classes != 1000:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    return model

def resnet18(weights, num_classes=1000):
    model = torchvision.models.resnet50(weights=weights)
    print("Model used: ResNet18")
    if num_classes != 1000:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    return model

def efficientnetb3_features(weights, avgpool_dim=(1, 1)):
    model = torchvision.models.efficientnet_b3(weights=weights)
    print("Model used: EffcientNetB3")

    model = nn.Sequential(*list(model.children())[:-1])
    model[-1].output_size = avgpool_dim

    return model

def efficientnetb3(weights, num_classes=1000):
    model = torchvision.models.efficientnet_b3(weights=weights)
    print("Model used: EffcientNetB3")

    model.classifier[1] = nn.Linear(1536, num_classes, bias=True)

    return model

def efficientnetb7(weights, num_classes=1000):
    model = torchvision.models.efficientnet_b7(weights=weights)
    print("Model used: EffcientNetB7")

    model.classifier[1] = nn.Linear(1536, num_classes, bias=True)

    return model


def resnet34_features(weights, avgpool_dim=(1, 1)):
    print("Avgpool_dim" + str(avgpool_dim))
    model = torchvision.models.resnet34(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model[-1].output_size = avgpool_dim

    return model


def resnet18_features(weights, avgpool_dim=(1, 1)):
    print("Avgpool_dim" + str(avgpool_dim))
    model = torchvision.models.resnet18(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model[-1].output_size = avgpool_dim
    return model


def vgg16_features(weights, avgpool_dim=(1, 1)):
    print("Avgpool_dim" + str(avgpool_dim))
    model = torchvision.models.vgg16(weights=weights)
    model = nn.Sequential(*list(model.children())[:-1])
    model[-1].output_size = avgpool_dim
    return model

def vgg16(weights, num_classes=1000):
    model = torchvision.models.vgg16(weights=weights)
    print("Model used: VGG16")

    model.classifier[6] = nn.Linear(4096, num_classes, bias=True)

    return model

class SoftAttn_patch(nn.Module):
    def __init__(self, backbone, patch_size, img_size, num_slices, num_classes):
        super(SoftAttn_patch, self).__init__()

        self.backbone = backbone
        self.num_slices = num_slices
        self.p = patch_size

        self.num_patches = (img_size[0] // self.p) * (img_size[1] // self.p)

        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, img_size[0], img_size[1])).shape))

        self.soft_attn = nn.Linear(self.embedding_dim, 1)

        # Custom init for soft attention layer
        with torch.no_grad():
            self.soft_attn.weight.fill_(0)
            self.soft_attn.bias.fill_(1)

        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, return_attn_weights=False):

        batch_size, n_slice, h, w = x.shape

        x = x.unfold(2, self.p, self.p).unfold(3, self.p, self.p)
        # shape [batch_size, n_slice, h/p, w/p, p, p]

        x = x.reshape(-1, 1, self.p, self.p)
        # num_patch = n_slice*h*w/p^2
        # shape [batch_size*num_patch, 1, p, p]
        x = self.backbone(x)

        x = x.view(batch_size, self.num_patches*n_slice, self.embedding_dim)
        # shape [batch_size, num_patch*num_slices, D]

        a = F.softmax(self.soft_attn(x), dim=1)
        # a shape [batch_size, num_patch, 1]

        x = torch.sum(x * a, axis=1)

        x = self.classifier(x)

        if return_attn_weights:
            return x, a.view(n_slice, (h // self.p), (w // self.p))
        else:
            return x

class SelfAttn_patch(nn.Module):
    def __init__(self, backbone, patch_size, img_size, num_classes, num_slices, num_heads):
        super(SelfAttn_patch, self).__init__()

        self.backbone = backbone

        self.p = patch_size

        self.num_patches = num_slices * (img_size[0] // self.p) * (img_size[1] // self.p)

        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, img_size[0], img_size[1])).shape))

        self.positional_embedding = nn.Parameter(torch.randn(self.num_patches + 1, self.embedding_dim))

        self.attn = nn.MultiheadAttention(self.embedding_dim, num_heads, batch_first=True)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, return_attn_weights=False):

        batch_size, n_slice, h, w = x.shape

        x = x.unfold(2, self.p, self.p).unfold(3, self.p, self.p)
        # shape [batch_size, n_slice, h/p, w/p, p, p]

        x = x.reshape(-1, 1, self.p, self.p)
        # num_patch = n_slice*h*w/p^2
        # shape [batch_size*num_patch, 1, p, p]

        x = self.backbone(x)

        x = x.view(batch_size, self.num_patches, self.embedding_dim)
        # shape [batch_size, num_patch, D]

        class_token = self.class_token.expand(batch_size, 1, self.embedding_dim)

        x = torch.cat((class_token, x), dim=1)

        x = x + self.positional_embedding

        x, a = self.attn(x, x, x, need_weights=return_attn_weights, average_attn_weights=True)

        class_token_out = x[:, 0]

        x = self.classifier(class_token_out)

        if return_attn_weights:
            return x, a
        else:
            return x

class NoAttn(nn.Module):
    def __init__(self, backbone, img_size, num_slices, num_classes):
        super(NoAttn, self).__init__()

        self.backbone = backbone
        self.num_slices = num_slices
        self.img_size = img_size


        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, img_size[0], img_size[1])).shape))

        self.classifier = nn.Linear(self.embedding_dim*num_slices, num_classes)

    def forward(self, x, return_attn_weights=False):

        batch_size, n_slice, h, w = x.shape

        x = x.reshape(-1, 1, self.img_size[0], self.img_size[1])
        # shape [batch_size*num_patch, 1, h, w]
        x = self.backbone(x)

        x = x.view(batch_size, self.embedding_dim*n_slice)
        # shape [batch_size, D*num_slices]

        x = self.classifier(x)

        return x


class NoAttn_Patch(nn.Module):
    def __init__(self, backbone, img_size, num_slices, num_classes, patch_size):
        super(NoAttn_Patch, self).__init__()

        self.backbone = backbone
        self.num_slices = num_slices
        self.img_size = img_size
        self.p = patch_size
        self.num_patches = num_slices * (img_size[0] // self.p) * (img_size[1] // self.p)

        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, img_size[0], img_size[1])).shape))

        self.classifier = nn.Linear(self.embedding_dim*self.num_patches, num_classes)

    def forward(self, x, return_attn_weights=False):
        batch_size, n_slice, h, w = x.shape

        x = x.unfold(2, self.p, self.p).unfold(3, self.p, self.p)
        # shape [batch_size, n_slice, h/p, w/p, p, p]

        x = x.reshape(-1, 1, self.p, self.p)
        # num_patch = n_slice*h*w/p^2
        # shape [batch_size*num_patch, 1, p, p]

        x = self.backbone(x)

        x = x.view(batch_size, self.num_patches * self.embedding_dim)
        # shape [batch_size, num_patch, D]

        x = self.classifier(x)

        return x

class SelfPatch_softSlice(nn.Module):
    def __init__(self, backbone, patch_size, img_size, num_classes, num_slices, num_heads):
        super(SelfPatch_softSlice, self).__init__()

        self.backbone = backbone

        self.p = patch_size
        self.num_slices = num_slices
        self.patches_per_slice = img_size[0] // self.p * img_size[1] // self.p

        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, img_size[0], img_size[1])).shape))

        self.positional_embedding = nn.Parameter(torch.randn(self.patches_per_slice + 1, self.embedding_dim))

        self.attn = nn.MultiheadAttention(self.embedding_dim, num_heads, batch_first=True)

        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.soft_attn = nn.Linear(self.embedding_dim, 1)

        # Custom init for soft attention layer
        with torch.no_grad():
            self.soft_attn.weight.fill_(0)
            self.soft_attn.bias.fill_(1)

        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, return_attn_weights=False):

        batch_size, n_slice, h, w = x.shape

        x = x.unfold(2, self.p, self.p).unfold(3, self.p, self.p)
        # shape [batch_size, n_slice, h/p, w/p, p, p]

        x = x.reshape(-1, 1, self.p, self.p)
        # num_patch = n_slice*h*w/p^2
        # shape [batch_size*num_patch, 1, p, p]

        x = self.backbone(x)

        x = x.view(batch_size * n_slice, self.patches_per_slice, self.embedding_dim)
        # shape [batch_size*num_slices, patches_per_slice, D]

        class_token = self.class_token.expand(batch_size * n_slice, 1, self.embedding_dim)

        x = torch.cat((class_token, x), dim=1)

        x = x + self.positional_embedding

        x, a_patches = self.attn(x, x, x, need_weights=return_attn_weights, average_attn_weights=True)

        x = x[:, 0].view(batch_size, n_slice, -1)

        a_slice = F.softmax(self.soft_attn(x), dim=1)
        # a shape [batch_size, num_patch, 1]

        x = torch.sum(x * a_slice, axis=1)

        x = self.classifier(x)

        if return_attn_weights:
            return x, (a_patches, a_slice)
        else:
            return x


class SelfEncoderLayerPatch_SoftSlice(nn.Module):
    def __init__(self, backbone, patch_size, img_size, num_classes, num_slices, num_heads):
        super(SelfEncoderLayerPatch_SoftSlice, self).__init__()

        self.backbone = backbone

        self.p = patch_size
        self.num_slices = num_slices
        self.patches_per_slice = img_size[0] // self.p * img_size[1] // self.p

        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, img_size[0], img_size[1])).shape))

        self.positional_embedding = nn.Parameter(torch.randn(self.patches_per_slice + 1, self.embedding_dim))

        self.attn = CustomTransformerEncoderLayer(self.embedding_dim, num_heads, batch_first=True)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.soft_attn = nn.Linear(self.embedding_dim, 1)

        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, return_attn_weights=False):

        batch_size, n_slice, h, w = x.shape

        x = x.unfold(2, self.p, self.p).unfold(3, self.p, self.p)
        # shape [batch_size, n_slice, h/p, w/p, p, p]

        x = x.reshape(-1, 1, self.p, self.p)
        # num_patch = n_slice*h*w/p^2
        # shape [batch_size*num_patch, 1, p, p]

        x = self.backbone(x)

        x = x.view(batch_size * n_slice, self.patches_per_slice, self.embedding_dim)
        # shape [batch_size*num_slices, patches_per_slice, D]

        class_token = self.class_token.expand(batch_size * n_slice, 1, self.embedding_dim)

        x = torch.cat((class_token, x), dim=1)

        x = x + self.positional_embedding

        x, a_patches = self.attn(x, return_attn_weights=return_attn_weights)

        x = x[:, 0].view(batch_size, n_slice, -1)

        a_slice = F.softmax(self.soft_attn(x), dim=1)
        # a shape [batch_size, num_patch, 1]

        x = torch.sum(x * a_slice, axis=1)

        x = self.classifier(x)

        if return_attn_weights:
            return x, (a_patches, a_slice)
        else:
            return x

class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation=F.relu,
                 layer_norm_eps: float = 1e-5,
                 batch_first: bool = False,
                 norm_first: bool = False,
                 device=None,
                 dtype=None,
                 need_weights=True,
                 average_attn_weights=True):
        super(CustomTransformerEncoderLayer, self).__init__(d_model,
                                                            nhead,
                                                            dim_feedforward,
                                                            dropout,
                                                            activation,
                                                            layer_norm_eps,
                                                            batch_first,
                                                            norm_first,
                                                            device,
                                                            dtype)

        self.need_weights = need_weights
        self.average_attn_weights = average_attn_weights
        self.attn_weights = None

    def _sa_block(self, x, attn_mask, key_padding_mask, is_causal):
        x, self.attn_weights = self.self_attn(x, x, x,
                                              attn_mask=attn_mask,
                                              key_padding_mask=key_padding_mask,
                                              need_weights=self.need_weights,
                                              average_attn_weights=self.average_attn_weights,
                                              is_causal=is_causal)
        return self.dropout1(x)

    def get_attn_weights(self):
        return self.attn_weights

class ResNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.model = r3d_18(pretrained=True)

        self.model.stem[0] = nn.Conv3d(
            in_channels, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class VIT25D(nn.Module):
    def __init__(self, backbone, patch_size, img_size, num_classes, num_slices, num_encoder_layers, num_heads):
        super(VIT25D, self).__init__()

        self.backbone = backbone

        self.p = patch_size
        self.num_slices = num_slices
        self.patches_per_slice = img_size[0] // self.p * img_size[1] // self.p

        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, self.p, self.p)).shape))

        self.positional_embedding = nn.Parameter(torch.randn(self.patches_per_slice * self.num_slices + 1, self.embedding_dim))

        self.encoder = nn.TransformerEncoder(
            CustomTransformerEncoderLayer(self.embedding_dim, num_heads, batch_first=True),
            num_layers=num_encoder_layers)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.classifier = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x, return_attn_weights=False):

        batch_size, n_slice, h, w = x.shape

        x = x.unfold(2, self.p, self.p).unfold(3, self.p, self.p)
        # shape [batch_size, n_slice, h/p, w/p, p, p]

        x = x.reshape(-1, 1, self.p, self.p)
        # num_patch = n_slice*h*w/p^2
        # shape [batch_size*num_patch, 1, p, p]

        x = self.backbone(x)

        x = x.view(batch_size, self.patches_per_slice * n_slice, self.embedding_dim)
        # shape [batch_size, num_slices * patches_per_slice, D]

        class_token = self.class_token.expand(batch_size, 1, self.embedding_dim)

        x = torch.cat((class_token, x), dim=1)

        x = x + self.positional_embedding

        x = self.encoder(x)

        x = x[:, 0]

        x = self.classifier(x)

        if return_attn_weights:
            return x, [layer.get_attn_weights() for layer in self.encoder.layers]
        else:
            return x