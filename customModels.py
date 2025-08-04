import torchvision
import torch.nn as nn
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision.models.video import r3d_18
import monai


class TransformerEncoderLayerWithWeights(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", batch_first=True):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation, batch_first=batch_first)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn_weights = self.self_attn(src, src, src, attn_mask=src_mask,
                                            key_padding_mask=src_key_padding_mask, need_weights=True)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


class TransformerEncoderWithWeights(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None):
        output = src
        attn_weights_all_layers = []
        for mod in self.layers:
            output, attn_weights = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)
            attn_weights_all_layers.append(attn_weights)
        if self.norm is not None:
            output = self.norm(output)
        return output, attn_weights_all_layers


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

def resnet101(weights, num_classes=1000):
    model = torchvision.models.resnet101(weights=weights)
    print("Model used: ResNet101")
    if num_classes != 1000:
        model.fc = nn.Linear(in_features=model.fc.in_features, out_features=num_classes)

    return model

def resnet152(weights, num_classes=1000):
    model = torchvision.models.resnet152(weights=weights)
    print("Model used: ResNet152")
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

class MVNet(nn.Module):
    def __init__(self, backbone):
        super(MVNet, self).__init__()

        self.model2d = backbone

    def forward(self, x):
        batch_size, n_slice, w, h = x.shape

        x = x.view(batch_size * n_slice, 1, w, h)

        x = self.model2d(x)

        x = x.view(batch_size, n_slice, -1)

        x = x.mean(1)

        return x

class SoftAttn_slice(nn.Module):
    def __init__(self, backbone, img_size, num_classes):
        super(SoftAttn_slice, self).__init__()

        self.backbone = backbone

        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, img_size[0], img_size[1])).shape))

        self.soft_attn = nn.Linear(self.embedding_dim, 1, bias=True)

        # Custom init for soft attention layer
        with torch.no_grad():
            self.soft_attn.weight.fill_(0)
            self.soft_attn.bias.fill_(1)

        self.classifier = nn.Linear(self.embedding_dim, num_classes, bias=True)

        # torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, x, return_attn_weights=False):

        batch_size, n_slice, w, h = x.shape

        x = x.view(batch_size * n_slice, 1, w, h)

        x = self.backbone(x)

        x = x.view(batch_size, n_slice, self.embedding_dim)

        a = F.softmax(self.soft_attn(x), dim=1)

        x = torch.sum(x * a, axis=1)

        x = self.classifier(x)

        if return_attn_weights:
            return x, a
        else:
            return x

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

class DoubleCNN_SelfPatch_softSlice(nn.Module):
    def __init__(self, backbones, patch_size, img_size, num_classes, num_slices, num_heads):
        super(DoubleCNN_SelfPatch_softSlice, self).__init__()

        self.soft_backbone = backbones[0]
        self.self_backbone = backbones[1]

        self.p = patch_size
        self.num_slices = num_slices
        self.patches_per_slice = img_size[0] // self.p * img_size[1] // self.p

        self.embedding_dim = int(np.prod(self.soft_backbone(torch.rand(1, 1, img_size[0], img_size[1])).shape))

        self.positional_embedding = nn.Parameter(torch.randn(self.patches_per_slice + 1, self.embedding_dim))

        self.self_attn = nn.MultiheadAttention(self.embedding_dim, num_heads, batch_first=True)

        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.soft_attn = nn.Linear(self.embedding_dim, 1)

        # Custom init for soft attention layer
        with torch.no_grad():
            self.soft_attn.weight.fill_(0)
            self.soft_attn.bias.fill_(1)

        self.classifier = nn.Linear(num_slices * self.embedding_dim, num_classes)

    def forward(self, x, return_attn_weights=False):

        input_soft = x
        input_self = x

        batch_size, n_slice, h, w = x.shape

        # ---- SOFT
        input_soft = input_soft.view(batch_size * n_slice, 1, w, h)
        # shape [batch_size*num_slices, 1, w, h]

        input_soft = self.soft_backbone(input_soft)
        # shape [batch_size*num_slices, 1, embedding_dim]

        input_soft = input_soft.view(batch_size, n_slice, self.embedding_dim)
        # shape [batch_size, num_slices, embedding_dim]

        a_slice = F.softmax(self.soft_attn(input_soft), dim=1)
        # a shape [batch_size, num_slices, 1]

        output_soft = torch.sum(input_soft * a_slice, axis=1)

        # ---- SELF
        input_self = input_self.unfold(2, self.p, self.p).unfold(3, self.p, self.p)
        # shape [batch_size, n_slice, h/p, w/p, p, p]

        input_self = input_self.reshape(-1, 1, self.p, self.p)
        # shape [batch_size*num_patch, 1, p, p]

        input_self = self.self_backbone(input_self)

        input_self = input_self.view(batch_size * n_slice, self.patches_per_slice, self.embedding_dim)
        # shape [batch_size*num_slices, patches_per_slice, D]

        class_token = self.class_token.expand(batch_size * n_slice, 1, self.embedding_dim)

        input_self = torch.cat((class_token, input_self), dim=1)

        input_self = input_self + self.positional_embedding

        output_self, a_patches = self.self_attn(input_self, input_self, input_self, need_weights=return_attn_weights, average_attn_weights=True)
        # shape [batch_size*num_slices, num_patches + 1, embedding_dim]

        output_self = output_self[:, 0].view(batch_size, n_slice, -1)

        output = output_soft.unsqueeze(1) * output_self

        output = output.view(batch_size, n_slice*self.embedding_dim)

        # I need to find a way to gather both output_self and output_soft in a single vector representation
        output = self.classifier(output)

        if return_attn_weights:
            return output, (a_patches, a_slice)
        else:
            return output

class CrossCentralAttn_patch(nn.Module):
    def __init__(self, backbone, patch_size, img_size, num_classes, num_slices, num_heads, dropout=0.4):
        super(CrossCentralAttn_patch, self).__init__()

        self.backbone = backbone

        self.p = patch_size

        self.num_patches = num_slices * (img_size[0] // self.p) * (img_size[1] // self.p)

        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, img_size[0], img_size[1])).shape))

        self.positional_embedding = nn.Parameter(torch.randn(self.num_patches + 1, self.embedding_dim))

        self.attn = nn.MultiheadAttention(self.embedding_dim, num_heads, batch_first=True)
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.embedding_dim))

        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=self.embedding_dim),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=self.embedding_dim, out_features=num_classes)
        )

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

        central_slice = int(x.shape[1]/2)

        class_token = self.class_token.expand(batch_size, 1, self.embedding_dim)

        x = torch.cat((class_token, x), dim=1)

        x = x + self.positional_embedding

        q = x[:, central_slice, :].unsqueeze(1)

        k = torch.cat((x[:, :central_slice, :], x[:, central_slice + 1:, :]), dim=1)
        v = k
        x, a = self.attn(q, k, v, need_weights=return_attn_weights, average_attn_weights=True)

        class_token_out = x[:, 0]

        x = self.classifier(class_token_out)

        if return_attn_weights:
            return x, a
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

class PatchModel(nn.Module):
    def __init__(self, backbone, patch_size, num_classes, d=512):
        super(PatchModel, self).__init__()

        self.backbone = backbone

        self.p = patch_size

        out = self.backbone(torch.rand(1, 1, self.p, self.p))
        self.embedding = nn.Linear(np.prod(out.shape).item(), d)

        self.soft_attn = nn.Linear(d, 1)

        self.fc = nn.Linear(d, num_classes)

        self.softmax = nn.Softmax(0)

    def image_forward(self, x):
        s, h, w = x.shape

        # x.shape = [S, H, W]
        # TODO: add assert p/2 int H/p int and W/p int

        # --- patch division ---

        x = x.unfold(1, self.p, self.p).unfold(2, self.p, self.p)

        # --- more patches ---

        # patches = x.unfold(1, self.p, self.p).unfold(2, self.p, self.p)
        # # patches.shape = [S, H/p, W/p, p, p]
        #
        # shifted_patches = x[:, self.p//2:-self.p//2, self.p//2:-self.p//2].unfold(1, self.p, self.p).unfold(2, self.p, self.p)
        # # shifted_ patches.shape = [S, H/p-1, W/p-1, p, p]
        #
        # x = torch.cat((patches.reshape(s, -1, self.p, self.p), shifted_patches.reshape(s, -1, self.p, self.p)), 1)
        # # x.shape = [S, (H/p)*(W/p) + (H/p-1)*(W/p-1), p, p)

        # # --- a lot of patches ---
        #
        # x = x.unfold(1, self.p, self.p//2).unfold(2, self.p, self.p//2)
        # # patches.shape = [S, H/p, W/p, p, p] (for the second approach)

        # --------------------------
        x = x.reshape(-1, 1, self.p, self.p)
        # num_patches = S * ((H/p)*(W/p) + (H/p-1)*(W/p-1))
        # x.shape = [num_patches, 1, p, p]

        x = self.backbone(x)
        # x.shape = [num_patches, ...]

        x = x.reshape(x.shape[0], -1)
        x = self.embedding(x)
        # x.shape = [num_patches, D]

        # TODO: apply position encoding/embedding

        a = self.soft_attn(x)
        # a.shape = [num_patches, 1]
        a = self.softmax(a)

        x = torch.sum(x.permute(1, 0) * a.permute(1, 0), dim=1)
        # x.shape = [D]

        x = self.fc(x)

        return x

    def forward(self, x):
        outputs = []

        # TODO: add assert
        for image in x:
            outputs.append(self.image_forward(image))

        return torch.stack(outputs)


class endtoend_objectdetection_test(nn.Module):
    def __init__(self):
        super(endtoend_objectdetection_test, self).__init__()

        self.retina = torchvision.models.detection.retinanet_resnet50_fpn()

    def forward(self, images, targets=None):

        if self.training:
            self.retina.train()
            # TODO: assert on targets
            losses = self.retina(images, targets)
            self.retina.eval()
            detection = self.retina(images)

            return losses, detection

        else:
            return self.retina(images)

class AwareNet(nn.Module):
    def __init__(self, num_classes, num_slices, return_attention_weights=False):
        super(AwareNet, self).__init__()
        # [1] base feature extractor
        self.basic_module = TimeDistributed(Basic_Fex())
        # [2] slice-aware module
        self.att_module = Attention(num_slices)
        # [3] Fusion feature extractor
        self.fusion_module = Fusion_Fex(num_classes=num_classes)
        self.return_attention_weights = return_attention_weights

    def forward(self, x):
        x = self.basic_module(x, init=True)
        x, slice_attn, local_attn, alpha = self.att_module(x)
        x = self.fusion_module(x)
        x = torch.mean(x, axis=1)
        Y_hat = torch.softmax(x, dim=1).argmax(dim=1)
        if self.return_attention_weights:
            return x, Y_hat, slice_attn, local_attn, alpha
        return x


# [1] Basic feature extractor
class Basic_Fex(nn.Module):
    def __init__(self, expansion=4):
        super(Basic_Fex, self).__init__()
        self.expansion = expansion
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.layer1 = self._make_layer(BasicBlock, 64, 2, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, 2)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x


# [2] Slice-aware module
class Attention(nn.Module):
    def __init__(self, num_slices):
        super(Attention, self).__init__()
        self.num_slices = num_slices
        self.loc_att = TimeDistributed(
            nn.Sequential(nn.Conv2d(128, 1, kernel_size=1, stride=1), nn.LeakyReLU(0.2, inplace=True)))
        self.k = nn.Conv2d(self.num_slices, self.num_slices, kernel_size=3, padding=1, groups=self.num_slices, bias=False)
        self.q = nn.Conv2d(self.num_slices, 1, kernel_size=1, bias=False)
        self.alpha = 32.0

    def forward(self, x):
        ori_x = x
        local_attention = self.loc_att(x)
        x = local_attention.squeeze(2)
        b, c, h, w = x.shape
        k = self.k(x).reshape(b, c, -1)
        q = self.q(x).reshape(b, 1, -1)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)).squeeze(1)
        attn *= self.alpha
        slice_attention = attn.softmax(dim=-1)
        attn = slice_attention.reshape(b, self.num_slices, 1, 1, 1).expand(b, self.num_slices, 128, 128, 64)
        out = attn * ori_x
        return out + ori_x, slice_attention, local_attention, self.alpha


# [3] Fusion feature extractor
class Fusion_Fex(nn.Module):
    def __init__(self, num_classes=2, expansion=4):
        super(Fusion_Fex, self).__init__()
        self.expansion = expansion
        self.in_channels = 128
        # shared 3D->2D
        self.layer1 = TimeDistributed(self._make_layer(BasicBlock, 256, 2, 2))
        self.layer2 = TimeDistributed(self._make_layer(BasicBlock, 512, 2, 2))
        self.global_avg = TimeDistributed(nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = TimeDistributed(
            nn.Sequential(*[nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, num_classes)]))

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    # slice shift module
    def TemporalShiftModule(self, x, fold_div):
        # shape of x: [N, T, C, H, W]
        c = x.size(1)
        out = torch.zeros_like(x)
        fold = c // fold_div
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out

    def forward(self, x):
        # slice shift 1
        x = self.TemporalShiftModule(x, 8)
        x = self.layer1(x)
        # slice shift 2
        x = self.TemporalShiftModule(x, 8)
        x = self.layer2(x)
        x = self.global_avg(x)
        x = self.classifier(x, fcmodule=True)
        return x


############################################### Basic network components ###############################################
# [C.1] Time distribute 3D->2D
class TimeDistributed(nn.Module):
    def __init__(self, module):
        super(TimeDistributed, self).__init__()
        self.module = module

    def forward(self, x, fcmodule=False, init=False):
        ''' x size: (batch_size, time_steps, in_channels, height, width) '''
        if init:
            x = x.unsqueeze(2)
        if fcmodule:
            batch_size, time_steps, C, _, _ = x.size()
            c_in = x.reshape(batch_size, time_steps, C)
            c_out = self.module(c_in)
        else:
            batch_size, time_steps, C, H, W = x.size()
            c_in = x.view(batch_size * time_steps, C, H, W)
            c_out = self.module(c_in)
            _, c, h, w = c_out.size()
            c_out = c_out.reshape(batch_size, time_steps, c, h, w)
        return c_out


#  [C.2] Basic resnet block
class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """
    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )
        # shortcut
        self.shortcut = nn.Sequential()
        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

def CustomSwinTransformer(model_type, num_classes=2, img_size=(1024,512)):

    model = timm.create_model(model_type, pretrained=True, img_size=img_size)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    print(model)

    return model

def CustomConvNextTransformer(model_type, num_classes=2, img_size=(1024,512)):

    model = timm.create_model(model_type, pretrained=True)
    model.head.fc = nn.Linear(model.head.fc.in_features, num_classes)
    print(model)
    print(model.default_cfg['input_size'])

    return model
# class CustomCNN_backbone_electric_boogaloo(nn.Module):
#     def __init__(self, channels, output_size=(1,1), in_channels=1, kernel_size=3, stride=1, padding=1):
#         super(CustomCNN_backbone_electric_boogaloo, self).__init__()
#         self.layers = nn.ModuleList()
#         self.skip_conns = nn.ModuleList()
#
#         channels = [in_channels] + channels
#
#         for in_c, out_c in zip(channels[:-1], channels[1:]):
#             self.layers.append(nn.Conv2d(in_c, out_c, kernel_size, stride, padding))
#             self.layers.append(nn.Conv2d(out_c, out_c, kernel_size, stride, padding))
#             self.layers.append(nn.ReLU(inplace=True))
#             self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
#             self.skip_conns.append(nn.Conv2d(1, out_c, kernel_size, stride, padding))
#
#         self.avgpool = nn.AdaptiveAvgPool2d(output_size=output_size)
#
#     def forward(self, x):
#         input = x
#
#         for layer in self.layers:
#             x = layer(x)
#
#         skip_connection = self.skip_conns[-1](input)
#
#         skip_connection = F.interpolate(skip_connection, size=(x.size(2), x.size(3)), mode='bilinear',
#                                         align_corners=False)
#
#         x = x + skip_connection
#
#         x = self.avgpool(x)
#
#         return x

def CNN_Encoder_Soft(cnn_channels, p, img_size, num_classes, slices, num_enc_layers, output_size=(1,1)):
    backbone = CustomCNN_backbone(cnn_channels, output_size)
    model = SelfEncoderPatch_SoftSlice(backbone, p, img_size, num_classes, slices, num_enc_layers)

    return model

class SelfEncoderPatch_SoftSlice(nn.Module):
    def __init__(self, backbone, patch_size, img_size, num_classes, num_slices, num_encoder_layer):
        super(SelfEncoderPatch_SoftSlice, self).__init__()

        self.backbone = backbone

        self.p = patch_size
        self.num_slices = num_slices
        self.patches_per_slice = img_size[0] // self.p * img_size[1] // self.p

        self.embedding_dim = int(np.prod(self.backbone(torch.rand(1, 1, self.p, self.p)).shape))

        self.positional_embedding = nn.Parameter(torch.randn(self.patches_per_slice + 1, self.embedding_dim))

        self.encoder = nn.TransformerEncoder(
            CustomTransformerEncoderLayer(self.embedding_dim, 4, batch_first=True),
            num_layers=num_encoder_layer)
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

        x = self.encoder(x)

        x = x[:, 0].view(batch_size, n_slice, -1)

        a_slice = F.softmax(self.soft_attn(x), dim=1)
        # a shape [batch_size, num_patch, 1]

        x = torch.sum(x * a_slice, axis=1)

        x = self.classifier(x)

        if return_attn_weights:
            return x, (a_slice, [layer.get_attn_weights() for layer in self.encoder.layers])
        else:
            return x

class CustomCNN_backbone(nn.Module):
    def __init__(self, channels, output_size=(1,1), in_channels=1, kernel_size=3, stride=1, padding=1):
        super(CustomCNN_backbone, self).__init__()
        self.layers = nn.ModuleList()

        channels = [in_channels] + channels

        for in_c, out_c in zip(channels[:-1], channels[1:]):
            self.layers.append(nn.Conv2d(in_c, out_c, kernel_size, stride, padding))
            self.layers.append(nn.Conv2d(out_c, out_c, kernel_size, stride, padding))
            self.layers.append(nn.ReLU(inplace=True))
            self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        self.avgpool = nn.AdaptiveAvgPool2d(output_size=output_size)

        # self.fc = nn.Linear(channels[-1], num_classes)  # Adjust according to input size

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.avgpool(x)
        # x = torch.flatten(x, start_dim=1)
        # x = self.fc(x)
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

# Modify the model for a new input size (1024x512)
# def CustomSwin(img_size=(1024, 512)):
#
#     model = torchvision.models.swin_t()
#     self.classifier = nn.Linear(swin_model.embed_dim, num_classes)
#
#     return model
class CustomSwin(nn.Module):
    def __init__(self, num_classes, img_size=(1024,512)):
        super(CustomSwin, self).__init__()
        self.model = torchvision.models.swin_t()

        # Step 2: Modify the final classification head (in case you need a different number of classes)
        self.num_classes = num_classes  # Change this to match your dataset
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)

        print(self.model)

    def forward(self, x):
        x = self.model(x)

        return x

class CustomConvNext(nn.Module):
    def __init__(self, num_classes, img_size=(1024,512)):
        super(CustomConvNext, self).__init__()
        self.model = torchvision.models.convnext_tiny()

        # Step 2: Modify the final classification head (in case you need a different number of classes)
        self.num_classes = num_classes  # Change this to match your dataset
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, num_classes)

        print(self.model)

    def forward(self, x):
        x = self.model(x)

        return x

# class VisionTransformer(nn.Module):
#     def __init__(self, image_size=(1024, 512), patch_size=16, num_channels=3, embed_dim=768, num_heads=12,
#                  num_layers=12):
#         super(VisionTransformer, self).__init__()
#
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
#
#         self.embedding = nn.Conv2d(in_channels=num_channels, out_channels=embed_dim, kernel_size=patch_size,
#                                    stride=patch_size)
#
#         # Positional embedding
#         self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))
#
#         # Transformer layers (assuming ViT-style architecture)
#         self.transformer_blocks = nn.ModuleList([
#             nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
#             for _ in range(num_layers)
#         ])
#
#         self.fc_out = nn.Linear(embed_dim, 1000)  # Assuming 1000 classes for classification
#
#     def forward(self, x):
#         # Assuming input image size: (batch_size, 3, 1024, 512)
#         x = self.embedding(x)  # Shape: (batch_size, embed_dim, num_patches_height, num_patches_width)
#         x = x.flatten(2).transpose(1, 2)  # Flatten and transpose to (batch_size, num_patches, embed_dim)
#
#         x += self.position_embedding  # Add positional embeddings
#
#         for block in self.transformer_blocks:
#             x = block(x)
#
#         x = self.fc_out(x.mean(dim=1))  # Global average pooling
#         return x

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


class VIT3D(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, num_slices, num_heads, hidden_size=1020, mlp_dim=4080):
        super().__init__()
        self.model = monai.networks.nets.ViT(in_channels=1,
                                             patch_size=(2, patch_size, patch_size),
                                             img_size=(num_slices, img_size[0], img_size[1]),
                                             proj_type='conv',
                                             classification=True,
                                             num_classes=num_classes,
                                             num_heads=num_heads,
                                             mlp_dim=mlp_dim,
                                             hidden_size=hidden_size)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model(x)
        return x[0]