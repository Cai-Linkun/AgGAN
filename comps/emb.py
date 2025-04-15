import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d


class DeformableConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(DeformableConvNet, self).__init__()

        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2

        self.offset_conv = nn.Conv2d(
            in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=self.padding
        )

        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.padding)

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        return out


class ConvBlock(torch.nn.Module):
    def __init__(self, idim, odim, d):
        super().__init__()
        self.dconv = torch.nn.Conv2d(idim, odim, kernel_size=3, stride=1, padding=d, dilation=d)
        self.relu = nn.ReLU()
        self.conv = torch.nn.Conv2d(idim, odim, kernel_size=3, stride=1, padding=1)
        self.dfconv = DeformableConvNet(idim, odim, 3)

    def forward(self, x):
        x1 = self.dconv(x)
        x2 = self.relu(x1)
        x3 = x + x2
        x4 = self.dfconv(x3)
        return x4


class RAC(torch.nn.Module):
    def __init__(self, idim, odim):
        super(RAC, self).__init__()
        self.cb1 = ConvBlock(idim, odim, d=1)
        self.cb2 = ConvBlock(idim, odim, d=2)
        self.cb3 = ConvBlock(idim, odim, d=3)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(3 * odim, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        d1 = self.cb1(x)
        d2 = self.cb2(x)
        d3 = self.cb3(x)

        d_cat = torch.cat([d1, d2, d3], dim=1)
        global_feat = self.global_pool(d_cat)
        global_feat = global_feat.view(global_feat.shape[0], -1)

        weights = self.fc1(global_feat)
        weights = self.softmax(weights)

        d = (
            weights[:, 0].view(-1, 1, 1, 1) * d1
            + weights[:, 1].view(-1, 1, 1, 1) * d2
            + weights[:, 2].view(-1, 1, 1, 1) * d3
        )
        return d


class PositionModulatedLabelEmbedding(nn.Module):
    def __init__(self, num_labels=117, label_dim=64, pos_dim=64, H=64, W=64):
        super().__init__()
        self.H = H
        self.W = W
        self.label_embed = nn.Embedding(num_labels, label_dim, padding_idx=0)

        self.pos_modulation = nn.Parameter(torch.randn(1, pos_dim, H, W))

        if label_dim != pos_dim:
            self.align_proj = nn.Linear(pos_dim, label_dim)
        else:
            self.align_proj = nn.Identity()

    def forward(self, label_map):
        B, _, _, _ = label_map.shape

        label_emb = self.label_embed(label_map.squeeze(1))

        pos_mod = self.pos_modulation.expand(B, -1, -1, -1).permute(0, 2, 3, 1)
        pos_mod = self.align_proj(pos_mod)

        mod_feat = label_emb * pos_mod
        return mod_feat.permute(0, 3, 1, 2)


class FeatureGatingFusion(nn.Module):
    def __init__(self, h, w, atlas, label_dim, pos_dim, dim):
        super().__init__()
        self.proj_asl = nn.Conv2d(in_channels=1, out_channels=dim, kernel_size=1)

        self.label_pos_encoder = PositionModulatedLabelEmbedding(
            num_labels=atlas, label_dim=label_dim, pos_dim=pos_dim, H=h, W=w
        )

        self.concat_fusion = nn.Sequential(
            nn.Conv2d(label_dim + dim, dim, kernel_size=1), nn.ReLU(), nn.Conv2d(dim, dim, kernel_size=1)
        )

        self.rac = RAC(dim, dim)

    def forward(self, asl, atla):
        asl_feat = self.proj_asl(asl)
        prior_feat = self.label_pos_encoder(atla)
        concat = torch.cat([asl_feat, prior_feat], dim=1)
        fused = self.concat_fusion(concat)
        return self.rac(fused)
