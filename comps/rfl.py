import torch
from torch.nn import functional as F


def compute_entropy(img_region, num_bins, eps=1e-6):
    hist = torch.histc(img_region, bins=num_bins, min=0.0, max=1.0)
    prob = hist / (hist.sum() + eps)
    entropy = -torch.sum(prob * torch.log(prob + eps))
    return entropy


def compute_region_weights(real, fake, region_map, regions, num_bins, alpha, beta):
    rec_err = []
    ent_err = []
    for k in range(regions):
        mask = region_map == k
        if mask.sum() == 0:
            rec_err.append(torch.tensor(0.0, device=real.device))
            ent_err.append(torch.tensor(0.0, device=real.device))
            continue

        error = F.l1_loss(real[mask], fake[mask], reduction="mean")
        entropy = compute_entropy(real[mask].float(), num_bins=num_bins)
        rec_err.append(error)
        ent_err.append(entropy)

    rec_err = torch.stack(rec_err)
    ent_err = torch.stack(ent_err)
    max_rec_err = torch.max(rec_err)
    max_ent_err = torch.max(ent_err)
    rec_err = rec_err / max_rec_err
    ent_err = ent_err / max_ent_err

    weight = alpha * rec_err + beta * ent_err
    weight = torch.nan_to_num(weight, 0)

    return weight


def region_focal_loss(real, fake, region_map, weights, lam=1.0):
    _, _, H, W = real.shape
    loss = 0.0
    l1_loss = F.l1_loss(real, fake, reduction="none") / (H * W)
    for k, w_k in enumerate(weights):
        mask = region_map == k
        if mask.sum() == 0:
            continue
        region_loss = l1_loss[mask].sum()
        loss += (1 + lam * w_k) * region_loss
    return loss


class RegionalFocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.07, beta=0.03, lam=1, num_bins=64):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.lam = lam
        self.num_bins = num_bins

    def forward(self, real, fake, region_map, regions=117):
        weights = compute_region_weights(real, fake, region_map, regions, self.num_bins, self.alpha, self.beta)
        return region_focal_loss(real, fake, region_map, weights, self.lam)


class RegionL1Loss_Max(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("max_l1_loss", torch.zeros(1))

    def calc_pixel_loss(self):
        return torch.nn.functional.l1_loss(self.real, self.fake, reduction="none")

    def calc_region_loss(self, rid):
        eps = 1e-6
        mask = self.region_map == rid
        count = torch.sum(mask)
        loss = self.whole_loss * mask
        loss_mean = torch.sum(loss) / (count + eps)
        self.max_l1_loss = torch.maximum(loss_mean.detach(), self.max_l1_loss)
        return loss, loss_mean

    def calc_region_weight(self, rid, rloss):
        eps = 1e-6
        gamma = 1e-3
        w = 1 + gamma * (rloss / (self.max_l1_loss + eps))
        return w

    def forward(self, real, fake, region_map, regions):
        self.real = real
        self.fake = fake
        self.region_map = region_map
        self.regions = range(0, 116)

        self.whole_loss = self.calc_pixel_loss()

        region_losses = {}
        weighted_loss = 0

        for rid in self.regions:
            rloss, rloss_mean = self.calc_region_loss(rid)
            region_losses[rid] = rloss, rloss_mean

        for rid, (rloss, rloss_mean) in region_losses.items():
            w = self.calc_region_weight(rid, rloss_mean)
            weighted_loss += w * rloss

        return weighted_loss.mean()
