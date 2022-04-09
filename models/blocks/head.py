# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import MODELS, build_linear_modules, build_loss


@MODELS.register()
class SaliencyHead(nn.Module):

    def __init__(self,
                 dims,
                 pred_indices=None,
                 loss_indices=None,
                 saliency_loss=dict(type='DynamicBCELoss', loss_weight=1.0),
                 **kwargs):
        super(SaliencyHead, self).__init__()

        self.saliency_pred = build_linear_modules(dims, **kwargs)
        self.saliency_loss = build_loss(saliency_loss)

        self.pred_indices = pred_indices or (-1, )
        self.loss_indices = loss_indices or self.pred_indices

    def forward(self, inputs, data, output, mode):
        mask = torch.where(data['saliency'] >= 0, 1, 0)

        pred_indices = [idx % len(inputs) for idx in self.pred_indices]
        loss_indices = [idx % len(inputs) for idx in self.loss_indices]

        out = []
        for i, x in enumerate(inputs):
            saliency_pred = self.saliency_pred(x).squeeze(-1)

            if i in pred_indices:
                saliency = saliency_pred.sigmoid() * mask
                out.append(saliency)

            if mode != 'test' and i in loss_indices:
                output[f'd{i}.saliency_loss'] = self.saliency_loss(
                    saliency_pred, data['saliency'], weight=mask)

        output['_out']['saliency'] = (sum(out) / len(out)).detach().cpu()
        return output


@MODELS.register()
class BoundaryHead(nn.Module):

    def __init__(self,
                 dims,
                 radius_factor=0.2,
                 sigma_factor=0.2,
                 kernel=1,
                 unit=2,
                 max_num_moments=100,
                 pred_indices=None,
                 loss_indices=None,
                 center_loss=dict(type='GaussianFocalLoss', loss_weight=1.0),
                 window_loss=dict(type='L1Loss', loss_weight=0.1),
                 offset_loss=dict(type='L1Loss', loss_weight=1.0),
                 **kwargs):
        super(BoundaryHead, self).__init__()

        self.center_pred = build_linear_modules(dims, **kwargs)
        self.window_pred = build_linear_modules(dims, **kwargs)
        self.offset_pred = build_linear_modules(dims, **kwargs)

        self.center_loss = build_loss(center_loss)
        self.window_loss = build_loss(window_loss)
        self.offset_loss = build_loss(offset_loss)

        self.radius_factor = radius_factor
        self.sigma_factor = sigma_factor
        self.kernel = kernel
        self.unit = unit
        self.max_num_moments = max_num_moments
        self.pred_indices = pred_indices or (-1, )
        self.loss_indices = loss_indices or self.pred_indices

    def get_targets(self, boundary, num_clips):
        batch_size = boundary.size(0)
        avg_factor = 0

        center_tgt = boundary.new_zeros(batch_size, num_clips)
        window_tgt = boundary.new_zeros(batch_size, num_clips)
        offset_tgt = boundary.new_zeros(batch_size, num_clips)
        weight = boundary.new_zeros(batch_size, num_clips)

        for batch_id in range(batch_size):
            batch_boundary = boundary[batch_id]
            batch_boundary[:, 1] -= self.unit

            keep = batch_boundary[:, 0] != -1
            batch_boundary = batch_boundary[keep] / self.unit

            num_centers = batch_boundary.size(0)
            avg_factor += num_centers

            centers = batch_boundary.mean(dim=-1).clamp(max=num_clips - 0.5)
            windows = batch_boundary[:, 1] - batch_boundary[:, 0]

            for i, center in enumerate(centers):
                radius = (windows[i] * self.radius_factor).int().item()
                sigma = (radius + 1) * self.sigma_factor
                center_int = center.int().item()

                heatmap = batch_boundary.new_zeros(num_clips)
                start = max(0, center_int - radius)
                end = min(center_int + radius + 1, num_clips)

                kernel = torch.arange(start - center_int, end - center_int)
                kernel = (-kernel**2 / (2 * sigma**2)).exp()
                heatmap[start:end] = kernel

                center_tgt[batch_id] = torch.max(center_tgt[batch_id], heatmap)
                window_tgt[batch_id, center_int] = windows[i]
                offset_tgt[batch_id, center_int] = center - center_int
                weight[batch_id, center_int] = 1

        return center_tgt, window_tgt, offset_tgt, weight, avg_factor

    def get_boundary(self, center_pred, window_pred, offset_pred):
        pad = (self.kernel - 1) // 2
        hmax = F.max_pool1d(center_pred, self.kernel, stride=1, padding=pad)
        keep = (hmax == center_pred).float()
        center_pred = center_pred * keep

        topk = min(self.max_num_moments, center_pred.size(1))
        scores, inds = torch.topk(center_pred, topk)

        center = inds + offset_pred.gather(1, inds).clamp(min=0)
        window = window_pred.gather(1, inds).clamp(min=0)

        boundry = center.unsqueeze(-1).repeat(1, 1, 2)
        boundry[:, :, 0] = center - window / 2
        boundry[:, :, 1] = center + window / 2
        boundry = boundry.clamp(min=0, max=center_pred.size(1) - 1) * self.unit
        boundry[:, :, 1] += self.unit

        boundary = torch.cat((boundry, scores.unsqueeze(-1)), dim=2)
        return boundary

    def forward(self, inputs, data, output, mode):
        mask = torch.where(data['saliency'] >= 0, 1, 0)

        pred_indices = [idx % len(inputs) for idx in self.pred_indices]
        loss_indices = [idx % len(inputs) for idx in self.loss_indices]

        out = []
        for i, x in enumerate(inputs):
            center_pred = self.center_pred(x).squeeze(-1).sigmoid() * mask
            window_pred = self.window_pred(x).squeeze(-1)
            offset_pred = self.offset_pred(x).squeeze(-1)

            if i in pred_indices:
                boundary = self.get_boundary(center_pred, window_pred,
                                             offset_pred)
                out.append(boundary)

            if mode != 'test' and i in loss_indices:
                tgts = self.get_targets(data['boundary'], mask.size(1))
                center_tgt, window_tgt, offset_tgt, weight, avg_factor = tgts

                output[f'd{i}.center_loss'] = self.center_loss(
                    center_pred,
                    center_tgt,
                    weight=mask,
                    avg_factor=avg_factor)
                output[f'd{i}.window_loss'] = self.window_loss(
                    window_pred,
                    window_tgt,
                    weight=weight,
                    avg_factor=avg_factor)
                output[f'd{i}.offset_loss'] = self.offset_loss(
                    offset_pred,
                    offset_tgt,
                    weight=weight,
                    avg_factor=avg_factor)

        output['_out']['boundary'] = torch.cat(out, dim=1).detach().cpu()
        return output
