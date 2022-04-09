# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch
import torch.nn as nn
from nncore.nn import (MODELS, build_linear_modules, build_model,
                       build_norm_layer)


@MODELS.register()
class UniModalEncoder(nn.Module):

    def __init__(self,
                 dims=None,
                 p=0.5,
                 pos_cfg=None,
                 enc_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(UniModalEncoder, self).__init__()

        drop_cfg = dict(type='drop', p=p) if p > 0 else None
        enc_dims = dims[-1] if isinstance(dims, (list, tuple)) else dims

        self.dropout = build_norm_layer(drop_cfg)
        self.mapping = build_linear_modules(dims, **kwargs)
        self.pos_enc = build_model(pos_cfg, enc_dims)
        self.encoder = build_model(enc_cfg, enc_dims, bundler='sequential')
        self.norm = build_norm_layer(norm_cfg, enc_dims)

    def forward(self, x, **kwargs):
        if self.dropout is not None:
            x = self.dropout(x)
        if self.mapping is not None:
            x = self.mapping(x)
        if self.encoder is not None:
            pe = None if self.pos_enc is None else self.pos_enc(x)
            x = self.encoder(x, pe=pe, **kwargs)
        if self.norm is not None:
            x = self.norm(x)
        return x


@MODELS.register()
class CrossModalEncoder(nn.Module):

    def __init__(self,
                 dims=None,
                 fusion_type='sum',
                 pos_cfg=None,
                 enc_cfg=None,
                 norm_cfg=None,
                 **kwargs):
        super(CrossModalEncoder, self).__init__()
        assert fusion_type in ('sum', 'mean', 'concat')

        map_dims = [2 * dims, dims] if fusion_type == 'concat' else None
        self.fusion_type = fusion_type

        self.pos_enc = build_model(pos_cfg, dims)
        self.encoder = build_model(enc_cfg, dims)
        self.mapping = build_linear_modules(map_dims, **kwargs)
        self.norm = build_norm_layer(norm_cfg, dims)

    def forward(self, a, b, **kwargs):
        if self.encoder is not None:
            pe = None if self.pos_enc is None else self.pos_enc(a)
            a, b = self.encoder(a, b, pe=pe, **kwargs)
        if self.fusion_type in ('sum', 'mean'):
            x = (a + b) / ((self.fusion_type == 'mean') + 1)
        else:
            x = torch.cat((a, b), dim=-1)
            x = self.mapping(x)
        if self.norm is not None:
            x = self.norm(x)
        return x
