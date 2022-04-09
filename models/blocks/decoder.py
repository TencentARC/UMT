# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch
import torch.nn as nn
from nncore.nn import (MODELS, build_linear_modules, build_model,
                       build_norm_layer)


@MODELS.register()
class QueryGenerator(nn.Module):

    def __init__(self, dims=None, p=0.3, enc_cfg=None, **kwargs):
        super(QueryGenerator, self).__init__()

        drop_cfg = dict(type='drop', p=p) if p > 0 else None
        enc_dims = dims[-1] if isinstance(dims, (list, tuple)) else dims

        self.dropout = build_norm_layer(drop_cfg)
        self.mapping = build_linear_modules(dims, **kwargs)
        self.encoder = build_model(enc_cfg, enc_dims)

    def forward(self, x, mem=None, **kwargs):
        if mem is None:
            mem = x.new_zeros(x.size(0), 10, x.size(2))
        else:
            mem[~mem.isfinite()] = 0
        mask = torch.where(mem[:, :, 0].isfinite(), 1, 0)
        if self.dropout is not None:
            mem = self.dropout(mem)
        if self.mapping is not None:
            mem = self.mapping(mem)
        if self.encoder is not None:
            x = self.encoder(x, mem, mask=mask, **kwargs)
        return x


@MODELS.register()
class QueryDecoder(nn.Module):

    def __init__(self, dims=None, pos_cfg=None, dec_cfg=None, norm_cfg=None):
        super(QueryDecoder, self).__init__()

        self.q_pos_enc = build_model(pos_cfg, dims)
        self.k_pos_enc = build_model(pos_cfg, dims)
        self.decoder = build_model(dec_cfg, dims, bundler='modulelist')
        self.norm = build_norm_layer(norm_cfg, dims)

    def forward(self, x, mem=None, **kwargs):
        out = [x]
        if self.decoder is not None:
            q_pe = None if self.q_pos_enc is None else self.q_pos_enc(x)
            k_pe = None if self.k_pos_enc is None else self.k_pos_enc(x)
            for dec in self.decoder:
                hid = dec(out[-1], mem=mem, q_pe=q_pe, k_pe=k_pe, **kwargs)
                out.append(hid)
        x = out if len(out) == 1 else out[1:]
        if self.norm is not None:
            x = [self.norm(h) for h in x]
        return x
