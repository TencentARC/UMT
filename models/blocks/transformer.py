# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch.nn as nn
from nncore.nn import (MODELS, FeedForwardNetwork, MultiHeadAttention,
                       Parameter, build_norm_layer)


@MODELS.register()
class BottleneckTransformerLayer(nn.Module):

    def __init__(self,
                 dims,
                 heads=8,
                 ratio=4,
                 p=0.1,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(BottleneckTransformerLayer, self).__init__()

        self.dims = dims
        self.heads = heads
        self.ratio = ratio
        self.p = p

        self.att1 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att2 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att3 = MultiHeadAttention(dims, heads=heads, p=p)
        self.att4 = MultiHeadAttention(dims, heads=heads, p=p)

        self.ffn1 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)
        self.ffn2 = FeedForwardNetwork(dims, ratio=ratio, p=p, act_cfg=act_cfg)

        self.norm1 = build_norm_layer(norm_cfg, dims=dims)
        self.norm2 = build_norm_layer(norm_cfg, dims=dims)
        self.norm3 = build_norm_layer(norm_cfg, dims=dims)
        self.norm4 = build_norm_layer(norm_cfg, dims=dims)
        self.norm5 = build_norm_layer(norm_cfg, dims=dims)
        self.norm6 = build_norm_layer(norm_cfg, dims=dims)

    def forward(self, a, b, t, pe=None, mask=None):
        da = self.norm1(a)
        db = self.norm2(b)
        dt = self.norm3(t)

        ka = da if pe is None else da + pe
        kb = db if pe is None else db + pe

        at = self.att1(dt, ka, da, mask=mask)
        bt = self.att2(dt, kb, db, mask=mask)

        t = t + at + bt
        dt = self.norm4(t)

        qa = da if pe is None else da + pe
        qb = db if pe is None else db + pe

        a = a + self.att3(qa, dt)
        b = b + self.att4(qb, dt)

        da = self.norm5(a)
        db = self.norm6(b)

        a = a + self.ffn1(da)
        b = b + self.ffn2(db)

        return a, b, t


@MODELS.register()
class BottleneckTransformer(nn.Module):

    def __init__(self, dims, num_tokens=4, num_layers=1, **kwargs):
        super(BottleneckTransformer, self).__init__()

        self.dims = dims
        self.num_tokens = num_tokens
        self.num_layers = num_layers

        self.token = Parameter(num_tokens, dims)
        self.encoder = nn.ModuleList([
            BottleneckTransformerLayer(dims, **kwargs)
            for _ in range(num_layers)
        ])

    def forward(self, a, b, **kwargs):
        t = self.token.expand(a.size(0), -1, -1)
        for enc in self.encoder:
            a, b, t = enc(a, b, t, **kwargs)
        return a, b
