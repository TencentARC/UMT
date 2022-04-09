# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import torch
import torch.nn as nn
from nncore.nn import MODELS, build_model, xavier_init_


@MODELS.register()
class UMT(nn.Module):

    def __init__(self,
                 video_enc=None,
                 audio_enc=None,
                 cross_enc=None,
                 query_gen=None,
                 query_dec=None,
                 pred_head=None):
        super(UMT, self).__init__()

        cnt = sum(e is None for e in (video_enc, audio_enc, cross_enc))
        assert not cnt % 2 and ((query_gen is None) == (query_dec is None))

        self.video_enc = build_model(video_enc)
        self.audio_enc = build_model(audio_enc)
        self.cross_enc = build_model(cross_enc)
        self.query_gen = build_model(query_gen)
        self.query_dec = build_model(query_dec)
        self.pred_head = build_model(pred_head, bundler='modulelist')

        self.apply(lambda m: xavier_init_(m)
                   if isinstance(m, nn.Linear) else None)

    def forward(self, data, mode):
        mask = torch.where(data['saliency'] >= 0, 1, 0)

        if self.video_enc is not None:
            d_emb = r_emb = v_emb = self.video_enc(data['video'], mask=mask)
        else:
            v_emb = data['video']

        if self.audio_enc is not None:
            d_emb = r_emb = a_emb = self.audio_enc(data['audio'], mask=mask)
        else:
            a_emb = data['audio']

        if self.cross_enc is not None:
            d_emb = r_emb = self.cross_enc(v_emb, a_emb, mask=mask)

        if self.query_gen is not None:
            q_emb = self.query_gen(r_emb, data.get('query'))
            d_emb = self.query_dec(q_emb, r_emb)

        output = dict(
            _avg_factor=mask.size(0), _out=dict(meta=data.get('meta')))

        for pred_head in self.pred_head:
            output = pred_head(d_emb, data, output, mode)

        return output
