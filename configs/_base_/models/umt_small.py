_base_ = 'models'
# model settings
norm_cfg = dict(type='LN')
model = dict(
    type='UMT',
    video_enc=dict(
        type='UniModalEncoder',
        dims=[2048, 256],
        pos_cfg=dict(type='PositionalEncoding'),
        enc_cfg=dict(type='TransformerEncoderLayer')),
    audio_enc=dict(
        type='UniModalEncoder',
        dims=[2048, 256],
        pos_cfg=dict(type='PositionalEncoding'),
        enc_cfg=dict(type='TransformerEncoderLayer')),
    cross_enc=dict(
        type='CrossModalEncoder',
        dims=256,
        pos_cfg=dict(type='PositionalEncoding'),
        enc_cfg=dict(type='BottleneckTransformer'),
        norm_cfg=norm_cfg),
    query_gen=dict(
        type='QueryGenerator',
        dims=[512, 256],
        enc_cfg=dict(type='MultiHeadAttention'),
        last_norm=True,
        norm_cfg=norm_cfg),
    query_dec=dict(
        type='QueryDecoder',
        dims=256,
        pos_cfg=dict(type='PositionalEncoding'),
        dec_cfg=dict(type='TransformerDecoderLayer'),
        norm_cfg=norm_cfg))
