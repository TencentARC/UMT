# model settings
model = dict(
    video_enc=dict(dims=[2816, 256]),
    pred_head=[
        dict(
            type='SaliencyHead',
            dims=[256, 1],
            saliency_loss=dict(type='DynamicBCELoss', loss_weight=3.0)),
        dict(type='BoundaryHead', dims=[256, 1])
    ])
