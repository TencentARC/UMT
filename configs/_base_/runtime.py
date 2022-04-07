# runtime settings
hooks = dict(
    type='EvalHook',
    high_keys=[
        'MR-full-mAP', 'HL-min-VeryGood-mAP', 'Rank1@0.5', 'Rank1@0.7',
        'Rank5@0.5', 'Rank5@0.7', 'mAP'
    ])
