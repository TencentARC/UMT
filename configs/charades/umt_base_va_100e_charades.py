_base_ = [
    '../_base_/models/umt_base.py', '../_base_/plugins/mr.py',
    '../_base_/datasets/charades.py', '../_base_/schedules/100e.py',
    '../_base_/runtime.py'
]
# dataset settings
data = dict(train=dict(modality='va'), val=dict(modality='va'))
