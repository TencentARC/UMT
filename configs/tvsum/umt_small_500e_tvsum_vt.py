_base_ = [
    '../_base_/models/umt_small.py', '../_base_/plugins/hd.py',
    '../_base_/datasets/tvsum.py', '../_base_/schedules/500e.py',
    '../_base_/runtime.py'
]
# dataset settings
data = dict(train=dict(domain='VT'), val=dict(domain='VT'))
