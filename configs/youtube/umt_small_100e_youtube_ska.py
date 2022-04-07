_base_ = [
    '../_base_/models/umt_small.py', '../_base_/plugins/hd.py',
    '../_base_/plugins/no_query.py', '../_base_/datasets/youtube.py',
    '../_base_/schedules/100e.py', '../_base_/runtime.py'
]
# dataset settings
data = dict(train=dict(domain='skating'), val=dict(domain='skating'))
