_base_ = [
    '../_base_/models/umt_base.py', '../_base_/plugins/mrhd.py',
    '../_base_/datasets/qvhighlights.py', '../_base_/schedules/100e.py',
    '../_base_/runtime.py'
]
# model settings
model = dict(pred_head=dict(_update_=0, saliency_loss=dict(loss_weight=0.0)))
# dataset settings
data_root = 'data/qvhighlights/'
data = dict(
    train=dict(
        label_path=data_root + 'subs_train.jsonl',
        query_path=data_root + 'clip_sub_features',
        loader=dict(batch_size=512)))
# runtime settings
stages = dict(validation='_delete_')
