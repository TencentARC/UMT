_base_ = 'datasets'
# dataset settings
dataset_type = 'QVHighlights'
data_root = 'data/qvhighlights/'
data = dict(
    train=dict(
        type=dataset_type,
        label_path=data_root + 'highlight_train_release.jsonl',
        video_path=[
            data_root + 'slowfast_features', data_root + 'clip_features'
        ],
        audio_path=data_root + 'pann_features',
        query_path=data_root + 'clip_text_features',
        loader=dict(batch_size=32, num_workers=4, shuffle=True)),
    val=dict(
        type=dataset_type,
        label_path=data_root + 'highlight_val_release.jsonl',
        video_path=[
            data_root + 'slowfast_features', data_root + 'clip_features'
        ],
        audio_path=data_root + 'pann_features',
        query_path=data_root + 'clip_text_features',
        loader=dict(batch_size=1, num_workers=4, shuffle=False)))
