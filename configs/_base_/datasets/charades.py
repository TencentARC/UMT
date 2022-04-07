_base_ = 'datasets'
# dataset settings
dataset_type = 'CharadesSTA'
data_root = 'data/charades/'
data = dict(
    train=dict(
        type=dataset_type,
        modality=None,
        label_path=data_root + 'charades_sta_train.txt',
        video_path=data_root + 'rgb_features',
        optic_path=data_root + 'opt_features',
        audio_path=data_root + 'audio_features',
        loader=dict(batch_size=8, num_workers=4, shuffle=True)),
    val=dict(
        type=dataset_type,
        modality=None,
        label_path=data_root + 'charades_sta_test.txt',
        video_path=data_root + 'rgb_features',
        optic_path=data_root + 'opt_features',
        audio_path=data_root + 'audio_features',
        loader=dict(batch_size=1, num_workers=4, shuffle=False)))
