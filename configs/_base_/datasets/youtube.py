_base_ = 'datasets'
# dataset settings
dataset_type = 'YouTubeHighlights'
data_root = 'data/youtube/'
data = dict(
    train=dict(
        type=dataset_type,
        domain=None,
        label_path=data_root + 'youtube_anno.json',
        video_path=data_root + 'video_features',
        audio_path=data_root + 'audio_features',
        loader=dict(batch_size=4, num_workers=4, shuffle=True)),
    val=dict(
        type=dataset_type,
        domain=None,
        label_path=data_root + 'youtube_anno.json',
        video_path=data_root + 'video_features',
        audio_path=data_root + 'audio_features',
        loader=dict(batch_size=1, num_workers=4, shuffle=False)))
