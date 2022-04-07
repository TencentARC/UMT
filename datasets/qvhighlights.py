import nncore
import torch
from nncore.dataset import DATASETS
from nncore.parallel import DataContainer
from torch.utils.data import Dataset

from .utils import eval_qvhighlights


@DATASETS.register()
class QVHighlights(Dataset):

    def __init__(self, label_path, video_path, audio_path, query_path):
        self.label = nncore.load(label_path)

        self.label_path = label_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.query_path = query_path

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video = self.get_video(idx)
        audio = self.get_audio(idx)
        query = self.get_query(idx)

        saliency = self.get_saliency(idx)
        boundary = self.get_boundary(idx)

        if saliency is None:
            num_clips = min(c.size(0) for c in (video, audio))
            saliency = torch.ones(num_clips)
        else:
            num_clips = min(c.size(0) for c in (video, audio, saliency))
            saliency = saliency[:num_clips]

        data = dict(
            video=DataContainer(video[:num_clips]),
            audio=DataContainer(audio[:num_clips]),
            query=DataContainer(query, pad_value=float('inf')),
            saliency=DataContainer(saliency, pad_value=-1),
            meta=DataContainer(self.label[idx], cpu_only=True))

        if boundary is not None:
            data['boundary'] = DataContainer(boundary, pad_value=-1)

        return data

    def get_video(self, idx):
        vid = self.label[idx]['vid']
        video = [
            nncore.load(nncore.join(path, f'{vid}.npz'))['features']
            for path in self.video_path
        ]
        num_clips = min(video[0].shape[0], video[1].shape[0])
        video = [torch.from_numpy(v[:num_clips]) for v in video]
        return torch.cat(video, dim=1).float()

    def get_audio(self, idx):
        vid = self.label[idx]['vid']
        audio = nncore.load(nncore.join(self.audio_path, f'{vid}.npy'))
        return torch.from_numpy(audio).float()

    def get_query(self, idx):
        qid = self.label[idx]['qid']
        query = nncore.load(nncore.join(self.query_path, f'qid{qid}.npz'))
        return torch.from_numpy(query['last_hidden_state']).float()

    def get_saliency(self, idx):
        if 'saliency_scores' in self.label[idx]:
            saliency = [0] * int(self.label[idx]['duration'] / 2)
            for clip_id, score in zip(self.label[idx]['relevant_clip_ids'],
                                      self.label[idx]['saliency_scores']):
                saliency[clip_id] = sum(score) / 12
            return torch.Tensor(saliency)

    def get_boundary(self, idx):
        if 'relevant_windows' in self.label[idx]:
            return torch.Tensor(self.label[idx]['relevant_windows'])

    def evaluate(self, blob, **kwargs):
        num_samples, collected = len(blob), []
        blob = nncore.to_dict_of_list(blob)

        for i in range(num_samples):
            pred = dict(
                qid=blob['meta'][i][0]['qid'], vid=blob['meta'][i][0]['vid'])

            if 'saliency' in blob:
                pred['pred_saliency_scores'] = blob['saliency'][i][0].tolist()

            if 'boundary' in blob:
                pred['pred_relevant_windows'] = blob['boundary'][i][0].tolist()

            collected.append(pred)

        label = nncore.load(self.label_path)
        results = eval_qvhighlights(collected, label)['brief']

        return results
