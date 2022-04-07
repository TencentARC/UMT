import nncore
import torch
from nncore.dataset import DATASETS
from nncore.parallel import DataContainer
from torch.utils.data import Dataset

from .utils import TVSUM_SPLITS


@DATASETS.register()
class TVSum(Dataset):

    SPLITS = TVSUM_SPLITS

    def __init__(self,
                 domain,
                 label_path,
                 video_path,
                 audio_path,
                 query_path=None):
        assert domain in self.SPLITS

        self.domain = domain
        self.label_path = label_path
        self.video_path = video_path
        self.audio_path = audio_path
        self.query_path = query_path

        self.label = nncore.load(label_path)

        video = nncore.join(video_path, '{}_rgb.npy')
        optic = nncore.join(video_path, '{}_opt.npy')
        audio = nncore.join(audio_path, '{}.npy')

        self.video = {k: nncore.load(video.format(k)) for k in self.label}
        self.optic = {k: nncore.load(optic.format(k)) for k in self.label}
        self.audio = {k: nncore.load(audio.format(k)) for k in self.label}

        self.video_id = {
            k: [s for s in self.SPLITS[domain][k] if s in self.label]
            for k in ('train', 'val')
        }

        self.set_state('train')

    def __len__(self):
        return len(self.video_id[self.state])

    def __getitem__(self, idx):
        video = self.get_video(idx)
        audio = self.get_audio(idx)
        saliency = self.get_saliency(idx)

        num_clips = min(c.size(0) for c in (video, audio, saliency))

        data = dict(
            video=DataContainer(video[:num_clips]),
            audio=DataContainer(audio[:num_clips]),
            saliency=DataContainer(saliency[:num_clips], pad_value=-1))

        if self.query_path is not None:
            query = self.get_query(idx)
            data['query'] = DataContainer(query, pad_value=float('inf'))

        return data

    def set_state(self, state):
        self.state = 'train' if state == 'train' else 'val'

    def get_video_id(self, idx):
        return self.video_id[self.state][idx]

    def get_video(self, idx):
        video_id = self.get_video_id(idx)
        video = torch.from_numpy(self.video[video_id]).float()
        optic = torch.from_numpy(self.optic[video_id]).float()
        return torch.cat((video, optic), dim=1)

    def get_audio(self, idx):
        video_id = self.get_video_id(idx)
        return torch.from_numpy(self.audio[video_id]).float()

    def get_query(self, idx):
        video_id = self.get_video_id(idx)
        query = nncore.load(nncore.join(self.query_path, f'{video_id}.npz'))
        return torch.from_numpy(query['token']).float()

    def get_saliency(self, idx):
        video_id = self.get_video_id(idx)
        saliency = torch.Tensor(self.label[video_id]['anno'])
        return (saliency.sum(dim=1) - 20) / 80

    def evaluate(self, blob, k=5, **kwargs):
        blob = nncore.to_dict_of_list(blob)
        collected = []

        for i in range(20):
            video_ap = []

            for idx, score in enumerate(blob['saliency']):
                inds = torch.argsort(score[0], descending=True)

                video_id = self.get_video_id(idx)
                label = torch.Tensor(self.label[video_id]['anno'])[:, i]
                label = torch.where(label > label.median(), 1.0, .0)
                label = label[inds].tolist()[:k]

                if (num_gt := sum(label)) == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc

                video_ap.append(ap)

            collected.append(sum(video_ap) / len(video_ap))

        mean_ap = sum(collected) / len(collected)
        results = dict(mAP=round(mean_ap, 5))

        return results
