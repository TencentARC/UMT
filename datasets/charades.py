# Copyright (c) THL A29 Limited, a Tencent company. All rights reserved.

import nncore
import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.dataset import DATASETS
from nncore.ops import temporal_iou
from nncore.parallel import DataContainer
from torch.utils.data import Dataset
from torchtext import vocab


@DATASETS.register()
class CharadesSTA(Dataset):

    def __init__(self,
                 modality,
                 label_path,
                 video_path,
                 optic_path=None,
                 audio_path=None):
        assert modality in ('va', 'vo')
        self.label = nncore.load(label_path)

        self.modality = modality
        self.label_path = label_path
        self.video_path = video_path
        self.optic_path = optic_path
        self.audio_path = audio_path

        self.vocab = vocab.pretrained_aliases['glove.6B.300d']()
        self.vocab.itos.extend(['<unk>'])
        self.vocab.stoi['<unk>'] = self.vocab.vectors.shape[0]
        self.vocab.vectors = torch.cat(
            (self.vocab.vectors, torch.zeros(1, self.vocab.dim)), dim=0)
        self.embedding = nn.Embedding.from_pretrained(self.vocab.vectors)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        video = self.get_video(idx)
        audio = self.get_audio(idx)
        query = self.get_query(idx)

        num_clips = min(c.size(0) for c in (video, audio))

        boundary = self.get_boundary(idx)
        saliency = torch.ones(num_clips)

        data = dict(
            video=DataContainer(video[:num_clips]),
            audio=DataContainer(audio[:num_clips]),
            query=DataContainer(query, pad_value=float('inf')),
            saliency=DataContainer(saliency, pad_value=-1),
            meta=DataContainer(self.label[idx], cpu_only=True))

        if boundary is not None:
            data['boundary'] = DataContainer(boundary, pad_value=-1)

        return data

    def parse_boundary(self, label):
        boundary = label.split('##')[0].split()[1:]
        if float(boundary[1]) < float(boundary[0]):
            boundary = [boundary[1], boundary[0]]
        return torch.Tensor([[float(s) for s in boundary]])

    def get_video(self, idx):
        vid = self.label[idx].split()[0]
        video = nncore.load(nncore.join(self.video_path, f'{vid}.npy'))
        return F.normalize(torch.from_numpy(video).float())

    def get_audio(self, idx):
        vid = self.label[idx].split()[0]
        path = self.audio_path if self.modality == 'va' else self.optic_path
        audio = nncore.load(nncore.join(path, f'{vid}.npy'))
        return F.normalize(torch.from_numpy(audio).float())

    def get_query(self, idx):
        query = self.label[idx].split('##')[-1][:-1]
        word_inds = torch.LongTensor(
            [self.vocab.stoi.get(w.lower(), 400000) for w in query.split()])
        return self.embedding(word_inds)

    def get_boundary(self, idx):
        return self.parse_boundary(self.label[idx])

    def evaluate(self,
                 blob,
                 method='gaussian',
                 nms_thr=0.3,
                 sigma=0.5,
                 rank=[1, 5],
                 iou_thr=[0.5, 0.7],
                 **kwargs):
        assert method in ('fast', 'normal', 'linear', 'gaussian')

        blob = nncore.to_dict_of_list(blob)
        results = dict()

        print('Performing temporal NMS...')
        boundary = []

        for bnd in blob['boundary']:
            bnd = bnd[0]

            if method == 'fast':
                iou = temporal_iou(bnd[:, :-1], bnd[:, :-1]).triu(diagonal=1)
                keep = iou.amax(dim=0) <= nms_thr
                bnd = bnd[keep]
            else:
                for i in range(bnd.size(0)):
                    max_idx = bnd[i:, -1].argmax(dim=0)
                    bnd = nncore.swap_element(bnd, i, max_idx + i)
                    iou = temporal_iou(bnd[i, None, :-1], bnd[i + 1:, :-1])[0]

                    if method == 'normal':
                        bnd[i + 1:, -1][iou >= nms_thr] = 0
                    elif method == 'linear':
                        bnd[i + 1:, -1] *= 1 - iou
                    else:
                        bnd[i + 1:, -1] *= (-iou.pow(2) / sigma).exp()

            boundary.append(bnd)

        for k in rank:
            for thr in iou_thr:
                print(f'Evaluating Rank{k}@{thr}...')
                hits = 0

                for idx, bnd in enumerate(boundary):
                    inds = torch.argsort(bnd[:, -1], descending=True)
                    keep = inds[:k]
                    bnd = bnd[:, :-1][keep]

                    gt = self.parse_boundary(blob['meta'][idx][0])
                    iou = temporal_iou(gt, bnd)

                    if iou.max() >= thr:
                        hits += 1

                results[f'Rank{k}@{thr}'] = hits / len(self.label)

        return results
