import nncore
import torch
from nncore.dataset import DATASETS

from .tvsum import TVSum
from .utils import YOUTUBE_SPLITS


@DATASETS.register()
class YouTubeHighlights(TVSum):

    SPLITS = YOUTUBE_SPLITS

    def get_saliency(self, idx):
        video_id = self.get_video_id(idx)
        saliency = [1 if s > 0 else 0 for s in self.label[video_id]['match']]
        return torch.Tensor(saliency)

    def evaluate(self, blob, **kwargs):
        blob = nncore.to_dict_of_list(blob)
        collected = []

        for idx, score in enumerate(blob['saliency']):
            inds = torch.argsort(score[0], descending=True)
            label = self.get_saliency(idx)[inds].tolist()

            if (num_gt := sum(label)) == 0:
                collected.append(0)
                continue

            hits = ap = rec = 0
            prc = 1

            for i, gt in enumerate(label):
                hits += gt

                _rec = hits / num_gt
                _prc = hits / (i + 1)

                ap += (_rec - rec) * (prc + _prc) / 2
                rec, prc = _rec, _prc

            collected.append(ap)

        mean_ap = sum(collected) / len(collected)
        results = dict(mAP=round(mean_ap, 5))

        return results
