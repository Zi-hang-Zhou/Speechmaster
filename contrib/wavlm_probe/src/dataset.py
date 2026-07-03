"""Dataset over cached WavLM features for CTC training.

Supports two modes:
  - continuous: load fp16 hidden states of a given layer -> [T, 768]
  - discrete:   map each frame to its k-means cluster id -> [T] long (embedded in model)
"""
import os
import sys

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))
from common import load_manifest, text_to_ids


class FeatDataset(Dataset):
    def __init__(self, manifest, feat_dir, layer, discrete_labels=None):
        self.items = load_manifest(manifest)
        self.feat_dir = feat_dir
        self.key = f"L{layer}"
        self.discrete = discrete_labels  # dict uttid -> np.int array of cluster ids, or None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        target = torch.tensor(text_to_ids(it["text"]), dtype=torch.long)
        if self.discrete is not None:
            feat = torch.tensor(self.discrete[it["uttid"]], dtype=torch.long)  # [T]
        else:
            arr = np.load(os.path.join(self.feat_dir, it["uttid"] + ".npz"))[self.key]
            feat = torch.tensor(arr.astype(np.float32))  # [T, 768]
        return feat, target, it["uttid"]


def collate_continuous(batch):
    feats, targets, uttids = zip(*batch)
    feat_lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    tgt_lens = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    T = feat_lens.max().item()
    D = feats[0].shape[1]
    padded = torch.zeros(len(feats), T, D)
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
    targets = torch.cat(targets)
    return padded, feat_lens, targets, tgt_lens, list(uttids)


def collate_discrete(batch):
    feats, targets, uttids = zip(*batch)
    feat_lens = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    tgt_lens = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)
    T = feat_lens.max().item()
    padded = torch.zeros(len(feats), T, dtype=torch.long)
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = f
    targets = torch.cat(targets)
    return padded, feat_lens, targets, tgt_lens, list(uttids)
