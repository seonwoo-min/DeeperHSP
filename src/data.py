# Written by Seonwoo Min, Seoul National University (mswzeus@gmail.com)

import json
import numpy as np

import torch
import torch.nn.functional as F


class HSP_dataset(torch.utils.data.Dataset):
    """ Pytorch dataloader for HSP dataset """
    def __init__(self, X, labels, split_idx):
        self.X = X
        self.labels = labels
        self.max_length = 5000

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        e = torch.from_numpy(np.load(self.X[i])).float()
        if len(e) < self.max_length:
            e = F.pad(e, (0, 0, 0, self.max_length - e.shape[0]), "constant", 0)

        return e, self.labels[i]


labels_dict = {"NonHSP":0, "HSP20":1, "HSP40":2, "HSP60":3, "HSP70":4, "HSP90":5, "HSP100":6}
def get_dataset_from_configs(data_cfg, split_idx=None, embedder_idx=None, sanity_check=False):
    """ load HSP dataset from config files """
    X, labels = [], []
    if "fold" in data_cfg.data_idx:
        for dataset, split in sorted(data_cfg.split.items()):
            if "test" in dataset: continue
            fold = int(data_cfg.data_idx[-1])
            for s in range(5):
                if (split_idx == "test" and s == fold) or (split_idx == "train" and s != fold):
                    for filename in split["split%d" % s]:
                        if sanity_check and len(X) == 300: break
                        X.append(data_cfg.path["data"] + "/%s/%s/%d.npy" % (embedder_idx, dataset, filename))
                        labels.append(labels_dict[dataset.split("_")[0]])

    else:
        for dataset, split in sorted(data_cfg.split.items()):
            if ((data_cfg.data_idx == "all" and "test" in dataset) or
                (data_cfg.data_idx != "all" and "test" not in dataset)): continue
            for filename in split["all"]:
                if sanity_check and len(X) == 300: break
                X.append(data_cfg.path["data"] + "/%s/%s/%d.npy" % (embedder_idx, dataset, filename))
                labels.append(labels_dict[dataset.split("_")[0]])

    labels = torch.from_numpy(np.array(labels)).long()
    dataset = HSP_dataset(X, labels, split_idx)

    return dataset
