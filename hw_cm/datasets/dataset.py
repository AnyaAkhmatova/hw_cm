import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio


class CMDataset(Dataset):
    def __init__(self, path_to_kaggle_dataset, split):
        self.path_to_protocol = path_to_kaggle_dataset + '/LA/LA/ASVspoof2019_LA_cm_protocols'
        self.path_to_flac = path_to_kaggle_dataset + '/LA/LA'
        if split == 'train':
            self.path_to_protocol += '/ASVspoof2019.LA.cm.train.trn.txt'
            self.path_to_flac += '/ASVspoof2019_LA_train/flac/'
        elif split == 'dev':
            self.path_to_protocol += '/ASVspoof2019.LA.cm.dev.trl.txt'
            self.path_to_flac += '/ASVspoof2019_LA_dev/flac/'
        else:
            self.path_to_protocol += '/ASVspoof2019.LA.cm.eval.trl.txt'
            self.path_to_flac += '/ASVspoof2019_LA_eval/flac/'
        df = pd.read_csv(self.path_to_protocol, header=None, sep=' ')
        self.filenames = df[1]
        self.labels = df[4]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio, _ = torchaudio.load(self.path_to_flac + self.filenames[idx] + '.flac')
        audio = audio[0]
        padded_audio = torch.zeros(64000)
        for left_border in range(0, padded_audio.shape[0], audio.shape[0]):
            right_border = min(padded_audio.shape[0], left_border + audio.shape[0])
            padded_audio[left_border: right_border] = audio[: (right_border - left_border)]
        label = 1 if self.labels[idx] == 'bonafide' else 0
        return padded_audio, label


def collate_fn(data):
    audio, label = zip(*data)

    audio = torch.stack(audio)
    label = torch.tensor(label)
    
    return {"audio": audio, "label": label}


def get_dataloaders(path_to_kaggle_dataset, splits, batch_sizes):
    dataloaders = {}
    for i in range(len(splits)):
        split = splits[i]
        batch_size = batch_sizes[i]
        dataset = CMDataset(path_to_kaggle_dataset=path_to_kaggle_dataset, split=split)
        if split == 'train':
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        else:
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate_fn)
        dataloaders[split] = dataloader
    return dataloaders

