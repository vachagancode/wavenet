import os

import torch
import librosa
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio

class WavenetDataset(Dataset):
    def __init__(self, datapath):
        super().__init__()
        self.datapath = datapath
        self.files = os.listdir(self.datapath)
        self.mu_law_transform = torchaudio.transforms.MuLawEncoding(quantization_channels=512) # leaving the quantization channels to default (256)

    def __getitem__(self, idx):
        audio = self.files[idx]
        song, sr = torchaudio.load(os.path.join(self.datapath, audio), normalize=True)
        song = song.float() # cast from int16 to float32
        length = int(song.shape[-1] // 2)
        # print(song.shape)
        input_song = song[:, :length]
        target_song = song[:, length:]
        if target_song.shape[-1] != length:
            target_song = F.pad(target_song, pad=(0, length - target_song.shape[-1]))
        # Apply Mu Law Transform on the target
        target_song = self.mu_law_transform(target_song)

        return input_song, target_song

    def __len__(self):
        return len(self.files)

def create_datasets():
    """
        Returns Train/Validation/Test datasets
    """
    dataset = WavenetDataset(datapath="data/")
    train_set, valid_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

    return train_set, valid_set, test_set


def create_dataloaders():
    train_set, valid_set, test_set = create_datasets()

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=8,
        shuffle=True,
        pin_memory=True
    )

    valid_dataloader = DataLoader(
        dataset=valid_set,
        batch_size=8,
        shuffle=True,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=8,
        shuffle=False,
        pin_memory=True
    )

    return train_dataloader, valid_dataloader, test_dataloader
