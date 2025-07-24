import os

import torch
import librosa
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio

class WavenetDataset(Dataset):
    def __init__(self, datapath, quantization_channels):
        super().__init__()
        self.datapath = datapath
        self.files = os.listdir(self.datapath)
        self.mu_law_transform = torchaudio.transforms.MuLawEncoding(quantization_channels=quantization_channels) # leaving the quantization channels to default (256)

    def __getitem__(self, idx):
        audio = self.files[idx]
        song, sr = torchaudio.load(os.path.join(self.datapath, audio), normalize=True)
        song = song.float() # cast from int16 to float32
        length = song.shape[-1]
        # print(song.shape)
        input_song = song[:, :-1]
        target_song = song[:, 1:]

        # Apply Mu Law Transform on the target
        target_song = self.mu_law_transform(target_song)

        return input_song, target_song

    def __len__(self):
        return len(self.files)

def create_datasets(quantization_channels):
    """
        Returns Train/Validation/Test datasets
    """
    dataset = WavenetDataset(datapath="data/", quantization_channels=quantization_channels)
    train_set, valid_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

    return train_set, valid_set, test_set


def create_dataloaders(quantization_channels):
    train_set, valid_set, test_set = create_datasets(quantization_channels)

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
        pin_memory=True,

    )

    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=8,
        shuffle=False,
        pin_memory=True
    )

    return train_dataloader, valid_dataloader, test_dataloader


if __name__ == "__main__":
    a, _, _ = create_dataloaders(256)
    print(next(iter(a)))
