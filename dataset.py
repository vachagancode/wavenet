import os

import torch
import librosa
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchaudio
import pandas as pd

class WavenetDataset(Dataset):
    def __init__(self, datapath, quantization_channels):
        super().__init__()
        self.df = pd.read_csv(datapath, sep="|")

        self.audio_path = "data/wavs/"

        self.sample_rate = 22050

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=1024,
            hop_length=256,
            n_mels=80
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

        self.mu_law_transform = torchaudio.transforms.MuLawEncoding(quantization_channels=quantization_channels) # leaving the quantization channels to default (256)

    def __getitem__(self, idx):
        audio_name = f"{self.df['name'].iloc[idx]}.wav"

        audio_path = os.path.join(self.audio_path, audio_name)

        # Load the audio
        waveform, sr = torchaudio.load(audio_path)
        
        if waveform.ndim == 2:
            waveform = waveform.mean(0, keepdim=True)

        # Apply Formattin

        src = waveform[:, 0:waveform.shape[-1]-1]
        tgt = self.mu_law_transform(waveform[:, 1:waveform.shape[-1]])

        mel_spectrogram = self.amp_to_db(self.mel_transform(waveform))

        return src, tgt, mel_spectrogram

    def __len__(self):
        return len(self.df)

def create_datasets(quantization_channels):
    """
        Returns Train/Validation/Test datasets
    """
    dataset = WavenetDataset(datapath="data/metadata.csv", quantization_channels=quantization_channels)
    train_set, valid_set, test_set = random_split(dataset, [0.8, 0.1, 0.1])

    return train_set, valid_set, test_set


def collate_fn(batch):
    src, tgt, mel_spectrograms = zip(*batch)
    
    trimmed_src = []
    trimmed_tgt = []
    trimmed_mel_spectrograms = []

    # <-- Trimming the waveforms -->
    # Get the lengths of waveforms
    src_lengths = [wf.shape[-1] for wf in list(src)]
    # Get the smallest one and get the index of it
    src_min_length = min(src_lengths)
    min_ix = src_lengths.index(src_min_length) # going to be the same for mel spectrograms also
    trimmed_src.append(src[min_ix])
    trimmed_tgt.append(tgt[min_ix])

    for s in src:
        if s.shape[-1] != src_min_length:
            trimmed_src.append(s[:, :src_min_length])
    for t in tgt:
        if t.shape[-1] != src_min_length:
            trimmed_tgt.append(t[:, :src_min_length])
    
    # <-- Trimming the Mel Spectrograms -->
    mel_lengths = [ms.shape[-1] for ms in list(mel_spectrograms)]
    # print(mel_lengths)
    mel_min_length = min(mel_lengths)
    trimmed_mel_spectrograms.append(mel_spectrograms[min_ix])
    for mel_spectrogram in mel_spectrograms:
        if mel_spectrogram.shape[-1] != mel_min_length:
            trimmed_mel_spectrograms.append(mel_spectrogram[:, :, :mel_min_length])

    
    trimmed_src = torch.stack(trimmed_src, dim=0)
    trimmed_tgt = torch.stack(trimmed_tgt, dim=0)
    trimmed_mel_spectrograms = torch.stack(trimmed_mel_spectrograms, dim=0).squeeze(1)

    return trimmed_src, trimmed_tgt, trimmed_mel_spectrograms

def create_dataloaders(quantization_channels):
    train_set, valid_set, test_set = create_datasets(quantization_channels)

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    valid_dataloader = DataLoader(
        dataset=valid_set,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        collate_fn=collate_fn
    )

    test_dataloader = DataLoader(
        dataset=test_set,
        batch_size=4,
        shuffle=False,
        pin_memory=True,
        collate_fn=collate_fn
    )

    return train_dataloader, valid_dataloader, test_dataloader