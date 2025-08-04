import torch
import torchaudio
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from pydub import AudioSegment
from tqdm import tqdm

from config import get_config

def trim_audio(path : str, chunk_size : int = 5000):
    audio = AudioSegment.from_file(path)
    audio = audio.set_channels(1)
    print(audio)
    chunks = []
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i+chunk_size]
        if len(chunk) < chunk_size:
            silence = AudioSegment.silent(chunk_size-len(chunk))
            chunk += silence
        filename = int(i/chunk_size)
        chunk.export(f"data/{filename}.wav", format='wav')

def create_optimizer_and_scheduler(model, dataloader, start_epoch, end_epoch, optimizer_state_dict=None, scheduler_state_dict=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.75,
        patience=3
    )

    if optimizer_state_dict is not None and scheduler_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    return optimizer, scheduler

def calculate_accuracy(logits, targets):
    correct = 0
    total = 0

    # apply softmax to the logits
    preds = F.softmax(logits, dim=-1).argmax(-1)

    total += logits.shape[1] * logits.shape[0]
    correct += (preds == targets).sum()

    accuracy = (correct / total) * 100

    return accuracy.item()

def create_summary_writer(model_name):
    writer_dir = f"./runs/{model_name}"
    writer = SummaryWriter(log_dir=writer_dir)
    print(f"[INFO] SummaryWriter created in {writer_dir}.")
    return writer

def save_generated_audio(audio, path, sr):
    if audio.ndim == 3:
        audio_save = audio.squeeze(0) # squeeze the batch dimension
    print(audio_save.shape)
    torchaudio.save(path, audio_save, sample_rate=sr)
    print(f"[INFO] Audio successfully saved to: {path}.")
