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

def sample_from_model(config : dict, path : str = None, length : int = 5, device : torch.device = torch.device("cpu")): # if path to the file is specified then use the file to sample from
    if path is not None:
        start_sample, sr = torchaudio.load(path, normalize=False)
        start_sample = start_sample.float().unsqueeze(0) # cast to float 32 and add the batch dimension
    else:
        # create a 0 tensor
        start_sample = torch.zeros(1, 1, 15)

    model = create_wavenet(config, device)
    model_outputs = [start_sample]

    for _ in range(length):
        model.eval()
        with torch.inference_mode():
            # Forward pass
            logits = model(start_sample)

            # Do the decoding
            decoded_data =  greedy_decode(logits)
            start_sample = decoded_data.float()

            model_outputs.append(decoded_data)

    output = torch.cat(tuple(model_outputs), dim=-1).squeeze(0)

    return output

def create_optimizer_and_scheduler(model, dataloader, start_epoch, end_epoch, optimizer_state_dict=None, scheduler_state_dict=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=0.5
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
    print(f"[INFO] Audio sucessfully saved to: {path}.")
