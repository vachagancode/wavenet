import torch
import torchaudio
import torch.nn.functional as F

from pydub import AudioSegment
from tqdm import tqdm

from wavenet import create_wavenet
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

def greedy_decode(logits):
    predictions = torch.argmax(logits, dim=-1, keepdim=True)
    return predictions.permute(0, 2, 1)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=0.01,
        total_steps=len(dataloader) * (end_epoch - start_epoch),
        anneal_strategy='cos',
    )

    if optimizer_state_dict is not None and scheduler_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        scheduler.load_state_dict(scheduler_state_dict)

    return optimizer, scheduler

def calculate_accuracy(logits, targets):
    correct = 0
    total = 0

    # apply softmax to the logits
    preds = F.softmax(logits, dim=-1).argmax(dim=-1)
    total += preds.shape[-1]
    correct += (preds == targets).sum()

    accuracy = (correct.item() / total) * 100

    return accuracy

def create_summary_writer(model_name):
    writer_dir = f"./runs/{model_name}"
    writer = SummaryWriter(log_dir=writer_dir)
    print(f"[INFO] SummaryWriter created in {writer_dir}.")
    return writer

def generate_audio(input_path, model, device, mu_decoder, steps=88200):
    predicted = []
    audio, sr = torchaudio.load(input_path)
    audio = audio.to(device)
    print(f"Sample Rate: {sr}")

    if audio.shape[0] != 1:
        audio = audio.mean(0, keepdim=True)
    audio = audio.unsqueeze(0)
    initial_audio = audio.clone()
    with torch.inference_mode():
        for step in tqdm(range(steps)):
            # print(step)

            logits = model(audio)
            # Apply softmax
            preds = F.softmax(logits, dim=-1)

            decoded = greedy_decode(preds)

            mu_law_decoded = mu_decoder(decoded)

            prediction = mu_law_decoded[:, :, -1].unsqueeze(0)
            # print(prediction)
            predicted.append(prediction)

            audio = torch.cat([(audio[:, :, 1:]), prediction], dim=-1)
        predictions = torch.stack(predicted, dim=-1).squeeze(0)

        # Concatenate with the audio
        final = torch.cat([initial_audio, predictions], dim=-1).squeeze(0)

        torchaudio.save("./output.wav", final, sr, bits_per_sample=16)

if __name__ == "__main__":
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_wavenet(config[0], device=device)
    model_state_dict = torch.load("./m256k5v1_final.pth", map_location=device)["model_state_dict"]
    model.load_state_dict(model_state_dict)

    mu_decoder = torchaudio.transforms.MuLawDecoding(quantization_channels=256)
    generate_audio("./test.wav", model, mu_decoder)
