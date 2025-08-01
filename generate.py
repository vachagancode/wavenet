import torch
import torchaudio

from config import get_config
from wavenet import create_wavenet

def generate_audio():
    pass


if __name__ == "__main__":

    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_wavenet(config[0], device)
    model.load_state_dict(torch.load("./m256k5v1_final.pth", map_location=device)["model_state_dict"])

    path = "./test.wav"

    # Load the audio
    audio, sr = torchaudio.load(path)

    if audio.shape[0] == 2:
        audio = audio.mean(0, keepdim=True)

    audio = audio.unsqueeze(0)
    model.generate(audio, initial_steps=torch.ones(1, 1, 5000))
