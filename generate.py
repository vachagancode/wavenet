import os
import torch
import torchaudio

from utils import save_generated_audio
from config import get_config
from wavenet import create_wavenet

def generate_audio(model_name, model_dir, input_file=None):
    print(1)
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(2)
    for ix, model_config in enumerate(config):
        if model_name.split("_")[0] in model_config.values():
            model_index = ix
            model_path = os.path.join(model_dir, f"{model_name}_final.pth")
            break

    print(3)
    model = create_wavenet(config[model_index], device)
    model.load_state_dict(torch.load(model_path, map_location=device)["model_state_dict"])

    print(4)
    if input_file != None:
        audio, sr = torchaudio.load(input_file)
        if audio.shape[0] == 2:
            audio = audio.mean(0, keepdim=True)
    else:
        audio = None
    audio = audio.unsqueeze(0)
    print(5)

    output = model.generate(initial_steps=audio)

    return output, sr

if __name__ == "__main__":

    generated, sample_rate = generate_audio(model_name="m256k5v1", model_dir="p./", input_file="./data/65.wav")
    print(generated)

    save_generated_audio(audio=generated, path="./generated.wav", sr=sample_rate)

