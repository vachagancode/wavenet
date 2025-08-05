import time
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from modules import DilatedCausalConvolutionalLayer, DilatedConvolutionalLayerStack, PostDilationLayer

from dataset import create_dataloaders
from tqdm import tqdm
from config import get_config


class WaveNet(nn.Module):
    def __init__(self, dconv_input : int, dconv_output : int, dconv_hidden : int, pconv_input : int, pconv_hidden : int, pconv_output : int, num_dilated : int, kernel_size : int, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.quantization_channels = pconv_output
        # self.embedding_layer = EmbeddingLayer(vocab_size=vocab_size, dim=embd_dim)
        self.dilated_conv_layer = DilatedConvolutionalLayerStack(in_channels=dconv_input, in_conditional=80, hidden_channels=dconv_hidden, out_channels=dconv_output, kernel_size=kernel_size, num_dilated=num_dilated)
        self.post_dilated_conv_layer = PostDilationLayer(in_channels=pconv_input, hidden_channels=pconv_hidden, out_channels=pconv_output)

        self.mu_decoding = torchaudio.transforms.MuLawDecoding(quantization_channels=pconv_output)

        # self.dilation_queues = [deque(torch.zeros(), maxlen=kernel_size*d) for d in [1, 2, 4, 8, 16, 32]]

    def forward(self, x, mel_spectrogram):
        if mel_spectrogram != None:
            mel_spectrogram = self.upsample_condition(mel_spectrogram, x.shape[-1])
        x = self.dilated_conv_layer(x, condition=mel_spectrogram)
        x = self.post_dilated_conv_layer(x)

        return x
    
    def upsample_condition(self, condition, target_length):
        condition = condition
        condition = F.interpolate(condition, size=target_length, mode="nearest")
        return condition
    
    def generate(self, mel_spectrogram, hop_length=256):

        self.eval()
        with torch.inference_mode():
            output = torch.zeros(1, 1, 1)

            output_length = 48000
            # Upsample the mel_spectrogram
            mel_spectrogram = self.upsample_condition(mel_spectrogram, output_length)

            for t in tqdm(range(output_length)):
                # Do the forward pass
                logits = self.forward(output[:, :, -1:], mel_spectrogram=mel_spectrogram[:, :, t:t+1])
                
                # Now sampling
                prediction = F.softmax(logits, dim=-1)
                arg_prediction = torch.multinomial(prediction, num_samples=1)
                output = torch.cat([output, arg_prediction], dim=-1)
            
            output = (output / (self.quantization_channels - 1)) * 2 - 1.0 

            output = self.mu_decoding(output)

            return output.squueze(1)



def create_wavenet(config, device=torch.device("cpu")):
    return WaveNet(
        dconv_input=config["dconv_input"],
        dconv_output=config["dconv_output"],
        dconv_hidden=config["dconv_output"],
        pconv_input=config["pconv_input"],
        pconv_hidden=config["pconv_hidden"],
        pconv_output=config["pconv_output"],
        kernel_size=config["kernel_size"],
        num_dilated=6
    ).to(device)

if __name__ == "__main__":
    audio, sr = torchaudio.load("./song.mp3")
    if audio.ndim == 2:
        audio = audio.mean(0, keepdim=True)
    print(f"Sample Rate: {sr}")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    )
    amp_to_db = torchaudio.transforms.AmplitudeToDB()

    mel_spectrogram = amp_to_db(mel_transform(audio))

    # print(mel_spectrogram.shape)


    wavenet = create_wavenet(get_config()[0])

    # print(wavenet(audio, mel_spectrogram).shape)

    output = wavenet.generate(mel_spectrogram.unsqueeze(0))
    print(output.shape)