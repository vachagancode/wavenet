import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchaudio

from modules import DilatedCausalConvolutionalLayer, DilatedConvolutionalLayerStack, PostDilationLayer

from dataset import create_dataloaders
from tqdm import tqdm
from config import get_config
from decoding import greedy_decode


class WaveNet(nn.Module):
    def __init__(self, dconv_input : int, dconv_output : int, dconv_hidden : int, pconv_input : int, pconv_hidden : int, pconv_output : int, kernel_size : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.embedding_layer = EmbeddingLayer(vocab_size=vocab_size, dim=embd_dim)
        self.dilated_conv_layer = DilatedConvolutionalLayerStack(in_channels=dconv_input, hidden_channels=dconv_hidden, out_channels=dconv_output, kernel_size=kernel_size)
        self.post_dilated_conv_layer = PostDilationLayer(in_channels=pconv_input, hidden_channels=pconv_hidden, out_channels=pconv_output)

        self.mu_decoding = torchaudio.transforms.MuLawDecoding(quantization_channels=pconv_output)

    def forward(self, x):
        # x = self.embedding_layer(x)
        # print(x.shape)
        x = self.dilated_conv_layer(x)
        x = self.post_dilated_conv_layer(x)

        return x

    def generate(self,
                 initial_steps=None,
                 num_steps=44100,
                 chunk_size=1000):

        predictions = []
        if initial_steps == None:
            input_steps = torch.zeros((1, 1, chunk_size))
        else:
            if initial_steps.shape[-1] > chunk_size:
                input_steps = initial_steps[:, :, initial_steps.shape[-1]-chunk_size:]
            elif initial_steps.shape[-1] < chunk_size:
                zero_pad = torch.zeros((1, 1, chunk_size - initial_steps.shape[-1]))
                input_steps = torch.cat([zero_pad, initial_steps], dim=-1)
            else:
                input_steps = initial_steps.unsqueeze(0)

        with torch.inference_mode():
            s = time.time()
            for _ in tqdm(range(num_steps)):
                # Do the forward pass
                logits = self.forward(input_steps)

                preds = F.softmax(logits, dim=-1)
                arg_preds = greedy_decode(preds)
                predictions.append(arg_preds[:, :, -1])
                # print(predictions)

                input_steps = torch.cat([input_steps[:, :, 1:], arg_preds[:, :, -1].unsqueeze(0)], dim=-1)

            final_predictions = torch.stack(predictions, dim=-1)
            decoded = self.mu_decoding(final_predictions)

            e = time.time()

            print(f"[INFO] Generation done in {(e-s):.3f} seconds.")

            return decoded



def create_wavenet(config, device=torch.device("cpu")):

    return WaveNet(
        dconv_input=config["dconv_input"],
        dconv_output=config["dconv_output"],
        dconv_hidden=config["dconv_output"],
        pconv_input=config["pconv_input"],
        pconv_hidden=config["pconv_hidden"],
        pconv_output=config["pconv_output"],
        kernel_size=config["kernel_size"]
    ).to(device)
