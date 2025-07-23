import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import DilatedCausalConvolutionalLayer, DilatedConvolutionalLayerStack, PostDilationLayer

from dataset import create_dataloaders
from tqdm import tqdm
from config import get_config

class WaveNet(nn.Module):
    def __init__(self, dconv_input : int, dconv_output : int, pconv_input : int, pconv_hidden : int, pconv_output : int, kernel_size : int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.embedding_layer = EmbeddingLayer(vocab_size=vocab_size, dim=embd_dim)
        self.dilated_conv_layer = DilatedConvolutionalLayerStack(in_channels=dconv_input, out_channels=dconv_output, kernel_size=kernel_size)
        self.post_dilated_conv_layer = PostDilationLayer(in_channels=pconv_input, hidden_channels=pconv_hidden, out_channels=pconv_output)

    def forward(self, x):
        # x = self.embedding_layer(x)
        # print(x.shape)
        x = self.dilated_conv_layer(x)
        x = self.post_dilated_conv_layer(x)

        return x

def create_wavenet(config, device=torch.device("cpu")):
    return WaveNet(
        dconv_input=config["dconv_input"],
        dconv_output=config["dconv_output"],
        pconv_input=config["pconv_input"],
        pconv_hidden=config["pconv_hidden"],
        pconv_output=config["pconv_output"],
        kernel_size=config["kernel_size"]
    ).to(device)

if __name__ == "__main__":
    config = get_config()
    wavenet = create_wavenet(config)
    _, valid_dataloader, _ = create_dataloaders()

    batchloader = tqdm(valid_dataloader)
    for batch in batchloader:
        src, tgt = batch
        src, tgt = src.float(), tgt.float()
        print(src.shape)
        logits = wavenet(src)
        print(logits.shape)
        break
