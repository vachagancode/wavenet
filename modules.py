import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

# First layer, embedding layer
# class EmbeddingLayer(nn.Module):
#     def __init__(self, vocab_size : int,  dim : int = 128, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.vocab_size = vocab_size
#         self.dim = dim
#         self.embedding = nn.Embedding(num_embeddings=self.vocab_size,  embedding_dim=self.dim)

#     def forward(self, x):
#         return self.embedding(x)

class DilatedCausalConvolutionalLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, dilation_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.causal_padding = (kernel_size - 1) * dilation_rate
        self.dilation_rate = dilation_rate

        self.conv_layer_1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.hidden_channels, kernel_size=self.kernel_size, padding=0, dilation=self.dilation_rate)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.conv_1x1 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.out_channels, kernel_size=1)
        self.skip_conv_1x1 = nn.Conv1d(in_channels=self.hidden_channels, out_channels=self.out_channels, kernel_size=1)
        self.residual_connections = lambda x, y: x + y

        self.dropout = nn.Dropout(0.35)

    def forward(self, x):
        if self.dilation_rate == 1:

            # x = x.permute(0, 2, 1)
            pass

        # apply padding manually
        x_padded = F.pad(x, (self.causal_padding, 0))

        x_conv = self.conv_layer_1(x_padded)
        # pass through activations
        x_act = self.tanh(x_conv) * self.sigmoid(x_conv)

        x_act_dropout = self.dropout(x_act)

        x_1x1 = self.conv_1x1(x_act_dropout)
        skip_1x1 = self.skip_conv_1x1(x_act_dropout)

        x = self.residual_connections(x_1x1, x)
        return x, skip_1x1

class DilatedConvolutionalLayerStack(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, num_dilated, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dilations = [2**i for i in range(num_dilated)]
        self.stack = nn.ModuleList()
        for ix, dilation in enumerate(self.dilations):
            if ix == 0:
                layer = DilatedCausalConvolutionalLayer(in_channels=1, out_channels=out_channels, hidden_channels=hidden_channels, kernel_size=kernel_size, dilation_rate=dilation)
            else:
                layer = DilatedCausalConvolutionalLayer(in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels, kernel_size=kernel_size, dilation_rate=dilation)
            self.stack.append(layer)

    def forward(self, x):
        global_skip_connections = []
        for layer in self.stack:
            # pass through the layers
            output, skip_connection = layer(x)

            global_skip_connections.append(skip_connection)
            x = output

        # sum the global skip connections
        global_skip_connections_sum = sum(global_skip_connections)

        return global_skip_connections_sum

class PostDilationLayer(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relu = nn.ReLU()
        self.conv_1x1_1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=1)
        self.conv_1x1_2 = nn.Conv1d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        self.mu_law_transform = torchaudio.transforms.MuLawEncoding()

    def forward(self, x):
        x = self.relu(x)
        x = self.conv_1x1_1(x)
        x = self.relu(x)
        x = self.conv_1x1_2(x)
        x = x.permute(0, 2, 1)
        # x = self.softmax(x)

        return x
