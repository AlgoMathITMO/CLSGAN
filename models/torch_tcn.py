import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import numpy as np

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()



class TemporalBlock(nn.Module):
    """Creates a temporal block.
    Args:
        n_inputs (int): number of inputs.
        n_outputs (int): size of fully connected layers.
        kernel_size (int): kernel size along temporal axis of convolution layers within the temporal block.
        dilation (int): dilation of convolution layers along temporal axis within the temporal block.
        padding (int): padding
        dropout (float): dropout rate
    Returns:
        tuple of output layers
    """
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.1):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(dropout)

        # self.convh = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=1, dilation=dilation, padding='same'))
        # self.reluh = nn.ReLU()
        # self.batchnorm1 = nn.BatchNorm1d(n_outputs)
        # self.batchnorm2 = nn.BatchNorm1d(n_outputs)

        # self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        # self.up1 = nn.ConvTranspose1d(n_hidden, n_hidden, kernel_size=2, stride=1, padding='same')
        # self.up2 = nn.ConvTranspose1d(n_hidden, n_hidden, kernel_size=2)

# self.convh, self.reluh, self.batchnorm,
        # if padding == 0:
        #     self.net = nn.Sequential(self.conv1, self.relu1, self.batchnorm1, self.dropout1,\
        #     self.conv2, self.relu2, self.batchnorm2, self.dropout2)
        # else:
        #     self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.batchnorm1, self.dropout1,\
        #     self.conv2, self.chomp2, self.relu2, self.batchnorm2, self.dropout2)
        if padding == 0:
            self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,\
            self.conv2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,\
            self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.LeakyReLU(0.2)
        # self.init_weights()

    def init_weights(self):
        # self.conv1.weight.data.normal_(0, 0.5)
        # self.conv2.weight.data.normal_(0, 0.5)
        torch.nn.init.xavier_normal_(self.conv1.weight)
        torch.nn.init.xavier_normal_(self.conv2.weight)
        if self.downsample is not None:
            # self.downsample.weight.data.normal_(0, 0.5)
            torch.nn.init.xavier_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        # return out, self.relu(out + res)
        return out, self.relu(out + res)



class Generator(nn.Module):
    def __init__(self,  z_dim, batch_size):
        super(Generator, self).__init__()
        self.hidden = 80
        self.tcn = nn.ModuleList([TemporalBlock(z_dim, self.hidden, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(self.hidden, self.hidden, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(self.hidden, 1, kernel_size=1, stride=1, dilation=1)
        self.layernorm = nn.BatchNorm1d(batch_size)
        # self.init_weights()

    def init_weights(self):
        # self.last.weight.data.normal_(0, 0.5)
        torch.nn.init.xavier_normal_(self.last.weight)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            x = self.layernorm(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return x


class Discriminator(nn.Module):
    def __init__(self, seq_len, batch_size):
        super(Discriminator, self).__init__()
        self.hidden = 80
        self.tcn = nn.ModuleList([TemporalBlock(1, self.hidden, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(self.hidden, self.hidden, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(self.hidden, 1, kernel_size=1, dilation=1)
        self.to_prob = nn.Sequential(nn.Linear(seq_len, 1),
        #                              nn.Sigmoid())
                                     # nn.Linear(2**6, 1),
                                     nn.Sigmoid())
        self.layernorm = nn.BatchNorm1d(batch_size)

        # self.init_weights()

    def init_weights(self):
        # self.last.weight.data.normal_(0, 0.5)
        torch.nn.init.xavier_normal_(self.last.weight)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            x = self.layernorm(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.to_prob(x).squeeze()

class Generator_normed(nn.Module):
    def __init__(self,  z_dim, batch_size):
        super(Generator_normed, self).__init__()
        self.hidden = 80
        self.tcn = nn.ModuleList([TemporalBlock(z_dim, self.hidden, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(self.hidden, self.hidden, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(self.hidden, 1, kernel_size=1, stride=1, dilation=1)
        self.tanh = nn.Tanh()
        self.layernorm = nn.BatchNorm1d(batch_size)
        # self.init_weights()

    def init_weights(self):
        # self.last.weight.data.normal_(0, 0.5)
        torch.nn.init.xavier_normal_(self.last.weight)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            x = self.layernorm(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.tanh(x)

class Supervisor(nn.Module):
    def __init__(self, z_dim, batch_size):
        super(Supervisor, self).__init__()
        self.hidden = 80
        self.tcn = nn.ModuleList([TemporalBlock(z_dim, self.hidden, kernel_size=1, stride=1, dilation=1, padding=0),
                                 *[TemporalBlock(self.hidden, self.hidden, kernel_size=2, stride=1, dilation=i, padding=i) for i in [1, 2, 4, 8, 16, 32]]])
        self.last = nn.Conv1d(self.hidden, 1, kernel_size=1, stride=1, dilation=1)
        self.layernorm = nn.BatchNorm1d(batch_size)
        self.tanh = nn.Tanh()
        # self.fc = nn.Sequential(nn.Linear(seq_len, 2**8),
        #                         nn.ReLU(),
        #                         nn.Linear(2**8, seq_len))

        # self.init_weights()

    def init_weights(self):
        # self.last.weight.data.normal_(0, 0.5)
        torch.nn.init.xavier_normal_(self.last.weight)

    def forward(self, x):
        skip_layers = []
        for layer in self.tcn:
            skip, x = layer(x)
            x = self.layernorm(x)
            skip_layers.append(skip)
        x = self.last(x + sum(skip_layers))
        return self.tanh(x)
        # out = self.fc(x)
        # return x
