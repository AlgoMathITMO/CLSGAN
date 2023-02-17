import pandas as pd
import numpy as np
import torch
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import torch.nn as nn
from torch.nn.utils import weight_norm
import sys
# sys.path.insert(1, '/Synth time series Project/tools/')
from tools.data_preprocessing import real_data_loading, split_ts
from tqdm import tqdm

def create_loader(ts, seq_len=2**7, batch_size=2**5):
    new_time_series = np.array(real_data_loading(ts, seq_len))
    loader = DataLoader(new_time_series, batch_size=batch_size, shuffle=False)
    return loader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def compute_gp(netD, real_data, fake_data):
        batch_size = real_data.size(0)
        # Sample Epsilon from uniform distribution
        eps = torch.rand(batch_size, 1, 1)
        eps = eps.expand_as(real_data)

        # Interpolation between real data and fake data.
        interpolation = eps * real_data + (1 - eps) * fake_data

        # get logits for interpolated images
        interp_logits = netD(interpolation)
        grad_outputs = torch.ones_like(interp_logits)

        # Compute Gradients
        gradients = torch.autograd.grad(
            outputs=interp_logits,
            inputs=interpolation,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
        )[0]

        # Compute and return Gradient Norm
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, 1)
        return torch.mean((grad_norm - 1) ** 2)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_hidden, n_outputs, kernel_size, dilation, padding):
        super(TemporalBlock, self).__init__()
        # self.add_channel = 10

        self.conv1 = nn.Conv1d(n_inputs, n_hidden, kernel_size, stride=1, dilation=dilation, padding='same')
        self.batchnorm1 = nn.BatchNorm1d(n_hidden)
        self.relu1 = nn.PReLU()
        self.chomp1 = Chomp1d(padding)
        self.dropout1 = nn.Dropout(0.05)

        self.conv2 = nn.Conv1d(n_hidden, n_outputs, kernel_size, stride=1, dilation=dilation, padding='same')
        self.batchnorm2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.PReLU()
        self.chomp2 = Chomp1d(padding)
        self.dropout2 = nn.Dropout(0.05)

        # self.convh = nn.Conv1d(n_hidden, n_hidden+self.add_channel, kernel_size, stride=1, dilation=dilation, padding='same')
        # self.reluh = nn.ReLU()
        # self.batchnorm = nn.BatchNorm1d(n_hidden+self.add_channel)

        # self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=1)
        # self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=1)
        # self.up1 = nn.ConvTranspose1d(n_hidden, n_hidden, kernel_size=2, stride=1, padding='same')
        # self.up2 = nn.ConvTranspose1d(n_hidden, n_hidden, kernel_size=2)

        # self.relu_out = nn.LeakyReLU(0.01)
        #
        # if padding == 0:
        #     self.net = nn.Sequential(self.conv1, self.relu1, \
        #     self.dropout1, self.conv2, self.relu2, self.dropout2)
            # self.net = nn.Sequential(self.conv1, self.relu1,\
             # self.maxpool1, self.convh, self.reluh, self.up1,  self.conv2, self.relu2, self.dropout2)
        # else:
        self.net = nn.Sequential(self.conv1, self.relu1, \
            self.dropout1, self.conv2, self.relu2, self.dropout2)
             # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,\
              # self.maxpool1, self.convh, self.reluh, self.up1,  self.conv2, self.chomp2, self.relu2, self.dropout2)

        # self.net = nn.Sequential(self.conv1, self.relu1, self.dropout,\
        #                          self.conv2, self.relu2, self.dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # torch.nn.init.xavier_normal_(self.conv1.weight)
        # torch.nn.init.xavier_normal_(self.conv2.weight)
        # self.convh.weight.data.normal_(0, 0.015)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
            # torch.nn.init.xavier_normal_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return out + res
        # return out, self.relu_out(out + res) #use for skip-connections on TCN

class TCN(nn.Module):
    def __init__(self, input_size, output_size, n_hidden=80):
        super(TCN, self).__init__()
        layers = []
        for i in range(7):
            num_inputs = input_size if i == 0 else n_hidden
            kernel_size = 2 if i > 0 else 1
            dilation = 2 * dilation if i > 1 else 1
            if i==0:
                padding = 0
            elif i==1:
                padding = 1
            else:
                padding = 2*padding
            layers += [TemporalBlock(num_inputs, n_hidden, n_hidden, kernel_size, dilation, padding)]

        self.conv = nn.Conv1d(n_hidden, output_size, 1)
        self.net = nn.Sequential(*layers)
        # self.net = nn.ModuleList([*layers]) #use for skip-connections on TCN
        self.init_weights()

    def init_weights(self):
        self.conv.weight.data.normal_(0, 0.01)
        # self.conv.weight.data.uniform_(-0.05, 0.05)
        # torch.nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        y1 = self.net(x.transpose(1, 2))
        return self.conv(y1).transpose(1, 2)
        # x = x.transpose(1, 2)
        # skips = []
        # for l in self.net:
        #     skip, x = l(x)
        #     skips.append(skip)
        # return self.conv(x + sum(skips)).transpose(1, 2)

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.net = TCN(input_size, output_size)
        self.fc = nn.Sequential(
                                #nn.Flatten()
                                # nn.Linear(output_size * seq_len, seq_len),
                                # nn.Linear(seq_len, 1),
                                nn.Tanh())
        # self.tanh = nn.Tanh()

    def forward(self, x):
        return self.fc(self.net(x))

class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, seq_len):
        super(Discriminator, self).__init__()
        self.net = TCN(input_size, output_size)
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(output_size * seq_len, 1),
                                nn.Sigmoid())

    def forward(self, x):
        return self.fc(self.net(x))

def train(ts, epochs = 50, seq_len=2**7-1, batch_size=2**5, n_fclayers=3, feature_dim=1, z_dim=3):
    loader = create_loader(ts, seq_len, batch_size)
    clip_value = 0.01
    # generator = Generator(z_dim, feature_dim, n_fclayers).to(device)
    # discriminator = Discriminator(feature_dim, n_fclayers).to(device)

    generator = Generator(z_dim, feature_dim).to(device)
    discriminator = Discriminator(feature_dim, 1, seq_len).to(device)

    optimizer_G = optim.RMSprop(generator.parameters(), lr=0.002)
    optimizer_D = optim.RMSprop(discriminator.parameters(), lr=0.002)

    scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[3, epochs-2], gamma=0.1)
    scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[3, epochs-2], gamma=0.1)

    tqdm_ticker = tqdm(range(epochs))
    for epoch in tqdm_ticker:
        for batch_idx, X in enumerate(loader):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # loss = torch.nn.BCELoss()
            # loss_mse = torch.nn.MSELoss()

            # Loss for real
            validity_real = discriminator(X.float().to(device))
            # d_real_loss = loss(validity_real, torch.ones_like(validity_real)).to(device)

            #Loss for fake
            # z = torch.FloatTensor(np.random.normal(0, 1, size=(batch_size, seq_len, z_dim))).to(device)
            batch_size, seq_len = X.shape[0], X.shape[1]
            z = torch.randn(batch_size, seq_len, z_dim, device=device)
            gen_res = generator(z)

            validity_fake = discriminator(gen_res)
            # d_fake_loss = loss(validity_fake, torch.zeros_like(validity_fake)).to(device)

            # Total discriminator loss
            # d_loss = (d_real_loss + d_fake_loss) / 2
            d_loss = (-torch.mean(validity_real) + torch.mean(validity_fake)).to(device)
            # d_loss = (-torch.mean(validity_real) + torch.mean(validity_fake)).to(device)+ \
                            # 0.5*compute_gp(discriminator, X.float(), gen_res)

            optimizer_D.zero_grad()
            # d_loss.backward(retain_graph=True)
            d_loss.backward()
            optimizer_D.step()

            for p in discriminator.parameters():
                p.data.clamp_(-clip_value, clip_value)

            # -----------------
            #  Train Generator
            # -----------------
            if batch_idx%5 == 0:
                validity = discriminator(generator(z))
                # g_loss = loss(validity, torch.ones_like(validity)).to(device)
                # g_loss_mse = loss_mse(X.float(), gen_res)
                g_loss = -torch.mean(validity).to(device)

                optimizer_G.zero_grad()
                # g_loss.backward(retain_graph=True)
                # g_loss_mse.backward(retain_graph=True)
                g_loss.backward()
                optimizer_G.step()
        scheduler_D.step()
        scheduler_G.step()

        # if epoch%10==0:
            # || MSE: {g_loss_mse:.{4}f}
        tqdm_ticker.set_description(f'Discriminator Loss: {d_loss:.{7}f} || Generator_loss: {g_loss:.{4}f}')
            # print(f'epoch {epoch}: Discriminator_loss: {d_loss:.{7}f} || Generator_loss: {g_loss:.{4}f}')
    return generator
