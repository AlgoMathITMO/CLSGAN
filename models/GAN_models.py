import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from models.utils.gaussianize import *
from models.torch_tcn import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

generator_path = f'./trained/'
file_name = 'gen'

class Loader32(Dataset):

    def __init__(self, data, length):
        assert len(data) >= length
        self.data = data
        self.length = length

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx:idx+self.length]).reshape(-1, self.length).to(torch.float32)

    def __len__(self):
        return max(len(self.data)-self.length, 0)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)


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

#df -- pd.Series
def quantgan_preprocessing(df):
    returns = df.shift(1)/df - 1
    log_returns = np.log(df/df.shift(1))[1:].to_numpy().reshape(-1, 1)
    standardScaler1 = StandardScaler()
    standardScaler2 = StandardScaler()
    gaussianize = Gaussianize()
    log_returns_preprocessed = standardScaler2.fit_transform(gaussianize.fit_transform(standardScaler1.fit_transform(log_returns)))

    return log_returns_preprocessed, log_returns, standardScaler1, standardScaler2, gaussianize

def quantgan_fit(log_returns_preprocessed, num_epochs=30, seq_len=127, batch_size=80, z_dim=3, train=True):
    clip= 0.01
    lr = 0.0009
    train = train
    # receptive_field_size = 127  # p. 17
    # data_size = log_returns.shape[0]
    generator = Generator(z_dim, batch_size).to(device)
    discriminator = Discriminator(seq_len, batch_size).to(device)

    dataset = Loader32(log_returns_preprocessed, seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    if train:
        # optimizer_D = optim.RMSprop(discriminator.parameters(), lr=lr)
        # optimizer_G = optim.RMSprop(generator.parameters(), lr=lr)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
        optimizer_G = optim.Adam(generator.parameters(), lr=lr)

        # scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[3, 7], gamma=0.1)
        # scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[3, 7], gamma=0.1)
        scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.96)
        scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.96)

        scores_gen = []
        scores_disc = []
        t = tqdm(range(num_epochs))
        for epoch in t:
            for idx, data in enumerate(dataloader, 0):
                # loss_mse = nn.MSELoss()
                # loss = torch.nn.BCELoss()

                real = data.to(device)
                batch_size, seq_len = real.size(0), real.size(2)
                noise = torch.randn(batch_size, z_dim, seq_len, device=device)
                fake = generator(noise).detach()

                # disc_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake)) +\
                            # 10*compute_gp(discriminator, real, generator(noise))

                disc_loss = torch.mean(torch.log(discriminator(real.detach())) + torch.log(1 - discriminator(fake)))

                optimizer_D.zero_grad()
                disc_loss.backward()
                optimizer_D.step()

                for dp in discriminator.parameters():
                    dp.data.clamp_(-clip, clip)

                if idx % 5 == 0:
                    gen_loss = torch.mean(torch.log(discriminator(generator(noise))))

                    # gen_loss = -torch.mean(discriminator(generator(noise)))

                    optimizer_G.zero_grad()
                    gen_loss.backward()
                    optimizer_G.step()

            scheduler_D.step()
            scheduler_G.step()
            # t.set_description('Discriminator Loss: %.8f Generator Loss: %.8f' % (disc_loss.item(), gen_loss.item()))
            t.set_description('Discriminator Loss: %.7f || Generator Loss: %.5f' % (disc_loss.item(), gen_loss.item()))
            scores_gen.append(gen_loss.item())
            scores_disc.append(disc_loss.item())
    else:
        pass
    return generator, np.array([scores_gen, scores_disc])

def clsgan_fit(log_returns_preprocessed, num_epochs=30, seq_len=127, batch_size=80, z_dim=3, train=True):
    clip= 0.01
    lr = 0.0009
    train = train
    # receptive_field_size = 127  # p. 17
    # data_size = log_returns.shape[0]
    generator = Generator_normed(z_dim, batch_size).to(device)
    supervisor = Supervisor(1, batch_size).to(device)
    discriminator = Discriminator(seq_len, batch_size).to(device)
    discriminator2 = Discriminator(seq_len, batch_size).to(device)

    dataset = Loader32(log_returns_preprocessed, seq_len)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    if train:
        optimizer_D = optim.Adam(discriminator.parameters(), lr=lr)
        optimizer_D2 = optim.Adam(discriminator2.parameters(), lr=lr)
        optimizer_G = optim.Adam(generator.parameters(), lr=lr)
        optimizer_S = optim.Adam(supervisor.parameters(), lr=0.001)

        # scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[5], gamma=0.1)
        # scheduler_D = optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[5], gamma=0.1)
        # scheduler_D2 = optim.lr_scheduler.MultiStepLR(optimizer_D2, milestones=[5], gamma=0.1)
        # scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, milestones=[5], gamma=0.1)
        scheduler_G = optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.96)
        scheduler_D = optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.96)
        scheduler_D2 = optim.lr_scheduler.ExponentialLR(optimizer_D2, gamma=0.96)
        scheduler_S = optim.lr_scheduler.ExponentialLR(optimizer_S, gamma=0.96)

        scores_gen = []
        scores_disc1 = []
        scores_super = []
        scores_disc2 = []
        # dataset = Loader32(log_returns_preprocessed, 127)
        # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        t = tqdm(range(num_epochs))
        for epoch in t:
            for idx, data in enumerate(dataloader, 0):
                loss_mse = nn.MSELoss()

                real = data.to(device)
                batch_size, seq_len = real.size(0), real.size(2)
                noise = torch.randn(batch_size, z_dim, seq_len, device=device)
                fake = generator(noise).detach()
                # fake = supervisor(generator(noise)).detach()

                real_grad = torch.autograd.Variable(real, requires_grad=True)

                # disc_loss = -torch.mean(discriminator(real)) + torch.mean(discriminator(fake))
                # disc2_loss = -torch.mean(discriminator2(real)) + torch.mean(discriminator2(supervisor(fake)))
                disc_loss = torch.mean(torch.log(discriminator(real.detach())) + torch.log(1 - discriminator(fake)))
                disc2_loss = torch.mean(torch.log(discriminator2(real.detach())) + torch.log(1 - discriminator2(supervisor(fake))))
                # print(supervisor(fake))
                # print(discriminator2(supervisor(fake)))
                # print(torch.log(1 - discriminator2(supervisor(fake))+ 1e-8))
                # print(torch.log(discriminator2(real.detach())))
                # print(disc2_loss)

                optimizer_D.zero_grad()
                disc_loss.backward()
                optimizer_D.step()

                optimizer_D2.zero_grad()
                disc2_loss.backward()
                optimizer_D2.step()

                for dp in discriminator.parameters():
                    dp.data.clamp_(-clip, clip)

                for dp in discriminator2.parameters():
                    dp.data.clamp_(-clip, clip)

                if idx % 5 == 0:
                    # gen_loss1 = -torch.mean(discriminator(generator(noise)))
                    # gen_loss2 = -torch.mean(discriminator2(supervisor(generator(noise))))
                    gen_loss1 = torch.mean(torch.log(discriminator(generator(noise))))
                    gen_loss2 = torch.mean(torch.log(discriminator2(supervisor(generator(noise)))))

                    gen_loss = 0.8 * gen_loss1 + 0.2 * gen_loss2

                    optimizer_G.zero_grad()
                    gen_loss.backward()
                    optimizer_G.step()
                    #
                    # for gp in generator.parameters():
                    #     gp.data.clamp_(-0.01, 0.01)

                if idx % 5 == 0:
                    # S_fft = torch.FloatTensor(np.fft.fft(supervisor(generator(noise).detach()).detach().cpu().numpy()).real)
                    # R_fft = torch.FloatTensor(np.fft.fft(real.detach().cpu().numpy()).real)
                    # s_loss_real = loss_mse(torch.autograd.Variable(S_fft, requires_grad=True), torch.autograd.Variable(R_fft, requires_grad=True))
                    #
                    #
                    # S_fft_im = torch.FloatTensor(np.fft.fft(supervisor(generator(noise).detach()).detach().cpu().numpy()).imag)
                    # R_fft_im = torch.FloatTensor(np.fft.fft(real.detach().cpu().numpy()).imag)
                    # s_loss_imag = loss_mse(torch.autograd.Variable(S_fft_im, requires_grad=True), torch.autograd.Variable(R_fft_im, requires_grad=True))
                    #
                    # s_loss_freq = (s_loss_real + s_loss_imag) / 2

                    # s_loss = loss_mse(supervisor(generator(noise).detach()), real_grad) +\
                    s_loss = torch.mean(torch.log(discriminator2(supervisor(generator(noise))))) + \
                             3*loss_mse(supervisor(generator(noise).detach()), real_grad)


                    optimizer_S.zero_grad()
                    s_loss.backward()
                    optimizer_S.step()

                    # for sp in supervisor.parameters():
                    #     sp.data.clamp_(-0.01, 0.01)
                    # torch.nn.utils.clip_grad_norm_(supervisor.parameters(), 1, norm_type=2.0)

            scheduler_D.step()
            scheduler_D2.step()
            scheduler_G.step()
            scheduler_S.step()
            # t.set_description('Discriminator Loss: %.8f Generator Loss: %.8f' % (disc_loss.item(), gen_loss.item()))
            t.set_description('D1 Loss: %.7f || D2 Loss: %.7f || G Loss: %.5f || S Loss: %.5f' % (disc_loss.item(), disc2_loss.item(), gen_loss.item(), s_loss.item()))
            scores_gen.append(gen_loss.item())
            scores_disc1.append(disc_loss.item())
            scores_disc2.append(disc2_loss.item())
            scores_super.append(s_loss.item())
    else:
        pass
    return generator, supervisor, np.array([scores_gen, scores_disc1, scores_disc2, scores_super])

def quantGAN_generate_synth(generator, log_returns, ts, standardScaler1, standardScaler2, gaussianize, z_dim):
    # generator.eval()
    noise = torch.randn(10, z_dim, len(ts)).to(device)
    y = generator(noise).cpu().detach().squeeze()

    y = (y - y.mean(axis=0))/y.std(axis=0)
    y = standardScaler2.inverse_transform(y)
    y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
    y = standardScaler1.inverse_transform(y)

    # some basic filtering to redue the tendency of GAN to produce extreme returns
    y = y[(y.max(axis=1) <= 2 * log_returns.max()) & (y.min(axis=1) >= 2 * log_returns.min())]
    y -= y.mean()

    return y[0]


#df -- pd.Series
def clsgan_preprocessing(df):
    returns = df.shift(1)/df - 1
    log_returns = np.log(df/df.shift(1))[1:].to_numpy().reshape(-1, 1)
    standardScaler1 = StandardScaler()
    standardScaler2 = StandardScaler()
    gaussianize = Gaussianize()
    log_returns_preprocessed = standardScaler2.fit_transform(gaussianize.fit_transform(standardScaler1.fit_transform(log_returns)))

    # max_el = 1
    max_el = np.max(np.abs(log_returns_preprocessed))
    log_returns_preprocessed /= max_el

    return log_returns_preprocessed, log_returns, standardScaler1, standardScaler2, gaussianize, max_el

def generate_sample_CLSGAN(generator, supervisor, log_returns, ts, standardScaler1, standardScaler2, gaussianize, m, z_dim):
    noise = torch.randn(10, z_dim, len(ts)).to(device)
    y = supervisor(generator(noise)).cpu().detach().squeeze()
    y = y * m
    y = (y - y.mean(axis=0))/y.std(axis=0)

    y = standardScaler2.inverse_transform(y)
    y = np.array([gaussianize.inverse_transform(np.expand_dims(x, 1)) for x in y]).squeeze()
    y = standardScaler1.inverse_transform(y)


    # some basic filtering to redue the tendency of GAN to produce extreme returns
    y = y[(y.max(axis=1) <= 2 * log_returns.max()) & (y.min(axis=1) >= 2 * log_returns.min())]
    y -= y.mean()
    return y[0]
