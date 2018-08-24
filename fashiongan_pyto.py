import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs('images', exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')

opt = parser.parse_known_args()
print(opt)

cuda = True if torch.cuda.is_available() else False

def weights_init_normal(m):
    classname= m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

config = dofile('./config_sr1.lua')

nt_input = config.nt_input
nt = config.nt

nc = config.n_map_all
ncondition = config.n_condition

nz = config.nz


ndf = 64
ngf = 64
inplace = true

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4

        self.conv_blocks_1 = nn.Sequential(
            nn.Conv2d(nt_input, nt, 1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.deconv_blocks_1 = nn.Sequential(
            nn.ConvTranspose2d(nz+nt, ngf*16, 4, stride=1, padding=0),
            nn.BatchNorm2d(ngf*16),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*16, ngf*8, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(inplace=True)
        )

        self.conv_blocks_2 = nn.Sequential(
            nn.Conv2d(ncondition, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.conv_deconv_blocks = nn.Sequential(
            nn.ConvTranspose2d(ngf*10, ngf*4, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.ReLU(inplace=True),

            nn.Conv2d(ngf*4, ngf*8, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf*8, ngf*8, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ngf*8, ngf*4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf*2, ngf, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, nc, 4, stride=2, padding=1)
        )

    def forward(self, {input_data, input_encode, input_condition}):

        h1 = self.conv_blocks_1(input_encode)
        input_data_encode = torch.cat((input_data, h1),1)
        g_extra = self.deconv_blocks_1(input_data_encode)
        f_extra = self.conv_blocks_2(input_condition)
        gf_extra = torch.cat((g_extra, f_extra),1)
        out = self.conv_deconv_blocks(gf_extra)

        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block_3(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 3, 1, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        def discriminator_block_4(in_filters, out_filters, bn=True):
            block = [   nn.Conv2d(in_filters, out_filters, 4, 2, 1)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.discri_modul_block_1 = nn.Sequential(
            *discriminator_block_4(nc, ndf, bn=False),
            *discriminator_block_4(ndf, ndf*2),
            *discriminator_block_4(ndf*2, ndf*4),
            *discriminator_block_3(ndf*4, ndf*8),
            *discriminator_block_3(ndf*8, ndf*8),
            *discriminator_block_3(ndf*8, ndf*4),
            *discriminator_block_4(ndf*4, ndf*8),
        )

        self.discri_modul_block_2 = nn.Sequential(
        	*discriminator_block_3(ncondition, ndf, bn=False),
        	*discriminator_block_3(ndf, ndf*2)
        )

        self.discri_modul_block_3 = nn.Sequential(
            *discriminator_block_4(ndf*10, ndf*8)
        )

        self.discri_modul_block_4 = nn.Sequential(
            nn.Conv2d(nt_input, nt, 1),
            nn.BatchNorm2d(nt),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.discri_modul_block_5 = nn.Sequential(
            nn.Conv2d(ndf*8+nt, ndf*8, 1),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True)

            nn.Conv2d(ndf*8+nt, 1, 4),
            nn.Sigmoid()
        )


    def forward(self, {output_data, output_encode, output_condition}):
    	output_data_softmax = nn.Softmax2d(output_data)
    	d4 = self.discri_modul_block_1(output_data_softmax)
    	c2 = self.discri_modul_block_2(output_condition)
    	d4_c2 = torch.cat((d4, c2),1)
    	d_extra = self.discri_modul_block_3(d4_c2)
    	b1 = self.discri_modul_block_4(output_encode)
    	d_extra_b1 = torch.cat((d_extra, b1),1)
    	d5 = self.discri_modul_block_5(d_extra_b1)

        return d5


# Loss function
adversarial_loss = torch.nn.BCELoss()

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
generator.apply(weights_init_normal)
discriminator.apply(weights_init_normal)

# Configure data loader
os.makedirs('../../data/mnist', exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST('../../data/mnist', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.Resize(opt.img_size),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ])),
    batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise as generator input
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

        # Generate a batch of images
        gen_imgs = generator(z)

        # Loss measures generator's ability to fool the discriminator
        g_loss = adversarial_loss(discriminator(gen_imgs), valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(real_imgs), valid)
        fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item()))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:25], 'images/%d.png' % batches_done, nrow=5, normalize=True)



