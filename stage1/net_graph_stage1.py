import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02 / 16)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


nt_input = 100
nt = 20
nz = 80
ndf = 64
ngf = 64
ncondition = 3
nc = 3


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_blocks_1 = nn.Sequential(
            nn.Conv2d(in_channels=nt_input, out_channels=nt, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.deconv_blocks_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=nz + nt, out_channels=ngf * 16, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(inplace=True),  # g1

            nn.ConvTranspose2d(in_channels=ngf * 16, out_channels=ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),  # g_extra

            nn.ConvTranspose2d(in_channels=ngf * 8, out_channels=ngf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),  # g2
        )

        self.conv_blocks_2 = nn.Sequential(
            nn.Conv2d(in_channels=ncondition, out_channels=ngf, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),  # f1

            nn.Conv2d(in_channels=ngf, out_channels=ngf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # f_extra

            nn.Conv2d(in_channels=ngf * 2, out_channels=ngf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.deconv_blocks_3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 12, ngf * 8, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
        )

        self.mid_blocks = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # mid1

            nn.Conv2d(ngf * 8, ngf * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # mid2

            nn.Conv2d(ngf * 8, ngf * 8, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # mid3

            nn.Conv2d(ngf * 8, ngf * 4, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # mid4

            nn.Conv2d(ngf * 4, ngf * 2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # mid5
        )

        self.deconv_blocks_4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, stride=2, padding=1),
            nn.BatchNorm2d(ngf * 1),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(ngf, nc, 4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, input_data, input_encode, input_condition):
        h1 = self.conv_blocks_1(input_encode)
        input_data_encode = torch.cat((input_data, h1), 1)
        g_2 = self.deconv_blocks_1(input_data_encode)
        f_2 = self.conv_blocks_2(input_condition)
        gf_extra = torch.cat((g_2, f_2), 1)
        g3 = self.deconv_blocks_3(gf_extra)
        mid5 = self.mid_blocks(g3)
        out = self.deconv_blocks_4(mid5)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_blocks_1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # d1

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # d2

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # d3
        )

        self.conv_blocks_2 = nn.Sequential(
            nn.Conv2d(ncondition, ndf, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),  # c1

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),  # c2

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),  # c3
        )

        self.conv_blocks_3 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # m1

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # m2

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # m3

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # m4

            nn.Conv2d(ndf * 8, ndf * 8, 3, 1, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # m5
        )

        self.conv_blocks_4 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),  # d4

            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),  # d_extra
        )

        self.conv_blocks_5 = nn.Sequential(
            nn.Conv2d(nt_input, nt, 1),
            nn.BatchNorm2d(nt),
            nn.LeakyReLU(0.2, inplace=True),

            #### 这里缺少 nn.Replicate(4, 3) - nn.Replicate(4, 4)
        )

        self.conv_blocks_6 = nn.Sequential(
            nn.Conv2d(ndf * 16 + nt, ndf * 16, 1),
            nn.BatchNorm2d(ndf * 16),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 16, 1, 4),
            nn.Sigmoid(),
        )

        def forward(self, output_data, output_condition, output_encode):
            d3 = self.conv_blocks_1(output_data)
            c3 = self.conv_blocks_2(output_condition)
            m = torch.cat((d3, c3), 1)
            m5 = self.conv_blocks_3(m)
            d_extra = self.conv_blocks_4(m5)

            b1 = self.conv_blocks_5(output_encode)
            print(b1.dim)
            d_extra_b1 = torch.cat((d_extra, b1), 1)
            d5 = self.conv_blocks_6(d_extra_b1)
            return d5
