import torch
import torch.utils.data as Data

from config_stage1 import config
from mydata import BatchDataLoader, AllDataLoader
from net_graph_stage1 import Generator, Discriminator, weights_init_normal
from pytorchutils.dispSurrogate import dispSurrogate

torch.backends.cudnn.benchmark = True  # use cudnn to speed up
torch.cuda.set_device(5)  # set GPU id

train_ind, ih, ih_mean, b_, text = AllDataLoader.loadata()

# loss function
criterion = torch.nn.BCELoss()
criterionAE = torch.nn.L1Loss()

G = Generator()
D = Discriminator()

G.apply(weights_init_normal)
D.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

noise = torch.Tensor(config.batchSize, config.nz, 1, 1)
label = torch.Tensor(config.batchSize, 1, 1, 1)
real_label = 1
fake_label = 0


index_dataset = Data.TensorDataset(torch.Tensor(train_ind))
index_loader = Data.DataLoader(dataset=index_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
batch_data_loader = BatchDataLoader(ih, text, b_, train_ind)

_lambda = 100

for epoch in range(config.n_epochs):
    for step, inds in enumerate(index_loader):

        #### init G and D

        # the dataset of index_loader is a list,so it return a list of data.
        index_list = [int(inds[0][i][0]) for i in range(config.batch_size)]

        noise.normal_(0, 1)
        input, condition, input_wrong, condition_wrong, encode = batch_data_loader.getData()
        fake = G.forward(noise, encode, condition)

        input_record = input.clone()
        condition_record = condition.clone()

        label.fill_(real_label)

        output = D.forward(input, encode, condition)
        errD_real = criterion.forward(output, label)
        de_do = criterion.backward(output, label)
        D.backward({input, encode, condition}, de_do)

        label.fill_(fake_label)
        output = D.forward(input_wrong, encode, condition_wrong)
        errD_wrong = config.lambda_mismatch * criterion.forward(output, label)
        de_do = criterion.backward(output, label)
        D.backward({input_wrong, encode, condition_wrong}, de_do)

        input.copy_(fake)
        label.fill_(fake_label)

        output = D.forward(input, encode, condition)
        errD_fake = config.lambda_fake * criterion.forward(output, label)
        de_do = criterion.backward(output, label)
        D.backward({input, encode, condition}, de_do)

        errD = (errD_real + errD_wrong + errD_fake) / 2

        output = D.output
        label.fill_(real_label)
        errG = criterion.forward(output, label)
        de_do = criterion.backward(output, label)
        de_dg = D.updateGradInput({input, encode, condition}, de_do)
        errL1 = criterionAE.forward(input, input_record)
        df_do_AE = criterionAE.backward(input, input_record)
        G.backward({noise, encode, condition}, de_dg[1] + df_do_AE.mul(_lambda))

        print(
            "[Epoch %d/%d] [Batch %d/%d] [ErrD: %.5f] [ErrD_real: %.5f] [ErrD_wrong: %.5f] [ErrD_fake: %.5f] [ErrG: %.5f] [ErrL1: %.5f]" % (
                epoch + 1, config.n_epochs, step + 1, len(index_loader), errD, errD_real, errD_wrong, errD_fake, errG,
                errL1))

    #### display  why base + 3,4,5 ???
    dispSurrogate(fake + ih_mean.repeatTensor(config.batchSize, 1, 1, 1), 3, 'fake')
    dispSurrogate(input_record + ih_mean.repeatTensor(config.batchSize, 1, 1, 1), 4, 'real')
    dispSurrogate(condition_record, 5, 'condition')

    ####  save model  checkpoint
