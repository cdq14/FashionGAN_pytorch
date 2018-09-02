import h5py
import numpy as np
import scipy.io
import torch
import torch.utils.data as Data
from easydict import EasyDict as edict

from config_stage1 import config
from mydata import MyDataLoader
from net_graph_stage1 import Generator, Discriminator, weights_init_normal
from pytorchutils.dispSurrogate import dispSurrogate

torch.backends.cudnn.benchmark = True
torch.cuda.set_device(5)

ind_mat_file = scipy.io.loadmat('../data_release/benchmark/ind.mat')
train_ind = ind_mat_file['train_ind']  # type numpy.ndarray , train_ind.shape = 70000 * 1

# local
# getNet = dofile('../codes_lua/getNet.lua')

theme = 'ih1'
_lambda = 100

h5file = h5py.File('../data_release/supervision_signals/G2.h5', 'r')
ih = h5file['/ih']  # type h5py._hl.dataset.Dataset shape (78979, 3, 128, 128)
ih = np.array(ih).transpose((0, 1, 3, 2))  # numpy.ndarray (78979, 3, 128, 128)
n_file = ih.shape[0]

ih_mean = h5file['/ih_mean']  # type h5py._hl.dataset.Dataset shape (3, 128, 128)
ih_mean = np.array(ih_mean).reshape(1, 3, 128, 128).transpose((0, 1, 3, 2))  # numpy.ndarray (1, 3, 128, 128)

b_ = h5file['/b_']
b_ = np.array(b_).transpose((0, 1, 3, 2))  # numpy.ndarray  (78979, 1, 128, 128)

hn2_mat_file = scipy.io.loadmat('../data_release/test_phase_inputs/encode_hn2_rnn_100_2_full.mat')  # keys: hn2
text = hn2_mat_file['hn2']
text = np.ascontiguousarray(text)

criterion = torch.nn.BCELoss()
criterionAE = torch.nn.L1Loss()

optimStateG = edict({'learningRate': config.lr, 'beta1': config.beta1})
optimStateD = edict({'learningRate': config.lr, 'beta1': config.beta1})

nz = config.nz

G = Generator()
D = Discriminator()

G.apply(weights_init_normal)
D.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(G.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))
optimizer_D = torch.optim.Adam(D.parameters(), lr=config.lr, betas=(config.beta1, config.beta2))

noise = torch.Tensor(config.batchSize, nz, 1, 1)
label = torch.Tensor(config.batchSize, 1, 1, 1)
real_label = 1
fake_label = 0

index_dataset = Data.TensorDataset(torch.Tensor(train_ind))
index_loader = Data.DataLoader(dataset=index_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
my_dataloader = MyDataLoader(ih, text, b_, train_ind)

for epoch in range(config.n_epochs):
    for step, inds in enumerate(index_loader):
        # the dataset of index_loader is a list,so it return a list of data.
        index_list = [int(inds[0][i][0]) for i in range(config.batch_size)]

        noise.normal_(0, 1)
        input, condition, input_wrong, condition_wrong, encode = my_dataloader.getData()
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

    # display  why base + 3,4,5 ???
    dispSurrogate(fake + ih_mean.repeatTensor(config.batchSize, 1, 1, 1), 3, 'fake')
    dispSurrogate(input_record + ih_mean.repeatTensor(config.batchSize, 1, 1, 1), 4, 'real')
    dispSurrogate(condition_record, 5, 'condition')
    # save model
