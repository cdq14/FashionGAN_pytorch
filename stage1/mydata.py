import random

import h5py
import numpy as np
import torch
from scipy import signal, io as scipyio

from config_stage1 import config


class AllDataLoader:
    def loadata(self):
        ind_mat_file = scipyio.loadmat('../data_release/benchmark/ind.mat')
        train_ind = ind_mat_file['train_ind']  # type numpy.ndarray , train_ind.shape = 70000 * 1

        h5file = h5py.File('../data_release/supervision_signals/G2.h5', 'r')
        ih = h5file['/ih']  # type h5py._hl.dataset.Dataset shape (78979, 3, 128, 128)
        ih = np.array(ih).transpose((0, 1, 3, 2))  # numpy.ndarray (78979, 3, 128, 128)

        ih_mean = h5file['/ih_mean']  # type h5py._hl.dataset.Dataset shape (3, 128, 128)
        ih_mean = np.array(ih_mean).reshape(1, 3, 128, 128).transpose((0, 1, 3, 2))  # numpy.ndarray (1, 3, 128, 128)

        b_ = h5file['/b_']
        b_ = np.array(b_).transpose((0, 1, 3, 2))  # numpy.ndarray  (78979, 1, 128, 128)

        hn2_mat_file = scipyio.loadmat('../data_release/test_phase_inputs/encode_hn2_rnn_100_2_full.mat')  # keys: hn2
        text = hn2_mat_file['hn2']
        text = np.ascontiguousarray(text)

        return train_ind, ih, ih_mean, b_, text


class BatchDataLoader:
    def __init__(self, images, text, b_, train_ind):
        assert isinstance(images, torch.Tensor) and isinstance(text, torch.Tensor) and isinstance(b_,
                                                                                                  torch.Tensor) and isinstance(
            train_ind, torch.Tensor)
        assert images.size(0) == text.size(0) == b_.size(0)
        self.images = images
        self.text = text
        self.b_ = b_
        self.train_ind = train_ind

        self.train_size = self.images.size(0)

        self.cb = torch.Tensor([[3, 2, 1, 1, 2, 3, 2],
                                [2, 3, 3, 2, 1, 1, 2],
                                [1, 1, 2, 3, 3, 2, 2]])
        self.cb = self.cb * 0.25

        self.H = torch.Tensor(
            [0.0030, 0.0133, 0.0219, 0.0133, 0.0030, 0.0133, 0.0596, 0.0983, 0.0596, 0.0133, 0.0219, 0.0983, 0.1621,
             0.0983,
             0.0219, 0.0133, 0.0596, 0.0983, 0.0596, 0.0133, 0.0030, 0.0133, 0.0219, 0.0133, 0.0030]).resize(5, 5)

    def getData(self, index_list):
        batchSize = len(index_list)
        assert batchSize == config.batchSize

        wrong_index_list = [(inx + random.randint(0, self.train_size) % self.train_size) for inx in index_list]

        input = torch.Tensor(config.batchSize, config.n_c, config.win_size, config.win_size)
        condition = torch.Tensor(config.batchSize, config.n_condition, config.win_size, config.win_size)
        input_wrong = torch.Tensor(config.batchSize, config.n_c, config.win_size, config.win_size)
        condition_wrong = torch.Tensor(config.batchSize, config.n_condition, config.win_size, config.win_size)
        encode = torch.Tensor(config.batchSize, config.nt_input, 1, 1)

        for i in range(batchSize):
            index_list[i] = self.train_ind[index_list[i]]
        for i in range(batchSize):
            wrong_index_list[i] = self.train_ind[wrong_index_list[i]]

        for i in range(batchSize):
            input[i] = self.images[index_list[i]]
            t = self.b_[index_list[i], 0, :, :]
            for j in range(config.n_condition):
                u = torch.Tensor(1, 1, config.win_size, config.win_size)
                for k in range(config.n_map_all):
                    u[t.eq(k)] = self.cb[j][k]
                # do blurring toward u
                torch.nn.functional.conv2d
                v = signal.convolve2d(u.squeeze(), self.H, 'same')
                condition[i, j, :, :] = v.reshape((1, 1, config.win_size, config.win_size))

            encode[i, :, :, :] = self.text[index_list[i], :].resize(1, config.nt_input, 1, 1)
        condition = condition - 0.5  # zero mean

        for i in range(batchSize):
            input_wrong[i] = self.images[wrong_index_list[i]]
            t = self.b_[wrong_index_list[i], 0, :, :]
            for j in range(config.n_condition):
                u = torch.Tensor(1, 1, config.win_size, config.win_size)
                for k in range(config.n_map_all):
                    u[t.eq(k)] = self.cb[j][k]
                # do blurring torward u
                v = signal.convolve2d(u.squeeze(), self.H, 'same')
                condition_wrong[{{i}, {j}, {}, {}}] = v.reshape((1, 1, config.win_size, config.win_size))
        condition_wrong = condition_wrong - 0.5

        return input, condition, input_wrong, condition_wrong, encode
