"""
usage:
    from config_stage1 import config
    print(config.batchSize)
"""

from easydict import EasyDict

config = EasyDict()

# learning batch
config.n_epochs = 500
config.batchSize = 4
config.test_batchSize = 10
config.win_size = 128

# specific size
config.n_map_all = 7
config.n_condition = 3
config.n_z = 80
config.nz = config.n_z
config.n_c = 3
config.nc = config.n_c
config.nt_input = 100

config.lambda_mismatch = 0.1
config.lambda_fake = 0.9

config.lr = 0.0002  # * 1e2 * 2.5

# torch.optim.Adam default beta1=0.9 beta2=0.999
config.beta1 = 0.5
config.beta2 = 0.999





