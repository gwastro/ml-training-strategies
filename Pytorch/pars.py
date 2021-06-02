import torch

# device ('cpu'/'cuda') and data type
device = 'cuda'
store_device = 'cpu'
dtype = torch.float32

### path to training, validation and test data
path = '/path/to/data/'

train_prefix = 'train_'
valid_prefix = 'val_'
test_prefix = 'thr_'

waveform_fname = 'signals.hdf'
noise_fname = 'noise.hdf'

# training parameters: learning rate, batch size, number of epochs, loss function
lr = 0.00001
batch_size = 32
epochs = 200

# how many independently initialized runs to do consecutively
runs_number = 1

train_noises_per_signal = 1
valid_noises_per_signal = 1
test_noises_per_signal = 1

train_signals = [0, 100000]
valid_signals = [0, 100000]
test_signals = [0, 10000]

train_combined_noises = [0, 100000]
valid_combined_noises = [0, 100000]
test_combined_noises = [0, 10000]

train_pure_noises = [100000, 200000]
valid_pure_noises = [100000, 200000]
test_pure_noises = [0, 400000]

train_index_array = [train_noises_per_signal, train_signals, train_combined_noises, train_pure_noises]
valid_index_array = [valid_noises_per_signal, valid_signals, valid_combined_noises, valid_pure_noises]
test_index_array = [test_noises_per_signal, test_signals, test_combined_noises, test_pure_noises]
