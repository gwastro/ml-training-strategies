### This file contains the function which initializes the network
# Currently contains the adapted Gabbard et al. network

from torch import nn
from pars import device, dtype

### Initialize the network; can give a device and data type as arguments
def get_network(device=device, dtype=dtype):
	Network = nn.Sequential(
		nn.BatchNorm1d(1),	# 1x2048
		nn.Conv1d(1, 8, 64),	# 8x1985
		nn.ELU(),
		nn.Conv1d(8, 8, 32),	# 8x1954
		nn.MaxPool1d(4),	# 8x488
		nn.ELU(),
		nn.Conv1d(8, 16, 32),	# 16x457
		nn.ELU(),
		nn.Conv1d(16, 16, 16),	# 16x442
		nn.MaxPool1d(3),	# 16x147
		nn.ELU(),
		nn.Conv1d(16, 32, 16),	# 32x132
		nn.ELU(),
		nn.Conv1d(32, 32, 16),	# 32x117
		nn.MaxPool1d(2),	# 32x58
		nn.ELU(),
		nn.Flatten(),	#  1856
		nn.Linear(1856, 64),	# 64
		nn.Dropout(p=.5),
		nn.ELU(),
		nn.Linear(64, 64),	# 64
		nn.Dropout(p=.5),
		nn.ELU(),
		nn.Linear(64, 2),	# 2
		nn.Softmax(dim=1)
	)
	Network.to(device=device, dtype=dtype)
	return Network

if __name__=='__main__':
    print(get_network())
