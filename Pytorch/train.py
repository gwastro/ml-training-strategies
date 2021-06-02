### import modules
import numpy as np
from numpy import fft as npfft

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torch.optim import lr_scheduler

# from tools import SavedDataset, UnsavedDataset, reg_BCELoss, SensitivityEstimator
from tools import load_dataset, reg_BCELoss
from network import get_network

from pars import *
from scheduler_pars import *

import sys
if len(sys.argv) > 1:
	i_run_init = int(sys.argv[1])
else:
	i_run_init = 0
import os

outfiles_dir = 'outfiles'
state_dicts_dir = 'state_dicts'

# range of optimal SNRs (uniform distribution)
# in the current curriculum learning implementation, this is irrelevant but required to define
snr_range = (0, 0)

# load datasets
DSs = [load_dataset(path, prefix+waveform_fname, prefix+noise_fname, snr_range, index_array, dtype=dtype, device=device, store_device=store_device) for prefix, index_array in ((train_prefix, train_index_array), (valid_prefix, valid_index_array))]

TrainDS, ValidDS = DSs

# initialize data loaders as training convenience
TrainDL = DataLoader(TrainDS, batch_size=batch_size, shuffle=True)
ValidDL = DataLoader(ValidDS, batch_size=1000, shuffle=True)

def run_training(i_run, DSs, epochs_done=0, network_state_dict=None, optim_state_dict=None):
	# regularized binary cross entropy to prevent exploding gradients (gradient clipping is also an option, though)
	crit = reg_BCELoss(dim=2, epsilon=1.e-6)

	print('# Initializing for run %04i' % i_run, flush=True)
	# output files; originally there were two in order to output all batch losses into one and validation losses into the second; can be reverted
	tr_outfile = open(os.path.join(outfiles_dir, 'out_train_%04i.txt' % i_run), 'w', buffering=1)

	# set model in training mode and move to the desired device (cpu/cuda)
	Network = get_network(device=device, dtype=dtype)
	Network.train()

    # if given, load network state from state dictionary
	if network_state_dict is None:
		pass
	else:
		network.load_state_dict(network_state_dict)

    # initialize optimizer and curriculum learning scheduler
	opt = optim.Adam(Network.parameters(), lr=lr)
	CLSched = CLSchedClass(snr_ranges, (TrainDS, ValidDS), optim=opt, **CLSched_kwargs)

    # if given, load optimizer state from state dictionary (momentum, second order moments in adaptive methods)
	if optim_state_dict is None:
		pass
	else:
		optim.load_state_dict(optim_state_dict)

	print('# Starting run %04i' % i_run, flush=True)
	# training loop
	min_valid_loss = 1.e100
	for e in range(epochs_done+1, epochs_done+epochs+1):
		# training epoch
		Network.train()
		train_loss = 0.
		batches = 0

        # optimization step
		for train_inputs, train_labels in TrainDL:
			opt.zero_grad()
			train_outputs = Network(train_inputs)
			loss = crit(train_outputs, train_labels)
			loss.backward()
			opt.step()
			train_loss += loss.detach().item()
			batches += 1
		train_loss /= batches

		# intermediate testing
		with torch.no_grad():
			Network.eval()
            # validation loss and accuracy
			valid_loss = 0.
			samples = 0
			valid_accuracy = 0
			batches = 0
			for valid_inputs, valid_labels in ValidDL:
				valid_outputs = Network(valid_inputs)
				valid_loss += crit(valid_outputs, valid_labels).detach().item()
				batches += 1
				for p1, p2 in zip(valid_labels, valid_outputs):
					samples += 1
					if torch.argmax(p1)==torch.argmax(p2):
						valid_accuracy += 1
			valid_loss /= batches
			valid_accuracy /= samples

		# file and stdout output of losses
		tr_outfile.write('%04i    %1.12e    %1.12e    %f\n' % (e, train_loss, valid_loss, valid_accuracy))
		print('%04i    %1.12e    %1.12e    %f' % (e, train_loss, valid_loss, valid_accuracy), flush=True)
		# save the best-performing (on validation set) model state and optimizer state to file
		if CLSched.done:
			if valid_loss<min_valid_loss:
				torch.save(Network.state_dict(), os.path.join(state_dicts_dir, 'best_state_dict_%04i.pt' % i_run))
				min_valid_loss = valid_loss
		torch.save(Network.state_dict(), os.path.join(state_dicts_dir, 'state_dict_run_%04i_epoch_%04i.pt' % (i_run, e)))
		torch.save(opt.state_dict(), os.path.join(state_dicts_dir, 'optim_state_dict_run_%04i_epoch_%04i.pt' % (i_run, e)))
		CLSched.step(valid_loss, valid_accuracy)
        # kill the training loop if the curriculum learning scheduler says so
		if CLSched.interrupt:
			break

        # close output files
	tr_outfile.close()

# run the training function multiple times
if __name__=='__main__':
	for i_run in range(i_run_init, i_run_init+runs_number):
		run_training(i_run, DSs) ### fill in arguments here

