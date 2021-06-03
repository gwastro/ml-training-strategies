### This script calculates the efficiency estimates of a set of runs which have already finished from the saved state dictionaries

from network import get_network
from tools import UnsavedDataset, EfficiencyEstimator, load_dataset
from pars import *

efficiency_snrs = list(range(1, 31))
faps = [0.1, 0.01, 0.001, 0.0001, 0.00001]

device = 'cuda'
remove_softmax = False
output_directory = 'efficiencies'

import torch
import sys
import os

if len(sys.argv)==1:
	indices_run = range(runs_number)
elif len(sys.argv)==2:
	indices_run = [int(sys.argv[1])]
elif len(sys.argv)==3:
	indices_run = range(int(sys.argv[1]), int(sys.argv[2]))
else:
	raise ValueError
print('indices_run = %s' % str(indices_run), flush=True)
indiceses_epoch = [range(1, epochs+1) for _ in range(len(indices_run))]

state_dict_fnameses = [[os.path.join('state_dicts', 'state_dict_run_%04i_epoch_%04i.pt' % (i_run, i_epoch)) for i_epoch in indices_epoch] for i_run, indices_epoch in zip(indices_run, indiceses_epoch)]	# in case of training split over multiple folders or inequal lengths of individual runs, modify this line

# range of optimal SNRs (uniform distribution)
# in the current curriculum learning implementation, this is irrelevant but required to define
snr_range = (0, 0)

# construct dataset and dataloader (PyTorch convenience for batches)
TestDS = load_dataset(path, test_prefix+waveform_fname, test_prefix+noise_fname, snr_range, test_index_array, dtype=dtype, device=device, store_device=store_device)

SEWaveDS = UnsavedDataset(TestDS.wave_tensor, TestDS.noise_tensor, (0., 0.), TestDS.wave_lim, TestDS.noise_comb_lim, (0, 0), noises_per_signal=TestDS.noises_per_signal, device=device, dtype=dtype, bool=True)
SENoiseDS = UnsavedDataset(TestDS.wave_tensor, TestDS.noise_tensor, (0., 0.), (0, 0), (0, 0), TestDS.noise_pure_lim, noises_per_signal=TestDS.noises_per_signal, device=device, dtype=dtype, bool=True)

EEst = EfficiencyEstimator(SEWaveDS, SENoiseDS, efficiency_snrs, faps=faps)

# main loop over runs, epochs, batches
for i_run, indices_epoch, state_dict_fnames in zip(indices_run, indiceses_epoch, state_dict_fnameses):
	for e, state_dict_fname in zip(indices_epoch, state_dict_fnames):
		ef_outfile = open(os.path.join(output_directory, 'out_efficiencies_run_%04i_epoch_%04i.txt' % (i_run, e)), 'w', buffering=1)
		ef_outfile.write('# FAPs: %f' % faps[0])
		for fap in faps[1:]:
			ef_outfile.write('    %f' % fap)
		ef_outfile.write('\n')
		print(state_dict_fname, flush=True)
		Network = get_network(device=device, dtype=dtype)
		Network.eval()
		Network.load_state_dict(torch.load(state_dict_fname, map_location=device))
		if remove_softmax:
			new_layer = torch.nn.Linear(2, 2, bias=False)
			new_layer._parameters['weight'] = torch.nn.Parameter(torch.Tensor([[1., -1.], [-1., 1.]]), requires_grad=False)
			new_layer.to(device=device)
			Network[-1] = new_layer

		with torch.no_grad():
			estimated_efficiencies = EEst(Network)
		for snr, effs in zip(efficiency_snrs, estimated_efficiencies):
			ef_outfile.write('%f' % snr)
			for num in effs:
				ef_outfile.write('    %f' % num)
			ef_outfile.write('\n')
		ef_outfile.close()
