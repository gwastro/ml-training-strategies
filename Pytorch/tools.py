import numpy as np
from numpy import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import h5py
import os

data_type = torch.float

### Base class for datasets
class BaseDataset(Dataset):
	def __init__(self, snr_range, wave_limits, noise_combined_limits, noise_pure_limits, noises_per_signal=1, device='cpu', dtype=data_type, bool=False):
		Dataset.__init__(self)
		assert (len(wave_limits)==2 and len(noise_combined_limits)==2 and len(noise_pure_limits)==2 and len(snr_range)==2)
		self.snr_range = snr_range
		self.wave_lim = wave_limits
		self.noise_comb_lim = noise_combined_limits
		self.noise_pure_lim = noise_pure_limits
		self.device = torch.device(device)
		self.dtype = dtype
		if bool:
			self.wave_label = True
			self.noise_label = False
		else:
			self.wave_label = torch.tensor([1., 0.], device=self.device, dtype=self.dtype)
			self.noise_label = torch.tensor([0., 1.], device=self.device, dtype=self.dtype)
		self.gen = random.default_rng()
		self.noises_per_signal = noises_per_signal
		self.signal_samples = (self.wave_lim[1]-self.wave_lim[0])*self.noises_per_signal
		assert (self.signal_samples==(self.noise_comb_lim[1]-self.noise_comb_lim[0]))

	def __len__(self):
		return ((self.noise_comb_lim[1]-self.noise_comb_lim[0])+(self.noise_pure_lim[1]-self.noise_pure_lim[0]))

	def __getitem__(self, index):
		if index<self.signal_samples:
			wave_i = index//self.noises_per_signal + self.wave_lim[0]
			noise_i = index + self.noise_comb_lim[0]
			snr = self.gen.uniform(low=self.snr_range[0], high=self.snr_range[1])
			return self.get_noise(noise_i) + snr*self.get_waveform(wave_i), self.wave_label
		else:
			noise_i = index - self.signal_samples + self.noise_pure_lim[0]
			return self.get_noise(noise_i), self.noise_label

	def get_noise(self, index):		# needs to be overwritten when subclassing
		raise NotImplementedError

	def get_waveform(self, index):	# needs to be overwritten when subclassing
		raise NotImplementedError

	def snrs(self, *args):
		if len(args)==0:
			return self.snr_range
		elif len(args)==1:
			self.snr_range = args[0]
		elif len(args)==2:
			self.snr_range = tuple(args)
		else:
			raise ValueError
		return


### Dataset class for PyTorch, uses data stored as torch tensors, injects at a randomly generated SNR and generates labels either (1., 0.)/(0., 1.), or True/False
### each noise sample is used once as noise and once with an injection
class UnsavedDataset(BaseDataset):
	def __init__(self, waveform_tensor, noise_tensor, *args, **kwargs):
		assert (torch.is_tensor(waveform_tensor) and torch.is_tensor(noise_tensor))
		BaseDataset.__init__(self, *args, **kwargs)
		self.wave_tensor, self.noise_tensor = waveform_tensor.to(dtype=self.dtype), noise_tensor.to(dtype=self.dtype)
		if 'store_device' in kwargs.keys():
			self.wave_tensor, self.noise_tensor = self.wave_tensor.to(device=kwargs['store_device']), self.noise_tensor.to(device=kwargs['store_device'])

	def get_noise(self, index):
		return self.noise_tensor[index].to(device=self.device)

	def get_waveform(self, index):
		return self.wave_tensor[index].to(device=self.device)


### used to load HDF5 format data
def load_dataset(path, waveform_fname, noise_fname, snr_range, index_array, dtype=torch.float32, device=torch.device('cpu'), store_device=torch.device('cpu')):
		waveform_tensor = torch.from_numpy(h5py.File(os.path.join(path, waveform_fname), 'r')['data/0'].__array__())[:index_array[1][1]]
		waveform_tensor = torch.transpose(waveform_tensor, 1, 2).to(device=store_device)
		noise_tensor = torch.from_numpy(h5py.File(os.path.join(path, noise_fname), 'r')['data/0'].__array__())[:max(index_array[2][1], index_array[3][1])]
		noise_tensor = torch.unsqueeze(noise_tensor, 1).to(device=store_device)
		return UnsavedDataset(waveform_tensor, noise_tensor, snr_range, index_array[1], index_array[2], index_array[3], noises_per_signal=index_array[0], device=device, dtype=dtype)


### regularized binary cross entropy loss to compensate for exploding gradients; a linear map is applied to the first argument: (0., 1.) -> (epsilon, 1.-epsilon*dim), so that the mapped probabilities still sum up to 1
class reg_BCELoss(nn.BCELoss):
	def __init__(self, *args, epsilon=0.001, dim=None, **kwargs):
		nn.BCELoss.__init__(self, *args, **kwargs)
		assert isinstance(dim, int)
		self.regularization_dim = dim
		self.regularization_A = epsilon	#constant term in transform
		self.regularization_B = 1. - epsilon*self.regularization_dim	#linear term in transform
	def forward(self, inputs, target, *args, **kwargs):
		assert inputs.shape[-1]==self.regularization_dim
		transformed_input = self.regularization_A + self.regularization_B*inputs
		return nn.BCELoss.forward(self, transformed_input, target, *args, **kwargs)


### Base class for curriculum learning schedulers
class CurriculumLearningScheduler:
	def __init__(self, snr_ranges, datasets, verbose=True, optim=None):
		self.snr_ranges = snr_ranges
		self.datasets = datasets
		self.verbose = verbose

		self.done = False
		self.interrupt = False

		self.reload_optimizer = not optim is None
		if self.reload_optimizer:
			self.optim = optim
			self.optim_init_state_dict = self.optim.state_dict()

		self.snr_iter = iter(self.snr_ranges)
		self.next_range = next(self.snr_iter)
		self.set_next_range()

	def set_next_range(self):
		for dataset in self.datasets:
			old_range = dataset.snrs()
			dataset.snrs(self.next_range)
		self.output_info(old_range, self.next_range)
		try:
			self.next_range = next(self.snr_iter)
		except StopIteration:
			self.done = True

		if self.reload_optimizer:
			self.optim.load_state_dict(self.optim_init_state_dict)

	def output_info(self, old_range, new_range):
		if self.verbose:
			print('# Reducing SNR range from %f-%f to %f-%f' % (old_range[0], old_range[1], new_range[0], new_range[1]))


### CL scheduler, takes a step when a given metric (loss/accuracy) has failed to improve for more than "patience" epochs
class PlateauCLScheduler(CurriculumLearningScheduler):
	def __init__(self, *args, patience=4, threshold=1.e-4, threshold_mode='rel', optimization_mode='min', metric_index=0, allow_interrupt=False, **kwargs):
		CurriculumLearningScheduler.__init__(self, *args, **kwargs)
		self.patience = patience
		self.threshold = threshold
		self.threshold_mode = threshold_mode
		self.optimization_mode = optimization_mode
		self.metric_index = metric_index
		self.allow_interrupt = allow_interrupt

		self.best = None
		self.num_bad_epochs = None

		print('# Initializing PlateauCLScheduler')
		return

	def is_better(self, a):
		if self.best is None:
			return True
		elif self.threshold_mode=='rel':
			if self.optimization_mode=='min':
				return a < self.best*(1.-self.threshold)
			elif self.optimization_mode=='max':
				return a > self.best*(1.+self.threshold)
			else:
				raise NotImplementedError
		elif self.threshold_mode=='abs':
			if self.optimization_mode=='min':
				return a < self.best - self.threshold
			elif self.optimization_mode=='max':
				return a > self.best + self.threshold
			else:
				raise NotImplementedError
		else:
			raise NotImplementedError

	def step(self, *args):
		current = float(args[self.metric_index])

		if self.is_better(current):
			self.best = current
			self.num_bad_epochs = 0
		else:
			self.num_bad_epochs += 1

		if self.num_bad_epochs > self.patience:
			if self.done:
				if self.allow_interrupt:
					self.interrupt = True
			else:
				self.set_next_range()
				self.best = None
				self.num_bad_epochs = None
		return


### CL scheduler, takes a step when a given metric (loss/accuracy) has improved beyond a given threshold
class ThresholdCLScheduler(CurriculumLearningScheduler):
	def __init__(self, *args, threshold=0.2, optimization_mode='min', metric_index=0, **kwargs):
		CurriculumLearningScheduler.__init__(self, *args, **kwargs)
		self.threshold = threshold
		self.optimization_mode = optimization_mode
		self.metric_index = metric_index

		print('# Initializing ThresholdCLScheduler')
		return

	def is_better(self, a):
		if self.optimization_mode=='min':
			return a<=self.threshold
		elif self.optimization_mode=='max':
			return a>=self.threshold
		else:
			raise NotImplementedError

	def step(self, *args):
		current = float(args[self.metric_index])

		if (self.is_better(current) and not self.done):
			self.set_next_range()
		return


### CL scheduler, takes a step after a given number of epochs
class EpochCLScheduler(CurriculumLearningScheduler):
	def __init__(self, *args, patience=4, **kwargs):
		CurriculumLearningScheduler.__init__(self, *args, **kwargs)
		self.patience = patience
		self.num_epochs = 0

		print('# Initializing EpochCLScheduler')
		return

	def step(self, *args):
		self.num_epochs += 1
		if (self.num_epochs>self.patience and not self.done):
			self.num_epochs = 0
			self.set_next_range()
		return


### computes the efficiencies of a network
class EfficiencyEstimator:
	def __init__(self, wave_dataset, noise_dataset, snrs, batch_size=1000, faps=(1.e-2, 1.e-3, 1.e-4)):
		self.snrs = snrs
		self.wave_dataset = wave_dataset
		self.noise_dataset = noise_dataset
		self.wave_dl = DataLoader(self.wave_dataset, batch_size=batch_size)
		self.noise_dl = DataLoader(self.noise_dataset, batch_size=batch_size)
		self.faps = faps

	def __call__(self, network):
		# gather noise outputs
		noise_outputs = []
		self.noise_dataset.snrs((0., 0.))
		for noise_inputs, noise_labels in self.noise_dl:
			assert not (torch.max(noise_labels).item())
			noise_outputs.append(network(noise_inputs)[:, 0])
		noise_outputs = torch.cat(noise_outputs, dim=0)
		noise_outputs = torch.sort(noise_outputs).values
		# calculate thresholds for the individual FAPs
		false_alarm_counts = (np.array(self.faps)*len(self.noise_dataset)).astype(int)
		thresholds = torch.tensor([noise_outputs[-fac] for fac in false_alarm_counts], device=self.wave_dataset.device)	### this needs some further discussion (although for FAPs high enough to be statistically significant, this should be irrelevant)
		expanded_thresholds = torch.unsqueeze(thresholds, 0)	# thresholds unsqueezed to shape (1, N) for broadcasting
		# compute efficiencies
		new_sensitivities = []
		for snr in self.snrs:
			self.wave_dataset.snrs((snr, snr))
			samples = 0
			detections = torch.zeros_like(thresholds, dtype=int)
			for wave_inputs, wave_labels in self.wave_dl:
				assert torch.min(wave_labels).item()
				wave_outputs = network(wave_inputs)[:, 0:1]	# network outputs are taken in shape (N, 1) to be broadcastable
				detections += torch.sum(wave_outputs > expanded_thresholds, 0)
				samples += len(wave_outputs)
			new_sensitivities.append(detections.cpu().detach().numpy()/samples)
		new_sensitivities = np.stack(new_sensitivities, axis=0)
		return new_sensitivities

