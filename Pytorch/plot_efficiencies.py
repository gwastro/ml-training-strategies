import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from pars import *
efficiency_snrs_to_plot = (6., 9., 15., 30.)
index_filecol = 4
indices_filerow_plot = (5, 8, 14, 29)
indices_filerow_eval = range(2, 30, 3)

extra_plots = 0
runs_number = 50

epochs = epochs
epoch_axis = range(1, epochs+1)

omit_error = False

lw_run = 0.8
lw_mean = 3.

c_run = 'grey'
c_mean = 'black'

yrange = (-0.05, 1.05)

same_lengths = True
first_length = None

def load_run(i_run, indices_epoch):
	# return np.stack([np.take(np.loadtxt('efficiencies/out_efficiencies_run_%04i_epoch_%04i.txt' % (i_run, i_epoch))[:, index_filecol], indices_filerow, axis=0) for i_epoch in epoch_axis], axis=0)
	return np.stack([np.loadtxt('efficiencies/out_efficiencies_run_%04i_epoch_%04i.txt' % (i_run, i_epoch))[:, index_filecol] for i_epoch in epoch_axis], axis=0)

infiles = []
for i_run in range(runs_number):
	try:
		infiles.append(load_run(i_run, epoch_axis))
	except:
		if omit_error:
			pass
		else:
			raise

for infile in infiles:
	if first_length is None:
		first_length = len(infile)
	else:
		if len(infile)!=first_length:
			same_lengths = False

if same_lengths:
	averages = sum(infiles)/len(infiles)
else:
	raise NotImplementedError

best_epoch_zi = np.argmax(np.mean(averages[:, indices_filerow_eval], axis=1))	# zi = zero-indexed
best_epoch_oi = epoch_axis[best_epoch_zi]	# oi = one-indexed
best_epoch_sensitivities = np.array([np.mean(infile[best_epoch_zi, indices_filerow_eval], axis=0) for infile in infiles])	# 1-dim np array: run

best_run = np.argmax(best_epoch_sensitivities)
worst_run = np.argmin(best_epoch_sensitivities)
middle_run = np.argmin((best_epoch_sensitivities-np.mean(best_epoch_sensitivities))**2, axis=0)

nums_file = open('epoch_run_nums.txt', 'w')
nums_file.write('#run, epoch\n')
nums_file.write('#best, middle, worst\n')
nums_file.write('%i    %i\n' % (best_run, best_epoch_oi))
nums_file.write('%i    %i\n' % (middle_run, best_epoch_oi))
nums_file.write('%i    %i\n' % (worst_run, best_epoch_oi))
nums_file.close()

print('Best epoch: %i' % best_epoch_oi)
print('Worst run: %i, best run: %i, middle run: %i' % (worst_run, best_run, middle_run), flush=True)



ncols = 2
nrows = int(np.ceil((len(efficiency_snrs_to_plot)+extra_plots)/ncols))
to_remove = ncols*nrows - len(efficiency_snrs_to_plot) - extra_plots
fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(15., 15.))
for i in range(to_remove):
	axes[-1, -i-1].remove()


snr_index_iter = iter(zip(efficiency_snrs_to_plot, indices_filerow_plot))

for axes_line in axes:
	for ax in axes_line:
		try:
			snr, i_filerow = next(snr_index_iter)
		except StopIteration:
			break
		ax.set_ylim(yrange)
		ax.grid(b=True)
		ax.set_title('Efficiency estimate for SNR=%f' % snr)
		for infile in infiles:
			ax.plot(epoch_axis, infile[:, i_filerow], linewidth=lw_run, color=c_run)
		ax.plot(epoch_axis, averages[:, i_filerow], linewidth=lw_mean, color=c_mean)
		ax.plot([best_epoch_oi, best_epoch_oi], yrange, linewidth=lw_mean, color='red', linestyle='dashed')

# after here, append parts that do "extra" plots

#ax = axes[-1, -to_remove-1]
#ax.set_ylim(yrange)
#ax.grid(b=True)
#ax.set_title('Sensitivity estimate average over previous SNRs')
#for infile in infiles:
#	ax.plot(infile[:, 0], np.mean(infile[:, 1:], axis=1), linewidth=lw_run, color=c_run)
#ax.plot(averages[:, 0], np.mean(averages[:, 1:], axis=1), linewidth=lw_mean, color=c_mean)

# end of "extra" plots section, save figure

fig.savefig('efficiency_plots.png')
