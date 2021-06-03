from tools import *

initial_snr_range = (90., 100.)
min_snr = 5.
lower_by = 0.9

# compute the list of SNR ranges
lower_snrs = [initial_snr_range[0]]
upper_snrs = [initial_snr_range[1]]

done = False
while not done:
	oldmin = lower_snrs[-1]
	oldmax = upper_snrs[-1]
	newmin = max(min_snr, oldmin*lower_by)
	if newmin<oldmin:
		newmax = newmin + (oldmax-oldmin)*lower_by
		lower_snrs.append(newmin)
		upper_snrs.append(newmax)
	else:
		done = True


snr_ranges = list(zip(lower_snrs, upper_snrs))


CLSchedClass = PlateauCLScheduler
CLSched_kwargs = {}
