from tools import *

initial_snr_range = (90., 100.)
final_snr_range = (5., 15.)
snr_steps = 17

### single SNR range training
initial_snr_range = final_snr_range
snr_steps = 0


# compute the list of SNR ranges
lower_snrs = np.linspace(initial_snr_range[0], final_snr_range[0], snr_steps+1)
upper_snrs = np.linspace(initial_snr_range[1], final_snr_range[1], snr_steps+1)
# snr_ranges = [(lower_snrs[i], upper_snrs[i]) for i in range(len(lower_snrs))]
snr_ranges = list(zip(lower_snrs, upper_snrs))


CLSchedClass = PlateauCLScheduler
CLSched_kwargs = {}
