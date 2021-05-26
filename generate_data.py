from BnsLib.data.generate_train import WaveformGetter, WhiteNoiseGenerator, NoiseGenerator
from BnsLib.data.transform import whiten, optimal_snr
from BnsLib.types.utils import NamedPSDCache
from BnsLib.utils import progress_tracker
from pycbc.filter import sigma
from pycbc.psd import aLIGOZeroDetHighPower as aPSD
import numpy as np
import h5py
import os
import sys
import time
from argparse import ArgumentParser
import logging
import matplotlib.pyplot as plt

class SignalGetter(WaveformGetter):
    def transform(self, wav):
       wav.prepend_zeros(5 * int(wav.sample_rate))
       wav.append_zeros(4 * int(wav.sample_rate))
       wav = whiten(wav, psd='aLIGOZeroDetHighPower')
       snr = sigma(wav, low_frequency_cutoff=20.)
       return (snr, wav)

class RawSignalGetter(WaveformGetter):
    def transform(self, wav):
        snr = optimal_snr(wav, psd='aLIGOZeroDetHighPower',
                          low_freq_cutoff=20.)
        return (snr, wav)

def generate_signals(file_path, variable_params, static_params,
                     detectors, tc_mean_position=0.7, verbose=True,
                     raw_data=False):
    calib_mean_time = 1. - tc_mean_position
    logging.debug(f'Got mean-merger time position {tc_mean_position} and calibrated it to {calib_mean_time}')
    sample_rate = int(1. / static_params['delta_t'])
    if raw_data:
        logging.info(f"Generating signals without whitening")
        wav_getter = RawSignalGetter(variable_params=variable_params,
                                     static_params=static_params,
                                     detectors=detectors)
    else:
        logging.info(f"Generating whitened signals")
        wav_getter = SignalGetter(variable_params=variable_params,
                                  static_params=static_params,
                                  detectors=detectors)
    
    tmp = wav_getter.generate(single_detector_as_list=False,
                              workers=None,
                              verbose=verbose)
    
    snr_dict = {}
    white_wavs = {}
    for det in tmp.keys():
        snr_dict[det] = []
        white_wavs[det] = []
        for pt in tmp[det]:
            snr_dict[det].append(pt[0])
            white_wavs[det].append(pt[1])
    
    logging.info('Calculating network SNRs for all signals')
    snrs = np.zeros(len(wav_getter))
    for i in range(len(wav_getter)):
        tmp = []
        for det in snr_dict.keys():
            tmp.append(snr_dict[det][i])
        snrs[i] = np.sqrt(np.sum(np.square(np.array(tmp))))
    
    signals = {}
    logging.info('Scaling all signals to SNR 1 and casting into array of correct size')
    for det in white_wavs.keys():
        signals[det] = []
        for i, sig in enumerate(white_wavs[det]):
            tmp = sig / snrs[i]
            tmp.append_zeros(sample_rate)
            eidx = int((calib_mean_time - float(tmp.start_time)) * tmp.sample_rate)
            sidx = eidx - sample_rate
            dat = np.zeros(sample_rate)
            if sidx < 0:
                dat[-sidx:] = tmp.data[:eidx]
            else:
                dat = tmp.data[sidx:eidx]
            signals[det].append(dat)
        signals[det] = np.vstack(signals[det])
    
    #Format signals to store them
    to_store = np.zeros((len(wav_getter), len(detectors), sample_rate))
    for i, det in enumerate(detectors):
        for j, sig in enumerate(signals[det]):
            to_store[j,i] = sig
    to_store = to_store.transpose(0, 2, 1)
    
    #Store signals
    with h5py.File(file_path, 'w') as fp:
        data_gr = fp.create_group('data')
        label_gr = fp.create_group('labels')
        params_gr = fp.create_group('params')
        for i, det in enumerate(detectors):
            dataset = data_gr.create_dataset(str(i), data=to_store)
            dataset.attrs['detector'] = det
        
        for key, val in wav_getter.get_params().items():
            if key == 'approximant':
                params_gr.create_dataset(key, data=np.array(val, dtype='S'))
            else:
                params_gr.create_dataset(key, data=np.array(val))
    logging.info(f'Saved signals at {file_path}')
    
    return len(wav_getter)

def generate_noise(file_path, static_params, detectors, verbose=True,
                   number_samples=1, chunk_size=None, seed=None,
                   raw_data=False):
    sample_rate = int(1. / static_params['delta_t'])
    
    if seed is None:
        seed = int(time.time())
    
    if chunk_size is None:
        logging.debug('Called noise generation without a chunk-size')
        if raw_data:
            logging.info(f"Generating noise without whitening")
            ngen = NoiseGenerator(int(sample_rate), 1. / sample_rate,
                                  psd_name='aLIGOZeroDetHighPower',
                                  low_frequency_cutoff=max(static_params['f_lower']-2, 0.))
        else:
            logging.info(f"Generating white noise")
            ngen = WhiteNoiseGenerator(int(sample_rate), 1. / sample_rate,
                                       psd_name='aLIGOZeroDetHighPower',
                                       low_frequency_cutoff=max(static_params['f_lower']-2, 0.))
        
        noise = {}
        for i, det in enumerate(detectors):
            tmp = ngen.generate(number_samples, seed=seed+i,
                                verbose=verbose)
            noise[det] = np.vstack([np.expand_dims(np.array(pt), axis=0) for pt in tmp])
        
        #Store noise
        file_path = file_path.format(file_number=0,
                                     number_samples=number_samples,
                                     chunk_size=number_samples)
        with h5py.File(file_path, 'w') as fp:
            data_gr = fp.create_group('data')
            for i, (key, val) in enumerate(noise.items()):
                dataset = data_gr.create_dataset(str(i), data=val)
                dataset.attrs['detector'] = key
        logging.info(f'Saved noise to {file_path}')
    else:
        logging.debug(f'Called noise generation with a chunk-size of {chunk_size}')
        base_name, extension = os.path.splitext(file_path)
        num_chunks = int(np.ceil(number_samples / float(chunk_size)))
        file_paths = [base_name + '_' + str(i) + extension for i in range(num_chunks)]
        tmp = [0]
        for i in range(num_chunks):
            tmp.append(min(tmp[-1] + chunk_size, number_samples))
        num_samps_list = [tmp[i+1]-tmp[i] for i in range(num_chunks)]
        seeds = [seed + i * len(detectors) for i in range(len(file_paths))]
        
        for i, (file_path, num_samps, cseed) in enumerate(zip(file_paths, num_samps_list, seeds)):
            print("In chunk {}/{}".format(i+1, num_chunks))
            file_path = file_path.format(file_number=i,
                                         number_samples=number_samples,
                                         chunk_size=num_chunks)
            generate_noise(file_path, static_params, detectors,
                           number_samples=num_samps, seed=cseed)

def main():
    parser = ArgumentParser()
    
    parser.add_argument('--mass-method', type=str, default='grid',
                        help="How to generate the masses. Options are `grid` and `uniform`. Default: `grid`.")
    parser.add_argument('--repeat-mass-parameters', type=int, default=1,
                        help="How often to repeat each drawn mass-pair. Can be used to generate the same mass component signal with different phases. Default: 1")
    parser.add_argument('--min-mass', type=float, default=10.,
                        help="The minimum mass to consider. Default: 10")
    parser.add_argument('--max-mass', type=float, default=50.,
                        help="The maximum mass to consider. Default: 50")
    parser.add_argument('--grid-spacing', type=float, default=0.2,
                        help="By how much to separate the grid points in a mass grid. This option is ignored for --mass-method uniform. Default: 0.2")
    parser.add_argument('--number-mass-draws', type=int, default=20000,
                        help="How many mass pairs to draw from the interval. This option is ignored if --mass-method grid. Default: 20,000")
    parser.add_argument('--approximant', type=str, default='SEOBNRv4_opt',
                        help="The waveform approximant to use. Default: SEOBNRv4_opt")
    parser.add_argument('--sample-rate', type=int, default=2048,
                        help="The sample rate at which data is generated. Default: 2048")
    parser.add_argument('--f-lower', type=float, default=20.,
                        help="The lower frequency cutoff for waveform and noise generation. Default: 20")
    parser.add_argument('--detectors', type=str, nargs='+', default=['H1'],
                        help="The detectors for which to generate the data. Default: H1")
    parser.add_argument('--tc-min', type=float, default=-0.1,
                        help="The lower limit from which time shifts are drawn from. Default: -0.1")
    parser.add_argument('--tc-max', type=float, default=0.1,
                        help="The upper limit from which time shifts are drawn from. Default: 0.1")
    parser.add_argument('--tc-mean-position', type=float, default=0.7,
                        help="The time in the 1s at which the merger time is located for tc=0. Default: 0.7")
    parser.add_argument('--random-phase', action='store_true',
                        help="Randomize the phase of each waveform.")
    parser.add_argument('--not-generate-signals', action='store_true',
                        help="Skip generating signals.")
    parser.add_argument('--seed', type=int,
                        help="The seed to use for generating signals and noise. Default: None")
    
    parser.add_argument('--not-generate-noise', action='store_true',
                        help="Skip generating the noise.")
    parser.add_argument('--number-noise-samples', type=int, default=400000,
                        help="The number of noise samples to generate. Ignored if --generate-noise is not set. Default: 400,000")
    parser.add_argument('--chunk-size', type=int,
                        help="Store up to this number of noise samples in a single file. This option is ignored if --generate-noise is not set. Default: None")
    
    parser.add_argument('--signal-output', type=str, default='signals.hdf',
                        help="The path under which the signals should be stored. Default: signals.hdf")
    parser.add_argument('--noise-output', type=str, default='{file_number}-noise-{chunk_size}-{number_samples}.hdf',
                        help="The path under which the noise files should be stored. May be a format-string, where the filled in variables are `file_number` (the current iteration number), `chunk_size` (the number of samples contained in this file) or `number_samples` (the total number of samples generated in this run). Default: {file_number}-noise-{chunk_size}-{number_samples}.hdf")
    parser.add_argument('--raw-data', action='store_true',
                        help="Generate the data without whitening it.")
    
    parser.add_argument('--store-command', type=str,
                        help="Path under which the command that generated the data will be stored. Do not set this option if you do not want to store the command.")
    parser.add_argument('--verbose', action='store_true',
                        help="Print progress updates.")
    parser.add_argument('--debug', action='store_true',
                        help="Print debugging information.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    log_level = logging.INFO if args.verbose else logging.WARN
    if args.debug:
        log_level = logging.DEBUG
    
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    logging.info(f'Program starting')
    
    #Test if files exist
    if os.path.isfile(args.signal_output) and not args.not_generate_signals and not args.force:
        raise IOError(f'File {args.signal_output} already exists. Set the flag --force to overwrite it.')
    if not args.not_generate_noise and os.path.isfile(args.noise_output) and not args.force:
        raise IOError(f'File {args.noise_output} already exists. Set the flag --force to overwrite it.')
    
    #Store the command that is being executed
    if args.store_command is not None:
        if os.path.isfile(args.store_command) and not args.force:
            raise IOError(f'File {args.store_command} already exists. Set the flag --force to overwrite it.')
        with open(args.store_command, 'w') as fp:
            fp.write(' '.join(sys.argv))
        logging.debug(f'Stored the command at {args.store_command}')
    
    sample_rate = args.sample_rate
    static_params = {'approximant': args.approximant,
                    'f_lower': args.f_lower,
                    'delta_t': 1. / sample_rate}
    logging.debug(f'Set static parameters {static_params} and sample rate {sample_rate}')
    
    if args.seed is None:
        logging.debug(f'No seed found. Generating new one.')
        args.seed = int(time.time() % 1e6)
    logging.info(f'Using seed {args.seed}')
    
    if not args.not_generate_signals:
        logging.info('Drawing parameters for signal samples')
        logging.debug(f'Min-mass: {args.min_mass}, Max-mass: {args.max_mass}')
        ##############################
        #Generate variable parameters#
        ##############################
        np.random.seed(args.seed)
        params = {}
        #Generate mass parameters
        if args.mass_method.lower() == 'grid':
            logging.debug(f'Using mass-method grid with grid-spacing {args.grid_spacing}')
            masses = np.arange(args.min_mass, args.max_mass, args.grid_spacing)
            logging.debug(f'Grid for individual masses contains {len(masses)} points')
            mass1 = []
            mass2 = []
            for m1 in masses:
                for m2 in masses:
                    if m2 > m1:
                        continue
                    mass1.append(m1)
                    mass2.append(m2)
            logging.debug(f'Found {len(mass1)} unique combinations with m1 >= m2')
        elif args.mass_method.lower() == 'uniform':
            logging.debug(f'Using mass-method uniform and drawing {args.number_mass_draws} combinations')
            masses = np.random.uniform(args.min_mass, args.max_mass,
                                    size=(2, args.number_mass_draws))
            masses.sort(axis=0)
            mass2 = list(masses[0])
            mass1 = list(masses[1])
        else:
            raise RuntimeError(f'Unrecognized mass-method {args.mass_method}.')
        params['mass1'] = []
        params['mass2'] = []
        logging.debug(f'Repeating the mass signals {args.repeat_mass_parameters} times')
        for _ in range(args.repeat_mass_parameters):
            params['mass1'].extend(mass1)
            params['mass2'].extend(mass2)
        
        #Generate phases
        if args.random_phase:
            logging.info('Using random coalescence phase')
            params['coa_phase'] = list(np.random.uniform(0, 2 * np.pi,
                                                        size=len(params['mass1'])))
        
        #Generate tc
        logging.debug(f'Drawing a random merger time from {args.tc_min} to {args.tc_max} for all signals')
        params['tc'] = list(np.random.uniform(args.tc_min, args.tc_max,
                                            size=len(params['mass1'])))
        
        generate_signals(args.signal_output, params, static_params,
                        args.detectors,
                        tc_mean_position=args.tc_mean_position,
                        verbose=args.verbose,
                        raw_data=args.raw_data)
    
    if not args.not_generate_noise:
        generate_noise(args.noise_output, static_params, args.detectors,
                       number_samples=args.number_noise_samples,
                       chunk_size=args.chunk_size, seed=args.seed,
                       verbose=args.verbose,
                       raw_data=args.raw_data)
    logging.info('Finished')

if __name__ == "__main__":
    main()
