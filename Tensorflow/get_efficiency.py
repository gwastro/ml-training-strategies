#!/usr/bin/env python

from argparse import ArgumentParser
import os
import numpy as np
from tensorflow import keras
from generator import get_generator
from reg_loss import reg_loss
from BnsLib.utils import inverse_string_format
from BnsLib.network.callbacks import SensitivityEstimator

def get_model_paths(base_dir, model_names=None, name_format=['{}'],
                    sort_by=None, dynamic_types=True):
    if model_names is None:
        model_names = os.listdir(base_dir)
    files = []
    for model_name in model_names:
        for form in name_format:
            inv_form = inverse_string_format(model_name, form,
                                             dynamic_types=dynamic_types)
            if inv_form is not None:
                files.append((model_name, inv_form))
                break
    if sort_by is not None:
        files = sorted(files, key=lambda inp: inp[1].get(sort_by, np.inf))
    return [os.path.join(base_dir, pt[0]) for pt in files]

def main():
    parser = ArgumentParser()
    
    parser.add_argument('--data-dir', type=str,
                        help="Path to the directory where the data is stored.")
    parser.add_argument('--base-dir', type=str,
                        help="Path to the directory in which the different models are stored.")
    parser.add_argument('--model-names', type=str, nargs='*',
                        help="Name(s) of the models to load. They have to be part of the --base-dir. If this option is not set, all models in --base-dir that fit any --model-name-format are loaded.")
    parser.add_argument('--model-name-format', type=str, nargs='+', default=['{name}'],
                        help="One or multiple formats for the file/directory names of the models. Defaults to: `{name}` (accepts any string)")
    parser.add_argument('--sort-by', type=str,
                        help="Set a key to sort the networks by. This option is ignored if no matching variable is found in --model-name-format.")
    parser.add_argument('--use-dynamic-types', action='store_true',
                        help="Automatically infer the type of variables in the format-string.")
    parser.add_argument('--fap', type=float, default=1e-4,
                        help="The false-alarm probability (FAP) at which to produce the efficiency curves.")
    parser.add_argument('--snrs', type=float, nargs='+', default=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
                        help="The signal-to-noise-ratios (SNRs) at which to generate the efficiency curves.")
    parser.add_argument('--output', type=str, default='sensitivity_estimate.csv',
                        help="The path at which the output-file should be stored.")
    parser.add_argument('--remove-softmax', action='store_true',
                        help="Replace the final softmax activation by a linear one.")
    parser.add_argument('--verbose', action='store_true',
                        help="Print status updates.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    if args.base_dir is None:
        args.base_dir = '.'
    if args.data_dir is None:
        args.data_dir = './data'
    
    if os.path.isfile(args.output) and not args.force:
        raise IOError(f'File {args.output} already exists. Set the flag --force to overwrite it.')
    
    model_paths = get_model_paths(args.base_dir,
                                  model_names=args.model_names,
                                  name_format=args.model_name_format,
                                  sort_by=args.sort_by)
    
    sens_sig_generator = get_generator(args.data_dir, 'thr', n_noise=0,
                                       noise_per_signal=1,
                                       shuffle=True)
    sens_noi_generator = get_generator(args.data_dir, 'thr', n_signals=0,
                                       shuffle=True,
                                       use_signal_files=False)
    
    if args.remove_softmax:
        transform = lambda inp: inp.T[0] - inp.T[1]
    else:
        transform = lambda inp: inp.T[0]
    sens_writer = SensitivityEstimator(sens_sig_generator,
                                       threshold=sens_noi_generator,
                                       file_path=args.output,
                                       transform_function=transform,
                                       snrs=args.snrs,
                                       verbose=int(args.verbose),
                                       fap=args.fap)
    #Setup header for the file
    sens_writer.on_train_begin()
    
    for i, mp in enumerate(model_paths):
        model = keras.models.load_model(mp,
                                        custom_objects={'reg_loss': reg_loss})
        if args.remove_softmax:
            model.layers[-1].activation = keras.activations.linear
            model.compile(loss=reg_loss)
        sens_writer.model = model
        sens_writer.on_epoch_end(i)
    return

if __name__ == "__main__":
    main()
