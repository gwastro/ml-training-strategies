#! /usr/bin/env python
import argparse
from tensorflow import keras
import numpy as np
import h5py
import os
from reg_loss import reg_loss
import logging
import sys

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--network', required=True, type=str,
                        help="The path to the network that should be evaluated.")
    parser.add_argument('--output-dir', required=True, type=str,
                        help="The path to a directory in which the output files will be stored.")
    parser.add_argument('--input-dir', required=True, type=str,
                        help="The path to a directory from which the files will be read.")
    parser.add_argument('--store-command', action='store_true',
                        help="Put a text file with name `command.txt` into the output directory. The text file contains the command line content.")
    parser.add_argument('--create-dirs', action='store_true',
                        help="Create missing output directories.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files in the output directory.")
    parser.add_argument('--verbose', action='store_true',
                        help="Output information on status.")
    parser.add_argument('--debug', action='store_true',
                        help="Print debugging information.")
    parser.add_argument('--remove-softmax', action='store_true',
                        help="Remove the final softmax activation from the network.")
    
    args = parser.parse_args()
    
    log_level = logging.INFO if args.verbose else logging.WARN
    if args.debug:
        log_level = logging.DEBUG
    
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    if args.store_command:
        if os.path.isfile(os.path.join(args.output_dir, 'command.txt')) and not args.force:
            raise IOError(f'File {os.path.join(args.output_dir, "command.txt")} already exists. Set the flag --force to overwrite it.')
    
    if not os.path.isdir(args.input_dir):
        raise ValueError('Unknown input directory {}.'.format(args.input_dir))
    
    if not os.path.isdir(args.output_dir):
        if not args.create_dirs:
            raise ValueError('Missing output directory {}.'.format(args.output_dir))
        os.makedirs(args.output_dir)
    
    if args.input_dir == args.output_dir:
        raise ValueError('Cannot use the input directory as the output directory.')
    
    model = keras.models.load_model(args.network,
                                    custom_objects={'reg_loss': reg_loss})
    if args.remove_softmax:
        logging.debug("Removing the final softmax")
        model.layers[-1].activation = keras.activations.linear
        model.compile(loss=reg_loss)
        logging.info("Successfully removed final softmax")
    
    file_list = os.listdir(args.input_dir)
    l = str(len(file_list))
    
    for i, fn in enumerate(file_list):
        num_str = str(i).ljust(len(l))
        inpath = os.path.join(args.input_dir, fn)
        outpath = os.path.join(args.output_dir, fn)
        if os.path.isfile(outpath) and not args.force:
            raise ValueError('File {} already exists. Use flag --force to overwrite it.'.format(outpath))
        with h5py.File(inpath, 'r') as fp:
            data = fp['H1/data'][()]
        
        logging.info("{}/{}: Loaded data from {}.".format(num_str, l, inpath))
        logging.debug("{}/{}: Evaluating model on data.".format(num_str, l))
        if args.verbose:
            res = model.predict(data, verbose=1)
        else:
            res = model.predict(data, verbose=0)
        
        with h5py.File(outpath, 'w') as fp:
            fp.create_dataset('data', data=res)
        logging.debug("{}/{}: Wrote data to {}.".format(num_str, l, outpath))
    
    if args.store_command:
        with open(os.path.join(args.output_dir, 'command.txt'), 'w') as fp:
            fp.write(' '.join(sys.argv))

if __name__ == "__main__":
    main()
