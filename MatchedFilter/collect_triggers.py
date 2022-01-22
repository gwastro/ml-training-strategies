#!/usr/bin/env python

from argparse import ArgumentParser
import h5py
import numpy as np
import os
import logging

from bnslib import inverse_string_format, progress_tracker

def main():
    parser = ArgumentParser()
    
    parser.add_argument('--dir', type=str, required=True,
                        help="The directory that contains the output files of the pycbc_inspiral analysis.")
    parser.add_argument('--output',type=str, required=True,
                        help="Path at which the output file will be stored.")
    parser.add_argument('--file-name', type=str, default='{start}-{end}-out.hdf',
                        help="Format string that was used to name the files. Set to `{name}` to consider all files in the --dir.")
    parser.add_argument('--start-time', type=float, default=0,
                        help="Time after which triggers are considered.")
    parser.add_argument('--end-time', type=float, default=2592000,
                        help="Time after which triggers are ignored.")
    parser.add_argument('--threshold', type=float, default=5,
                        help="Consider triggers only if their SNR exceeds this value.")
    parser.add_argument('--verbose', action='store_true',
                        help="Print status updates.")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing files.")
    
    args = parser.parse_args()
    
    log_level = logging.INFO if args.verbose else logging.WARN
    
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    if os.path.isfile(args.output) and not args.force:
        raise IOError(f'File {args.output} already exists. Set the flag --force to overwrite it.')
    
    paths = []
    logging.info('Looking for files')
    for fn in os.listdir(args.dir):
        if inverse_string_format(fn, args.file_name) is None:
            continue
        paths.append(os.path.join(args.dir, fn))
    logging.info(f'Found {len(paths)} files matching the file name {args.file_name}')
    
    trig_times = []
    trig_vals = []
    if args.verbose:
        bar = progress_tracker(len(paths), name='Loading triggers')
    for path in paths:
        with h5py.File(path, 'r') as fp:
            vals = fp['H1/snr'][()]
            idxs = np.where(vals > args.threshold)[0]
            if len(idxs) < 1:
                bar.iterate()
                continue
            vals = vals[idxs]
            times = fp['H1/end_time'][idxs]
            idxs = np.where(np.logical_and(args.start_time <= times,
                                           times <= args.end_time))[0]
            if len(idxs) < 1:
                bar.iterate()
                continue
            trig_times.append(times[idxs])
            trig_vals.append(vals[idxs])
        bar.iterate()
    
    trig_times = np.concatenate(trig_times)
    trig_vals = np.concatenate(trig_vals)
    
    logging.info(f'Found {len(trig_times)} eligable triggers')
    
    logging.info(f'Sorting triggers by their times of occurence')
    
    idxs = trig_times.argsort()
    
    logging.info(f'Saving triggers to {args.output}')
    mode = 'w' if args.force else 'x'
    with h5py.File(args.output, mode) as fp:
        fp.create_dataset('data', data=trig_times[idxs])
        fp.create_dataset('trigger_values', data=trig_vals[idxs])
    
    logging.info('Finished')
    return

if __name__ == "__main__":
    main()
