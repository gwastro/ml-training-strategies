#! /usr/bin/env python

#Basic imports
import argparse
import numpy as np
import h5py
import os
import logging

#PyCBC imports
from pycbc.types import TimeSeries

#BnsLib imports
from BnsLib.testing import *
from BnsLib.utils import progress_tracker

SEC_PER_MONTH = 30 * 24 * 60 * 60

def get_start_time(fn):
    start = int(fn.split('-')[1])
    if start == 0:
        return start
    else:
        return start + 0.1

def load_data(path, epoch_offset=0., verbose=False, delta_t=0.1,
              data_activation='linear', target_activation='softmax'):
    if not os.path.isdir(path):
        raise ValueError('Path {} for loading data not found.'.format(path))
    files = os.listdir(path)
    out = []
    if verbose:
        bar = progress_tracker(len(files), name='Loading data')
    for fn in files:
        tmp_path = os.path.join(path, fn)
        if not os.path.isfile(tmp_path):
            if verbose:
                bar.iterate()
            continue
        try:
            with h5py.File(tmp_path, 'r') as fp:
                data = fp['data'][()]
                epoch = get_start_time(fn) + epoch_offset
                if data_activation == 'linear':
                    if target_activation == 'linear':
                        out.append(TimeSeries(data.T[0] - data.T[1],
                                              delta_t=delta_t,
                                              epoch=epoch))
                    elif target_activation == 'softmax':
                        exp0 = np.exp(data.T[0])
                        exp1 = np.exp(data.T[1])
                        ts = TimeSeries(exp0 / (exp0 + exp1),
                                        delta_t=delta_t,
                                        epoch=epoch)
                        out.append(ts)
                    else:
                        raise RuntimeError(f'Unrecognized target_activation {target_activation}.')
                elif data_activation == 'softmax':
                    if target_activation == 'softmax':
                        out.append(TimeSeries(data.T[0], delta_t=delta_t,
                                              epoch=epoch))
                    elif target_activation == 'linear':
                        raise ValueError('Cannot use target activation `linear` if data was generated with a softmax activation.')
                    else:
                        raise RuntimeError(f'Unrecognized target_activation {target_activation}.')
                else:
                    raise RuntimeError(f'Unrecognized data_activation {data_activation}.')
                if verbose:
                    bar.iterate()
        except:
            if verbose:
                bar.iterate()
            continue
    out = sorted(out, key=lambda ts: ts.start_time)
    return out

def assemble_time_series(ts_list):
    start = float(min(ts_list, key=lambda ts: ts.start_time).start_time)
    end = float(max(ts_list, key=lambda ts: ts.end_time).end_time)
    dts = {ts.delta_t for ts in ts_list}
    assert len(dts) == 1
    dt = dts.pop()
    ret = TimeSeries(np.zeros(int((end - start) / dt)+1),
                     delta_t=dt, epoch=start)
    assert int((end - float(ret.end_time)) / dt) == 0, 'Got end {} and end_time {} with {} samples difference.'.format(end, float(ret.end_time), int(abs(end - float(ret.end_time)) // dt))
    for ts in ts_list:
        start_idx = int(float(ts.start_time - ret.start_time) / dt)
        end_idx = start_idx + len(ts)
        ret.data[start_idx:end_idx] = ts.data[:]
    return ret

def custom_get_event_list_from_triggers(triggers, cluster_boundaries,
                                        verbose=False):
    events = []
    sort_idxs = np.argsort(triggers[0])
    sorted_triggers = (triggers.T[sort_idxs]).T
    if verbose:
        bar = progress_tracker(len(cluster_boundaries),
                               name='Calculating events')
    for cstart, cend in cluster_boundaries:
        sidx = np.searchsorted(sorted_triggers[0], cstart, side='left')
        eidx = np.searchsorted(sorted_triggers[0], cend, side='right')
        if sidx == eidx:
            logging.warn(f'Got a cluster that lies between two samples. Cluster: {(cstart, cend)}, Indices: {(sidx, eidx)}')
            continue
        idx = sidx + np.argmax(sorted_triggers[1][sidx:eidx])
        events.append((sorted_triggers[0][idx], sorted_triggers[1][idx]))
        if verbose:
            bar.iterate()
    return events

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--trigger-threshold', type=float, default=0.1,
                        help="The threshold value to determine triggers.")
    parser.add_argument('--cluster-tolerance', type=float, default=0.2,
                        help="The maximum distance (in seconds) between a trigger and the cluster boundaries for both to be clustered together.")
    parser.add_argument('--event-tolerance', type=float, default=0.3,
                        help="The maximum time (in seconds) between an event and an injection for them to be considered of the same origin.")
    parser.add_argument('--injection-file', required=True, type=str,
                        help="Path to the file containing the injections for this data.")
    parser.add_argument('--data-dir', type=str,
                        help="Path to the directory containing the output of the network. All files in this directory will be loaded.")
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing output files.')
    parser.add_argument('--delta-t', type=float, default=0.1,
                        help="The (actual) time (in seconds) between two slices. (By actual nsamples / sample_rate is meant)")
    parser.add_argument('--start-time-offset', type=float, default=0.75,
                        help="The time from the start of each processed window to the central point of the interval in which the merger time is varied.")
    parser.add_argument('--duration', type=float,
                        help="The duration of the data that is analyzed. Only required if triggers or events are loaded.")
    parser.add_argument('--test-data-activation', choices=['linear', 'softmax'], default='linear',
                        help="Which activation function was used to create the output. Default: `linear`")
    parser.add_argument('--ranking-statistic', choices=['softmax', 'linear'], default='softmax',
                        help="How should the output of the network be used to rate events? (This option may only be set to `linear`, if --test-data-activation is set to `linear`)")
    parser.add_argument('--verbose', action='store_true',
                        help="Print status updates.")
    parser.add_argument('--trigger-file-name', type=str, default='triggers.hdf',
                        help="The name of the trigger file that is stored in the --data-dir.")
    parser.add_argument('--event-file-name', type=str, default='events.hdf',
                        help="The name of the event file that is stored in the --data-dir.")
    parser.add_argument('--stats-file-name', type=str, default='statistics.hdf',
                        help="The name of the statistics file that is stored in the --data-dir.")
    parser.add_argument('--load-triggers', type=str,
                        help="Start analysis from the given trigger-file. (Argument must be the path to the file)")
    parser.add_argument('--load-events', type=str,
                        help="Start analysis from the given event-file. (Argument must be the path to the file)")
    
    args = parser.parse_args()
    
    log_level = logging.INFO if args.verbose else logging.WARN
    
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')
    
    logging.info('Started evaluation process')
    
    if args.ranking_statistic == 'linear':
        if args.test_data_activation != 'linear':
            raise ValueError(f'Can only use a linear ranking statistic if the test data was produced from a linear activation.')
    
    if args.load_triggers is None and args.load_events is None:
        if args.data_dir is None:
            raise ValueError(f'Must provide a directory from which to load the data when no triggers or events are loaded.')
        logging.info('Starting to load data')
        data = load_data(args.data_dir, epoch_offset=args.start_time_offset,
                        verbose=args.verbose, delta_t=args.delta_t,
                        data_activation=args.test_data_activation,
                        target_activation=args.ranking_statistic)
        logging.info(f'Loading complete. Loaded {len(data)} files.')
        logging.info('Assembling total timeseries')
        ts = assemble_time_series(data)
        logging.info('Assembling complete')
        if args.duration is None:
            args.duration = ts.duration
    else:
        if args.data_dir is None:
            args.data_dir = '.'
        if args.duration is None:
            raise ValueError(f'Duration required if data is not loaded.')
    
    if args.load_triggers is None and args.load_events is None:
        #Calculate triggers
        logging.info('Calculating triggers')
        triggers = get_triggers(ts, args.trigger_threshold)
        logging.info('Found {} triggers'.format(len(triggers[0])))

        #Write triggers to file
        trigger_path = os.path.join(args.data_dir,
                                    args.trigger_file_name)
        if os.path.isfile(trigger_path):
            if not args.force:
                msg = 'Cannot overwrite trigger file at {}. Set the flag '
                msg += '--force if you want to overwrite the file anyways.'
                msg = msg.format(trigger_path)
                raise IOError(msg)
        with h5py.File(trigger_path, 'w') as fp:
            fp.create_dataset('data', data=triggers[0])
            fp.create_dataset('trigger_values', data=triggers[1])
        logging.info("Wrote triggers to {}.".format(trigger_path))
    elif args.load_triggers is not None:
        with h5py.File(args.load_triggers, 'r') as fp:
            triggers = np.vstack([fp['data'][()],
                                  fp['trigger_values']])
        logging.info(f"Loaded {len(triggers[0])} triggers from {args.load_triggers}")
    
    if args.load_events is None:
        #Calculate events
        logging.info('Calculating cluster boundaries')
        cb = get_cluster_boundaries(triggers,
                                    boundarie_time=args.cluster_tolerance)
        logging.info('Found {} clusters.'.format(len(cb)))
        
        logging.info('Calculating events')
        events = custom_get_event_list_from_triggers(triggers, cb,
                                                    verbose=args.verbose)
        logging.info('Found {} events.'.format(len(events)))
        
        #Write events to file
        event_path = os.path.join(args.data_dir,
                                args.event_file_name)
        if os.path.isfile(event_path):
            if not args.force:
                msg = 'Cannot overwrite event file at {}. Set the flag '
                msg += '--force if you want to overwrite the file anyways.'
                msg = msg.format(event_path)
                raise IOError(msg)
        with h5py.File(event_path, 'w') as fp:
            fp.create_dataset('times', data=np.array(get_event_times(events)))
            fp.create_dataset('values', data=np.array([event[1] for event in events]))
        logging.info("Wrote events to {}.".format(event_path))
    else:
        with h5py.File(args.load_events, 'r') as fp:
            events = np.vstack([fp['times'][()],
                                fp['values'][()]])
        events = [tuple(pt) for pt in events.T]
    
    #Read injection file
    with h5py.File(args.injection_file, 'r') as fp:
        inj_times = fp['tc'][()]
        inj_idxs = np.arange(len(inj_times))
        mass1 = fp['mass1'][()]
        mass2 = fp['mass2'][()]
        dist = fp['distance'][()]
    
    #Calculate sensitivities and false-alarm rates
    logging.info('Splitting all events into true- and false-positives.')
    tp, fp = split_true_and_false_positives(events, inj_times,
                                            tolerance=args.event_tolerance)
    logging.info(f'Found {len(tp)} true and {len(fp)} false positives')
    logging.info(f'Sorting true and false positives by their ranking statistic')
    tp = np.array(sorted(tp, key=lambda inp: inp[1]))
    fp = np.array(sorted(fp, key=lambda inp: inp[1]))

    tptimes, tpvals = tp.T

    rank = []
    far = []
    sens_frac = []
    tidxs = []
    if args.verbose:
        bar = progress_tracker(len(fp), name='Calculating ranking steps and false-alarm rate')
    for i, event in enumerate(fp):
        t, val = event
        if len(rank) == 0:
            rank.append(val)
            far.append((len(fp) - i) / args.duration * SEC_PER_MONTH)
            tidx = np.searchsorted(tpvals, val, side='right')
            sens_frac.append(1 - float(tidx) / len(tp))
            tidxs.append(tidx)
            if args.verbose:
                bar.iterate()
            continue
        if val < rank[-1]:
            raise RuntimeError
        if rank[-1] == val:
            far[-1] = (len(fp) - i - 1) / args.duration * SEC_PER_MONTH
        else:
            rank.append(val)
            far.append((len(fp) - i - 1) / args.duration * SEC_PER_MONTH)
            tidx = np.searchsorted(tpvals, val, side='right')
            sens_frac.append(1 - float(tidx) / len(tp))
            tidxs.append(tidx)
        if args.verbose:
            bar.iterate()

    logging.info(f'Getting base-level found and missed indices')
    _, base_fidxs = get_closest_injection_times(inj_times, tptimes,
                                                        return_indices=True)

    logging.info(f'Starting to calculate sensitive volumes')
    #Calculations based on pycbc.sensitivity.volume_montecarlo
    max_distance = dist.max()
    mc_vtot = (4. / 3.) * np.pi * max_distance**3.
    Ninj = len(dist)
    mc_norm = float(Ninj)
    mc_prefactor = mc_vtot / mc_norm
    sens_vol = []
    sens_vol_err = []
    sens_dist = []
    if args.verbose:
        bar = progress_tracker(len(tidxs), name='Calculating sensitive volume')
    for idx in tidxs:
        mc_sum = len(base_fidxs) - idx
        mc_sample_variance = mc_sum / Ninj - (mc_sum / Ninj) ** 2
        vol = mc_prefactor * mc_sum
        vol_err = mc_prefactor * (Ninj * mc_sample_variance) ** 0.5
        
        sens_vol.append(vol)
        sens_vol_err.append(vol_err)
        rad = (3 * vol / (4 * np.pi))**(1. / 3.)
        sens_dist.append(rad)
        if args.verbose:
            bar.iterate()
    
    #Write FAR and sensitivity to file
    stat_path = os.path.join(args.data_dir, args.stats_file_name)
    if os.path.isfile(stat_path):
        if not args.force:
            msg = 'Cannot overwrite statistics file at {}. Set the flag '
            msg += '--force if you want to overwrite the file anyways.'
            msg = msg.format(stat_path)
            raise IOError(msg)
    with h5py.File(stat_path, 'w') as fp:
        fp.create_dataset('ranking', data=np.array(rank))
        fp.create_dataset('far', data=np.array(far))
        fp.create_dataset('sens-frac', data=np.array(sens_frac))
        fp.create_dataset('sens-dist', data=np.array(sens_dist))
        fp.create_dataset('sens-vol', data=np.array(sens_vol))
        fp.create_dataset('sens-vol-err', data=np.array(sens_vol_err))
    logging.info("Wrote statistics to {}.".format(stat_path))
    
    logging.info('Finished')
    return

if __name__ == "__main__":
    main()
