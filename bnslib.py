import numpy as np
from queue import Queue
import multiprocessing as mp
import datetime
import sys
import time
import warnings
from functools import wraps
from collections import OrderedDict

from pycbc.detector import Detector
from pycbc.sensitivity import volume_montecarlo
from pycbc.waveform import get_td_waveform, get_fd_waveform
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.noise import noise_from_string
from pycbc.psd import from_string, interpolate, inverse_spectrum_truncation
from pycbc.psd import aLIGOZeroDetHighPower as aPSD
from pycbc.filter import sigma

def optimal_snr(signal, psd='aLIGOZeroDetHighPower',
                low_freq_cutoff=None, high_freq_cutoff=None):
    """Calculate the optimal signal-to-noise ratio for a given signal.
    
    Arguments
    ---------
    signal : pycbc.TimeSeries or pycbc.FrequencySeries
        The signal of which to calculate the signal-to-noise ratio.
    psd : {str or None or pycbc.FrequencySeries, 'aLIGOZeroDetHighPower}
        A power spectral density to use for the noise-model. If set to a
        string, a power spectrum will be generated using
        pycbc.psd.from_string. If set to None, no noise will be assumed.
        If a frequency series is given, the user has to make sure that
        the delta_f and length match the signal.
    low_freq_cutoff : {float or None, None}
        The lowest frequency to consider. If a value is given, the power
        spectrum will be generated with a lower frequency cutoff 2 below
        the given one. (0 at minimum)
    high_freq_cutoff : {float or None, None}
        The highest frequency to consider.
    
    Returns
    -------
    float
        The optimal signal-to-noise ratio given the signal and the noise
        curve (power spectrum).
    """
    if psd is not None:
        if isinstance(psd, str):
            df = signal.delta_f
            if isinstance(signal, TimeSeries):
                flen = len(signal) // 2 + 1
            elif isinstance(signal, FrequencySeries):
                flen = len(signal)
            psd_low = 0. if low_freq_cutoff is None else max(low_freq_cutoff - 2., 0.)
            psd = from_string(psd, length=flen, delta_f=df,
                              low_freq_cutoff=psd_low)
    return sigma(signal, psd=psd, low_frequency_cutoff=low_freq_cutoff,
                 high_frequency_cutoff=high_freq_cutoff)

def whiten(strain_list, low_freq_cutoff=20., max_filter_duration=4.,
           psd=None):
    """Returns the data whitened by the PSD.
    
    Arguments
    ---------
    strain_list : pycbc.TimeSeries or list of pycbc.TimeSeries
        The data that should be whitened.
    low_freq_cutoff : {float, 20.}
        The lowest frequency that is considered during calculations. It
        must be >= than the lowest frequency where the PSD is not zero.
        Unit: hertz
    max_filter_duration : {float, 4.}
        The duration to which the PSD is truncated to in the
        time-domain. The amount of time is removed from both the
        beginning and end of the input data to avoid wrap-around errors.
        Unit: seconds
    psd : {None or str or pycbc.FrequencySeries, None}
        The PSD that should be used to whiten the data. If set to None
        the pycbc.psd.aLIGOZeroDetHighPower PSD will be used. If a PSD
        is provided which does not fit the delta_f of the data, it will
        be interpolated to fit. If a string is provided, it will be
        assumed to be known to PyCBC.
    
    Returns
    -------
    pycbc.TimeSeries or list of pycbc.TimeSeries
        Depending on the input type it will return a list of TimeSeries
        or a single TimeSeries. The data contained in this time series
        is the whitened input data, where the inital and final seconds
        as specified by max_filter_duration are removed.
    """
    org_type = type(strain_list)
    if not org_type == list:
        strain_list = [strain_list]
    
    ret = []
    for strain in strain_list:
        df = strain.delta_f
        f_len = int(len(strain) / 2) + 1
        if psd is None:
            psd = aPSD(length=f_len,
                       delta_f=df,
                       low_freq_cutoff=low_freq_cutoff-2.)
        elif isinstance(psd, str):
            psd = from_string(psd,
                              length=f_len,
                              delta_f=df,
                              low_freq_cutoff=low_freq_cutoff-2.)
        else:
            if not len(psd) == f_len:
                msg = 'Length of PSD does not match data.'
                raise ValueError(msg)
            elif not psd.delta_f == df:
                psd = interpolate(psd, df)
        max_filter_len = int(max_filter_duration * strain.sample_rate) #Cut out the beginning and end
        psd = inverse_spectrum_truncation(psd,
                                          max_filter_len=max_filter_len,
                                          low_frequency_cutoff=low_freq_cutoff,
                                          trunc_method='hann')
        f_strain = strain.to_frequencyseries()
        kmin = int(low_freq_cutoff / df)
        f_strain.data[:kmin] = 0
        f_strain.data[-1] = 0
        f_strain.data[kmin:] /= psd[kmin:] ** 0.5
        strain = f_strain.to_timeseries()
        ret.append(strain[max_filter_len:len(strain)-max_filter_len])
    
    if not org_type == list:
        return(ret[0])
    else:
        return(ret)

def list_length(inp):
    """Returns the length of a list or 1, if the input is not a list.
    
    Arguments
    ---------
    inp : list or other
        The input.
    
    Returns
    -------
    int
        The length of the input, if the input is a list. Otherwise
        returns 1.
    
    Notes
    -----
    -A usecase for this function is to homologize function inputs. If
     the function is meant to operate on lists but can also accept a
     single instance, this function will give the length of the list the
     function needs to create. (Useful in combination with the function
     input_to_list)
    """
    if isinstance(inp, list):
        return len(inp)
    else:
        return 1

def input_to_list(inp, length=None):
    """Convert the input to a list of a given length.
    If the input is not a list, a list of the given length will be
    created. The contents of this list are all the same input value.
    
    Arguments
    ---------
    inp : list or other
        The input that should be turned into a list.
    length : {int or None, None}
        The length of the output list. If set to None this function will
        call list_length to determine the length of the list.
    
    Returns
    -------
    list
        Either returns the input, when the input is a list of matching
        length or a list of the wanted length filled with the input.
    """
    if length is None:
        length = list_length(inp)
    if isinstance(inp, list):
        if len(inp) != length:
            msg = f'Length of list {len(inp)} does not match the length'
            msg += f' requirement {length}.'
            raise ValueError(msg)
        else:
            return inp
    else:
        return [inp] * length

SECONDS_PER_MONTH = 60 * 60 * 24 * 30

def get_trigger_times(ts, thresh):
    """Generates an array of times that exceed the given threshold.
    
    Arguments
    ---------
    ts : pycbc.TimeSeries
        The time series to which a threshold should be applied.
    thresh : float
        The threshold value
    
    Returns
    -------
    numpy.array:
        An array of sample times, where the threshold is exceeded.
    """
    idxs = np.where(ts > thresh)[0]
    if len(idxs) == 0:
        return np.array([])
    else:
        return np.array(ts.sample_times[idxs])

def get_triggers(ts, thresh):
    """Generates an array of times that exceed the given threshold.
    
    Arguments
    ---------
    ts : pycbc.TimeSeries
        The time series to which a threshold should be applied.
    thresh : float
        The threshold value
    
    Returns
    -------
    numpy.array:
        A 2D array. The row with index 0 contains the sample times where
        the threshold was exceeded, the row with index 1 contains the
        according values.
    """
    idxs = np.where(ts > thresh)[0]
    if len(idxs) == 0:
        return np.array([[], []])
    else:
        ret = np.zeros((2, len(idxs)))
        ret[0] = np.array(ts.sample_times[idxs])
        ret[1] = np.array(ts.data[idxs])
        return ret

def get_cluster_boundaries(triggers, boundarie_time=1.):
    """A basic clustering algorithm that generates a list start and end
    times for every cluster.
    
    Arguments
    ---------
    triggers : iterable of floats or 2D array
        A list or array containing the times of a time series that
        exceed a given threshold. (As returned by get_trigger_times or
        get_triggers)
    boundarie_time : {float, 1.}
        A time in seconds around the cluster boundaries that may not
        contain any triggers for the cluster to be complete.
    
    Returns
    -------
    list of list of float:
        Returns a list that contains the boundarie times of all
        clusters. As such each entry is a list of length 2. The first
        of which is the inital time of the cluster, the second is the
        final time of the cluster.
    
    Note
    ----
    This is a very basic clustering algorithm that simply expands the
    boundaries of all clusters until there are no triggers within an
    accepted range.
    """
    if np.ndim(triggers) == 1:
        trigger_times = triggers
    elif np.ndim(triggers) == 2:
        trigger_times = triggers[0]
    else:
        raise RuntimeError
    i = 0
    clusters = []
    current_cluster = []
    while i < len(trigger_times):
        if len(current_cluster) == 0:
            current_cluster.append(trigger_times[i])
        elif len(current_cluster) == 1:
            if trigger_times[i] - current_cluster[0] < boundarie_time:
                current_cluster.append(trigger_times[i])
            else:
                current_cluster.append(current_cluster[0])
                clusters.append(current_cluster)
                current_cluster = [trigger_times[i]]
        elif len(current_cluster) == 2:
            if trigger_times[i] - current_cluster[1] < boundarie_time:
                current_cluster[1] = trigger_times[i]
            else:
                clusters.append(current_cluster)
                current_cluster = [trigger_times[i]]
        i += 1
    if len(current_cluster) == 2:
        clusters.append(current_cluster)
    elif len(current_cluster) == 1:
        clusters.append([current_cluster[0], current_cluster[0]])
    return clusters

def get_event_list(ts, cluster_boundaries):
    """Turns a list of clusters into events.
    
    Arguments
    ---------
    ts : pycbc.TimeSeries
        The time series the clusters are derived from.
    cluster_boundaries : list of list of float
        A list of cluster boundaries as returned by
        get_cluster_boundaries.
    
    Returns
    -------
    list of tuples of float:
        Returns a list of events. A event is a tuple of size two. The
        first entry is the time of the event, the second is the value
        of the time series at the corresponding event.
    
    Notes
    -----
    -Each event corresponds to a cluster. The algorithm takes the
     time and value of the maximum of the time series within each
     cluster as event.
    """
    events = []
    samp_times = np.array(ts.sample_times)
    for cstart, cend in cluster_boundaries:
        start_idx = int(float(cstart - ts.start_time) / ts.delta_t)
        end_idx = int(float(cend - ts.start_time) / ts.delta_t)
        idx = start_idx + np.argmax(ts[start_idx:end_idx+1])
        events.append((samp_times[idx], ts[idx]))
    return events

def get_event_list_from_triggers(triggers, cluster_boundaries):
    events = []
    sort_idxs = np.argsort(triggers[0])
    sorted_triggers = (triggers.T[sort_idxs]).T
    for cstart, cend in cluster_boundaries:
        sidx = np.searchsorted(sorted_triggers[0], cstart, side='left')
        eidx = np.searchsorted(sorted_triggers[0], cend, side='right')
        if sidx == eidx:
            continue
        idx = sidx + np.argmax(sorted_triggers[1][sidx:eidx])
        events.append((sorted_triggers[0][idx], sorted_triggers[1][idx]))
    return events

def events_above_threshold(event_list, thresh):
    """Filter events by a threshold on their value.
    
    Arguments
    ---------
    event_list : list of tuples of float
        A list of events as returned by get_event_list.
    thresh : float
        A threshold value to filter events.
    
    Returns
    -------
    list of tuples of float
        A list of events that exceed the given threshold.
    """
    ret = []
    for event in event_list:
        if event[1] > thresh:
            ret.append(event)
    return ret

def get_false_positives(event_list, injection_times, tolerance=3.):
    """Find a list of falsely identified events.
    
    Arguments
    ---------
    event_list : list of tuple of float
        A list of events as returned by get_event_list.
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    tolerance : {float, 3.}
        The maximum time in seconds an injection time may be away from
        an event time to be counted as a true positive.
    
    Returns
    -------
    list of tuples of float
        A list of events that were falsely identified as events.
    """
    ret = []
    for event in event_list:
        if np.min(np.abs(injection_times - event[0])) > tolerance:
            ret.append(event)
    return ret

def get_true_positives(event_list, injection_times, tolerance=3.):
    """Find a list of correctly identified events.
    
    Arguments
    ---------
    event_list : list of tuple of float
        A list of events as returned by get_event_list.
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    tolerance : {float, 3.}
        The maximum time in seconds an injection time may be away from
        an event time to be counted as a true positive.
    
    Returns
    -------
    list of tuples of float
        A list of events that were correctly identified as events.
    """
    ret = []
    for event in event_list:
        if np.min(np.abs(injection_times - event[0])) <= tolerance:
            ret.append(event)
    return ret

def split_true_and_false_positives(event_list, injection_times,
                                   tolerance=3., assume_sorted=False,
                                   workers=0):
    """Find a list of correctly identified events.
    
    Arguments
    ---------
    event_list : list of tuple of float
        A list of events as returned by get_event_list.
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    tolerance : {float, 3.}
        The maximum time in seconds an injection time may be away from
        an event time to be counted as a true positive.
    assume_sorted : {bool, False}
        Assume that the injection_times are sorted in an ascending
        order. (If this is false the injection times are sorted
        internally)
    workers : {int or None, 0}
        How many processes to use to split the events. If set to 0, the
        events are analyzed sequentially. If set to None spawns as many
        processes as there are CPUs available.
    
    Returns
    -------
    true_positives : list of tuples of float
        A list of events that were correctly identified as events.
    false_positives : list of tuples of float
        A list of events that were falsely identified as events.
    """
    if assume_sorted:
        injtimes = injection_times
    else:
        injtimes = injection_times.copy()
        injtimes.sort()

    def worker(sub_event_list, itimes, tol, output, wid):
        tp = []
        fp = []
        for event in sub_event_list:
            t, v = event
            idx = np.searchsorted(itimes, t, side='right')
            if idx == 0:
                diff = abs(t - itimes[0])
            elif idx == len(itimes):
                diff = abs(t - itimes[-1])
            else:
                diff = min(abs(t - itimes[idx-1]), abs(t - itimes[idx]))
            if diff <= tol:
                tp.append(event)
            else:
                fp.append(event)
        output.put((wid, tp, fp))

    if workers == 0:
        queue = Queue()
        worker(event_list, injtimes, tolerance, queue, 0)
        _, tp, fp = queue.get()
        return tp, fp
    else:
        if workers is None:
            workers = mp.cpu_count()
        idxsrange = int(len(event_list) // workers)
        overhang = len(event_list) - workers * idxsrange
        prev = 0
        queue = mp.Queue()
        jobs = []
        for i in range(workers):
            if i < overhang:
                end = prev + idxsrange + 1
            else:
                end = prev + idxsrange
            p = mp.Process(target=worker,
                           args=(event_list[prev:end],
                                 injtimes,
                                 tolerance,
                                 queue,
                                 i))
            prev = end
            jobs.append(p)

        for p in jobs:
            p.start()

        results = [queue.get() for p in jobs]

        for p in jobs:
            p.join()

        results = sorted(results, key=lambda inp: inp[0])
        tp = []
        fp = []
        for res in results:
            tp.extend(res[1])
            fp.extend(res[2])
        return tp, fp

def get_event_times(event_list):
    """Extract the event times from a list of events.
    
    Arguments
    ---------
    event_list : list of tuples of float
        A list of events as returned by get_event_list.
    
    Returns
    -------
    list of float
        A list containing the times of the events given by the
        event_list.
    """
    return [event[0] for event in event_list]

def get_closest_injection_times(injection_times, times,
                                return_indices=False,
                                assume_sorted=False):
    """Return a list of the closest injection times to a list of input
    times.
    
    Arguments
    ---------
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    times : iterable of floats
        A list of times. The function checks which injection time was
        closest to every single one of these times.
    return_indices : {bool, False}
        Return the indices of the found injection times.
    assume_sorted : {bool, False}
        Assume that the injection times are sorted in ascending order.
        (If set to false, the injection times are sorted internally)
    
    Returns
    -------
    numpy.array of float:
        Returns an array containing the injection times that were
        closest to the provided times. The order is given by the order
        of the input times.
    numpy.array of int, optional:
        Return an array of the corresponding indices. (Only returned if
        return_indices is true)
    """
    if assume_sorted:
        injtimes = injection_times
        sidxs = np.arange(len(injtimes))
    else:
        sidxs = injection_times.argsort()
        injtimes = injection_times[sidxs]

    ret = []
    idxs = []
    for t in times:
        idx = np.searchsorted(injtimes, t, side='right')
        if idx == 0:
            ret.append(injtimes[idx])
            idxs.append(sidxs[idx])
        elif idx == len(injtimes):
            ret.append(injtimes[idx-1])
            idxs.append(sidxs[idx-1])
        else:
            if abs(t - injtimes[idx-1]) < abs(t - injtimes[idx]):
                idx -= 1
            ret.append(injtimes[idx])
            idxs.append(sidxs[idx])
    if return_indices:
        return np.array(ret), np.array(idxs, dtype=int)
    else:
        return np.array(ret)

def get_missed_injection_times(event_list, injection_times,
                               tolerance=3., return_indices=False):
    """Find the injection times that are not present in a provided list
    of events.
    
    Arguments
    ---------
    event_list : list of tuples of float
        A list of events as returned by get_event_list.
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    tolerance : {float, 3.}
        The maximum time in seconds an injection time may be away from
        an event time to be counted as a true positive.
    return_indices : {bool, False}
        Return the indices of the missed injection times.
    
    Returns
    -------
    numpy.array of floats:
        Returns an array containing injection times that were not
        contained in the list of events, considering the tolerance.
    numpy.array of int, optional:
        Return an array of the corresponding indices. (Only returned if
        return_indices is true)
    """
    ret = []
    idxs = []
    event_times = np.array(get_event_times(event_list))
    if len(event_times) == 0:
        return injection_times
    for idx, inj_time in enumerate(injection_times):
        if np.min(np.abs(event_times - inj_time)) > tolerance:
            ret.append(inj_time)
            idxs.append(idx)
    if return_indices:
        return np.array(ret), np.array(idxs, dtype=int)
    else:
        return np.array(ret)
    

def false_alarm_rate(ts, injection_times, trigger_thresh=0.2,
                     ranking_thresh=0.5, cluster_tolerance=1.,
                     event_tolerance=3.):
    """Calculate the false-alarm rate of a search at given thresholds.
    
    Arguments
    ---------
    ts : pycbc.TimeSeries
        The time series that is output by the search.
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    trigger_thresh : {float, 0.2}
        The threshold that is used to determine triggers from the
        provided time series. (See the documentation of
        get_trigger_times for more details)
    ranking_thresh : {float, 0.5}
        The threshold that is applied to the list of events to determine
        which are significant. (See the documentation of
        events_above_threshold for more details)
    cluster_tolerance : {float, 1.}
        The maximum separation of two triggers for them to be considered
        part of the same cluster. (See the documentation of
        get_cluster_boundaries for more details)
    event_tolerance : {float, 3.}
        The maximum separation between an event time and a trigger time
        for the event to be counted as a true positive. (See the
        documentation of get_false_positives for more details)
    
    Returns
    -------
    float:
        A false-alarm rate as false alarms per month.
    
    Notes
    -----
    -This function is usually applied to the data with multiple
     different values for the ranking_thresh. By doing so one obtains
     the false-alarm rate as a function of a ranking statistic.
    """
    triggers = get_trigger_times(ts, trigger_thresh)
    clusters = get_cluster_boundaries(triggers,
                                      boundarie_time=cluster_tolerance)
    events = get_event_list(ts, clusters)
    significant_events = events_above_threshold(events, ranking_thresh)
    fp = get_false_positives(significant_events, injection_times,
                             tolerance=event_tolerance)
    
    far = len(fp) / ts.duration * SECONDS_PER_MONTH
    return far

def sensitive_fraction(ts, injection_times, trigger_thresh=0.2,
                       ranking_thresh=0.5, cluster_tolerance=1.,
                       event_tolerance=3.):
    """Calculate the sensitivity as a true positive rate.
    
    Arguments
    ---------
    ts : pycbc.TimeSeries
        The time series that is output by the search.
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    trigger_thresh : {float, 0.2}
        The threshold that is used to determine triggers from the
        provided time series. (See the documentation of
        get_trigger_times for more details)
    ranking_thresh : {float, 0.5}
        The threshold that is applied to the list of events to determine
        which are significant. (See the documentation of
        events_above_threshold for more details)
    cluster_tolerance : {float, 1.}
        The maximum separation of two triggers for them to be considered
        part of the same cluster. (See the documentation of
        get_cluster_boundaries for more details)
    event_tolerance : {float, 3.}
        The maximum separation between an event time and a trigger time
        for the event to be counted as a true positive. (See the
        documentation of get_true_positives for more details)
    
    Returns
    -------
    float:
        The fraction of injected signals that were detected.
    """
    triggers = get_trigger_times(ts, trigger_thresh)
    clusters = get_cluster_boundaries(triggers,
                                      boundarie_time=cluster_tolerance)
    events = get_event_list(ts, clusters)
    significant_events = events_above_threshold(events, ranking_thresh)
    
    tp = get_true_positives(significant_events, injection_times,
                            tolerance=event_tolerance)
    return float(len(tp)) / len(injection_times)

def filter_times(injection_times, times, assume_sorted=False):
    """Returns an index array. The indices point to positions in the
    injection times the corresponding time of the times list can be
    found.
    
    Arguments
    ---------
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    times : iterable of floats
        A list of times.
    assume_sorted : {bool, False}
        Assume that the array of injection times is sorted.
        Significantly speeds up the filtering process.
    
    Returns
    -------
    numpy.array of int:
        An array containing indices. The indices give the position of
        the time from the times list in the injection_times array.
    """
    ret = []
    msg = 'Found non-matching time {} in injection times {}.'
    if assume_sorted:
        for time in times:
            idx = np.searchsorted(injection_times, time)
            if idx == len(injection_times) - 1:
                if time == injection_times[-1]:
                    ret.append(len(injection_times)-1)
                else:
                    msg = msg.format(time, injection_times)
                    raise RuntimeError(msg)
            else:
                if time == injection_times[idx]:
                    ret.append(idx)
                elif time == injection_times[idx+1]:
                    ret.append(idx+1)
                else:
                    msg = msg.format(time, injection_times)
                    raise RuntimeError(msg)
    else:
        for time in times:
            idxs = np.where(injection_times == time)[0]
            if len(idxs) > 0:
                ret.append(idxs[0])
            else:
                msg = msg.format(time, injection_times)
                raise RuntimeError(msg)
    return np.array(ret)

def mchirp(m1, m2):
    """Calculate the chirp mass of the given component masses.
    
    Arguments
    ---------
    m1 : float or numpy.array of float
        The primary mass.
    m2 : float or numpy.array of float
        The secondary mass.
    
    Returns
    -------
    float or numpy.array of float:
        The corresponding chirp-mass(es)
    """
    return (m1 * m2) ** (3. / 5.) / (m1 + m2) ** (1. / 5.)

def sensitive_distance(ts, injection_times, injection_m1, injection_m2,
                       injection_dist, trigger_thresh=0.2,
                       ranking_thresh=0.5, cluster_tolerance=1.,
                       event_tolerance=3.):
    """Calculate the distance out to which the search is sensitive to
    gravitational-wave sources.
    
    Arguments
    ---------
    ts : pycbc.TimeSeries
        The time series that is output by the search.
    injection_times : numpy.array of floats
        An array containing the times at which a signal was actually
        present in the data.
    injection_m1 : numpy.array of floats
        An array containing the primary masses of the injected signals.
        (in solar masses)
    injection_m2 : numpy.array of floats
        An array containing the secondary masses of the injected
        signals. (in solar masses)
    injection_dist : numpy.array of floats
        An array containing the distances of the injected signals. (in
        solar mega Parsec)
    trigger_thresh : {float, 0.2}
        The threshold that is used to determine triggers from the
        provided time series. (See the documentation of
        get_trigger_times for more details)
    ranking_thresh : {float, 0.5}
        The threshold that is applied to the list of events to determine
        which are significant. (See the documentation of
        events_above_threshold for more details)
    cluster_tolerance : {float, 1.}
        The maximum separation of two triggers for them to be considered
        part of the same cluster. (See the documentation of
        get_cluster_boundaries for more details)
    event_tolerance : {float, 3.}
        The maximum separation between an event time and a trigger time
        for the event to be counted as a true positive. (See the
        documentation of get_true_positives for more details)
    
    Returns
    -------
    float:
        The distance out to which the search is able to detect
        gravitational waves. (an average, in mega Parsec)
    """
    triggers = get_trigger_times(ts, trigger_thresh)
    clusters = get_cluster_boundaries(triggers,
                                      boundarie_time=cluster_tolerance)
    events = get_event_list(ts, clusters)
    significant_events = events_above_threshold(events, ranking_thresh)
    
    tp = get_true_positives(significant_events, injection_times,
                            tolerance=event_tolerance)
    found_times, found_idxs = get_closest_injection_times(injection_times,
                                                          get_event_times(tp),
                                                          return_indices=True)
    #missed_times = get_missed_injection_times(significant_events,
                                              #injection_times,
                                              #tolerance=event_tolerance)
    missed_idxs = np.setdiff1d(np.arange(len(injection_times)), found_idxs)
    
    #found_idxs = filter_times(injection_times, found_times)
    if len(found_idxs) > 0:
        found_m1 = injection_m1[found_idxs]
        found_m2 = injection_m2[found_idxs]
        found_dist = injection_dist[found_idxs]
        found_mchirp = mchirp(found_m1, found_m2)
    else:
        found_m1 = np.array([1.])
        found_m2 = np.array([1.])
        found_dist = np.array([0.])
        found_mchirp = np.array([1.])
    
    #missed_idxs = filter_times(injection_times, missed_times)
    if len(missed_idxs) > 0:
        missed_m1 = injection_m1[missed_idxs]
        missed_m2 = injection_m2[missed_idxs]
        missed_dist = injection_dist[missed_idxs]
        missed_mchirp = mchirp(missed_m1, missed_m2)
    else:
        missed_m1 = np.array([1.])
        missed_m2 = np.array([1.])
        missed_dist = np.array([1.])
        missed_mchirp = np.array([np.inf])
    
    vol, vol_err = volume_montecarlo(found_dist,
                                     missed_dist,
                                     found_mchirp,
                                     missed_mchirp,
                                     'distance',
                                     'volume',
                                     'distance')
    
    rad = (3 * vol / (4 * np.pi))**(1. / 3.)
    return rad

class progress_tracker():
    """A class that implements and prints a dynamic progress bar to
    stdout.
    
    Arguments
    ---------
    num_of_steps : int
        The number of iterations that is expected to occur.
    name : {str, 'Progress'}
        The name for the header of the progress bar. It will be followed
        by a colon ':' when printed.
    steps_taken : {int, 0}
        The number of steps that are already completed.
    """
    def __init__(self, num_of_steps, name='Progress', steps_taken=0):
        self.t_start = datetime.datetime.now()
        self.num_of_steps = num_of_steps
        self.steps_taken = steps_taken
        self.name = name
        self._printed_header = False
        self.last_string_length = 0
    
    def __len__(self):
        return self.num_of_steps
    
    @property
    def eta(self):
        now = datetime.datetime.now()
        return(int(round(float((now - self.t_start).seconds) / float(self.steps_taken) * float(self.num_of_steps - self.steps_taken))))
    
    @property
    def percentage(self):
        return(int(100 * float(self.steps_taken) / float(self.num_of_steps)))
    
    def get_print_string(self):
        curr_perc = self.percentage
        real_perc = self.percentage
        #Length of the progress bar is 25. Hence one step equates to 4%.
        bar_len = 25
        
        if not curr_perc % 4 == 0:
            curr_perc -= curr_perc % 4
        
        if int(curr_perc / 4) > 0:
            s = '[' + '=' * (int(curr_perc / 4) - 1) + '>' + '.' * (bar_len - int(curr_perc / 4)) + ']'
        else:
            s = '[' + '.' * bar_len + ']'
        
        tot_str = str(self.num_of_steps)
        curr_str = str(self.steps_taken)
        curr_str = ' ' * (len(tot_str) - len(curr_str)) + curr_str
        eta = str(datetime.timedelta(seconds=self.eta)) + 's'
        perc_str = ' ' * (len('100') - len(str(real_perc))) + str(real_perc)
        
        out_str = curr_str + '/' + tot_str + ': ' + s + ' ' + perc_str + '%' + ' ETA: ' + eta
        
        if self.last_string_length > len(out_str):
            back = '\b \b' * (self.last_string_length - len(out_str))
        else:
            back = ''
        
        #back = '\b \b' * self.last_string_length
        
        self.last_string_length = len(out_str)
        
        return(back + '\r' + out_str)
        #return(back + out_str)
    
    def print_progress_bar(self, update=True):
        if not self._printed_header:
            print(self.name + ':')
            self._printed_header = True
        
        if update:
            sys.stdout.write(self.get_print_string())
            sys.stdout.flush()
            if self.steps_taken == self.num_of_steps:
                self.print_final(update=update)
        else:
            print(self.get_print_string())
            if self.steps_taken == self.num_of_steps:
                self.print_final(update=update)
    
    def iterate(self, iterate_by=1, print_prog_bar=True, update=True):
        if iterate_by > 0:
            self.steps_taken += iterate_by
            if print_prog_bar:
                self.print_progress_bar(update=update)
    
    def print_final(self, update=True):
        final_str = str(self.steps_taken) + '/' + str(self.num_of_steps) + ': [' + 25 * '=' + '] 100% - Time elapsed: ' + str(datetime.timedelta(seconds=(datetime.datetime.now() - self.t_start).seconds)) + 's'
        if update:
            clear_str = '\b \b' * self.last_string_length
            
            sys.stdout.write(clear_str + final_str + '\n')
            sys.stdout.flush()
        else:
            print(final_str)

class mp_progress_tracker(progress_tracker):
    """A class that implements and prints a dynamic progress bar to
    stdout. This special case is multiprocessing save.
    
    Arguments
    ---------
    num_of_steps : int
        The number of iterations that is expected to occur.
    name : {str, 'Progress'}
        The name for the header of the progress bar. It will be followed
        by a colon ':' when printed.
    steps_taken : {int, 0}
        The number of steps that are already completed.
    """
    def __init__(self, num_of_steps, name='Progress', steps_taken=0):
        self._printed_header_val = mp.Value('i', False)
        self.last_string_length_val = mp.Value('i', 0)
        super().__init__(num_of_steps, name=name,
                         steps_taken=steps_taken)
        self.steps_taken = mp.Value('i', steps_taken)
    
    @property
    def _printed_header(self):
        return bool(self._printed_header_val.value)
    
    @_printed_header.setter
    def _printed_header(self, boolean):
        with self._printed_header_val.get_lock():
            self._printed_header_val.value = int(boolean)
    
    @property
    def last_string_length(self):
        return self.last_string_length_val.value
    
    @last_string_length.setter
    def last_string_length(self, length):
        with self.last_string_length_val.get_lock():
            self.last_string_length_val.value = length
    
    @property
    def eta(self):
        now = datetime.datetime.now()
        return(int(round(float((now - self.t_start).seconds) / float(self.steps_taken.value) * float(self.num_of_steps - self.steps_taken.value))))
    
    @property
    def percentage(self):
        return(int(100 * float(self.steps_taken.value) / float(self.num_of_steps)))
    
    def get_print_string(self):
        curr_perc = self.percentage
        real_perc = self.percentage
        #Length of the progress bar is 25. Hence one step equates to 4%.
        bar_len = 25
        
        if not curr_perc % 4 == 0:
            curr_perc -= curr_perc % 4
        
        if int(curr_perc / 4) > 0:
            s = '[' + '=' * (int(curr_perc / 4) - 1) + '>' + '.' * (bar_len - int(curr_perc / 4)) + ']'
        else:
            s = '[' + '.' * bar_len + ']'
        
        tot_str = str(self.num_of_steps)
        curr_str = str(self.steps_taken.value)
        curr_str = ' ' * (len(tot_str) - len(curr_str)) + curr_str
        eta = str(datetime.timedelta(seconds=self.eta)) + 's'
        perc_str = ' ' * (len('100') - len(str(real_perc))) + str(real_perc)
        
        out_str = curr_str + '/' + tot_str + ': ' + s + ' ' + perc_str + '%' + ' ETA: ' + eta
        
        if self.last_string_length > len(out_str):
            back = '\b \b' * (self.last_string_length - len(out_str))
        else:
            back = ''
        
        #back = '\b \b' * self.last_string_length
        
        self.last_string_length = len(out_str)
        
        return(back + '\r' + out_str)
        #return(back + out_str)
    
    def print_progress_bar(self, update=True):
        if not self._printed_header:
            print(self.name + ':')
            self._printed_header = True
        
        if update:
            sys.stdout.write(self.get_print_string())
            sys.stdout.flush()
            if self.steps_taken.value == self.num_of_steps:
                self.print_final(update=update)
        else:
            print(self.get_print_string())
            if self.steps_taken.value == self.num_of_steps:
                self.print_final(update=update)
    
    def iterate(self, iterate_by=1, print_prog_bar=True, update=True):
        with self.steps_taken.get_lock():
            if iterate_by > 0:
                self.steps_taken.value += iterate_by
                if print_prog_bar:
                    self.print_progress_bar(update=update)
    
    def print_final(self, update=True):
        final_str = str(self.steps_taken.value) + '/' + str(self.num_of_steps) + ': [' + 25 * '=' + '] 100% - Time elapsed: ' + str(datetime.timedelta(seconds=(datetime.datetime.now() - self.t_start).seconds)) + 's'
        if update:
            clear_str = '\b \b' * self.last_string_length
            
            sys.stdout.write(clear_str + final_str + '\n')
            sys.stdout.flush()
        else:
            print(final_str)

def multi_wave_worker(idx, wave_params, projection_params,
                      detector_names, transform, domain, progbar,
                      output):
    """A helper-function to generate multiple waveforms using
    multiprocessing.
    
    Arguments
    ---------
    idx : int
        The index given to the process. This is returned as the first
        part of the output to identify which parameters the waveforms
        belong to.
    wave_params : list of dict
        A list containing the keyword-arguments for each waveform that
        should be generated. Each entry of the list is passed to
        get_td/fd_waveform using unwrapping of a dictionary.
    projection_params : list of list
        A list containing all the positional arguments to project the
        waveform onto the detector. Each entry should contain the
        following information in order:
        ['ra', 'dec', 'pol']
        Can be empty, if detector_names is set to None.
    detector_names : list of str or None
        A list of detectors names onto which the waveforms should be
        projected. Each entry has to be understood by
        pycbc.detector.Detector. If set to None the waveforms will not
        be projected and the two polarizations will be returned instead.
    transform : function
        A transformation function that should be applied to every
        waveform. (Can be the identity.)
    domain : 'time' or 'frequency'
        Whether to return the waveforms in the time- or
        frequency-domain.
    progbar : BnsLib.utils.progress_bar.mp_progress_tracker or None
        If a progress bar is desired, the instance can be passed here.
        When set to None, no progress will be reported.
    output : multiprocessing.Queue
        The Queue into which the outputs of the waveform generating code
        will be inserted. Contents are of the form:
        (index, data)
        Here `data` is a dictionary. The keys are the different detector
        names and the values are lists storing the generated waveforms.
    
    Returns
    -------
    None (see argument `output` for details)
    """
    if detector_names is None:
        detectors = None
    else:
        detectors = [Detector(det) for det in detector_names]
    ret = DictList()
    for wav_params, proj_params in zip(wave_params, projection_params):
        sig = signal_worker(wav_params,
                            proj_params,
                            detectors,
                            transform,
                            domain=domain)
        for key, val in sig.items():
            if key in ret:
                ret.append(key, value=val)
            else:
                ret.append(key, value=[val])
        #ret.append(ret)
        if progbar is not None:
            progbar.iterate()
    output.put((idx, ret.as_dict()))

def signal_worker(wave_params, projection_params, detectors, transform,
                  domain='time'):
    tc = wave_params.pop('tc', 0.)
    if domain.lower() == 'time':
        hp, hc = get_td_waveform(**wave_params)
    elif domain.lower() == 'frequency':
        hp, hc = get_fd_waveform(**wave_params)
    else:
        msg = 'Domain must be either "time" or "frequency".'
        raise ValueError(msg)
    
    hp.start_time = float(hp.start_time) + tc
    hc.start_time = float(hc.start_time) + tc
    
    if not isinstance(detectors, list):
        detectors = [detectors]
    ret = {}
    if detectors is None:
        ret['plus'] = hp
        ret['cross'] = hc
    else:
        st = float(hp.start_time)
        projection_params.append(st)
        #print(projection_params)
        req_opt = [np.isnan(pt) for pt in projection_params[:2]]
        if any(req_opt):
            opt_ra, opt_dec = detectors[0].optimal_orientation(st)
            if req_opt[0]:
                projection_params[0] = opt_ra
            if req_opt[1]:
                projection_params[1] = opt_dec
        for det in detectors:
            fp, fc = det.antenna_pattern(*projection_params)
            ret[det.name] = transform(fp * hp + fc * hc)
    return ret

def multi_noise_worker(length, delta_t, psd_name, flow, number, seed,
                       transform, bar, output):
    ret = []
    if psd_name.lower() == 'simple':
        sample_rate = int(1. / delta_t)
        nyquist = int(1. / (2 * delta_t))
        scale = np.sqrt(nyquist)
        total_noise = np.random.normal(loc=0., scale=scale,
                                       size=(number, length))
        for noise in total_noise:
            ret.append(transform(TimeSeries(noise, delta_t=delta_t)))
            if bar is not None:
                bar.iterate()
    else:
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e7, size=number, dtype=int)
        for i in range(number): 
            noise = noise_from_string(psd_name, length, delta_t,
                                      seed=int(seeds[i]),
                                      low_frequency_cutoff=flow)
            ret.append(transform(noise))
            if bar is not None:
                bar.iterate()
    output.put(ret)

class WaveformGetter(object):
    """Class to generate waveforms from given parameters. It can only
    generate as many waveforms as there are values for each parameter in
    the variable_params attribute.
    
    Arguments
    ---------
    variable_params : {None or dict, None}
        The table containing the waveform parameters. These may include
        all parameters accepted by pycbc.waveform.generate_td_waveform
        as well as 'ra', 'dec' and 'pol' to specify a sky-position and
        'tc' to place the merger time. The table is given as a
        dictionary where the keys are the parameter names and the values
        are iterables of floats that specify the value for each
        waveform.
    static_params : {None or dict, None}
        Parameters that are constant for all waveforms. These may
        commonly include the waveform approximant, the delta_t or the
        lower frequency bound. The keys are the parameter names and may
        contain any keys that could be put into the variable_params as
        well.
    domain : {'time' or 'freq', 'time'}
        Whether to generate the waveforms in the time ('time') or
        frequency ('freq') domain. [May be abbreviated to 't' and 'f'
        respectifevly]
    detectors : {None or str or list of str, 'H1'}
        A list of detectors onto which the waveform is projected. If set
        to None, the two polarizations of the waveform will be
        generated. Detector names must be known to
        pycbc.detector.Detector.
    
    Usage
    -----
    Generate a set of sources with different component masses from the
    same sky-location:
    >>> from BnsLib.data import WaveformGetter
    >>> import numpy
    >>> variable_params = {}
    >>> variable_params['mass1'] = numpy.random.uniform(15., 50., size=10)
    >>> variable_params['mass2'] = numpy.random.uniform(15., 50., size=10)
    >>> static_params = {}
    >>> static_params['approximant'] = 'TaylorF2'
    >>> static_params['delta_t'] = 1. / 2048
    >>> static_params['f_lower'] = 15.
    >>> static_params['ra'] = numpy.pi
    >>> static_params['dec'] = numpy.pi / 2
    >>> static_params['distance'] = 1000.
    >>> getter = WaveformGetter(variable_params=variable_params,\
    >>>                         static_params=static_params,\
    >>>                         detectors=['H1', 'L1'])
    >>> waves = getter.generate(verbose=False)
    """
    def __init__(self, variable_params=None, static_params=None,
                 domain='time', detectors='H1'):
        self.variable_params = variable_params
        self.static_params = static_params
        self.domain = domain
        self.detectors = detectors
        self._it_index = 0
    
    def __len__(self):
        if len(self.variable_params) == 0:
            if len(self.static_params) == 0:
                return 0
            else:
                return 1
        else:
            key = list(self.variable_params.keys())[0]
            return len(self.variable_params[key])
    
    def __getitem__(self, index):
        return self.generate(index=index, workers=0, verbose=False)
    
    def __next__(self):
        if self._it_index < len(self):
            ret = self[self._it_index]
            self._it_index += 1
            return ret
        else:
            raise StopIteration
    
    def __iter__(self):
        return self
    
    def generate(self, index=None, single_detector_as_list=True,
                 workers=None, verbose=True):
        """Generates one or multiple waveforms.
        
        Arguments
        ---------
        index : {int or slice or None, None}
            Which waveforms to generate. If set to None, all waveforms
            will be generated. Indices point to the given lists
            variable_params.
        single_detector_as_list : {bool, True}
            Usually this function will return a dictionary of lists,
            where each entry corresponds to one of multiple detectors.
            If only a single detector is used it is not necessary to
            use a dictionary. If this option is set to true, only the
            value of the dictionary will be returned when a single
            detector is used.
        worker : {None or int >= 0, None}
            How many processes to spawn to generate the waveforms. Set
            to None in order to use as many processes as there are CPU
            cores available. Set to 0 to disable multiprocessing.
            (Turning off multiprocessing is useful for debugging.)
        verbose : {bool, True}
            Print a pogressbar for the waveform generation.
        
        Returns
        -------
        dict of list or list or pycbc.TimeSeries:
            The return type depends on the index and the option
            `single_detector_as_list`. If multiple detectors are used
            and the index is a slice, a dictionary of lists will be
            returned. The keys to the dictionary contain the detector
            prefixes and the lists contain transformed waveforms [1].
            If the index is an integer instead the values of the
            dictionary will not be lists but the transformed waveform
            instead. If the option `single_detector_as_list` is set to
            True and only a single detector is provided the function
            will return just the waveform and no dictionary.
        """
        if index is None:
            index = slice(None, None)
        was_int = False
        if isinstance(index, int):
            index = slice(index, index+1)
            was_int = True
        
        if workers is None:
            workers = mp.cpu_count()
        
        indices = list(range(*index.indices(len(self))))
        
        #create input to signal worker
        wave_params = []
        projection_params = []
        for i in indices:
            params = self.get_params(i)
            wave_params.append(params)
            if self.detectors is None:
                projection_params.append([])
            else:
                if 'ra' in params:
                    ra_key = 'ra'
                elif 'right_ascension' in params:
                    ra_key = 'right_ascension'
                else:
                    ra_key = 'ra'
                    params['ra'] = np.nan
                if 'dec' in params:
                    dec_key = 'dec'
                elif 'declination' in params:
                    dec_key = 'declination'
                else:
                    dec_key = 'dec'
                    params['dec'] = np.nan
                if 'pol' in params:
                    pol_key = 'pol'
                elif 'polarization' in params:
                    pol_key = 'polarization'
                else:
                    pol_key = 'pol'
                    params['pol'] = 0.
                projection_params.append([params[key] for key in [ra_key, dec_key, pol_key]])
        
        if self.detectors is None:
            detector_names = None
        else:
            detector_names = [det.name for det in self.detectors]
        
        #Generate the signals
        if workers == 0:
            if verbose:
                progbar = progress_tracker(len(wave_params),
                                           name='Generating waveforms')
            if detector_names is None:
                detectors = None
            else:
                detectors = [Detector(det) for det in detector_names]
            ret = DictList()
            for wav_params, proj_params in zip(wave_params, projection_params):
                sig = signal_worker(wav_params,
                                    proj_params,
                                    detectors,
                                    self.transform,
                                    domain=self.domain)
                for key, val in sig.items():
                    if key in ret:
                        ret.append(key, value=val)
                    else:
                        ret.append(key, value=[val])
                if verbose:
                    progbar.iterate()
            ret = ret.as_dict()
        else:
            waves_per_process = [len(indices) // workers] * workers
            if sum(waves_per_process) < len(indices):
                for i in range(len(indices) - sum(waves_per_process)):
                    waves_per_process[i] += 1
            waves_per_process = np.cumsum(waves_per_process)
            wpp = [0]
            wpp.extend(waves_per_process)
            
            wave_boundaries = [slice(wpp[i], wpp[i+1]) for i in range(workers)]
            wb = wave_boundaries
            
            bar = None
            if verbose:
                bar = mp_progress_tracker(len(indices),
                                        name='Generating waveforms')
            
            jobs = []
            output = mp.Queue()
            for i in range(workers):
                p = mp.Process(target=multi_wave_worker,
                            args=(i,
                                    wave_params[wb[i]],
                                    projection_params[wb[i]],
                                    detector_names,
                                    self.transform,
                                    self.domain,
                                    bar,
                                    output))
                jobs.append(p)
            
            for p in jobs:
                p.start()
            
            results = [output.get() for p in jobs]
            
            for p in jobs:
                p.join()
            
            results.sort()
            ret = DictList()
            for pt in results:
                ret.extend(pt[1])
            ret = ret.as_dict()
        
        if was_int:
            ret = {key: val[0] for (key, val) in ret.items()}
        
        if self.detectors is None:
            return ret
        
        if single_detector_as_list and len(self.detectors) == 1:
            return ret[self.detectors[0].name]
        return ret
    
    #Legacy function
    generate_mp = generate
    
    def get_params(self, index=None):
        if index is None:
            index = slice(None, None)
        ret = {}
        if isinstance(index, int):
            for key, val in self.static_params.items():
                ret[key] = val
            for key, val in self.variable_params.items():
                ret[key] = val[index]
        elif isinstance(index, slice):
            slice_size = len(range(len(self))[index])
            for key, val in self.static_params.items():
                ret[key] = [val for _ in range(slice_size)]
            for key, val in self.variable_params.items():
                ret[key] = val[index]
        return ret
    
    def transform(self, wav):
        return wav
    
    @property
    def variable_params(self):
        return self._variable_params
    
    @variable_params.setter
    def variable_params(self, variable_params):
        if variable_params is None:
            self._variable_params = {}
        if not isinstance(variable_params, dict):
            msg = 'variable_params must be a dictionary containing '
            msg += 'iterables of the same length. Got type '
            msg += f'{type(variable_params)} instead.'
            raise TypeError(msg)
        parts = list(variable_params.values())
        if not all([len(pt) == len(parts[0]) for pt in parts]):
            msg = 'variable_params must be a dictionary containing '
            msg += 'iterables of the same length. Got lengths '
            msg_dict = {key: len(val) for (key, val) in variable_params.items()}
            msg += f'{msg_dict}.'
            raise ValueError(msg)
        self._variable_params = variable_params
    
    @property
    def static_params(self):
        return self._static_params
    
    @static_params.setter
    def static_params(self, static_params):
        if static_params is None:
            self._static_params = {}
        if not isinstance(static_params, dict):
            msg = 'static_params must be a dictionary. Got type '
            msg += f'{type(static_params)} instead.'
            raise TypeError(msg)
        self._static_params = static_params
    
    @property
    def domain(self):
        return self._domain
    
    @domain.setter
    def domain(self, domain):
        time_domains = ['time', 't']
        freq_domains = ['frequency', 'freq', 'f']
        poss_domains = time_domains + freq_domains
        if domain.lower() not in poss_domains:
            msg = f'domain must be one of {poss_domains}, not {domain}.'
            raise ValueError(msg)
        if domain.lower() in time_domains:
            self._domain = 'time'
        
        if domain.lower()in freq_domains:
            self._domain = 'frequency'
    
    @property
    def detectors(self):
        return self._detectors
    
    @detectors.setter
    def detectors(self, detectors):
        if detectors is None:
            self._detectors = None
            return
        detectors = input_to_list(detectors, length=list_length(detectors))
        self._detectors = []
        for det in detectors:
            if isinstance(det, Detector):
                self._detectors.append(det)
            elif isinstance(det, str):
                self._detectors.append(Detector(det))
            else:
                msg = 'Detectors must be specified either as a '
                msg += f'pycbc.Detector or a string. Got {type(det)} '
                msg += 'instead.'
                raise TypeError(msg)
    
    @classmethod
    def from_config(cls, config_file, number_samples):
        return

class NoiseGenerator(object):
    """A class that efficiently generates time series noise samples of
    equal length.
    
    Arguments
    ---------
    length : int
        The length of each noise in number of samples.
    delta_t : float
        The time between two samples in seconds.
    psd_name : {str, 'simple'}
        The name of the power spectral densitiy that should be used to
        color the noise. If set to 'simple' gaussian noise with a
        standard deviation of sqrt(1 / (2 * delta_t)) will be generated.
    low_frequency_cutoff : {float, 20.}
        The low frequency cutoff. Below this frequency the noise will be
        set to 0.
    """
    def __init__(self, length, delta_t, psd_name='simple',
                 low_frequency_cutoff=20.):
        self.length = length
        self.delta_t = delta_t
        self.psd_name = psd_name
        self.flow = low_frequency_cutoff
    
    def generate(self, number, workers=None, verbose=True, seed=None):
        """Generate a list of independently drawn noise samples.
        
        Arguments
        ---------
        number : int
            The number of noise samples that should be generated.
        workers : {int or None, None}
            This function may run in parallel. When setting this
            argument to an integer the user specifies how many processes
            will be spawned. If set to None the code will spawn as many
            processes as there are CPU-cores available. To run the code
            in serial set this argument to 0.
        verbose : {bool, True}
            Whether or not to print a dynamic progress bar.
        seed : {int or None, None}
            The seed to use for noise generation. If set to None the
            current time in milliseconds will be used as seed.
        
        Returns
        -------
        list of TimeSeries:
            Returns a list of pycbc.types.TimeSeries objects that
            contain noise. The list will be of length `number`.
        """
        if seed is None:
            seed = int(time.time())
        
        if workers is None:
            workers = mp.cpu_count()
        
        if workers == 0:
            class PutList(object):
                def __init__(self):
                    self.content = []
                
                def put(self, content):
                    self.content.extend(content)
            
            bar = None
            if verbose:
                bar = progress_tracker(number, name='Generating noise')
            
            output = PutList()
            
            multi_noise_worker(self.length, self.delta_t, self.psd_name,
                               self.flow, number, seed, self.transform,
                               bar, output)
            
            return output.content
        
        noise_per_worker = [number // workers] * workers
        if sum(noise_per_worker) < number:
            for i in range(number - sum(noise_per_worker)):
                noise_per_worker[i] += 1
        
        bar = None
        if verbose:
            bar = mp_progress_tracker(number, name='Generating noise')
        
        jobs = []
        output = mp.Queue()
        np.random.seed(seed)
        seeds = np.random.randint(0, 1e7, size=workers)
        for i in range(workers):
            p = mp.Process(target=multi_noise_worker,
                           args=(self.length,
                                 self.delta_t,
                                 self.psd_name,
                                 self.flow,
                                 noise_per_worker[i],
                                 seeds[i],
                                 self.transform,
                                 bar,
                                 output))
            jobs.append(p)
        
        for p in jobs:
            p.start()
        
        results = [output.get() for p in jobs]
        
        for p in jobs:
            p.join()
        
        ret = []
        for pt in results:
            ret.extend(pt)
        
        return ret
    
    def transform(self, noise):
        return noise

class WhiteNoiseGenerator(NoiseGenerator):
    """A class that efficiently generates white time series noise. If a
    power spectrum is given to color the noise, the output will be
    whitened by the same power spectrum.
    
    Arguments
    ---------
    length : int
        The length of each noise in number of samples.
    delta_t : float
        The time between two samples in seconds.
    psd_name : {str, 'simple'}
        The name of the power spectral densitiy that should be used to
        color the noise. If set to 'simple' gaussian noise with a
        standard deviation of sqrt(1 / (2 * delta_t)) will be generated.
    low_frequency_cutoff : {float, 20.}
        The low frequency cutoff. Below this frequency the noise will be
        set to 0.
    
    Notes
    -----
    -When subclassing this class and applying a different transform,
     make sure to call the transform method of this class first:
     
     >>> class CustomWhiteNoise(WhiteNoiseGenerator):
     >>>    def transform(self, noise):
     >>>        noise = super().transform(noise)
     >>>        #Your custom operations
     >>>        return noise
    """
    def __init__(self, length, delta_t, psd_name='simple',
                 low_frequency_cutoff=20.):
        if psd_name.lower() != 'simple':
            length += 8 * int(1. / delta_t)
        super().__init__(length, delta_t, psd_name=psd_name,
                         low_frequency_cutoff=low_frequency_cutoff)
    
    def transform(self, noise):
        if self.psd_name.lower() == 'simple':
            return noise
        return whiten(noise, low_freq_cutoff=self.flow,
                      psd=self.psd_name)

class DictList(object):
    """A table-like object. It is a dictionary where each value is a
    list.
    
    Arguments
    ---------
    dic : {dict or None, None}
        A dictionary from which to start
    
    Attributes
    ----------
    dic : dict
        A dictionary where each entry is a list. This attribute is
        returned by the function `as_dict`.
    """
    def __init__(self, dic=None):
        self.dic = dic
    
    def __getitem__(self, key):
        return self.dic[key]
    
    def __contains__(self, key):
        return key in self.dic
    
    def __len__(self):
        return len(self.dic)
    
    def __add__(self, other):
        ret = self.copy()
        return ret.join(other)
    
    def __radd__(self, other):
        if isinstance(other, dict):
            tmp = DictList(dic=other)
        else:
            tmp = other
        if not isinstance(tmp, type(self)):
            msg = 'Can only add dict or DictList to a DictList. '
            msg += 'Got type {} instead.'.format(type(other))
            raise TypeError(msg)
        ret = tmp.copy()
        return ret.join(self)
        
    def copy(self):
        return DictList(dic=self.dic.copy())
    
    def get(self, k, d=None):
        #Not sure if this implementation is correct. Does this allow for
        #setdefault to work?
        return self.dic.get(k, d)
    
    def pop(self, k, *d):
        return self.dic.pop(k, *d)
    
    @property
    def dic(self):
        return self._dic
    
    @dic.setter
    def dic(self, dic):
        self._dic = {}
        if dic is None:
            return
        elif isinstance(dic, dict):
            for key, val in dic.items():
                if isinstance(val, list):
                    self._dic[key] = val
                else:
                    self._dic[key] = [val]
        else:
            msg = 'The input has to be a dict.'
            raise TypeError(msg)
    
    def append(self, key, value=None):
        """Append data to the DictList. If some keys are not known to
        this DictList it will create a new entry.
        
        Arguments
        ---------
        key : hashable object or dict
            If this entry is a dictionary, all values of the keys will
            be appended to the DictList. If this entry is a hashable
            object, it will be understood as a key to the DictList and
            the optional argument 'value' will be appended to the
            DictList.
        value : {None or object, None}
            If 'key' is not a dictionary, this value will be appended to
            the DictList.
        
        Returns
        -------
        None
        """
        #print(f"Appending key of type {type(key)} in DictList")
        if isinstance(key, dict):
            for k, val in key.items():
                if k in self.dic:
                    self.dic[k].append(val)
                elif isinstance(val, list):
                    self.dic[k] = val
                else:
                    self.dic[k] = [val]
        elif key in self.dic:
            self.dic[key].append(value)
        elif isinstance(value, list):
            self.dic[key] = value
        else:
            self.dic[key] = [value]
    
    def as_dict(self):
        return self.dic
    
    def keys(self):
        return self.dic.keys()
    
    def values(self):
        return self.dic.values()
    
    def items(self):
        return self.dic.items()
    
    def extend(self, key, value=None):
        if isinstance(key, (dict, type(self))):
            for k, val in key.items():
                if k in self.dic:
                    self.dic[k].extend(val)
                else:
                    self.append(k, value=val)
        else:
            if key in self:
                if value is None:
                    return
                else:
                    self.dic[key].extend(value)
            else:
                self.append(key, value=value)
    
    def join(self, other):
        if isinstance(other, dict):
            to_join = DictList(other)
        else:
            to_join = other
        if not isinstance(to_join, type(self)):
            msg = 'Can only join a dictionary or DictList to a DictList.'
            msg += ' Got instance of type {} instead.'
            msg = msg.format(type(to_join))
            raise TypeError(msg)
        for okey, ovalue in to_join.items():
            if okey in self:
                self.dic[okey] = self.dic[okey] + to_join[okey]
            else:
                self.append(okey, value=to_join[okey])
    
    def count(self, item, keys=None):
        """Return the number of occurences of item in the DictList.
        
        Arguments
        ---------
        item : object
            The value to search for.
        keys : {iterable of keys or 'all' or None, None}
            Which dictionary entries to consider. If set to None, all
            keys will be considered but only the sum of all the
            individual counts will be returned. If set to 'all', all
            keys will be considered and a dictionary with {key: count}
            will be returned. This dictionary specifies the counts for
            each individual entry. If an iterable of keys is provided
            a dictionary with the keys and the according counts is
            returned.
        
        Returns
        -------
        ret : int or dict
            Either an integer specifying the count over all keys or a
            dictionary, where the count for each key is given
            explicitly.
        """
        if keys is None:
            return sum([val.count(item) for val in self.values()])
        if isinstance(keys, str) and keys.lower() == 'all':
            keys = list(self.keys())
        ret = {}
        for key in keys:
            if key in self:
                ret[key] = self[key].count(item)
            else:
                ret[key] = 0
        return ret

class MPCounter(object):
    def __init__(self, val=0):
        assert isinstance(val, int), 'Initial value has to be an integer.'
        self.val = mp.Value('i', val)
    
    def increment(self, n=1):
        with self.val.get_lock():
            self.val.value += n
    
    def __add__(self, other):
        if not isinstance(other, (int, type(self))):
            msg = 'Can only add an integer or MPCounter object to an '
            msg += 'MPCounter object.'
            raise TypeError(msg)
        if isinstance(other, int):
            return MPCounter(val=self.value+other)
        return MPCounter(val=self.value+other.value)
    
    def __iadd__(self, other):
        if not isinstance(other, (int, type(self))):
            msg = 'Can only add an integer or MPCounter object to an '
            msg += 'MPCounter object.'
            raise TypeError(msg)
        if isinstance(other, int):
            self.increment(other)
        else:
            self.increment(other.value)
    
    def __eq__(self, other):
        if not isinstance(other, (int, type(self))):
            msg = 'Can only compare to int or MPCounter.'
            raise TypeError(msg)
        if isinstance(other, int):
            return self.value == other
        return self.value == other.value
    
    @property
    def value(self):
        return self.val.value

class NamedPSDCache(object):
    def __init__(self, psd_names=None):
        if psd_names is None:
            self.psd_cache = {}
        else:
            self.psd_cache = {key: {} for key in input_to_list(psd_names)}
    
    def get(self, length, delta_f, low_freq_cutoff, psd_name=None):
        if psd_name is None:
            if len(self.psd_cache) > 1:
                msg = 'A PSD-name must be provided when {} stores more '
                msg += 'than one type of PSD.'
                msg = msg.format(self.__class__.__name__)
                raise ValueError(msg)
            else:
                psd_name = list(self.psd_cache.keys())[0]
            
        ident = (length, delta_f, low_freq_cutoff)
        if psd_name not in self.psd_cache:
            self.psd_cache[psd_name] = {}
        
        curr_cache = self.psd_cache[psd_name]
        if ident in curr_cache:
            return curr_cache[ident]
        else:
            psd = from_string(psd_name, *ident)
            self.psd_cache[psd_name][ident] = psd
            return psd
    
    def get_from_timeseries(self, timeseries, low_freq_cutoff,
                            psd_name=None):
        if not isinstance(timeseries, TimeSeries):
            msg = 'Input must be a pycbc.types.TimeSeries. Got type {} '
            msg += 'instead.'
            msg = msg.format(type(timeseries))
            raise TypeError(msg)
        length = len(timeseries) // 2 + 1
        delta_f = timeseries.delta_f
        return self.get(length, delta_f, low_freq_cutoff,
                        psd_name=psd_name)
    
    def get_from_frequencyseries(self, frequencyseries, low_freq_cutoff,
                                 psd_name=None):
        if not isinstance(frequencyseries, FrequencySeries):
            msg = 'Input must be a pycbc.types.FrequencySeries. Got type'
            msg += ' {} instead.'
            msg = msg.format(type(frequencyseries))
            raise TypeError(msg)
        length = len(frequencyseries)
        delta_f = frequencyseries.delta_f
        return self.get(length, delta_f, low_freq_cutoff,
                        psd_name=psd_name)
