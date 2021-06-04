import numpy as np
from pycbc.sensitivity import volume_montecarlo
from queue import Queue
import multiprocessing as mp

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
