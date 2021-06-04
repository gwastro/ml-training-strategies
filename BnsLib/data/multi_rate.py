import numpy as np
from scipy.signal import resample
from pycbc.types import TimeSeries
from pycbc.waveform import get_fd_waveform
from pycbc.waveform.spa_tmplt import spa_length_in_time

def multi_rate_sample(ts, samples_per_part, sample_rates, reverse=False, keep_end=True):
    """Function to re-sample a given time series at multiple rates.
    
    Arguments
    ---------
    ts : pycbc.TimeSeries
        The time series to be re-sampled.
    samples_per_part : int
        How many samples each part should contain.
    sample_rates : list of int
        A list of the sample rates that should be used.
        If reverse is False the first entry corresponds
        to the sample rate that should be used for the
        final part of the time series. The second entry
        corresponds to the part prior to the final part
        and does not overlap.
    reverse : {bool, False}
        Set this to True, if the first sample rate in
        sample_rates should re-sample the inital part of
        the time series. (i.e. re-sampling happens in
        a time-ordered manner)
    keep_end : {bool, True}
        If the re-sampled time series is shorter than the
        original time series, this option specifies if
        the original time-series is cropped in the beginning
        or end. (Default: cropped in the beginning)
    
    Returns
    -------
    re-sampled : list of pycbc.TimeSeries
        A list of pycbc.TimeSeries containing the re-sampled
        data. The time series are ordered, such that the
        first list entry corresponds to the initial part
        of the time series and the final entry corresponds
        to the final part of the waveform.
    
    Examples
    --------
    -We want to re-sample a time series of duration 10s.
     Each part should 400 samples. We want to sample
     the initial part of the time series with a sample
     rate of 50 and the part after that using a sample
     rate of 200. Therefore second 0 to 8 would be sampled
     at a rate of 50 and second 8 to 10 using a rate of 200.
     We could use the call:
     multi_rate_sample(ts, 400, [200, 50])
     or
     multi_rate_sample(ts, 400, [50, 200], reverse=True)
     
     We would receive the return
     [TimeSeries(start_time=0, end_time=8, sample_rate=50),
      TimeSeries(start_time=8, end_time=10, sample_rate=200)]
     in both cases.
    
    -We want to re-sample a time series of duration 10s. We want
     each part to contain 400 samples and want to use the sample
     rates [400, 50]. The re-sampled time series would be of
     total duration 9s, as sampling 400 samples with a rate of
     400 yields 1s and sampling 400 samples with a rate of 50 would
     yield 8s. The function call would be either
     multi_rate_sample(ts, 400, [400, 50])
     or
     multi_rate_sample(ts, 400, [50, 400], reverse=True)
     
     with the output
     [TimeSeries(start_time=1, end_time=9, sample_rate=50),
      TimeSeries(start_time=9, end_time=10, sample_rate=400)]
    """
    if reverse:
        sample_rates = sample_rates.copy()
        sample_rates.reverse()
    sample_rates = np.array(sample_rates, dtype=int)
    samples_per_part = int(samples_per_part)
    durations = float(samples_per_part) / sample_rates
    total_duration = sum(durations)
    if total_duration > ts.duration:
        msg = 'Cannot re-sample a time series of duration '
        msg += f'{ts.duration} with sample-rates {sample_rates} and '
        msg += f'samples per part {samples_per_part}.'
        ValueError(msg)
    parts = []
    last_time = ts.end_time
    if not keep_end:
        sample_rates = list(sample_rates)
        sample_rates.reverse()
        sample_rates = np.array(sample_rates, dtype=int)
        last_time = ts.start_time
    for i, sr in enumerate(sample_rates):
        resampled_data, resampled_t = resample(ts.data,
                                               int(len(ts) / ts.sample_rate * sr),
                                               t=ts.sample_times)
        delta_t = resampled_t[1] - resampled_t[0]
        epoch = resampled_t[0]
        resampled = TimeSeries(resampled_data, delta_t=delta_t,
                               epoch=epoch)
        if keep_end:
            diff = samples_per_part / sr
            st = max(last_time-2*diff, ts.start_time)
            tmp = resampled.time_slice(st, last_time)
            sidx = len(tmp) - samples_per_part
            eidx = len(tmp)
            parts.append(tmp[sidx:eidx])
            last_time = parts[-1].start_time
        else:
            diff = samples_per_part / sr
            et = min(last_time+2*diff, ts.end_time)
            tmp = resampled.time_slice(last_time, et)
            sidx = 0
            eidx = samples_per_part
            parts.append(tmp[sidx:eidx])
            last_time = parts[-1].end_time
    if keep_end:
        parts.reverse()
    return parts

def get_ideal_sample_rates(flow, fhigh, samples_per_part, mass1_min,
                           mass2_min, min_time=32., time_variance=0.25):
    """Generate the optimal sample-rates to use to resolve a signal
    generated by TyalorF2. The sample-rates are optimal in the sense
    that the total number of data-samples is minimal and a certain
    minimal time is covered.
    
    THIS METHOD DOES NOT NECESSARILY RETURN REASONABLE SAMPLE-RATES FOR
    ANY OTHER WAVEFORM APPROXIMANTS!
    
    Arguments
    ---------
    flow : float
        The lower frequency cutoff that is used.
    fhigh : float
        The highest frequency that should be resolved. This means that
        the Nyquist-frequency will be twice this frequency.
    samples_per_part : int
        The signal is sampled using multiple rates. Each part of the
        signal that has a different sample-rate will have the same
        number of samples. The number of samples for each part is given
        in this argument.
    mass1_min : float
        The minimum primary mass in units solar masses.
    mass2_min : float
        The minimum secondary mass in units solar masses.
    min_time : {float, 32.}
        The minimal time that should be spanned by that final template.
        (in seconds)
    time_variance : {float, 0.25}
        For neural networks the peak-amplitude of the signals are
        usually varied around a central position by +- some time x. This
        argument gives the amount of time (in seconds) by which the peak
        is varied.
    
    Returns
    -------
    sample_rates : list of int
        The optimal sample rates to use for the waveform. The first
        sample-rate corresponds to the final part of the waveform.
    time until merger : list of float
        The amount of time (in seconds) covered by the waveform when the
        re-sampling is cutoff at the corresponding rate.
    """
    def t_from_f(f):
        return spa_length_in_time(mass1=mass1_min,
                                  mass2=mass2_min,
                                  f_lower=f,
                                  phase_order=-1)
    
    hpt, hct = get_fd_waveform(approximant='TaylorF2',
                               mass1=mass1_min,
                               mass2=mass2_min,
                               f_lower=flow,
                               delta_f=1./512
                              )
    
    f = np.array(hpt.sample_frequencies)
    idxs = np.where(f > flow)
    f = f[idxs]
    f = list(f)
    f.reverse()
    f = np.array(f)
    t = np.array([t_from_f(pt) for pt in f])

    def f_from_t(time):
        return np.interp(time, t, f)
    
    sr = [int(np.ceil(2 * fhigh))] #sample rate (sr)
    time_variance = 2 * time_variance #Adjust for the fact that the waveform is shifted by +/- time_variance
    init_time_till_merger = samples_per_part / sr[-1] - time_variance
    if init_time_till_merger <= 0:
        msg = 'Using too few samples per part.'
        raise ValueError(msg)
    ttm = [init_time_till_merger] #time till merger (ttm) [cumulative]
    while ttm[-1] < min_time:
        best_sr = int(np.ceil(f_from_t(ttm[-1]))) * 2
        sr.append(best_sr)
        time_till_merger = ttm[-1] + samples_per_part / sr[-1] - time_variance
        ttm.append(time_till_merger)
    #print(ttm)
    return sr, ttm
