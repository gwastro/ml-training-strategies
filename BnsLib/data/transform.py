import numpy as np
from pycbc.types import TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation, from_string
from pycbc.psd import aLIGOZeroDetHighPower as aPSD
from pycbc.filter import sigma
from BnsLib.utils.formatting import input_to_list, list_length

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

def rescale_snr(signal, old_snr, new_snr):
    """Rescale a pycbc.TimeSeries or pycbc.FrequencySeries to a given
    signal-to-noise ratio.
    
    Arguments
    ---------
    signal : pycbc.TimeSeries or pycbc.FrequencySeries
        The data that should be rescaled.
    old_snr : float
        The signal-to-noise ratio of the input signal.
    new_snr : float
        The signal-to-noise ratio the output signal should have.
    
    Returns
    -------
    pycbc.TimeSeries or pycbcFrequencySeries
        Returns the signal (same data-type as input) after rescaling.
    """
    return signal / old_snr * new_snr

def rescale_signal(signal, new_snr, old_snr=None,
                   psd='aLIGOZeroDetHighPower', low_freq_cutoff=None,
                   high_freq_cutoff=None):
    """Rescale a pycbc.TimeSeries or pycbc.FrequencySeries to a given
    signal-to-noise ratio.
    
    Arguments
    ---------
    signal : pycbc.TimeSeries or pycbc.FrequencySeries
        The data that should be rescaled.
    new_snr : float
        The signal-to-noise ratio the output signal should have.
    old_snr : {float or None, None}
        The signal-to-noise ratio of the input signal. If None the
        optimal SNR of the signal will be calculated assuming the given
        PSD.
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
    pycbc.TimeSeries or pycbcFrequencySeries
        Returns the signal (same data-type as input) after rescaling.
    """
    if old_snr is None:
        old_snr = optimal_snr(signal, psd=psd,
                              low_freq_cutoff=low_freq_cutoff,
                              high_freq_cutoff=high_freq_cutoff)
    return rescale_snr(signal, old_snr, new_snr)

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

def optimal_network_snr(signals, psds='aLIGOZeroDetHighPower',
                        low_freq_cutoffs=None, high_freq_cutoffs=None):
    """Returns the optimal network signal-to-noise ratio for a list
    of signals and noise curves.
    
    signals : list of (pycbc.TimeSeries or pycbc.FrequencySeries)
        A list of the signals to analyze. This argument may be a single
        time or frequency series.
    psds : {list of (str or None or pycbc.FrequencySeries), 'aLIGOZeroDetHighPower'}
        A list of power spectrums to use for the different signals. The
        length has to match the length of signals. May be a single value
        instead of a list. If a single value is provided, the same value
        is assumed for all signals. All entries have to be interpretable
        by the function optimal_snr.
    low_freq_cutoffs : {list of (float or None), None}
        A list of low frequency cutoffs. See the documentation of
        optimal_snr for more details on this option. May be a single
        value. If a single value is provided, the same value is assumed
        for all signals.
    high_freq_cutoffs : {list of (float or None), None}
        A list of high frequency cutoffs. See the documentation of
        optimal_snr for more details on this option. May be a single
        value. If a single value is provided, the same value is assumed
        for all signals.
    
    Returns
    -------
    float
        The optimal network signal-to-noise ratio given the signals and
        the noise curves (power spectra).
    """
    length = list_length(signals)
    signals = input_to_list(signals, length=length)
    psds = input_to_list(psds, length=length)
    low_freq_cutoffs = input_to_list(low_freq_cutoffs, length=length)
    high_freq_cutoffs = input_to_list(high_freq_cutoffs, length=length)
    snrs = []
    for i in range(length):
        snrs.append(optimal_snr(signals[i],
                                psd=psds[i],
                                low_freq_cutoff=low_freq_cutoffs[i],
                                high_freq_cutoff=high_freq_cutoffs[i]))
    return np.sqrt(np.sum(np.square(snrs)))
