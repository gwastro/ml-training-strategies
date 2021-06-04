import numpy as np
from pycbc.waveform import get_td_waveform, get_fd_waveform
from BnsLib.utils.formatting import input_to_list, list_length
from functools import wraps
from pycbc.detector import Detector
import warnings
from BnsLib.utils.progress_bar import progress_tracker, mp_progress_tracker
from BnsLib.types.utils import DictList, MPCounter
import multiprocessing as mp
from pycbc.types import TimeSeries
import datetime
from pycbc.noise import noise_from_string
from BnsLib.data.transform import whiten
import time

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

from pycbc.workflow import WorkflowConfigParser
from pycbc.distributions import read_params_from_config
from pycbc.distributions import read_distributions_from_config
from pycbc.distributions import read_constraints_from_config
from pycbc.distributions import JointDistribution
from pycbc.transforms import read_transforms_from_config, apply_transforms
class WFParamGenerator(object):
    """A class that takes in a configuration file and creates parameters
    from the described distributions.
    
    Arguments
    ---------
    config_file : str
        Path to the config-file that should be used.
    seed : {int, 0}
        Which seed should be used for the parameter-generation.
    
    Attributes
    ----------
    var_args : list of str
        A list containing the names of the variable arguments.
    static : dict
        A dictionary containing the static parameters. The keys are the
        names of the static parameters, whereas the according values are
        the values of the parameters.
    trans : list of pycbc.Transformation
        A list of transformations that are applied to the variable
        arguments.
    pval : pycbc.JointDistribution
        The joint distribution of the variable arguments. Parameters are
        drawn from this distribution and transformed according to the
        transformations.
    """
    def __init__(self, config_file, seed=None):
        if seed is None:
            seed = int(time.time())
        np.random.seed(seed)
        config_file = input_to_list(config_file)
        config_file = WorkflowConfigParser(config_file, None)
        self.var_args, self.static = read_params_from_config(config_file)
        constraints = read_constraints_from_config(config_file)
        dist = read_distributions_from_config(config_file)

        self.trans = read_transforms_from_config(config_file)
        self.pval = JointDistribution(self.var_args, *dist, 
                                **{"constraints": constraints})   
    
    def __contains__(self, name):
        """Returns true if the given name is a parameter name known to
        the generator.
        
        Arguments
        ---------
        name : str
            The name to search for.
        
        Returns
        -------
        bool:
            True if the name is either in the static parameters, the
            variable arguments or any of the transform outputs.
        """
        in_params = (name in self.var_args) or (name in self.static)
        in_trans = any([(name in trans.input) or (name in trans.output) for trans in self.trans])
        return in_params or in_trans
    
    def draw(self):
        """Draw a single set of parameters.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        pycbc.io.record.FieldArray:
        A field array, where each column consists of a numpy array with
        a single entry.
        """
        return apply_transforms(self.pval.rvs(), self.trans)
    
    def draw_multiple(self, num):
        """Draw multiple parameters at once. This approach is preferable
        over calling draw multiple times.
        
        Arguments
        ---------
        num : int
            The number of parameters to draw from the distribution.
        
        Returns
        -------
        pycbc.io.record.FieldArray:
        A field array, where each column consists of a numpy array with
        n entries. (n as specified by `num`)
        """
        return apply_transforms(self.pval.rvs(size=num), self.trans)
    
    def keys(self):
        """Returns the list of keys to the output.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        list of str:
            The list of keys as they are known for the output.
        """
        params_keys = set(self.var_args)
        static_keys = set(self.static.keys())
        trans_input = set()
        trans_output = set()
        for trans in self.trans:
            trans_input = trans_input.union(trans.input)
            trans_output = trans_output.union(trans.output)
        ret = params_keys.union(static_keys)
        ret = ret.difference(trans_input)
        ret = ret.union(trans_output)
        return list(ret)

class WaveformGenerator(WaveformGetter):
    def __init__(self, config_file, seed=0, domain='time',
                 detectors='H1'):
        self.params = WFParamGenerator(config_file, seed=seed)
        vp = {key: [] for key in self.params.pval.variable_args}
        sp = self.params.static
        super().__init__(variable_params=vp,
                         static_params=sp,
                         domain=domain,
                         detectors=detectors)
    
    def __next__(self):
        params = self.params.draw()
        for key in self.variable_params.keys():
            if key in params:
                self.variable_params[key].append(params[key][0])
        return self[len(self)-1]

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
