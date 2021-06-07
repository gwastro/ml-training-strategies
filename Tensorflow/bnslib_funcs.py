from tensorflow import keras
import csv
import warnings
import numpy as np
import configparser

special_values={
    'pi': np.pi,
    'Pi': np.pi,
    'PI': np.pi,
    'np.pi': np.pi,
    'numpy.pi': np.pi,
    'e': np.e,
    'np.e': np.e,
    'numpy.e': np.e,
    'null': None,
    'none': None,
    'None': None,
    'true': True,
    'True': True,
    'false': False,
    'False': False
    }

known_functions={
    'sin': (np.sin, 1),
    'cos': (np.cos, 1),
    'tan': (np.tan, 1),
    'exp': (np.exp, 1),
    '+': (lambda x, y: x+y, 2),
    '-': (lambda x, y: x-y, 2),
    '*': (lambda x, y: x*y, 2),
    '/': (lambda x, y: x/y, 2)
    }

def safe_min(inp):
    try:
        return min(inp)
    except TypeError:
        return min([inp])

def safe_max(inp):
    try:
        return max(inp)
    except TypeError:
        return max([inp])

def get_config_value(inp, constants=None, functions=None):
    """Convert the string returned by ConfigParser to an Python
    expression.
    
    This function uses some special formatting rules. The string may
    contain some special known value like Pi or None. These will be
    converted accordingly. Furthermore, certain operations (+, -, ...)
    and named functions (sin, cos, ...) are supported.
    
    Arguments
    ---------
    inp : str
        The string returned by ConfigParser.get.
    constants : {dict or None, None}
        A dictionary containing constants. The keys specify sub-strings
        of the input string that should be replaced by the values. All
        values are cast to floats.
    functions : {dict or None, None}
        A dictionary containing functions. The keys specify sub-strings
        of the input string that should be replaced by the functions.
        If a function takes more than one argument the value must be a
        tuple, where the first entry is the callable function and the
        second entry specifies the number of arguments the function
        accepts.
    
    Returns
    -------
    expression:
        Tries to interpret the string as a Python expression. This is
        done by parsing the string.
    
    Notes
    -----
    -For details on how the string is parsed see
     BnsLib.utils.config.ExpressionString.
    -To make sure the config-value is understood as a string encapsulate
     it in quotation marks, i.e. "string" or 'string'.
    -The parsing of all strings that are not encapsulated in quotation
     marks is handled by BnsLib.utils.config.ExpressionString. If this
     parser cannot resolve the string to a Python expression the value
     is returned as a string.
    """
    inp = inp.lstrip().rstrip()
    if len(inp) > 1:
        #Test for string
        if inp[0] in ["'", '"'] and inp[-1] in ["'", '"']:
            return inp[1:-1]
        #Test for list
        elif inp[0] == '[' and inp[-1] == ']':
            #TODO: Add this functionality to ExpressionString?
            ret = []
            if len(inp[1:-1]) == 0:
                return ret
            parts = ['']
            in_string = False
            open_brackets = 0
            for char in inp[1:-1]:
                if char == ',' and open_brackets == 0 and not in_string:
                    parts.append('')
                elif char in ['(', '[', '{']:
                    open_brackets += 1
                    parts[-1] += char
                elif char in [')', ']', '{']:
                    open_brackets -= 1
                    parts[-1] += char
                elif char in ['"', "'"]:
                    if in_string:
                        in_string = False
                    else:
                        in_string = True
                    parts[-1] += char
                else:
                    parts[-1] += char
            for pt in parts:
                ret.append(get_config_value(pt,
                                            constants=constants,
                                            functions=functions))
            return ret
        #Test for dict
        elif inp[0] == '{' and inp[-1] == '}':
            #TODO: Add this functionality to ExpressionString?
            ret = {}
            if len(inp[1:-1]) == 0:
                return ret
            parts = ['']
            in_string = False
            open_brackets = 0
            for char in inp[1:-1]:
                if char == ',' and open_brackets == 0 and not in_string:
                    parts.append('')
                elif char in ['(', '[', '{']:
                    open_brackets += 1
                    parts[-1] += char
                elif char in [')', ']', '}']:
                    open_brackets -= 1
                    parts[-1] += char
                elif char in ['"', "'"]:
                    if in_string:
                        in_string = False
                    else:
                        in_string = True
                    parts[-1] += char
                else:
                    parts[-1] += char
            #return parts
            for pt in parts:
                tmp = pt.split(':')
                if len(tmp) != 2:
                    raise ValueError
                key = get_config_value(tmp[0],
                                       constants=constants,
                                       functions=functions)
                value = get_config_value(tmp[1],
                                         constants=constants,
                                         functions=functions)
                
                ret[key] = value
            return ret
    
    if constants is None:
        constants = {}
    
    if functions is None:
        functions = {}
    
    try:
        exp = ExpressionString()
        exp.add_named_values(constants)
        exp.add_functions(functions)
        res = exp.parse_string(inp)
    except:
        res = inp
    return res

def config_to_dict(*file_paths, split_sections=True, constants=None,
                   functions=None):
    """Convert one or multiple config files to a Python dictionary.
    
    Arguments
    ---------
    file_paths : str
        One or multiple paths to config files.
    split_sections : {bool, True}
        If this option is set to True the output dictionary will contain
        a key for every section and the value will be another dictionary
        that contains the key-value-pairs from the config files. If set
        to False a single dictionary will be returned. This dictionary
        contains the key-value-pairs from all sections.
    constants : {dict or None, None}
        A dictionary containing constants. The keys specify sub-strings
        of strings that should be replaced by the values. All values are
        cast to floats.
    functions : {dict or None, None}
        A dictionary containing functions. The keys specify sub-strings
        of strings that should be replaced by the functions. If a
        function takes more than one argument the value must be
        a tuple, where the first entry is the callable function and the
        second entry specifies the number of arguments the function
        accepts.
    
    Returns
    -------
    dict:
        If split_sections is True returns a dictionary where each key
        corresponds to a section.Each entry is furthermore a dictionary
        where each key corresponds to a name in the config file and the
        value is the value of the variable in the config file. The
        values are parsed to be Python data-types and if possible
        converted to int or float.
        If split_sections is False a single dictionary will be returned.
        This dictionary contains the variables and their values from all
        sections.
    
    Notes
    -----
    -See BnsLib.utils.config.get_config_value and
     BnsLib.utils.config.ExpressionString for details on the conversion
     to Python data-types.
    """
    ret = {}
    conf = configparser.ConfigParser()
    conf.optionxform = str
    conf.read(*file_paths)
    for sec in conf.sections():
        tmp = {}
        for key in conf[sec].keys():
            tmp[key] = get_config_value(conf[sec][key],
                                        constants=constants,
                                        functions=functions)
        if split_sections:
            ret[sec] = tmp.copy()
        else:
            ret.update(tmp)
    return ret

def dict_to_string_dict(inp_dict):
    return {key: str(val) for (key, val) in inp_dict.items()}

def split_str_by_vars(string):
    """Split a string into variables and strings, where each variable is
    is engulfed in curly brackets.
    
    Arguments
    ---------
    string : str
        The string to split into parts.
    
    Returns
    -------
    list of str
        A list of strings. Each string is either a variable or a
        substring.
    
    Example
    -------
    >>> split_str_by_vars('{hello} world')
        ['{hello}', ' world']
    
    Notes
    -----
    Escape the variable characters { and } by prepending them with \.
    """
    if len(string) == 0:
        return [string]
    if string[0] == '{':
        ret = []
    else:
        ret = ['']
    in_var = False
    esc = False
    for c in string:
        if c == '}' and not esc:
            if in_var:
                ret[-1] += c
                ret.append('')
                in_var = False
                if esc:
                    esc = False
            else:
                raise ValueError
        elif c == '{' and not esc:
            if in_var:
                raise ValueError
            else:
                ret.append('')
                in_var = True
                ret[-1] += c
                if esc:
                    esc = False
        elif c == '\\':
            esc = True
        else:
            ret[-1] += c
            if esc:
                esc = False
    else:
        if in_var:
            raise ValueError('Could not parse the string.')
    
    if ret[-1] == '':
        ret.pop(-1)
    return ret

def inverse_string_format(string, form, types=None,
                          dynamic_types=False):
    """Returns None if the string does not match the format. Otherwise
    returns a dictionary of the values.
    
    Arguments
    ---------
    string : str
        The string that should be checked against the format string.
    form : str
        The format string that should be matched with the input string.
    types : {dict or None, None}
        A dictionary containing the types of each variable in the format
        string. If a variable is not in this dictionary behavior is
        dictated by the argument dynamic_types. If set to None no
        variable types are assumed and the behavior is entirely dictated
        by the argument dynamic_types.
    dynamic_types : {bool, False}
        Try to interpret the types of variables which do not have a type
        stated explicitly. If set to True a the function
        BnsLib.utils.config.get_config_value will be used to infer the
        type of the variable. If set to False the return type will be
        string.
    
    Returns
    -------
    ret : dict
        A dictionary where the key is the variable name and the value is
        the value of said variable.
    
    Examples
    --------
    >>> inverse_string_format('hello_world', '{greet}_{target}')
        {'greet': 'hello', 'target': 'world'}
    >>>
    >>> inverse_string_format('file-0', 'file-{num}', types={'num': int})
        {'num': 0}
    >>>
    >>> inverse_string_format('fil-0', 'file-{num}', types={'num': int})
        None
    >>> inverse_string_format('file-0', 'file-{num}', dynamic_types=False)
        {'num': '0'}
    >>> inverse_string_format('file-0', 'file-{num}', dynamic_types=True)
        {'num': 0}
    """
    parts = split_str_by_vars(form)
    var_names = []
    str_parts = []
    for pt in parts:
        if pt.startswith('{') and pt.endswith('}'):
            var_names.append(pt[1:-1])
            str_parts.append(None)
        else:
            var_names.append(None)
            str_parts.append(pt)
    
    if types is None:
        types = {}
    for var_name in var_names:
        if var_name not in types:
            if var_name is not None:
                if dynamic_types:
                    types[var_name] = get_config_value
                else:
                    types[var_name] = str
    
    if len(parts) == 1:
        if str_parts[0] is None == 1:
            return {var_names[0]: types[var_names[0]](string)}
        elif var_names[0] is None:
            if string == str_parts[0]:
                return {}
            else:
                return None
    
    values = []
    work_string = string
    part = 0
    if str_parts[0] is not None:
        if string.startswith(str_parts[0]):
            part = 1
            values.append(str_parts[0])
            work_string = work_string[len(str_parts[0]):]
        else:
            return None
    
    buff = ''
    work = ''
    i = 0
    sidx = 0
    while part + 1 < len(var_names):
        c = work_string[i]
        work += c
        if str_parts[part+1][sidx] == c:
            sidx += 1
        else:
            buff += work
            work = ''
            sidx = 0
        if len(work) == len(str_parts[part+1]):
            values.append(buff)
            values.append(work)
            buff = ''
            work = ''
            sidx = 0
            part += 2
        i += 1
        if i >= len(work_string):
            break;
    if part >= len(var_names) - 1:
        if var_names[-1] is not None:
            values.append(work_string[i:])
        elif i < len(work_string) -1:
            return None
            
    elif i >= len(work_string):
        return None
    
    if not len(values) == len(parts):
        raise RuntimeError
    
    ret = {}
    for i in range(len(parts)):
        if var_names[i] is not None:
            key = var_names[i]
            ret[key] = types[key](values[i])
    
    return ret

class PlateauDetection(keras.callbacks.Callback):
    """A callback that can be used to detect plateaus in a logged
    quantity.
    
    To use this Callback overwrite the method `on_plateau`, which is
    called whenever a plateau is detected.
    
    Arguments
    ---------
    patience : {int, 5}
        The number of iterations the monitored quantity is allowed to
        not improve before it is detected as plateau. (Meaning of
        iteration is set by `frequency`)
    threshold : {float, 1e-4}
        An improvement is only detected once it is bigger than the
        threshold. The threshold can be either an absolute or relative
        value. (Set in `mode`)
    best : {'min' or 'max', 'min'}
        Whether the monitored quantity should be minimized ('min') or
        maximized ('max').
    mode : {'abs' or 'rel', 'rel'}
        Whether the threshold is understood as an absolute or relative
        value. (Improvement to: old ± threshold ['abs'] or
        old * (1 ± threshold) ['rel'])
    monitor : {str, 'val_loss'}
        A quantity that is logged by the model during training.
    frequency : {int or str or tuple, 'epoch'}
        The frequency of iterations. May be in units of batches or
        epochs. To set write '<x> epochs' or '<x> batches', where you
        replace <x> by some integer. If <x> is not set, the default is
        1. If only an integer is given, epochs are assumed. A tuple has
        to contain a int in the first entry and either 'epoch' or
        'batch' in the second entry.
    """
    def __init__(self, patience=5, threshold=1.e-4, best='min',
                 mode='rel', monitor='val_loss', frequency='epoch',
                 **kwargs):
        super().__init__(**kwargs)
        self.patience = int(patience)
        self.threshold = float(threshold)
        assert mode.lower() in ['abs', 'rel']
        self.mode = mode.lower()
        assert best.lower() in ['min', 'max']
        self.best = best.lower()
        self.best_val = None
        self.monitor = str(monitor)
        self.set_frequency(frequency)
        self.last_improvement = 0
    
    def set_frequency(self, frequency):
        def remove_suffixes(string, suffixes):
            ret = ''
            for suffix in suffixes:
                if string.endswith(suffix):
                    ret = string[:-len(suffix)]
                    break
            return ret
        
        ep_suffixes = ['epoch', 'epochs', 'e', 'ep']
        ba_suffixes = ['batch', 'batches', 'b', 'ba']
        
        if isinstance(frequency, int):
            #If a number is give, default to every epoch
            self.frequency = (frequency, 'e')
        elif isinstance(frequency, str):
            frequency = frequency.replace(' ', '').lower()
            if any([frequency.endswith(pt) for pt in ep_suffixes]):
                frequency = remove_suffixes(frequency, ep_suffixes)
                if frequency == '':
                    frequency = '1'
                self.frequency = (int(frequency), 'e')
            elif any([frequency.endswith(pt) for pt in ba_suffixes]):
                frequency = remove_suffixes(frequency, ba_suffixes)
                if frequency == '':
                    frequency = '1'
                self.frequency = (int(frequency), 'b')
            else:
                msg = 'When setting the frequency of {} through a string'
                msg += ', the string must be an integer followed by the '
                msg += 'following suffixes: {}.'
                msg = msg.format(self.__class__.__name__, ep_suffixes + ba_suffixes)
                raise ValueError(msg)
        elif isinstance(frequency, tuple):
            if isinstance(frequency[0], int) and frequency[1].lower() in ep_suffixes:
                self.frequency = (frequency[0], 'e')
            elif isinstance(frequency[0], int) and frequency[1].lower() in ba_suffixes:
                self.frequency = (frequency[0], 'b')
            else:
                raise ValueError
        else:
            msg = 'The frequency of {0} must be either an integer or a '
            msg += 'string. See help({0}) for more information.'
            msg = msg.format(self.__class__.__name__)
            raise TypeError(msg)
    
    def check_plateau(self, val):
        if self.best_val is None:
            self.best_val = val
            self.last_improvement = 0
            return False
        old = self.best_val
        new = val
        if self.mode == 'rel':
            if self.best == 'min':
                better = (new < (1 - self.threshold) * old)
            elif self.best == 'max':
                better = (new > (1 + self.threshold) * old)
            else:
                raise RuntimeError('Unkown comaprison.')
        elif self.mode == 'abs':
            if self.best == 'min':
                better = (new < (old - self.threshold))
            elif self.best == 'max':
                better = (new > (old + self.threshold))
            else:
                raise RuntimeError('Unkown comaprison.')
        else:
            raise RuntimeError('Unknown mode {}.'.format(self.mode))
        if better:
            self.best_val = val
            self.last_improvement = 0
            return False
        else:
            self.last_improvement += 1
            return self.last_improvement > self.patience
    
    def on_train_batch_end(self, batch, logs=None):
        if self.frequency[1] == 'b' and batch % self.frequency[0] == 0:
            if self.check_plateau(logs[self.monitor]):
                self.on_plateau(batch, logs=logs)
        return
    
    def on_epoch_end(self, epoch, logs=None):
        if self.frequency[1] == 'e' and epoch % self.frequency[0] == 0:
            if self.check_plateau(logs[self.monitor]):
                self.on_plateau(epoch, logs=logs)
        return
    
    def on_plateau(self, stat, logs=None):
        return

class SensitivityEstimator(keras.callbacks.Callback):
    """A callback that monitors the sensitivity of the network.
    
    Arguments
    ---------
    signal_generator : keras.utils.Sequence
        A datatype that can be understood by keras.models.Model.predict.
        It must contain all signals that should be used for the
        analysis. If `snrs` is not None this generator needs to have a
        `rescale` method that takes a list of length 2 as input. The
        method is expected to rescale the signals to a specific
        SNR-range.
    threshold : {int or float or keras.utils.Sequence, 0.5}
        Either a int or a float to set a given threshold or a generator
        that can be understood by keras.models.Model.predict. If the
        threshold is not fixed, i.e. if threshold is a generator, the
        threshold will be determined dynamically. To do so the generator
        is expected to provide only noise samples. The loudest value
        predicted by the network will then be used as threshold value.
        All signals that are predicted with a higher statistic than this
        threshold are counted as true positive.
    file_path : {str, 'sensitivity_estimate.csv'}
        The path at which the sensitivities should be stored. (format is
        fixed to a CSV-file)
    save_freq : {int, 1}
        The number of epochs that should pass between estimating the
        sensitivity. After the sensitivity is estimated it will be
        written to the CSV-file.
    transform_function : {None or callable, None}
        A function that transforms the output of the network to a 1D
        numpy array. The contents are assumed to be the ranking
        statistic. (i.e. larger value ~= more confident detection)
        If set to None the output of the network must be a 1D numpy
        array.
    snrs : {None or list of int/float or list of list, None}
        A list of SNRs to which the generator will be rescaled. If the
        list contains int or float the generator will be rescaled by
        `generator.rescale([val, val])`. If the list contains anything
        but int or float it will be passed to the rescale function of
        the generator directly. If set to None the generator will not be
        rescaled and only a single column will be created in the CSV
        file.
    header : {bool or list or None, None}
        The header to use for the CSV-file. If set to None a standard
        header will be generated and used. If set to False no header
        will be generated. If set to a list, this list will be used as a
        header if it contains the correct number of items.
        (len(snrs) + 2 for
        ['epoch', snr sensitivities, 'mean of snr sensitivities'])
    verbose : {int, 1}
        How much information is printed during evaluation of the
        sensitivity.
    fap : {None or int > 0, None}
        The false-alarm probability at which the efficiency should be
        calculated. This argument is ignored when threshold is no
        keras.utils.Sequence. When set to None a false-alarm probability
        of 1 / #noise samples is used. If the requested false-alarm
        probability is smaller than 1 / #noise samples, the false-alarm
        probability 1 / #noise samples is used.
    **kwargs : 
        Remaining keyword-arguments are passed to the base-class.
    """
    def __init__(self, signal_generator, threshold=0.5,
                 file_path='sensitivity_estimate.csv', save_freq=1,
                 transform_function=None, snrs=None, header=None,
                 verbose=1, fap=None, **kwargs):
        super().__init__(**kwargs)
        self.signal_generator = signal_generator
        self.threshold = threshold
        self.file_path = file_path
        self.save_freq = save_freq
        self.snrs = snrs
        self.verbose = verbose
        self.fap = fap
        if header is None:
            self.header = None
        else:
            target_length = 3 if snrs is None else len(snrs) + 2
            if len(header) != target_length:
                msg = 'The provided header does not contain the correct'
                msg += ' number of columns. Expected {} columns but got'
                msg += ' {} instead. ({} for different SNRs, 1 for '
                msg += 'epochs, 1 for the mean sensitivity) Using the '
                msg += 'default header.'
                msg = msg.format(target_length, len(header), target_length - 2)
                warnings.warn(msg, RuntimeWarning)
                self.header = None
            else:
                self.header = header
        if transform_function is None:
            self.transform = lambda res: res
        else:
            self.transform = transform_function
    
    def on_train_begin(self, logs=None):
        if isinstance(self.header, bool) and not self.header:
            return
        if self.header is None or self.header == True:
            header = ["Epoch (one based)"]
            if self.snrs is None:
                header.append("True positive rate for entire sample range")
            else:
                for snr in self.snrs:
                    header.append("True positive rate for SNR {}".format(snr))
            header.append("Mean true positive rate")
        else:
            header = self.header
        self.header = header
        
        with open(self.file_path, 'w') as fp:
            csv_writer = csv.writer(fp, delimiter=',', quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(header)
    
    def on_train_end(self, logs=None):
        return
    
    def on_epoch_end(self, epoch, logs=None):
        epoch = epoch + 1
        if epoch % self.save_freq != 0:
            return
        old_target = self.signal_generator.target
        if self.verbose > 0:
            print("Estimating the sensitivity of the network")
        if isinstance(self.threshold, (int, float)):
            threshold = self.threshold
        else:
            noise_floor = self.model.predict(self.threshold,
                                             verbose=self.verbose)
            noise_floor = self.transform(noise_floor)
            if self.fap is None or 1 / self.fap > len(noise_floor):
                threshold = noise_floor.max()
            else:
                noise_floor.sort()
                loudest_n = int(len(noise_floor) * self.fap)
                threshold = noise_floor[-loudest_n]
        
        sens = []
        if self.snrs is None:
            signals = self.model.predict(self.signal_generator,
                                         verbose=self.verbose)
            signals = self.transform(signals)
            true_positives = len(np.where(signals > threshold)[0])
            sens.append(float(true_positives) / len(signals))
        else:
            for snr in self.snrs:
                if isinstance(snr, (int, float)):
                    snr = [snr, snr]
                self.signal_generator.rescale(snr)
                signals = self.model.predict(self.signal_generator,
                                             verbose=self.verbose)
                signals = self.transform(signals)
                true_positives = len(np.where(signals > threshold)[0])
                sens.append(float(true_positives) / len(signals))
        self.write_to_file(epoch, sens)
        if self.verbose > 0:
            print(self.header)
            print([epoch] + sens + [np.mean(sens)])
            print('')
        self.signal_generator.rescale(old_target)
    
    def write_to_file(self, epoch, sens):
        with open(self.file_path, 'a') as fp:
            csv_writer = csv.writer(fp, delimiter=',',
                                    quotechar='"',
                                    quoting=csv.QUOTE_MINIMAL)
            row = [epoch] + sens + [np.mean(sens)]
            csv_writer.writerow(row)

class SnrPlateauScheduler(PlateauDetection):
    """A scheduler that lowers the target SNR of a given generator by
    a given amount when a plateau in the monitored metric is detected.
    
    Arguments
    ---------
    generator : keras.utils.Sequence or Generator
        The generator that provides the data-samples to the network.
        This generator needs to have an attribute `target` and a method
        `rescale`. For details see the notes section.
    lower_by : {float, 5.}
        The value by which the SNR should be lowered.
    lower_strat : {'abs' or 'rel', 'abs'}
        Whether to lower the target SNR by an absolute value ('abs') or
        relative to the current value. If set to 'rel' the new target
        will be `old_target * lower_by`. Thus, lower_by should be set to
        a value between 0 and 1 in this case.
    min_snr : {float, 5.}
        The SNR-value at which the training SNR will not be lowered
        anymore.
    patience : {int, 5}
        The number of iterations the monitored quantity is allowed to
        not improve before it is detected as plateau. (Meaning of
        iteration is set by `frequency`)
    threshold : {float, 1e-4}
        An improvement is only detected once it is bigger than the
        threshold. The threshold can be either an absolute or relative
        value. (Set in `mode`)
    best : {'min' or 'max', 'min'}
        Whether the monitored quantity should be minimized ('min') or
        maximized ('max').
    mode : {'abs' or 'rel', 'rel'}
        Whether the threshold is understood as an absolute or relative
        value. (Improvement to: old ± threshold ['abs'] or
        old * (1 ± threshold) ['rel'])
    monitor : {str, 'val_loss'}
        A quantity that is logged by the model during training.
    frequency : {int or str or tuple, 'epoch'}
        The frequency of iterations. May be in units of batches or
        epochs. To set write '<x> epochs' or '<x> batches', where you
        replace <x> by some integer. If <x> is not set, the default is
        1. If only an integer is given, epochs are assumed. A tuple has
        to contain a int in the first entry and either 'epoch' or
        'batch' in the second entry.
    
    Notes
    -----
    -The `target` attribute of the generator should be a list of length
     2, where the first entry gives the lower bound and the second entry
     gives the upper bound. The SNR will be drawn from this range. To
     use a single value simply set the upper bound equal to the lower
     bound.
    -The `rescale` method must take a list of length 2 as argument. The
     list contains the new target value.
    """
    def __init__(self, generator, min_snr=5., lower_by=5.,
                 lower_strat='abs', **kwargs):
        super().__init__(**kwargs)
        self.generator = generator
        self.min_snr = min_snr
        self.lower_by = lower_by
        assert lower_strat.lower() in ['abs', 'rel']
        self.lower_strat = lower_strat.lower()
    
    def lower_snr(self):
        target = self.generator.target
        minsnr = safe_min(target)
        maxsnr = safe_max(target)
        if self.lower_strat == 'abs':
            newmin = max(self.min_snr, minsnr - self.lower_by)
            newmax = newmin + maxsnr - minsnr
        elif self.lower_strat == 'rel':
            newmin = max(self.min_snr, minsnr * self.lower_by)
            if newmin < minsnr:
                newmax = newmin + (maxsnr - minsnr) * self.lower_by
            else:
                newmax = maxsnr
        if newmin == newmax:
            new_target = newmin
        else:
            new_target = [newmin, newmax]
        self.generator.rescale(new_target)
        print('\nSet SNR of generator {} to {}.'.format(self.generator.__class__.__name__, new_target))
        return
    
    def on_plateau(self, stat, logs=None):
        self.lower_snr()
        self.last_improvement = 0
        self.best_val = None

class SnrCurriculumLearningScheduler(keras.callbacks.Callback):
    """A callback that schedules when the SNR of the data-samples should
    be re-scaled when training a neural network.
    
    Arguments
    ---------
    generator : keras.utils.Sequence or Generator
        The generator that provides the data-samples to the network.
        This generator needs to have an attribute `target` and a method
        `rescale`. For details see the notes section.
    lower_by_monitor : {floar or None, None}
        The target value for the monitored quantity. If this value is
        surpassed (in the direction specified by `mode`) the SNR is
        lowered by the specified amount. If set to None the SNR will not
        be lowered due to the monitored value. If `lower_by_epoch` is
        not None as well as this attribute, the SNR will be lowered when
        either target is hit.
    lower_by_epoch : {int or None, None}
        
    lower_at : {int or float, 10}
        DEPRECATED: Specifies when the SNR should be lowered. If set to
        an integer the SNR will be lowered after that number of epochs
        regardless of the monitored quantity. When set to a float the
        SNR will be lowered once the monitored quantity drops below this
        value. Please use `lower_by_epoch` or `lower_by_monitor`
        instead.
    lower_by : {float, 5.}
        The value by which the SNR should be lowered.
    lower_strat : {'abs' or 'rel', 'abs'}
        Whether to lower the target SNR by an absolute value ('abs') or
        relative to the current value. If set to 'rel' the new target
        will be `old_target * lower_by`. Thus, lower_by should be set to
        a value between 0 and 1 in this case.
    min_snr : {float, 5.}
        The SNR-value at which the training SNR will not be lowered
        anymore.
    monitor : {str, 'val_loss'}
        The quantity to monitor. Has to be a key of the logs that are
        produced by Keras. (i.e. any loss or any metric)
    mode : {'min' or 'max', 'min'}
        When to schedule a new training target. If set to 'min' a new
        target will be set when the monitored quantity falls below the
        given threshold value. If set to 'max' a new target will be set
        when the monitored quantity rises above the given threshold.
    
    Notes
    -----
    -The `target` attribute of the generator should be a list of length
     2, where the first entry gives the lower bound and the second entry
     gives the upper bound. The SNR will be drawn from this range. To
     use a single value simply set the upper bound equal to the lower
     bound.
    -The `rescale` method must take a list of length 2 as argument. The
     list contains the new target value.
    """
    def __init__(self, generator, lower_at=None, lower_by_monitor=None,
                 lower_by_epoch=None, lower_by=5., min_snr=5.,
                 monitor='val_loss', mode='min', lower_strat='abs'):
        self.generator = generator
        self.lower_by_monitor = lower_by_monitor
        self.lower_by_epoch = int(lower_by_epoch) if lower_by_epoch is not None else None
        if self.lower_by_epoch is None and self.lower_by_monitor is None:
            if lower_at is not None:
                msg = 'The attribute `lower_at` is deprecated and will '
                msg += 'be dropped in a future version. Please use '
                msg += '`lower_by_epoch` or `lower_by_monitor` instead.'
                warnings.warn(msg, DeprecationWarning)
                if isinstance(lower_at, int):
                    self.lower_by_epoch = lower_at
                elif isinstance(lower_at, float):
                    self.lower_by_monitor = lower_at
        self.lower_by = lower_by
        self.min_snr = min_snr
        self.monitor = monitor
        assert mode.lower() in ['min', 'max']
        self.mode = mode.lower()
        assert lower_strat.lower() in ['abs', 'rel']
        self.lower_strat = lower_strat.lower()
        self.last_update_epoch = -1
    
    def on_epoch_begin(self, epoch, logs=None):
        print("Training with target: {}".format(self.generator.target))
    
    def on_epoch_end(self, epoch, logs={}):
        lower = False
        if self.lower_by_epoch is not None:
            lower = lower or (epoch > 0) and (epoch - self.last_update_epoch > self.lower_by_epoch)
        if self.lower_by_monitor is not None:
            monitor = logs.get(self.monitor, 0.)
            if self.mode == 'min':
                lower = lower or (monitor < self.lower_by_monitor)
            elif self.mode == 'max':
                lower = lower or (monitor > self.lower_by_monitor)
            else:
                raise RuntimeError
        if lower:
            target = self.generator.target
            minsnr = safe_min(target)
            maxsnr = safe_max(target)
            if self.lower_strat == 'abs':
                newmin = max(self.min_snr, minsnr - self.lower_by)
                newmax = newmin + maxsnr - minsnr
            elif self.lower_strat == 'rel':
                newmin = max(self.min_snr, minsnr * self.lower_by)
                if newmin < minsnr:
                    newmax = newmin + (maxsnr - minsnr) * self.lower_by
                else:
                    newmax = maxsnr
            if newmin == newmax:
                new_target = newmin
            else:
                new_target = [newmin, newmax]
            self.generator.rescale(new_target)
            self.last_update_epoch = epoch
            print('\nSet SNR of generator {} to {} on epoch {}.'.format(self.generator.__class__.__name__, new_target, epoch))

class FileHandler(object):
    """Base class for interfacing with a single file. This class takes
    care of opening, closing, formatting and indexing.
    
    This class should not be used directly.
    
    To use this class standalone all methods should be implemented.
    To use this class as part of a MultiFileHandeler, the two methods
    `__enter__` and `__exit__` can be omitted.
    
    Arguments
    ---------
    file_path : str
        Path to the file that should be opened.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.file = None
    
    def __contains__(self, item):
        raise NotImplementedError
    
    def __enter__(self, mode='r'):
        raise NotImplementedError
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError
    
    def open(self, mode='r'):
        raise NotImplementedError
    
    def close(self):
        raise NotImplementedError

#To correct my earlier spelling error and keep backwards compatability
FileHandeler = FileHandler

class MultiFileHandler(object):
    """Base class for handeling multiple files at once. With this class
    the index-range or even a single index may span multiple files.
    
    Under specific circumstances (i.e. no formatting needs to be done)
    this class may be used directly.
    
    To implement this class, the attributes input_shape and output_shape
    need to be set in the __init__ method.
    The main functions one will want to overwrite are
    `split_index_to_groups` and `format_return`.
    
    Arguments
    ---------
    file_handlers : {list of FileHandeler or None, None}
        The list of FileHandelers that this class will have access to.
    mode : {str, 'r'}
        The mode in which files will be opened. (leave this as default
        if unsure of the consequences)
    """
    def __init__(self, file_handlers=None, mode='r'):
        self._init_file_handlers(file_handlers)
        self.mode = mode
        self.input_shape = None #Shape the network expects as input
        self.output_shape = None #Shape the network expects as labels
    
    def __contains__(self, idx):
        """Check if a return value can be constructed from the index
        using the FileHandelers known to this class.
        
        Arguments
        ---------
        idx : index as understood by `self.split_index_to_groups`
            An abstract index that will be interpreted by this class. It
            may in principle be of any type but has to be understood by
            `self.split_index_to_groups` and the results have to be
            understood by the FileHandelers.
        """
        contains = []
        index_split = self.split_index_to_groups(idx)
        for key, index in index_split.items():
            curr = False
            for file_handler in self.file_handler_groups[key]:
                if index in file_handler:
                    curr = True
                    break
            contains.append(curr)
        return all(contains)
    
    def __enter__(self):
        """Initialization code at the beginning of a `with` statement.
        """
        for file_handler in self.file_handlers:
            file_handler.open(mode=self.mode)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        """Cleanup-code when exiting a `with` statement.
        """
        for file_handler in self.file_handlers:
            file_handler.close()
    
    def __getitem__(self, idx):
        """Return the formatted item corresponding to the index.
        
        Arguments
        ---------
        idx : index as understood by `self.split_index_to_groups`
            An abstract index that will be interpreted by this class. It
            may in principle be of any type but has to be understood by
            `self.split_index_to_groups` and the results have to be
            understood by the FileHandelers.
        
        Returns
        -------
        object:
            Returns the formatted output as defined by
            `self.format_return`.
        """
        split_index = self.split_index_to_groups(idx)
        ret = {}
        for key, index in split_index.items():
            ret[key] = None
            for file_handler in self.file_handler_groups[key]:
                if index in file_handler:
                    ret[key] = file_handler[index]
        if any([val is None for val in ret.values()]):
            msg = 'The index {} was not found in any of the provided files.'
            raise IndexError(msg.format(idx))
        else:
            return self.format_return(ret)
    
    @property
    def file_handlers(self):
        """A list of all FileHandelers known to this MultiFileHandeler.
        """
        return self.file_handler_groups['all']
    file_handelers = file_handlers
    
    @property
    def file_handeler_groups(self):
        return self.file_handler_groups
    
    def _init_file_handlers(self, file_handlers):
        """Initialize the dictionary of known FileHandelers and their
        corresponding groups.
        
        Arguments
        ---------
        file_handlers : None or list of FileHandeler or dict
            If None, the dictionary of known FileHandelers will be
            initialized with a single, empty group called "all".
            If the argument is a list of FileHandelers the dictionary of
            known FileHandelers will contain a single group called "all"
            and all FileHandelers will belong only to this group.
            If the argument is a dict the keys are expected to be
            strings that specify the group of the FileHandelers. The
            values have to be either a single FileHandeler or a list of
            FileHandelers. All FileHandelers will be added to their
            specified group and the group "all".
        """
        if file_handlers is None:
            self.file_handler_groups = {'all': []}
        elif isinstance(file_handlers, list):
            self.file_handler_groups = {'all': file_handlers}
        elif isinstance(file_handlers, dict):
            self.file_handler_groups = {'all': []}
            for key in file_handlers.keys():
                if isinstance(file_handlers[key], FileHandeler):
                    self.add_file_handler(file_handlers[key],
                                           group=key)
                else:
                    for file_handler in file_handlers[key]:
                        self.add_file_handler(file_handler, group=key)
    
    def add_file_handler(self, file_handler, group=None):
        """Add a new FileHandeler to this MultiFileHandeler.
        
        Arguments
        ---------
        file_handler : FileHandeler
            The FileHandeler that should be added.
        group : {None or hashable, None}
            The group this FileHandeler should be added to. If None the
            FileHandeler will only be added to the group "all". If a
            group is specified the FileHandeler will be added to that
            group. If the group does not yet exist, it will be created.
            Even if a group is specified the FileHandeler will also be
            added to the group "all".
        """
        if group is not None:
            if group in self.file_handler_groups:
                self.file_handler_groups[group].append(file_handler)
            else:
                self.file_handler_groups[group] = [file_handler]
        self.file_handler_groups['all'].append(file_handler)
    add_file_handeler = add_file_handler
    
    def remove_file_handler(self, file_handler):
        """Remove a FileHandeler from the MultiFileHandeler.
        
        Arguments
        ---------
        file_handler : FileHandeler
            The FileHandeler that should be removed.
        """
        for group in self.file_handler_groups.values():
            if file_handler in group:
                group.remove(file_handler)
    
    remove_file_handeler = remove_file_handler
    
    def split_index_to_groups(self, idx):
        """Process an abstract index and split it into the different
        groups.
        
        Arguments
        ---------
        idx : abstract index
            An abstract index that may be of any type.
        
        Returns
        -------
        dict:
            Returns a dictionary of indices. The keys correspond to the
            different groups. The values represent abstract indices that
            may be understood by all FileHandelers of the corresponding
            group.
        
        Examples
        --------
        -Suppose there are two distinct FileHandelers. The
         MultiFileHandeler should return the sum of two values from the
         two FileHandelers. Thus the index is not a simple integer but a
         tuple of two integers.
         Suppose the two FileHandelers are in different groups "val1"
         and "val2". Then this method could return the indices for both
         individual FileHandelers:
            def split_index_to_groups(self, idx):
                return {"val1": idx[0], "val2": idx[1]}
        -Suppose there is data split into two files. For some reason
         both FileHandelers are put into two different groups. Now
         FileHandeler 1 contains indices 0 to 5 and FileHandeler 2
         contains indices 6 to 10. However, both FileHandelers
         internally expect their indices starting at 0. One could
         therefore use this method to achieve this processing:
            def split_index_to_groups(self, idx):
                if idx < 6:
                    return {"group1": idx}
                else:
                    return {"group2": idx - 6}
        """
        return {'all': idx}
    
    def format_return(self, inp):
        """Format the output of the different groups to form a single
        input that can be interpreted by the FileGenerator.
        
        Arguments
        ---------
        inp : dict
            A dictionary containing the ouputs of the different groups.
        
        Returns
        -------
        object:
            Returns the formatted output. For the FileGenerator it
            should be of form
            (input to network, labels for network)
            or
            (input to network, labels for network, sample weights)
            where the input to network has to be a numpy array or list
            of numpy arrays of the shape given by model.input_shape. The
            labels of network has to be of shape given by
            model.output_shape. Sample weights only need to be returned
            if the FileGenerator sets use_sample_weights to True.
        """
        return inp['all']

MultiFileHandeler = MultiFileHandler

class FileGenerator(keras.utils.Sequence):
    """A Generator as required by Keras. It generates the samples from
    a FileHandeler or MultiFileHandeler.
    
    Arguments
    ---------
    file_handeler : FileHandeler or MultiFileHandeler
        The FileHandeler from which samples are retrieved.
    index_list : list of abstract indices
        The list of indices that will be considered by the generator.
        The indices may be abstract but have to be understood by the
        FileHandeler or MultiFileHandeler.
    batch_size : {int, 32}
        The size of each batch.
    shuffle : {bool, True}
        Whether or not to shuffle the index list on every epoch.
        (different order of samples on each epoch)
    use_sample_weights : {bool, False}
        Whether or not to return sample weights. (See details on sample
        weight in the documentation of keras.models.Model.fit)
    """
    #TODO: Write a function that logs the indices used and the file-path
    def __init__(self, file_handeler, index_list, 
                 batch_size=32, shuffle=True, use_sample_weights=False):
        self.file_handeler = file_handeler
        self.index_list = index_list
        self.indices = np.arange(len(self.index_list), dtype=int)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.use_sample_weights = use_sample_weights
        self.on_epoch_end()
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(float(len(self.index_list)) / self.batch_size))
    
    def __getitem__(self, idx):
        if (idx + 1) * self.batch_size > len(self.indices):
            batch = self.indices[idx*self.batch_size:]
        else:
            batch = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_size = len(batch)
        if isinstance(self.file_handeler.input_shape, list):
            X = [np.zeros([batch_size] + list(shape)) for shape in self.file_handeler.input_shape]
        else:
            X = np.zeros([batch_size]+ list(self.file_handeler.input_shape))
        
        if isinstance(self.file_handeler.output_shape, list):
            Y = [np.zeros([batch_size] + list(shape)) for shape in self.file_handeler.output_shape]
        else:
            Y = np.zeros([batch_size] + list(self.file_handeler.output_shape))
        
        if self.use_sample_weights:
            if isinstance(self.file_handeler.output_shape, list):
                W = [np.zeros(batch_size) for _ in len(self.file_handeler.output_shape)]
            else:
                W = np.zeros(batch_size)
        
        for num, i in enumerate(batch):
            if self.use_sample_weights:
                inp, out, wei = self.file_handeler[self.index_list[i]]
                if isinstance(wei, list):
                    for part, w in zip(wei, W):
                        w[num] = part
                else:
                    for w in W:
                        w[num] = wei
            else:
                inp, out = self.file_handeler[self.index_list[i]]
            
            if isinstance(inp, list):
                for part, x in zip(inp, X):
                    x[num] = part
            else:
                X[num] = inp
        
            if isinstance(out, list):
                for part, y in zip(out, Y):
                    y[num] = part
            else:
                Y[num] = out
        
        if self.use_sample_weights:
            return X, Y, W
        else:
            return X, Y

class BinaryTree(object):
    """A class that implements a basic binary tree that can be sorted by
    a few functions.
    
    Arguments
    ---------
    content : object
        The content this node of a tree should hold.
    parent : {BinaryTree or None, None}
        The BinaryTree-node of which this node is a child.
    
    Attributes
    ----------
    content : object
        The content held by this node.
    parent : BinaryTree or None
        The parent BinaryTree of which this node is a child.
    left : BinaryTree or None
        The left child of this node.
    right : BinaryTree or None
        The right child of this node.
    root : bool
        Whether or not this node is the root of the tree.
    leaf : bool
        True if this node has no children.
    """
    def __init__(self, content, parent=None):
        self.content = content
        self.parent = parent
        self.left = None
        self.right = None
    
    def copy(self):
        """Returns a new instance of itself.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        copy : BinaryTree
            A new instance of the tree from which it was called.
        """
        ret = BinaryTree(self.content, parent=self.parent)
        ret.set_left_child(self.left)
        ret.set_right_child(self.right)
        return ret
    
    def add_left_leaf(self, content):
        """Set the left child of this node to be a BinaryTree without
        any children.
        
        Arguments
        ---------
        content : object
            The content of the child node.
        
        Returns
        -------
        None
        """
        self.left = BinaryTree(content, parent=self)
    
    def add_right_leaf(self, content):
        """Set the right child of this node to be a BinaryTree without
        any children.
        
        Arguments
        ---------
        content : object
            The content of the child node.
        
        Returns
        -------
        None
        """
        self.right = BinaryTree(content, parent=self)
    
    def set_left_child(self, left):
        """Set the left child of this node.
        
        Arguments
        ---------
        left : BinaryTree
            The tree to be the left child of this node.
        
        Returns
        -------
        None
        """
        self.left = left
    
    def set_right_child(self, right):
        """Set the right child of this node.
        
        Arguments
        ---------
        left : BinaryTree
            The tree to be the right child of this node.
        
        Returns
        -------
        None
        """
        self.right = right
    
    @property
    def root(self):
        """Return True if this node has no parent node and thus is a
        root of a tree.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        bool:
            True if this node has no parent node and thus is a root of a
            tree.
        """
        return self.parent is None
    
    @property
    def leaf(self):
        """Return True if the node is a leaf of a tree.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        bool:
            True if the node has no children.
        """
        return self.left is None and self.right is None
    
    def mid_left_right(self):
        """Starting from this node go through the tree and return the
        contents of the node. Go in order node-content -> content of
        left sub-tree -> content of right sub-tree.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        list of object:
            The contents of the tree ordered:
            node content -> left content -> right content
        """
        ret = [self.content]
        if self.left is not None:
            ret.extend(self.left.mid_left_right())
        if self.right is not None:
            ret.extend(self.right.mid_left_right())
        return ret
    
    def left_right_mid(self):     
        """Starting from this node go through the tree and return the
        contents of the node. Go in order content of left sub-tree ->
        content of right sub-tree -> node-content.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        list of object:
            The contents of the tree ordered:
            left content -> right content -> node content
        """
        ret = []
        if self.left is not None:
            ret.extend(self.left.left_right_mid())
        if self.right is not None:
            ret.extend(self.right.left_right_mid())
        ret.append(self.content)
        return ret
    
    def right_left_mid(self):
        """Starting from this node go through the tree and return the
        contents of the node. Go in order content of right sub-tree ->
        content of left sub-tree -> node-content.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        list of object:
            The contents of the tree ordered:
            right content -> left content -> node content
        """
        ret = []
        if self.right is not None:
            ret.extend(self.right.right_left_mid())
        if self.left is not None:
            ret.extend(self.left.right_left_mid())
        ret.append(self.content)
        return ret

class ExpressionString(object):
    """A class that processes strings from config-files.
    
    This class expects the string to contain information that can be
    interpreted as floats, integers or functions applied to those. It
    provides a semi-safe environment that allows to parse only a set of
    pre-defined functions.
    
    This class is meant to handle "easy" cases, i.e. expressions which
    are not too complicated or convoluted. However, the parser is
    written in a way that should be robust to many situations and
    complicated structures. The main limitation is the requirement that
    the output, after applying all functions and replacing all named
    constants, must be convertable to a float or integer.
    
    Arguments
    ---------
    string : {str or None, None}
        The string that should be processed.
    
    Attributes
    ----------
    orig_string : str
        The original input to the class, if it is not overwritten by
        `set_string`.
    string : str
        The string that is worked on by the class. All parsing and
        manipulation to the content of the string is applied to this
        copy of the original string.
    level : nd.array of int
        An internal representation of execution order.
    min : int
        The minimum value of level.
    max : int
        The maximum value of level.
    
    Examples
    --------
    >>> from BnsLib.utils.config import ExpressionString
    >>> exstr = ExpressionString('2')
    >>> exstr.parse()
        2
    >>>
    >>> exstr.set_string('cos(pi)')
    >>> exstr.parse()
        -1.0
    >>>
    >>> exstr.parse_string('sin(2+1)*cos(3+pi)-e')
        -2.578574079359582
    >>>
    >>> exstr.add_named_value('a', 3)
    >>> exstr.parse_string('a')
        3.0
    """
    def __init__(self, string=None):
        self.set_string(string)
        self.known_functions = known_functions
        self.special_values = special_values
    
    def set_string(self, inp):
        """Set the main string of this class. All parsing function will
        apply to this string. (The exception being `parse_string`)
        
        Arguments
        ---------
        inp : str or None
            The string which should be parsed. If set to None a empty
            string will be used.
        
        Returns
        -------
        None
        """
        if inp == None:
            inp = ''
        else:
            inp = str(inp)
        self.orig_string = inp
        self.string = inp.replace(' ', '')
        self.level = np.zeros(len(inp))
    
    def __len__(self):
        """Returns the length of the working-string.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        int:
            The length of the working string (in number of characters).
        """
        return len(self.string)
    
    @property
    def min(self):
        """Returns the minimal value of level. The corresponding atomic
        expression will be parsed last and thus is the final action that
        takes place.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        int:
            Minimal value of level. 
        """
        return self.level.min()
    
    @property
    def max(self):
        """Returns the maximal value of level. The corresponding atomic
        expression will be parsed fiest and thus is the first action
        that takes place.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        int:
            Maximal value of level. 
        """
        return self.level.max()
    
    def add_named_value(self, name, value):
        """Add a value that can be replaced by the parser.
        
        Arguments
        ---------
        name : str
            The name of the value. The parser will search for this
            character sequence in the input string. If found and if the
            string is an atomic expression it will be replaced by the
            value.
        value : float or int
            The value to replace the string by. Its contents are cast to
            be floats.
        
        Returns
        -------
        None
        """
        self.special_values[name] = float(value)
    
    def add_named_values(self, value_dict):
        """Add multiple values to the parser at once.
        
        Arguments
        ---------
        value_dict : dict
            The dictionary that stores the named values. The key, value
            pairs will be passed to add_named_value. (See references
            therein)
        
        Returns
        -------
        None
        """
        for name, value in value_dict.items():
            self.add_named_value(name, value)
    
    def add_function(self, name, function, num_arguments=1):
        """Add a function to the parser.
        
        Arguments
        ---------
        name : str
            The name of the function that is searched for.
        function : callable
            The function to invoke.
        num_arguments : int
            The number of arguments the function takes.
        
        Returns
        -------
        None
        """
        #if num_arguments > 2:
            #msg = 'Currently functions with more than 2 arguments are '
            #msg += 'not supported. They may work, but the behavior of '
            #msg += 'the parser is relatively unpredictable and the '
            #msg += 'success may depend on the order of the arguments.'
            #warnings.warn(msg, RuntimeWarning)
        self.known_functions[name] = (function, num_arguments)
    
    def add_functions(self, function_dict):
        """Add multiple functions at once to the parser.
        
        Arguments
        ---------
        function_dict : dict
            A dictionary containing the keys and function specifier.
            The function specifier must be either a callable or a tuple
            of length two. In the later case the fisr entry must be a
            callable function and the second argument has to be an
            integer specifying the number of arguments the function
            takes. If just a callable is provided the number of
            arguments is expected to be 1.
        
        Returns
        -------
        None
        """
        for name, function_part in function_dict.items():
            try:
                func, num = function_part
            except TypeError:
                func = function_part
                num = 1
            self.add_function(name, func, num_arguments=num)
    
    def parse_commas(self):
        string = []
        ret_level = []
        level = list(self.level)
        for idx, char in enumerate(self.string):
            if char == ',':
                left_bound = 0
                for i in range(idx).__reversed__():
                    if level[i] <= level[idx]:
                        left = i
                level_diff = max(level[left:idx])-level[idx]+1
                for i in range(idx+1, len(self.string)):
                    if level[i] >= level[idx]:
                        level[i] += level_diff
                    else:
                        break
            else:
                string.append(char)
                ret_level.append(level[idx])
        self.string = ''.join(string)
        self.level = np.array(ret_level)
    
    def parse_brackets(self):
        string = []
        level = []
        current_level = 0
        for i in range(len(self.string)):
            if self.string[i] in ['(', '[']:
                current_level += 1
            elif self.string[i] in [')', ']']:
                current_level-= 1
            else:
                level.append(current_level)
                string.append(self.string[i])
        self.string = ''.join(string)
        self.level = np.array(level, dtype=int)
    
    def parse_summation(self):
        insert_zero = []
        for i_sub in range(len(self)):
            i = len(self)-1-i_sub
            if self.string[i] in ['+', '-']:
                left = i
                run = True
                while run:
                    if left <= 0:
                        run = False
                    elif self.level[left] < self.level[i]:
                        left += 1
                        run = False
                    else:
                        left -= 1
                right = i
                run = True
                while run:
                    if right >= len(self):
                        run = False
                    elif self.level[right] < self.level[i]:
                        run = False
                    else:
                        right += 1
                if self.string[i] == '-' and left == i:
                    insert_zero.append([i, self.level[i]+1])
                self.level[left:i] += 1
                self.level[i+1:right] += 1
        
        tmp_string = list(self.string)
        tmp_level = list(self.level)
        while len(insert_zero) > 0:
            curr_idx, curr_level = insert_zero.pop(0)
            for i in range(len(insert_zero)):
                insert_zero[i][0] += 1
            tmp_string.insert(curr_idx, '0')
            curr_level = self.level[curr_idx] + 1
            tmp_level.insert(curr_idx, curr_level)
        self.string = ''.join(tmp_string)
        self.level = np.array(tmp_level, dtype=int)
    
    def parse_product(self):
        for i in range(len(self)):
            if self.string[i] in ['*', '/']:
                left = i
                run = True
                while run:
                    if left <= 0:
                        run = False
                    elif self.level[left] < self.level[i]:
                        left += 1
                        run = False
                    else:
                        left -= 1
                right = i
                run = True
                while run:
                    if right >= len(self):
                        run = False
                    elif self.level[right] < self.level[i]:
                        run = False
                    else:
                        right += 1
                self.level[left:i] += 1
                self.level[i+1:right] += 1
    
    def parse_atomic(self, atomic):
        if atomic in self.known_functions:
            return self.known_functions[atomic]
        elif atomic in self.special_values:
            return self.special_values[atomic]
        else:
            #At this point we assume it is a number
            if '.' in atomic:
                return float(atomic)
            else:
                return int(atomic)
    
    def parse_to_atomic(self):
        atomics = [self.string[0]]
        atomic_levels = [self.level[0]]
        for char, level in zip(self.string[1:], self.level[1:]):
            if atomic_levels[-1] == level:
                atomics[-1] += char
            else:
                atomics.append(char)
                atomic_levels.append(level)
        atomics = [self.parse_atomic(atomic) for atomic in atomics]
        return atomics, atomic_levels
    
    def get_child_indices(self, atomics, atomic_levels, idx):
        left = (None, self.max+1)
        left_part = atomic_levels[:idx]
        if len(left_part) > 0:
            for i in range(1, len(left_part)+1):
                level = atomic_levels[idx - i]
                if level <= atomic_levels[idx]:
                    break
                elif level < left[1]:
                    left = (idx-i, level)
        
        right = (None, self.max+1)
        right_part = atomic_levels[idx+1:]
        if len(right_part) > 0:
            for i in range(1, len(right_part)+1):
                level = atomic_levels[idx+i]
                if level <= atomic_levels[idx]:
                    break
                elif level < right[1]:
                    right = (idx+i, level)
        
        return left[0], right[0]
    
    def tree_from_index(self, atomics, atomic_levels, idx, parent=None):
        if idx is None:
            return None
        ret = BinaryTree(atomics[idx], parent=parent)
        left_idx, right_idx = self.get_child_indices(atomics, atomic_levels, idx)
        ret.set_left_child(self.tree_from_index(atomics, atomic_levels, left_idx, parent=ret))
        ret.set_right_child(self.tree_from_index(atomics, atomic_levels, right_idx, parent=ret))
        return ret
    
    def atomics_to_tree(self, atomics, atomic_levels):
        if len(list(filter(lambda level: level == 0, atomic_levels))) > 1:
            raise RuntimeError('Binary tree needs to have a unique root.')
        
        root = np.where(np.array(atomic_levels, dtype=int) == 0)[0][0]
        tree = self.tree_from_index(atomics, atomic_levels, root)
        
        return tree
    
    def as_tree(self):
        atomics, atomic_levels = self.parse_to_atomic()
        return self.atomics_to_tree(atomics, atomic_levels)
    
    def carry_out_operations(self):
        tree = self.as_tree()
        order_operations = tree.right_left_mid()
        
        stack = []
        for i, operation in enumerate(order_operations):
            if isinstance(operation, tuple):
                args = []
                for i in range(operation[1]):
                    args.append(stack.pop())
                stack.append(operation[0](*args))
            else:
                stack.append(operation)
        
        return stack.pop()
    
    def parse(self):
        if self.orig_string == '':
            return None
        self.parse_brackets()
        self.parse_commas()
        self.parse_summation()
        self.parse_product()
        return self.carry_out_operations()
    
    def parse_string(self, string):
        backup_orig_string = self.orig_string
        backup_string = self.string
        backup_level = self.level
        self.set_string(string)
        res = self.parse()
        self.orig_string = backup_orig_string
        self.string = backup_string
        self.level = backup_level
        return res
    
    def print(self):
        print_string = ''
        for row in range(self.max+1):
            print_string += str(row) + ' '
            for col in range(len(self)):
                if row == self.level[col]:
                    print_string += self.string[col]
                else:
                    print_string += ' '
            print_string += '\n'
        print(print_string)
