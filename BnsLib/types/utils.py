import multiprocessing as mp
from collections import OrderedDict
import sys
import numpy as np
from pycbc.psd import from_string

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

class NamedPSDCache(object):
    def __init__(self, psd_names=None):
        from BnsLib.utils.formatting import input_to_list
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
        from pycbc.types import TimeSeries
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
        from pycbc.types import FrequencySeries
        if not isinstance(frequencyseries, FrequencySeries):
            msg = 'Input must be a pycbc.types.FrequencySeries. Got type'
            msg += ' {} instead.'
            msg = msg.format(type(frequencyseries))
            raise TypeError(msg)
        length = len(frequencyseries)
        delta_f = frequencyseries.delta_f
        return self.get(length, delta_f, low_freq_cutoff,
                        psd_name=psd_name)

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

def memsize(obj):
    """Tries to approximate the size of an object. If the object is not
    a list or a tuple, the returned size will be sys.getsizeof(obj).
    Otherwise the function will recursively call itself until the
    argument is not a list or tuple anymore.
    
    Arguments
    ---------
    obj : object
        The object of which to infer the size of.
    
    Returns
    -------
    DataSize:
        The size of the object in bytes.
    """
    if isinstance(obj, (list, tuple)):
        return DataSize(sys.getsizeof(obj) + sum([memsize(pt) for pt in obj]))
    else:
        return DataSize(sys.getsizeof(obj))

class DataSize(object):
    """A data-type used to store and compare file-sizes.
    
    Arguments
    ---------
    size : int or DataSize or None
        The size of an object or file.
    unit : {str, None}
        The unit in which the size was given. (from bits: 'b', over
        bytes: 'B', up to petabytes 'PB')
    """
    dtypes = ['b', 'B', 'kb', 'kB', 'Mb', 'MB', 'Gb', 'GB', 'Tb', 'TB',
              'Pb', 'PB']
    to_bytes = {'b': lambda size: size // 8,
                'B': lambda size: size,
                'kb': lambda size: (size * 1000) // 8,
                'kB': lambda size: size * 1000,
                'Mb': lambda size: (1e6 * size) // 8,
                'MB': lambda size: 1e6 * size,
                'Gb': lambda size: (1e9 * size) // 8,
                'GB': lambda size: 1e9 * size,
                'Tb': lambda size: (1e12 * size) // 8,
                'TB': lambda size: 1e12 * size,
                'Pb': lambda size: (1e15 * size) // 8,
                'PB': lambda size: 1e15 * size,
               }
    from_bytes = {'b': lambda size: size * 8,
                  'B': lambda size: size,
                  'kb': lambda size: int((size / 1000) * 8),
                  'kB': lambda size: int(size / 1000),
                  'Mb': lambda size: int((size / 1e6) * 8),
                  'MB': lambda size: int(size / 1e6),
                  'Gb': lambda size: int((size / 1e9) * 8),
                  'GB': lambda size: int(size / 1e9),
                  'Tb': lambda size: int((size / 1e12) * 8),
                  'TB': lambda size: int(size / 1e12),
                  'Pb': lambda size: int((size / 1e15) // 8),
                  'PB': lambda size: int(size / 1e15),
                 }
    def __init__(self, size, unit='B'):
        if size is None:
            self.size = None
        elif isinstance(size, type(self)):
            self.size = self.from_bytes[unit](size.as_bytes())
        else:
            assert isinstance(size, (int, float))
            self.size = size
        assert unit in self.dtypes
        self.unit = unit
    
    def as_bytes(self):
        if self.size is None:
            return np.inf
        else:
            return self.to_bytes[self.unit](self.size)
    
    def convert(self, unit):
        return DataSize(self.from_bytes[unit](self.as_bytes()), unit=unit)
    convert_to = convert
    
    def __lt__(self, other):
        if isinstance(other, type(self)):
            return self.as_bytes().__lt__(other.as_bytes())
        else:
            return self.as_bytes().__lt__(other)
    
    def __le__(self, other):
        if isinstance(other, type(self)):
            return self.as_bytes().__le__(other.as_bytes())
        else:
            return self.as_bytes().__le__(other)
    
    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.as_bytes().__eq__(other.as_bytes())
        else:
            return self.as_bytes().__eq__(other)
    
    def __ne__(self, other):
        if isinstance(other, type(self)):
            return self.as_bytes().__ne__(other.as_bytes())
        else:
            return self.as_bytes().__ne__(other)
    
    def __gt__(self, other):
        if isinstance(other, type(self)):
            return self.as_bytes().__gt__(other.as_bytes())
        else:
            return self.as_bytes().__gt__(other)
    
    def __ge__(self, other):
        if isinstance(other, type(self)):
            return self.as_bytes().__ge__(other.as_bytes())
        else:
            return self.as_bytes().__ge__(other)
    
    def get_best_unit(self):
        conversion = {0: 'B',
                      3: 'kB',
                      6: 'MB',
                      9: 'GB',
                      12: 'TB',
                      15: 'PB'}
        base = int(np.log10(self.as_bytes()))
        best_fitting = 0
        for unit in sorted(list(conversion.keys())):
            if unit < base:
                best_fitting = unit
        return conversion[best_fitting]
    
    def __str__(self):
        best_unit = self.get_best_unit()
        size = self.from_bytes[best_unit](self.as_bytes())
        return '%.2f %s' % (size, best_unit)
    
    def __repr__(self):
        return 'DataSize({}, unit={})'.format(self.size, self.unit)
    
    def __add__(self, other):
        if isinstance(other, type(self)):
            return DataSize(self.as_bytes() + other.as_bytes(), unit='B')
        else:
            return DataSize(self.as_bytes() + other, unit='B')
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __sub__(self, other):
        if isinstance(other, type(self)):
            return DataSize(self.as_bytes() - other.as_bytes(), unit='B')
        else:
            return DataSize(self.as_bytes() - other, unit='B')
    
    def __rsub__(self, other):
        return DataSize(other - self.as_bytes(), unit='B')
    
    def __mul__(self, other):
        if isinstance(other, type(self)):
            raise TypeError('Cannot multiply datasize by datasize.')
        else:
            return DataSize(self.as_bytes() * other, unit='B')
    
    def __rmul__(self, other):
        if isinstance(other, type(self)):
            raise TypeError('Cannot multiply datasize by datasize.')
        else:
            return DataSize(self.as_bytes() * other, unit='B')
    
    def __truediv__(self, other):
        if isinstance(other, type(self)):
            raise TypeError('Cannot divide datasize by datasize.')
        else:
            return DataSize(self.as_bytes() / other, unit='B')
    
    def __rtruediv__(self, other):
        if isinstance(other, type(self)):
            raise TypeError('Cannot divide datasize by datasize.')
        else:
            return DataSize(other / self.as_bytes(), unit='B')
    
    def __floordiv__(self, other):
        if isinstance(other, type(self)):
            raise TypeError('Cannot divide datasize by datasize.')
        else:
            return DataSize(self.as_bytes() // other, unit='B')
    
    def __rfloordiv__(self, other):
        if isinstance(other, type(self)):
            raise TypeError('Cannot divide datasize by datasize.')
        else:
            return DataSize(other // self.as_bytes(), unit='B')

class LimitedSizeDict(OrderedDict):
    """A dictionary that only allows entries to require a given amount
    of memory.
    
    This dictionary must be initialized empty and can only be added to
    via the __setitem__ method or its abbreviation `dict[key] = val`.
    
    Arguments
    ---------
    size_limit : {int or DataSize or None, None}
        The memory size limit for the dict.
    unit : {str, 'B'}
        The unit of the size limit. Must be a unit understood by
        DataSize.
    error_on_overflow : {bool, False}
        Whether or not to raise an error if an attempt is made to insert
        a value with a size that exceeds the size limit.
    
    Notes
    -----
    -Only the size of the values is monitored. Therefore, the
     LimitedSizeDict may significantly outgrow its size-limit if keys of
     large memory size are used.
    -The function memsize is used to calculate the size of the items.
    """
    def __init__(self, size_limit=None, unit='B',
                 error_on_overflow=False):
        super().__init__()
        if size_limit is None:
            self._size_limit = None
        elif isinstance(size_limit, DataSize):
            self._size_limit = size_limit
        else:
            self._size_limit = DataSize(size_limit, unit=unit)
        self._remaining_size = self._size_limit
        self._error_on_overflow = error_on_overflow
    
    def __setitem__(self, key, value):
        if self._size_limit is None:
            super().__setitem__(key, value)
        else:
            item_size = memsize(value)
            if item_size > self._size_limit:
                if self._error_on_overflow:
                    msg = 'Cannot insert an item of size {} into a size '
                    msg += 'limited dict with size limit {}.'
                    msg = msg.format(item_size, self._size_limit)
                    raise ValueError(msg)
                else:
                    return
            
            if key in self:
                item_size -= memsize(self[key])
            
            #Pop items until size limit is reached.
            while self._remaining_size < item_size:
                popped_item = self.popitem(last=False)
                self._remaining_size += memsize(popped_item)
            self._remaining_size -= item_size
            super().__setitem__(key, value)
    
    def fits(self, value):
        """Check if memory size of the value is smaller than the
        size-limit of the dictionary.
        
        Arguments
        ---------
        value : object
            The object of which the size should be checked.
        
        Returns
        -------
        bool:
            Returns True if it is possible to insert this value into the
            dictionary.
        """
        return self._size_limit >= memsize(value)
    
    def fits_lossless(self, value):
        """Check if the value can be inserted into the dictionary
        without deleting any previously inserted values.
        
        Arguments
        ---------
        value : object
            The object of which the size should be checked.
        
        Returns
        -------
        bool:
            Returns True if the value can be inserted into the
            dictionary without deleting any other values from the
            dictionary.
        """
        return self._remaining_size >= memsize(value)
