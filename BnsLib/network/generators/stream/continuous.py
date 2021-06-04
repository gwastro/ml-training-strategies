from tensorflow.keras.utils import Sequence
import numpy as np
from pycbc.types import TimeSeries

class TimeSeriesGenerator(Sequence):
    """A Keras generator that takes a list of pycbc TimeSeries as input
    and generates the correctly whitened and formatted input to the
    network.
    
    Arguments
    ---------
    ts : list of pycbc.TimeSeries or pycbc.TimeSeries
        List of TimeSeries with the same duration and sample rate.
    time_step : {float, 0.25}
        The step-size with which the sliding window moves in seconds.
    batch_size : {int, 32}
        The batch-size to use, i.e. how many subsequent windows will be
        returned for each call of __getitem__.
    dt : {None or float, None}
        If ts is not a list of pycbc.TimeSeries but rather a list of
        arrays, the array will be cast to a TimeSeries with delta_t = dt.
    window_duration : {float, 48.0}
        The duration of each slice that is passed to the transform
        function.
    """
    def __init__(self, ts, time_step=0.1, batch_size=32, dt=None,
                 window_duration=48.0):
        self.batch_size = batch_size
        self.time_step = time_step
        if not isinstance(ts, list):
            ts = [ts]
        self.ts = []
        self.dt = []
        for t in ts:
            if isinstance(t, TimeSeries):
                self.dt.append(t.delta_t)
                self.ts.append(t)
            elif isinstance(t, type(np.array([]))):
                if dt == None:
                    msg  = 'If the provided data is not a pycbc.types.TimeSeries'
                    msg += 'a value dt must be provided.'
                    raise ValueError(msg)
                else:
                    self.dt.append(dt)
                    self.ts.append(TimeSeries(t, delta_t=dt))
            else:
                msg  = 'The provided data needs to be either a list or a '
                msg += 'single instance of either a pycbc.types.TimeSeries'
                msg += 'or a numpy.array.'
                raise ValueError(msg)
        
        for delta_t in self.dt:
            if not delta_t == self.dt[0]:
                raise ValueError('All data must have the same delta_t.')
        
        #delta_t of all TimeSeries
        self.dt = self.dt[0]
        
        #How big is the window that is shifted over the data
        #(64s + 8s for cropping when whitening)
        self.window_size_time = window_duration
        
        #Window size in samples
        self.window_size = int(self.window_size_time / self.dt)
        
        #How many points are shifted each step
        self.stride = int(self.time_step / self.dt)
        
        #total number of window shifts
        self.window_shifts = int(np.floor(float(len(self.ts[0])-self.window_size + self.stride) / self.stride))
    
    def __len__(self):
        """Returns the number of batches provided by this generator.
        
        Arguments
        ---------
        None
        
        Returns
        -------
        int:
            Number of batches contained in this generator.
        """
        return(int(np.ceil(float(self.window_shifts) / self.batch_size))-1)
    
    def __getitem__(self, index):
        """Return batch of index.
        
        Arguments
        ---------
        index : int
            The index of the batch to retrieve. It has to be smaller
            than len(self).
        
        Returns
        -------
        list of arrays
            A list containing the input for the network. Each array is
            of shape (batch_size, 2048, 2).
        """
        min_stride = index * self.batch_size
        max_stride = min_stride + self.batch_size
        
        #Check for last batch and adjust size if necessary
        if index == len(self) - 1:
            len_data = (index + 1) * self.stride * self.batch_size
            if len_data > len(self.ts[0]):
                max_stride -= int(np.floor(float(len(self.ts[0]) - len_data + self.batch_size) / self.stride))
        
        #Calculate the indices of the slices
        index_range = np.zeros((2, max_stride - min_stride), dtype=int)
        index_range[0] = np.arange(min_stride * self.stride, max_stride * self.stride, self.stride)
        index_range[1] = index_range[0] + self.window_size
        index_range = index_range.transpose()
        
        #Generate correctly formatted input data
        X = self._gen_slices(index_range)
        
        return X
    
    def _gen_slices(self, index_range):
        """Slice the input data and apply the transformation.
        
        Arguments
        ---------
        index_range : numpy.array
            Array of shape (num_samples, 2), where each row contains the
            start- and stop-index of the slice.
        
        Returns
        -------
        array or list of arrays:
            The data that can be passed to the network.
        """      
        
        #Slice the time-series
        for in_batch, idx in enumerate(index_range):
            low, up = idx
            if len(self.ts) > 1:
                detector_slices = [self.ts[det][low:up] for det in range(len(self.ts))]
            else:
                detector_slices = self.ts[0][low:up]
            
            #Apply transformation
            transformed = self.transform(detector_slices)
            
            #Put output into list that will later be stacked.
            if in_batch == 0:
                if isinstance(transformed, list):
                    ret = [[pt] for pt in transformed]
                else:
                    ret = [transformed]
            else:
                if isinstance(transformed, list):
                    for i, pt in enumerate(transformed):
                        ret[i].append(pt)
                else:
                    ret.append(transformed)
        
        #Format the return data
        if isinstance(ret[0], list):
            ret = [np.vstack(pt) for pt in ret]
        else:
            ret = np.vstack(ret)
        
        return ret
    
    def transform(self, detector_slices):
        """A transformation to apply to every slice.
        
        Arguments
        ---------
        detector_slices : list of pycbc.types.TimeSeries
            Each list entry is a slice of a time series of a single
            detector. It is a slice of the requested length.
        
        Returns
        -------
        numpy array or list of numpy array:
            The transformed input sample for the network. If the network
            expects the input to be of shape (batch-size, rest), the
            return value of this function must be of shape (rest, ).
            If the network has multiple inputs, it will expect those as
            a list. In this case the output of this function may be a
            list of single samples for each input.
        """
        return detector_slices
    
    def on_epoch_end(self):
        return
