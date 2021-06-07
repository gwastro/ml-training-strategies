from bnslib_funcs import FileHandler, MultiFileHandler, FileGenerator
import h5py
import numpy as np
import os

class SignalHandler(FileHandler):
    def __init__(self, file_path, ref_index=0, slice=None):
        super().__init__(file_path)
        self.ref_index = ref_index
        self.slice = slice
        self.target = None
        self.load_data()
    
    def load_data(self):
        with h5py.File(self.file_path, 'r') as fp:
            if self.slice is None:
                self.data = fp['data/0'][()]
            else:
                self.data = fp['data/0'][self.slice]
    
    def translate_index(self, index):
        return index - self.ref_index
    
    def __len__(self):
        return len(self.data)
    
    def __contains__(self, idx):
        idx = self.translate_index(idx)
        if idx < 0:
            return False
        return idx < len(self)
    
    def __getitem__(self, index):
        index = self.translate_index(index)
        if index >= len(self):
            msg = 'Index {} out of range. Max index {}.'.format(index, len(self) - 1)
            raise IndexError(msg)
        if index < 0:
            msg = 'Index {} out of range. Min index 0.'.format(index)
            raise IndexError(msg)
        
        data = self.data[index].copy()
        
        if self.target is not None:
            target_snr = np.random.uniform(*self.target)
            data = data * target_snr
        else:
            target_snr = 1
        
        return data, np.array([1, 0]), target_snr
    
    def open(self, mode='r'):
        return
    
    def close(self):
        return
    
    def rescale(self, target):
        self.target = target

class NoSignalHandler(FileHandler):
    def __init__(self, no_signal_snr=3., num_detectors=1):
        super().__init__('')
        self.no_signal_snr = no_signal_snr
        self.num_detectors = num_detectors
    
    def __len__(self):
        return 1
    
    def __contains__(self, idx):
        return idx == -1
    
    def __getitem__(self, idx):
        return np.zeros((1, 2048, self.num_detectors)), np.array([0, 1]), self.no_signal_snr
    
    def open(self, mode='r'):
        return
    
    def close(self):
        return
    
    def rescale(self, new_target):
        return

class NoiseHandler(FileHandler):
    def __init__(self, file_path, ref_index=0, slice=None):
        super().__init__(file_path)
        self.ref_index = ref_index
        self.slice = slice
        self.load_data()
    
    def load_data(self):
        with h5py.File(self.file_path, 'r') as fp:
            if self.slice is None:
                self.data = fp['data/0'][()]
            else:
                self.data = fp['data/0'][self.slice]
    
    def translate_index(self, index):
        return index - self.ref_index
    
    def __len__(self):
        return len(self.data)
    
    def __contains__(self, idx):
        idx = self.translate_index(idx)
        if idx < 0:
            return False
        return idx < len(self)
    
    def __getitem__(self, idx):
        idx = self.translate_index(idx)
        data = self.data[idx]
        return np.expand_dims(data, axis=-1)
    
    def open(self, mode='r'):
        return
    
    def close(self):
        return
    
    def rescale(self, new_target):
        return

class MultiHandler(MultiFileHandler):
    def __init__(self, *args, num_detectors=1, freq_data=False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.freq_data = freq_data
        if self.freq_data:
            self.input_shape = (1025, 2*num_detectors)
        else:
            self.input_shape = (2048, num_detectors)
        self.output_shape = (2,)
        self.target = None
    
    def split_index_to_groups(self, index):
        try:
            return {'signal': index[0],
                    'noise': index[1]}
        except TypeError:
            return {'signal': -1,
                    'noise': index}
    
    def format_return(self, inp):
        signal, label, snr = inp['signal']
        noise = inp['noise']
        if self.freq_data:
            data = signal + noise
            if np.ndim(data) == 2 and data.shape[0] != 1:
                data = np.expand_dims(data, axis=0)
            fdata = np.fft.rfft(data, axis=1)
            return np.concatenate([np.real(fdata), np.imag(fdata)], axis=-1), label
        else:
            return signal + noise, label
    
    def rescale(self, new_target):
        self.target = new_target
        for fh in self.file_handelers:
            fh.rescale(new_target)

class ScalabelFileGenerator(FileGenerator):
    def __init__(self, file_handeler, index_list, batch_size=32,
                 shuffle=True, use_sample_weights=False,
                 target=[90, 100]):
        super().__init__(file_handeler, index_list,
                         batch_size=batch_size, shuffle=shuffle,
                         use_sample_weights=use_sample_weights)
        self.target = target
    
    def __getitem__(self, index):
        if self.file_handeler.target != self.target:
             self.file_handeler.rescale(self.target)
        return super().__getitem__(index)
    
    def rescale(self, new_target):
        self.target = new_target
        self.file_handeler.rescale(self.target)

def file_len(path):
    try:
        with h5py.File(path, 'r') as fp:
            ret = len(fp['data/0'])
        return ret
    except:
        return 0

def get_generator(directory, file_prefix, n_signals=None, n_noise=None,
                  noise_per_signal=1, use_signal_files=True,
                  use_noise_files=True, use_no_signal_handler=True,
                  batch_size=32, shuffle=True, target=[90, 100],
                  load_required=True, freq_data=False):
    #Get data-paths
    if not file_prefix.endswith('_'):
        file_prefix += '_'
    sig_prefix = file_prefix + 'signal'
    noi_prefix = file_prefix + 'noise'
    sig_files = []
    noi_files = []
    if use_signal_files:
        for fn in filter(lambda path: path.startswith(sig_prefix), os.listdir(directory)):
            sig_files.append(os.path.join(directory, fn))
    if use_noise_files:
        for fn in filter(lambda path: path.startswith(noi_prefix), os.listdir(directory)):
            noi_files.append(os.path.join(directory, fn))
    
    multi = MultiHandler(freq_data=freq_data)
    
    #Determine how many samples are required
    available_signals = 0
    for sig_file in sig_files:
        available_signals += file_len(sig_file)
    
    available_noise = 0
    for noi_file in noi_files:
        available_noise += file_len(noi_file)
    
    if n_signals is None:
        n_signals = available_signals
    n_signal_noise = n_signals * noise_per_signal
    if n_noise is None:
        if use_signal_files:
            n_noise = n_signal_noise
        else:
            n_noise = available_noise
    load_n_noise = n_signal_noise + n_noise
    if available_signals < n_signals:
        msg = 'Requesting more signals than are available. Requested '
        msg += f'{n_signals} but only {available_signals} are available.'
        raise ValueError(msg)
    if available_noise < n_signal_noise + n_noise:
        msg = 'Requesting more noise than available. Requested '
        msg += f'{n_signal_noise + n_noise} but only {available_noise} samples are available.'
        raise ValueError(msg)
    print(f"n_signals: {n_signals}")
    print(f"n_noise: {n_noise}")
    
    #Load signals
    ref_idx = 0
    for sig_file in sig_files:
        if ref_idx >= n_signals and load_required:
            continue
        sigs_in_file = file_len(sig_file)
        if sigs_in_file <= n_signals - ref_idx or not load_required:
            handler = SignalHandler(sig_file, ref_index=ref_idx)
        else:
            handler = SignalHandler(sig_file, ref_index=ref_idx,
                                    slice=slice(0, n_signals-ref_idx))
        multi.add_file_handler(handler, group='signal')
        ref_idx += len(handler)
    if use_no_signal_handler:
        multi.add_file_handler(NoSignalHandler(), group='signal')
    
    #Load noise
    ref_idx = 0
    for noi_file in noi_files:
        if ref_idx >= load_n_noise and load_required:
            continue
        noi_in_file = file_len(noi_file)
        if noi_in_file <= load_n_noise - ref_idx or not load_required:
            handler = NoiseHandler(noi_file, ref_index=ref_idx)
        else:
            handler = NoiseHandler(noi_file, ref_index=ref_idx,
                                   slice=slice(0, load_n_noise-ref_idx))
        multi.add_file_handler(handler, group='noise')
        ref_idx += len(handler)
    
    #Calculate the index list that is used by the generator
    sig_list = np.vstack([np.repeat(np.arange(n_signals), noise_per_signal),
                          np.arange(n_signal_noise)]).T
    noi_list = np.vstack([-np.ones(n_noise),
                          n_signal_noise + np.arange(n_noise)]).T
    index_list = np.concatenate([sig_list, noi_list], axis=0)
    index_list = index_list.astype(int)
    
    #Instantiate generator
    generator = ScalabelFileGenerator(multi, index_list=index_list,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      target=target)
    return generator
