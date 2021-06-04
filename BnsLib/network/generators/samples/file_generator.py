import tensorflow.keras as keras
import numpy as np

"""This module contains code to generate samples to train a network
from one or multiple files.
"""

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
