from tensorflow import keras
import numpy as np

class JointGenerator(keras.utils.Sequence):
    #TODO: Add all possible callback hooks, e.g. on_train_begin
    """A generator that combines multiple generators into one.
    
    Arguments
    ---------
    generators : Generator
        One or multiple generators that generate samples.
    
    Attributes
    ----------
    generators : list of generators
        The generators which generate the samples.
    lengths : list of int
        A list containing the lengths of all generators in order.
    cumlen : list of int
        The cumulative sum of the lengths.
    
    Notes
    -----
    -The samples are generated in order in which the generators where
     given.
    """
    def __init__(self, *generators):
        self.generators = generators
        self.lengths = [len(gen) for gen in self.generators]
        self.cumlen = np.cumsum(self.lengths)
    
    def __len__(self):
        return self.cumlen[-1]
    
    def on_epoch_end(self):
        for gen in self.generators:
            gen.on_epoch_end()
    
    def __getitem__(self, index):
        genidx = np.searchsorted(self.cumlen, index, side='right')
        if genidx > 0:
            index -= self.cumlen[genidx-1]
        return self.generators[genidx][index]
