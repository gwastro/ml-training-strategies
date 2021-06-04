from tensorflow import keras
import numpy as np

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
