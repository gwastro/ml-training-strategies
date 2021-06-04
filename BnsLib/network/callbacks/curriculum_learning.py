from tensorflow import keras
from BnsLib.utils.math import safe_min, safe_max
from .plateau import PlateauDetection
import warnings

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
