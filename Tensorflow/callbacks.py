from tensorflow import keras
import numpy as np
from BnsLib.network.callbacks import SnrCurriculumLearningScheduler
from BnsLib.network.callbacks import SnrPlateauScheduler
from BnsLib.network.callbacks import SensitivityEstimator
import os

standard_start = [90., 100.]

keys = {'acc': {'monitor': 'val_acc',
                'lower_by_monitor': 0.95,
                'mode': 'max',
                'start_snr': standard_start},
        'acc_rel': {'monitor': 'val_acc',
                    'lower_by_monitor': 0.95,
                    'mode': 'max',
                    'lower_strat': 'rel',
                    'lower_by': 0.9,
                    'start_snr': standard_start},
        'epochs': {'lower_by_epoch': 5,
                   'start_snr': standard_start},
        'epochs_rel': {'lower_by_epoch': 5,
                       'lower_strat': 'rel',
                       'lower_by': 0.9,
                       'start_snr': standard_start},
        'loss': {'monitor': 'val_loss',
                 'lower_by_monitor': 0.2,
                 'mode': 'min',
                 'start_snr': standard_start},
        'loss_rel': {'monitor': 'val_loss',
                     'lower_by_monitor': 0.2,
                     'mode': 'min',
                     'lower_strat': 'rel',
                     'lower_by': 0.9,
                     'start_snr': standard_start},
        'plateau_acc': {'plateau': True,
                        'monitor': 'val_acc',
                        'best': 'max',
                        'start_snr': standard_start},
        'plateau_acc_rel': {'plateau': True,
                            'monitor': 'val_acc',
                            'best': 'max',
                            'lower_strat': 'rel',
                            'lower_by': 0.9,
                            'start_snr': standard_start},
        'plateau_loss': {'plateau': True,
                         'monitor': 'val_loss',
                         'best': 'min',
                         'start_snr': standard_start},
        'plateau_loss_rel': {'plateau': True,
                            'monitor': 'val_loss',
                            'best': 'min',
                            'lower_strat': 'rel',
                            'lower_by': 0.9,
                            'start_snr': standard_start},
        'fixed_15': {'start_snr': [15., 15.]},
        'fixed_30': {'start_snr': [30., 30.]},
        'fixed_8': {'start_snr': [8., 8.]},
        'fixed_full': {'start_snr': [5., 100.]},
        'fixed_low': {'start_snr': [5., 15.]}}

def get_callbacks(train_generator, val_generator, signal_generator,
                  noise_generator, base_path='.', key='loss'):
    kwargs = keys[key].copy()
    start_snr = kwargs.pop('start_snr', standard_start)
    plateau = kwargs.pop('plateau', False)
    train_generator.rescale(start_snr)
    val_generator.rescale(start_snr)
    use_scaling_callbacks = (len(kwargs) > 0)
    
    if plateau:
        sched = SnrPlateauScheduler
    else:
        sched = SnrCurriculumLearningScheduler
    
    if use_scaling_callbacks:
        training_scaler = sched(train_generator, **kwargs)
        validation_scaler = sched(val_generator, **kwargs)
    check_path = os.path.join(base_path, 'curriculum_{epoch:d}.hf5')
    checkpointer = keras.callbacks.ModelCheckpoint(check_path,
                                                   verbose=1,
                                                   save_freq='epoch',
                                                   save_best_only=False)
    csv_path = os.path.join(base_path, 'curriculum_history.csv')
    csv_logger = keras.callbacks.CSVLogger(csv_path)
    
    trans = lambda inp: inp.T[0]
    sens_estimate = SensitivityEstimator(signal_generator,
                                         threshold=noise_generator,
                                         transform_function=trans,
                                         snrs=[int(pt) for pt in np.arange(3, 31, 3)],
                                         file_path=os.path.join(base_path, 'sensitivity_estimate.csv'),
                                         fap=1e-4
                                         )
    if use_scaling_callbacks:
        return [checkpointer,
                csv_logger,
                training_scaler,
                validation_scaler,
                sens_estimate]
    else:
        return [checkpointer,
                csv_logger,
                sens_estimate]
