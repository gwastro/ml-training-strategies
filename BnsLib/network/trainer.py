from tensorflow import keras
import numpy as np
import time
import datetime
import os
import pandas
import h5py
from pycbc.sensitivity import volume_montecarlo

from .callbacks import SensitivityEstimator
from .generators import JointGenerator
from ..utils import input_to_list, inverse_string_format
from ..testing import mchirp

SECONDS_PER_MONTH = 60 * 60 * 24 * 30

class Trainer(object):
    """A class that is tailored to train and evaluate networks that do
    signal detection.
    
    This class applies the most important callbacks to the network, uses
    the generated statistics to rank the networks and checks the best
    one on a test set.
    
    model : keras.models.Model or list of keras.models.Model
        The model which should be trained. If the argument is a list,
        all models from that list are trained sequentially. All models
        are finally passed simultaniously to the Evaluator.
    train_generator : generator or list of generators
        A generator that is understood by keras.models.Model.fit. This
        generator is used to train the network. If a list is given, it
        must contain the same number of generators as there are models.
        Each generator is used to train one model. (must have a
        `rescale` method, which scales signals to a given SNR)
    val_signal_generator : generator
        A generator that is understood by keras.models.Model.evaluate.
        This generator is supposed to only supply signal examples.
        During training, this generator is combined with
        val_noise_generator to do validation. If a list is given, it
        must contain the same number of generators as there are models.
        Each generator is used to train one model. (must have a
        `rescale` method, which scales signals to a given SNR)
    val_noise_generator : generator
        A generator that is understood by keras.models.Model.evaluate.
        This generator is supposed to only supply noise examples.
        During training, this generator is combined with
        val_signal_generator to do validation. If a list is given, it
        must contain the same number of generators as there are models.
        Each generator is used to train one model. (must have a
        `rescale` method, which scales signals to a given SNR)
    evaluator : Evaluator
        An evaluator subclassed from BnsLib.network.Evaluator. This
        class is supposed to handle evaluation of the provided
        test-data, i.e. do pre- and post-processing of the network
        input/output. It will return a set of event times which are
        compared to the test set.
    test_files : str or list of str
        Path to one or multiple files that contain test data. These
        paths will be passed to the Evaluator. All processing happens
        there.
    output_directory : {str or None, None}
        The directory at which the models and metrics are stored. If set
        to None the path will be TRAINER_<timestamp>.
    callbacks : {list of keras.callbacks.Callback or None, None}
        Callbacks to attach to all model.fit calls.
    description : {str or None, None}
        A description of what this trainer is training for. May be a
        path to file. If the str can be interpreted as such a path and a
        file exists at this path, the file will be copied to the output
        directory. If set to None, no description will be stored. If the
        string can not be interpreted as a file, the string will be
        stored in a file called `description.txt` in the output
        directory.
    name : {str or None, None}
        A name given to this trainer.
    transform_function : {function or list of function or None, None}
        Function(s) that transform the output of the models to a ranking
        statistic, i.e. a 1d array, where larger values signify higher
        probability of signal detection. If given as a list the length
        must match the number of models given to the trainer.
    efficiency_snrs : {list of int or None, None}
        The SNRs at which the efficiency (true-positive probability) is
        calculated on every epoch.
    fap : {float, 1e-4}
        The false-alarm probability (fap) at which the efficiency is
        calculated.
    event_tolerance : {float, 1.}
        The distance between an event and an injection time for which
        the event is assigned to be a true-positive. (I.e. events that
        are within ±event_tolerance of an injection are detections. If
        they are further away, they are false positives.)
        [unit: seconds]
    inj_files : {str or list of str or None, None}
        Path(s) to the file(s) containing the injection parameters for
        the test data. Must be a HDF5-file with at least the datasets
        `mass1`, `mass2`, `distance` and `tc`. If no files are given,
        all events returned by the Evaluator are understood as false
        positives.
    """
    def __init__(self, model, train_generator, val_signal_generator,
                 val_noise_generator, evaluator, test_files,
                 output_directory=None, callbacks=None, 
                 description=None, name=None, transform_function=None,
                 efficiency_snrs=None, fap=1e-4, event_tolerance=1.,
                 inj_files=None):
        self.models = input_to_list(model)
        self.train_generators = input_to_list(train_generator)
        assert len(self.models) == len(self.train_generators)
        self.val_signal_generators = input_to_list(val_signal_generator)
        assert len(self.models) == len(self.val_signal_generators)
        self.val_noise_generators = input_to_list(val_noise_generator)
        assert len(self.models) == len(self.val_noise_generators)
        self.evaluator = evaluator
        self.test_files = input_to_list(test_files)
        self.callbacks = callbacks if callbacks is not None else []
        self.setup_directory_structure(output_directory)
        self.description = description
        self.name = name
        if transform_function is None:
            self.transform_functions = input_to_list(lambda inp: inp,
                                                     length=len(self.models))
        else:
            self.transform_functions = input_to_list(transform_function,
                                                     length=len(self.models))
        self.efficiency_snrs = efficiency_snrs
        self.fap = fap
        self.event_tolerance = event_tolerance
        self.inj_files = input_to_list(inj_files) if inj_files is not None else []
    
    def setup_directory_structure(self, output_directory):
        if output_directory is None:
            self.output_directory = f'TRAINER_{int(time.time())}'
        else:
            self.output_directory = output_directory
        for i in range(len(self.models)):
            os.makedirs(os.path.join(self.output_directory,
                                     str(i)),
                        exist_ok=True)
        if self.description is not None:
            if isinstance(self.description, str) and os.path.isfile(self.description):
                fn = os.path.split(self.description)[1]
                copy(self.description, os.path.join(self.output_directory, fn))
            else:
                path = os.path.join(self.output_directory, 'description.txt')
                try:
                    with open(path, 'w') as fp:
                        fp.write(str(self.description))
                except:
                    pass
    
    def train(self, epochs, verbose=1):
        start = datetime.datetime.now()
        sens_paths = []
        train_start = time.time()
        for i, (model, tr_gen, val_sig_gen, val_noi_gen, tf) in enumerate(zip(self.models,
                                                                              self.train_generators,
                                                                              self.val_signal_generators,
                                                                              self.val_noise_generators,
                                                                              self.transform_functions)):
            
            val_gen = JointGenerator(val_sig_gen,
                                     val_noi_gen)
        
            callbacks = self.callbacks
            sens_path = os.path.join(self.output_directory,
                                     str(i),
                                     'efficiency.csv')
            if self.efficiency_snrs is not None:
                eff = SensitivityEstimator(val_sig_gen,
                                           threshold=val_noi_gen,
                                           file_path=sens_path,
                                           save_freq=1,
                                           transform_function=tf,
                                           snrs=self.efficiency_snrs,
                                           verbose=verbose,
                                           fap=self.fap)
                callbacks = [eff] + callbacks
                sens_paths.append(sens_path)
            else:
                sens_paths.append(None)
        
            if not any([isinstance(cb, keras.callbacks.ModelCheckpoint) for cb in callbacks]):
                filepath = os.path.join(self.output_directory,
                                        str(i),
                                        'models',
                                        'epoch_{epoch:d}')
                callbacks.append(keras.callbacks.ModelCheckpoint(filepath=filepath,
                                                                 save_weights_only=False,
                                                                 save_best_only=False,
                                                                 verbose=verbose,
                                                                 save_freq='epoch'))
            if not any([isinstance(cb, keras.callbacks.CSVLogger) for cb in callbacks]):
                filepath = os.path.join(self.output_directory,
                                        str(i),
                                        'history.csv')
                callbacks.append(keras.callbacks.CSVLogger(filepath))
        
            model.fit(tr_gen,
                      validation_data=val_gen,
                      callbacks=callbacks,
                      verbose=verbose,
                      epochs=epochs)
        train_end = time.time()
        
        model_paths = []
        for i, sens_path in enumerate(sens_paths):
            if sens_path is None:
                files = os.listdir(os.path.join(self.output_directory,
                                                str(i),
                                                'models'))
                def max_func(fn):
                    var = inverse_string_format(fn,
                                                'epoch_{epoch}',
                                                types={'epoch': int})
                    if var is None:
                        return -np.inf
                    else:
                        return var['epoch']
                max_model = max(files, key=max_func)
                model_paths.append(os.path.join(self.output_directory,
                                                str(i),
                                                'models',
                                                max_model))
                
            else:
                df = pandas.read_csv(sens_path)
                
                epoch = df['Epoch (one based)'][df['Mean true positive rate'].argmax()]
                model_path = os.path.join(self.output_directory,
                                          str(i),
                                          'models',
                                          f'epoch_{epoch}')
                model_paths.append(model_path)
        evalu = self.evaluator(model_paths)
        
        eval_start = time.time()
        event_list = evalu(self.test_files)
        eval_end = time.time()
        
        with h5py.File(os.path.join(self.output_directory, 'events.hdf'), 'w') as fp:
            fp.create_dataset('times', data=np.array([event[0] for event in event_list]))
            fp.create_dataset('values', data=np.array([event[1] for event in event_list]))
        
        injtimes = []
        mchirps = []
        dists = []
        duration = 0
        for path in self.inj_files:
            with h5py.File(path, 'r') as fp:
                injtimes.append(fp['tc'][()])
                dists.append(fp['distance'][()])
                m1 = fp['mass1'][()]
                m2 = fp['mass2'][()]
                mchirps.append(mchirp(m1, m2))
        injtimes = np.concatenate(injtimes)
        idxs = injtimes.argsort()
        injtimes = injtimes[idxs]
        dists = np.concatenate(dists)[idxs]
        mchirps = np.concatenate(mchirps)[idxs]
        
        duration = 0
        for path in self.test_files:
            with h5py.File(path, 'r') as fp:
                for i, key in enumerate(fp.keys()):
                    if i > 0:
                        continue
                    duration += (len(fp[key]['data']) * fp[key].attrs['delta_t'])
        
        sevents = sorted(event_list, key=lambda e: e[1])
        tevents = []
        found = []
        idxs = []
        for event in sevents:
            idx = np.searchsorted(injtimes, event[0])
            if idx == 0:
                istrue = (abs(injtimes[0] - event[0]) < self.event_tolerance)
            elif idx == len(injtimes):
                istrue = (abs(injtimes[idx-1] - event[0]) < self.event_tolerance)
            elif (abs(injtimes[idx-1] - event[0]) < self.event_tolerance):
                istrue = True
                idx -= 1
            else:
                istrue = (abs(injtimes[idx] - event[0]) < self.event_tolerance)
            idxs.append(idx)
            if istrue:
                found.append(idx)
            tevents.append(istrue)
        found = np.array(found)
        idxs = np.array(idxs, dtype=int)
        event_times = np.array([event[0] for event in sevents])
        
        ttp = tevents.count(True) #total true positives
        tfp = len(tevents) - ttp #total false positives
        ctp = ttp #current true positives
        cfp = tfp #current false positives
        thresholds = []
        tpr = [] #true positive rate
        far = [] #false positive rate
        sens = [] #Sensitive-distance
        tts = [] #true trigger separation
        ats = [] #all trigger separation
        closest_inj_times = injtimes[idxs]
        inj_diff = np.abs(event_times - closest_inj_times)
        for i, (true_event, event) in enumerate(zip(tevents, sevents)):
            thresholds.append(event[1])
            if true_event:
                ctp -= 1
            else:
                cfp -= 1
            tpr.append(ctp / len(injtimes))
            far.append(cfp / duration * SECONDS_PER_MONTH)
            
            fidxs = np.unique(idxs[i:][tevents[i:]])
            midxs = np.setdiff1d(np.arange(len(injtimes)), fidxs)
            found_dist = dists[fidxs]
            if len(found_dist) == 0:
                found_dist = np.array([0.])
            missed_dist = dists[midxs]
            if len(missed_dist) == 0:
                missed_dist = np.array([np.inf])
            found_mchirp = mchirps[fidxs]
            if len(found_mchirp) == 0:
                found_mchirp = np.array([1.])
            missed_mchirp = mchirps[midxs]
            if len(missed_mchirp) == 0:
                missed_mchirp = np.array([1.])
            
            vol, vol_err = volume_montecarlo(found_dist,
                                             missed_dist,
                                             found_mchirp,
                                             missed_mchirp,
                                             'distance',
                                             'volume',
                                             'distance')
            sens.append(vol)
            
            
            ats.append(np.mean(inj_diff[i:]))
            tts.append(np.mean(inj_diff[i:][tevents[i:]]))
        
        with h5py.File(os.path.join(self.output_directory, 'stats.hdf'), 'w') as fp:
            fp.create_dataset('ranking', data=np.array(thresholds))
            fp.create_dataset('far', data=np.array(far))
            fp.create_dataset('sens-frac', data=np.array(tpr))
            fp.create_dataset('sens-dist', data=np.array(sens))
            fp.create_dataset('true-event-separation', data=np.array(tts))
            fp.create_dataset('all-event-separation', data=np.array(ats))
            fp.create_dataset('closest-injection-time', data=closest_inj_times)
            
            fp.attrs['train-time'] = train_end - train_start
            fp.attrs['test-time'] = eval_end - eval_start
            fp.attrs['start'] = str(start)
            fp.attrs['end'] = str(datetime.datetime.now())
            fp.attrs['observation-time'] = float(duration)
            fp.attrs['number-injections'] = int(len(injtimes))

class Evaluator(object):
    """A class that handles model and data-loading for continuous test
    data. It applies the loaded model to the data and calculates events
    in that data. It returns a list of events.
    
    Arguments
    ---------
    paths : list of str
        A list of paths to the models that should be loaded.
    """
    def __init__(self, paths):
        self.paths = paths
        self.load_models()
    
    def load_models(self):
        """Sets self.models and compiles them, if necessary.
        
        It is generally assumed that the models are loaded from
        self.paths.
        """
        self.models = [keras.models.load_model(path) for path in self.paths]
    
    def __call__(self, file_paths):
        """Calculate events as detected by the network.
        
        Arguments
        ---------
        file_paths : list of str
            A list of paths to files containing continuous data that
            should be evaluated.
        
        Returns
        -------
        event_list : list of events
            A list of tuples, where the first entry is the GPS-time and
            the second is a ranking statistic. (higher value = more
            likely to contain an injection)
        """
        raise NotImplementedError
