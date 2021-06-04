from tensorflow import keras
import csv
import warnings
import numpy as np

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
