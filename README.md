# Training Strategies for Deep Learning Gravitational-Wave Searches

Marlin B. Schäfer<sup>1, 2</sup>, Ondřej Zelenka<sup>3, 4</sup>, Alexander H. Nitz<sup>1, 2</sup>, Frank Ohme<sup>1, 2</sup>, Bernd Brügmann<sup>3, 4</sup>

<sub>1. [Albert-Einstein-Institut, Max-Planck-Institut for Gravitationsphysik, D-30167 Hannover, Germany](http://www.aei.mpg.de/obs-rel-cos)</sub><br>
<sub>2. Leibniz Universität Hannover, D-30167 Hannover, Germany</sub><br>
<sub>3. Friedrich-Schiller-Universität Jena, D-07743 Jena, Germany</sub>
<sub>4. Michael Stifel Center Jena, D-07743 Jena, Germany</sub>

## Introduction

Put abstract here

## Contents of this repository

This repository contains the code required to reproduce the analysis presented in [`[1]`](#publication). It is split into two parts, which can be found in the separate directories `Tensorflow` and `PyTorch`. It contains scripts to generate the training, validation and test data and scripts to analyze the efficiency and sensitive volume of the networks. All final results and plots we used for the publication can be found in the `Results` folder.
The repository also includes a copy of the BnsLib [`[2]`](#bnslib)

Below we provide a step-by-step guide on how to reproduce our analysis.

## 1 Generate data required for training

Call the script `generate_data.py` to generate 3 sets of data; a training set, a validation set and a data set used to determine the efficiencies during training.
The script has many options which can be viewed by calling `./generate_data.py -h`. Below we list the options we used to create our datasets. Note, that the prefixes `train`, `val` and `thr` are important for the rest of the code.

<b>Training set:</b>
```
./generate_data.py \
--mass-method uniform \
--repeat-mass-parameters 5 \
--random-phase \
--number-noise-samples 200000 \
--signal-output ./data/train_signals.hdf \
--noise-output ./data/train_noise.hdf \
--store-command ./data/train_command.txt \
--verbose
```

<b>Validation set:</b>
```
./generate_data.py \
--mass-method uniform \
--repeat-mass-parameters 5 \
--random-phase \
--number-noise-samples 200000 \
--signal-output ./data/val_signals.hdf \
--noise-output ./data/val_noise.hdf \
--store-command ./data/val_command.txt \
--verbose
```

<b>Efficiency set:</b>
```
./generate_data.py \
--mass-method uniform \
--repeat-mass-parameters 1 \
--random-phase \
--number-mass-draws 10000 \
--number-noise-samples 400000 \
--signal-output ./data/thr_signals.hdf \
--noise-output ./data/thr_noise.hdf \
--store-command ./data/thr_command.txt \
--verbose
```

Two kinds of files are generated, both of which are HDF5 files. One contains pure signals. Each of these signals is whitened and scaled to a signal-to-noise ratio of 1. The other file contains white noise. Both of these files contain a group called `data`.

The `data` group contains the input samples that will be read by the network. The dataset in that group is called `'0'`.
For signal files these will be of shape `(number samples, 2048, 1)`. For noise files these will be of shape `(number samples, 2048)`.

The signal files also contain two more groups.

The `labels` group is empty.

The `params` group contains all parameters needed to generate the individual signals.

## 2 Train the networks

### 2.1 Tensorflow

To train a specific network call the script `Tensorflow/train.py`. The script has many options. For a full list refer to `./Tensorflow/train.py -h`. For most applications it is sufficient to set the following options.
```
./Tensorflow/train.py \
--store-path Tensorflow/training_output \
--data-path ./data \
--run-key fixed_low \
--network-author gabbard
```
The above call to the script will train a single instance of the network described in detail in [`[1]`](#publication) on the data stored in the `data` directory. It will train the network scaling the training signals to SNRs drawn from the fixed interval [5, 15]. The individual epochs, the efficiency and the history of the loss and accuracy will be stored in the directory `Tensorflow/training_output`.

 * `--store-path`: A path at which to store the individual epochs of the network during training as well as a history of the loss and accuracy and the efficiency.
 * `--data-path`: Path from which to read the training data. This data is assumed to be of the format generated by the `generate_data.py` script. This path is expected to be a directory containing training (prefix: train) and validation (prefix: val) data as well as data to calculate the efficiency (prefix: thr).
 * `--run-key`: The training strategy that should be used to train the networks. One of `acc`, `acc_rel`, `epochs`, `epochs_rel`, `loss`, `loss_rel`, `plateau_acc`, `plateau_acc_rel`, `plateau_loss`, `plateau_loss_rel`, `fixed_8`, `fixed_15`, `fixed_30`, `fixed_full`, `fixed_low`. To add custom keys, edit the `callbacks.py` module.
 * `--network-author`: Specifies the network that should be used. One of `gabbard`, `george`. To add custom networks, edit the `model.py` module.

To reproduce the results shown in [`[1]`](#publication) this script needs to be called 50 times for each of the 15 available run-keys.

### 2.2 PyTorch

## 3 Generate test data

Code to generate the test data is located in the directory `TestData`.
To generate test data we make use of the pycbc.workflow module. This is not strictly required but makes it easy to generate large datasets on a compute-cluster.
If you want to use the workflow, adjust the file `wf.ini` and `wrun.sh` by inserting full from which to load scripts and store data, where indicated.

To manually create long stretches of test data, follow these steps:
1. Generate an injection file. To do so run
    ```
    pycbc_create_injections \
    --ninjections 162000 \
    --config-files TestData/injection.ini \
    --output-file TestData/tmp_injections.hdf
    ```
2. Stack the injection-times in the file. This is a small workaround.
    ```
    ./TestData/pycbc_stack_injections.py \
    --injection-file TestData/tmp_injections.hdf \
    --output-file TestData/injections.hdf
    ```
3. Slice and whiten the strain. This is the step which requires the most amount of computation. Depending on the amount of memory available to you, you have to adjust the duration that can be sliced at once. The general call will look like
    ```
    ./TestData/pycbc_slice_strain \
    --gps-start-time <start-time> \
    --gps-end-time <end-time> \
    --first-slice-position <start-position> \
    --slice-size 1 \
    --step-size 0.1 \
    --detector H1 \
    --fake-strain H1:aLIGOZeroDetHighPower \
    --low-frequency-cutoff 15 \
    --sample-rate 2048 \
    --channel-name H1:H1:CHANNEL \
    --force \
    --output TestData/sliced/sliced-{gpsStart}-{gpsEnd}-{nsamples}.hdf \
    --whiten
    ```
The parameters `<start-time>` and `<end-time>` have to be integers. If you analyze chunks of 4096 seconds duration the first call would have `<start-time> = 0`, `<end-time> = 4096` and the second call would have `<start-time> = 4095`, `<end-time> = 8192`. Notice that for the second call the `<start-time>` is the previous `<end-time>` minus 1. This is required due to the internal workings of the code. So for all but the first call the new `<start-time>` is the old `<end-time>` minus 1. The parameter `<start-position>` is also required due to the internal workings of the code. For the first call this can be set to 0. From the second call on it has to be the new `<start-time>` plus 0.1. (The values that have to be subtracted/added are determined by the slice-size and the step-size.) Call the above function with the adjusted parameters until the `<end-time>` is equal or larger than 2,592,000.

## 4 Apply the network to test data

### 4.1 Tensorflow

To run a network over the sliced data call the script `Tensorflow/test_network.py`. For a full list of options refer to `Tensorflow/test_network.py -h`. A sample call to this function is
```
./Tensorflow/test_network.py \
--network ./Tensorflow/training_output/curriculum_0.hf5 \
--input-dir ./TestData/sliced \
--output-dir ./TestData/output \
--remove-softmax \
--store-command \
--verbose \
--create-dirs
```
The option `--remove-softmax` must be set in order to evaluate the sensitive distance for the network both with and without the final Softmax activation. Otherwise, it is only possible to evaluate the sensitive distance for the network with the final Softmax activation.

### 4.2 PyTorch

## 5 Evaluate the network

To evaluate the output of the network call the script `evaluate_test_data.py`. It calculates triggers, clusters them into events and compares them to the injections to estimate false-alarm rates and corresponding sensitivities.
To reproduce the analysis done in [`[1]`](#publication) call
```
./evaluate_test_data.py \
--injection-file TestData/injections.hdf \
--data-dir TestData/output \
--start-time-offset 0.7 \
--test-data-activation linear \
--ranking-statistic softmax \
--delta-t 0.099609375 \
--trigger-file-name triggers_softmax.hdf \
--event-file-name events_softmax.hdf \
--stats-file-name stats_softmax.hdf \
--verbose
```
This evaluates the network with a final Softmax activation. In case the injection-file contains injection times outside the analysis region (i.e. tc > last slice position), these triggers <b>must be cropped manually</b> from the injection file. It will otherwise lead to overestimated false-alarm rates.
Note that the option `--delta-t` must be set to the given value. This stems from the fact that the script `pycbc_slice_strain` internally has a step size converted from time to samples. This conversion is `int(step size * sample rate)`. A step size of 0.1s converts to 204 samples. With a sample rate of 2048 this converts to an actual step size of `204 / 2048 = 0.099609375` seconds. If `--delta-t` is set to `0.1` instead, the error accumulates and offsets returned events too much.

To reproduce the results after removing the final activation call
```
./evaluate_test_data.py \
--trigger-threshold -2.2 \
--injection-file TestData/injections.hdf \
--data-dir TestData/output \
--start-time-offset 0.7 \
--test-data-activation linear \
--ranking-statistic linear \
--delta-t 0.099609375 \
--trigger-file-name triggers_linear.hdf \
--event-file-name events_linear.hdf \
--stats-file-name stats_linear.hdf \
--verbose
```

The scripts will create 3 files each. Each of them is a HDF5 file. The first file contains information about times at which the output of the network exceeds some threshold value. The times are given in the `data` group, the corresponding network output in the `trigger_values` group. The second file contains the events. The times of the events are given in the `times` group, the corresponding ranking statistic in the `values` group. The last file contains the actual statistics. It consists of 6 groups. The groups contain the following information
| Group        | Content                                                         |
|--------------|-----------------------------------------------------------------|
| ranking      | The ranking statistic corresponding to a given false-alarm rate |
| far          | The false-alarm rates                                           |
| sens-frac    | The fraction of recovered signals                               |
| sens-dist    | The distance out to which the search is sensitive               |
| sens-vol     | The volume to which the search is sensitive                     |
| sens-vol-err | The Monte-Carlo error of the volume                             |

## Matched filter

To construct the matched filter comparison a template bank is required. We include the bank file used in our analysis as `MatchedFilter/template_bank.hdf`. You can construct your own by running `./MatchedFilter/gen_template_bank.sh`.

To run the search you can call the script `MatchedFilter/run_search.sh`. The script expects two positional arguments. The first must be the `<start-time>-80`, where `<start-time>` is the time at which analysis should begin. The second must be `<end-time>+16`, where `<end-time>` is the time at which to stop the analysis. Due to memory constraints it is infeasible to analyze the entire month of data at once. We chose a block size of 4096 seconds and thus called the script 633 times. Note that the analysis block must be larger then 512 seconds. To analyze the segment from 0 to 4096 seconds the call to the script would be
```
./MatchedFilter/run_search.sh -80 4112
```
which would output the file `MatchedFilter/output/-80-4112-out.hdf`.
Note, that the script expects the injection file to be located at `TestData/injections.hdf` and to be called from the root directory. Adjust the paths as required.

The triggers returned by the above script are not yet in a format that can be read by `evaluate_test_data.py`. To get them into a correct format, remove triggers below certain SNRs and outside a given analysis time use the script `MatchedFilter/collect_triggers.py`. Since the script `MatchedFilter/run_search.sh` is expected to be called more than once multiple output files are expected to exists and, therefore, the script `MatchedFilter/collect_triggers.py` loads triggers from all files in a specified directory that match a name pattern. To collect the triggers from the above example matched filter analysis use
```
./MatchedFilter/collect_triggers.py \
--dir MatchedFilter/output \
--output MatchedFilter/triggers.hdf \
--file-name {start}-{stop}-out.hdf \
--start-time 0 \
--end-time 2592000 \
--threshold 5 \
--verbose
```
Notice that we throw away any triggers with SNR < 5. This reduces file sizes and the required analysis time substantially.

The resulting trigger file can finally be analyzed using `evaluate_test_data.py`. To use the trigger-file just created you need to set the option `--load-triggers` and set the duration of the data analyzed in total. An example call would be
```
./evaluate_test_data.py \
--injection-file TestData/injections.hdf \
--load-triggers MatchedFilter/triggers.hdf \
--duration 2592000 \
--event-file-name MatchedFilter/events.hdf \
--stats-file-name MatchedFilter/stats.hdf \
--verbose
```
Note, that the injection file may only contain injections within the time analyzed by the matched filter search. If this is not the case, the injections have to be cropped manually. Also, the duration must match the time of the data which was analyzed. Otherwise, the resulting false-alarm rates may be far off.
The formats of the files are explained above and can be directly compared to the output of the machine learning analysis.

## Requirements

### Tensorflow

### PyTorch

## Acknowledgments

Put acknowledgments here.

## References
<a name="publication"></a>`[1]`: [Training Strategies for Deep Learning Gravitational-Wave Searches](https://arxiv.org)<br>
<a name="bnslib"></a>`[2]`: [BnsLib](https://github.com/MarlinSchaefer/BnsLib)
