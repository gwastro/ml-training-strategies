# Training Strategies for Deep Learning Gravitational-Wave Searches

Marlin B. Schäfer<sup>1, 2</sup>, Ondřej Zelenka<sup>3, 4</sup>, Alexander H. Nitz<sup>1, 2</sup>, Frank Ohme<sup>1, 2</sup>, Bernd Brügmann<sup>3, 4</sup>

<sub>1. [Albert-Einstein-Institut, Max-Planck-Institut for Gravitationsphysik, D-30167 Hannover, Germany](http://www.aei.mpg.de/obs-rel-cos)</sub><br>
<sub>2. Leibniz Universität Hannover, D-30167 Hannover, Germany</sub><br>
<sub>3. Friedrich-Schiller-Universität Jena, D-07743 Jena, Germany</sub><br>
<sub>4. Michael Stifel Center Jena, D-07743 Jena, Germany</sub>

## Table of contents

 * [Introduction](#Introduction)
 * [Contents of this repository](#contents-of-this-repository)
 * [Generate data required for training](#1-generate-data-required-for-training>)
 * [Train the networks](#2-train-the-networks)
   * [Tensorflow](#2.1-Tensorflow)
   * [Pytorch](#2.2-Pytorch)
 * [Generate test data](#3-Generate-test-data)
 * [Apply the network to test data](#4-Apply-the-network-to-test-data)
   * [Tensorflow](#4.1-Tensorflow)
   * [Pytorch](#4.2-Pytorch)
 * [Evaluate the network](#5-Evaluate-the-network)
 * [Matched filter](#Matched-filter)
 * [Requirements](#Requirements)
 * [Citation](#Citation)
 * [Acknowledgments](#Acknowledgments)
 * [References](#References)

## Introduction

Compact binary systems emit gravitational radiation which is potentially detectable by current Earth bound detectors. Extracting these signals from the instruments' background noise is a complex problem and the computational cost of most current searches depends on the complexity of the source model. Deep learning may be capable of finding signals where current algorithms hit computational limits. Here we restrict our analysis to signals from non-spinning binary black holes and systematically test different strategies by which training data is presented to the networks. To assess the impact of the training strategies, we re-analyze the first published networks and directly compare them to an equivalent matched-filter search. We find that the deep learning algorithms can generalize low signal-to-noise ratio (SNR) signals to high SNR ones but not vice versa. As such, it is not beneficial to provide high SNR signals during training, and fastest convergence is achieved when low SNR samples are provided early on. During testing we found that the networks are sometimes unable to recover any signals when a false alarm probability <10^-3 is required. We resolve this restriction by applying a modification we call unbounded Softmax replacement (USR) after training. With this alteration we find that the machine learning search retains >= 97.5% of the sensitivity of the matched-filter search down to a false-alarm rate of 1 per month.

## Contents of this repository

This repository contains the code required to reproduce the analysis presented in [`[1]`](#publication). It is split into two parts, which can be found in the separate directories `Tensorflow` and `Pytorch`. It contains scripts to generate the training, validation and test data and scripts to analyze the efficiency and sensitive volume of the networks. All final results and plots we used for the publication can be found in the `Results` folder.
The repository also includes parts of the BnsLib [`[2]`](#bnslib)

Below we provide a step-by-step guide on how to reproduce our analysis.

## 1 Generate data required for training

Call the script `generate_data.py` to generate 3 sets of data; a training set, a validation set and a data set used to determine the efficiencies during training (efficiency set).
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

The script produces only the efficiencies with the final Softmax activation still in place. To generate the efficiencies for the unbounded Softmax replacement as well, call the script `Tensorflow/get_efficiency.py` in the following way
```
./Tensorflow/get_efficiency.py \
--data-dir ./data \
--base-dir Tensorflow/training_output \
--model-name-format curriculum_{epoch}.hdf \
--sort-by epoch \
--use-dynamic-types \
--output Tensorflow/training_output/linear_efficiencies.csv \
--remove-softmax \
--verbose
```

For the above call to work the training/validation/efficiency data must be stored in `./data` and named with the correct prefixes. Also all training output must be located at `Tensorflow/training_output`. The option `--remove-softmax` is responsible for the unbounded Softmax replacement modification. For a list of all options and a description see `./Tensorflow/get_efficiency.py -h`.

### 2.2 Pytorch

To train a specific network call the script `Pytorch/train.py`. Output directories are set on lines 24 and 25 of the script, other parameters are set in the file `Pytorch/pars.py`. The script is set up to start a series of runs as in [`[1]`](#publication). The number of runs to be launched is specified on line 24 of `Pytorch/pars.py` and the index of the first run can be supplied as a command-line argument when launching `Pytorch/train.py`; if not supplied, it defaults to zero. To launch a set of 50 runs, one can either set `runs_number = 50` in `Pytorch/pars.py` and run:
```
python train.py
```
a single time to produce a series of non-parallelized runs indexed from 0 to 49 (optionally `python train.py $i` to start the indexing at `i`), or set, e.g., `runs_number = 1` and run:
```
python train.py 0
python train.py 1
...
python train.py 49
```
This way, the experiment can be parallelized over multiple GPUs. The output directories `state_dicts` and `outfiles` must be created before running the script.

The training strategy is specified using the file `Pytorch/scheduler_pars.py`. By default, the `fixed_low` strategy is used. To use a different strategy, replace the `scheduler_pars.py` file with the one from the corresponding directory under `Pytorch/strategies/`.

Efficiencies can be calculated using the script `Pytorch/calculate_efficiencies.py`. Options are set on lines 7-12, the output directory (`efficiencies` by default) must be created before running the script. Option on line 11 can be changed to `True` to use the softmax removal technique. By launching the script a single time as
```
python calculate_efficiencies.py
```
the efficiencies of all the runs specified in `Pytorch/pars.py` as `runs_number` are computed consecutively. Other options are to call
```
python calculate_efficiencies.py $i
```
to only calculate efficiencies of a single run with index `i`, or
```
python calculate_efficiencies.py $i $j
```
to calculate efficiencies of runs `i`, `i+1`, ..., `j-1`.

They can then be plotted using the script `Pytorch/plot_efficiencies.py`. Options are set on lines 8-16, `runs_number` is specified separately from `Pytorch/pars.py` due to the option of parallelization. The variable `extra_plots` should only be changed if one is adding another subplot, which should then be implemented at the end of the script, just before `fig.savefig()` is called. The option `indices_filerow_plot` and `index_filecol` point to the lines containing the corresponding SNRs (commented lines are omitted!) and the column containing the desired FAP, respectively, in the efficiency files. The option `indices_filerow_eval` specifies which lines in the efficiency files are to be used to determine the "High", "Mean" and "Low" network states, whose indices are printed in `stdout` as well as saved in the `epoch_run_nums.txt` file. The resulting plot is then saved as `efficiency_plots.png`.

## 3 Generate test data

Code to generate the test data is located in the directory `TestData`.
To generate test data we make use of the pycbc.workflow module. This is not strictly required but makes it easy to generate large datasets on a compute-cluster.
If you want to use the workflow, adjust the file `wf.ini` and `wrun.sh` by inserting full paths from which to load scripts and store data, where indicated.

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
--network ./Tensorflow/training_output/curriculum_1.hf5 \
--input-dir ./TestData/sliced \
--output-dir ./TestData/output \
--remove-softmax \
--store-command \
--verbose \
--create-dirs
```
This call uses the first epoch that finished training to evaluate the data. To use a different epoch adjust `<epoch>` in `--network ./Tensorflow/training_output/curriculum_<epoch>.hf5`. to the desired epoch. The option `--remove-softmax` must be set in order to evaluate the sensitive distance for the network both with and without the final Softmax activation. Otherwise, it is only possible to evaluate the sensitive distance for the network with the final Softmax activation.

### 4.2 Pytorch

To run a network over the sliced data call the script `Pytorch/test_network.py`. For a full list of options refer to `Pytorch/test_network.py -h`. A sample call to this function
```
python test_network.py \
--model-file-path /path/to/state_dicts/fixed_low_high.pt \
--input-dir ./TestData/sliced \
--output-dir ./TestData/output \
--verbose
```
Unlike the TensorFlow version, the usual way to store models in PyTorch only keeps the weights, while the layers are defined separately; thus, the `Pytorch/test_network.py` script explicitly contains the network definition and the option `--model-file-path` only specifies where the weights are stored. If one wishes to modify the network, it's necessary to also modify its definition in the test script, otherwise it will crash or produce incorrect results. Furthermore, the `--remove-softmax` option can be added, in which case the final Softmax is replaced by a subtraction layer to produce the ranking statistic *x<sub>0</sub>-x<sub>1</sub>* as its first output component.

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

The scripts will create 3 files each. Each of them is a HDF5 file. The first file (`triggers_<x>.hdf`) contains information about times at which the output of the network exceeds some threshold value. The times are given in the `data` group, the corresponding network output in the `trigger_values` group. The second file (`events_<x>.hdf`) contains the events. The times of the events are given in the `times` group, the corresponding ranking statistic in the `values` group. The last file(`stats_<x>.hdf`) contains the actual statistics. It consists of 6 datasets. The datasets contain the following information
| Dataset      | Content                                                         |
|--------------|-----------------------------------------------------------------|
| ranking      | The ranking statistic corresponding to a given false-alarm rate |
| far          | The false-alarm rates                                           |
| sens-frac    | The fraction of recovered signals                               |
| sens-dist    | The distance out to which the search is sensitive               |
| sens-vol     | The volume to which the search is sensitive                     |
| sens-vol-err | The Monte-Carlo error of the volume                             |

## Matched filter

To construct the matched filter comparison a template bank is required. We include the bank file used in our analysis as `MatchedFilter/template_bank.hdf`. You can construct your own by running `./MatchedFilter/gen_template_bank.sh`.

To run the search you can call the script `MatchedFilter/run_search.sh`. The script expects two positional arguments. The first must be the `<start-time> - 80`, where `<start-time>` is the time at which analysis should begin. The second must be `<end-time> + 16`, where `<end-time>` is the time at which to stop the analysis. Due to memory constraints it is infeasible to analyze the entire month of data at once. We chose a block size of 4096 seconds and thus called the script 633 times. Note that the analysis block must be larger then 512 seconds. To analyze the segment from 0 to 4096 seconds the call to the script would be
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

To run the code you need to install the following packages and their requirements

```
lalsuite 6.75
pycbc 1.18.0
tensorflow 2.3.0
torch 1.8.1
```

## Citation

If you use any of the material here, please cite the paper as
```
@article{Schafer:2021fea,
    author = {Sch\"afer, Marlin B. and Zelenka, Ond\v{r}ej and Nitz, Alexander H. and Ohme, Frank and Br\"ugmann, Bernd},
    title = "{Training strategies for deep learning gravitational-wave searches}",
    eprint = "2106.03741",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.IM",
    doi = "10.1103/PhysRevD.105.043002",
    journal = "Phys. Rev. D",
    volume = "105",
    number = "4",
    pages = "043002",
    year = "2022"
}
```

## Acknowledgments

We acknowledge the Max Planck Gesellschaft and the Atlas cluster computing team at Albert-Einstein Institut (AEI) Hannover for support, as well as the ARA cluster team at the URZ Jena. F.O. was supported by the Max Planck Society's Independent Research Group Programme. O.Z. thanks the Carl Zeiss Foundation for the financial support within the scope of the program line "Breakthroughs".

## References
<a name="publication"></a>`[1]`: [Training Strategies for Deep Learning Gravitational-Wave Searches](https://doi.org/10.1103/PhysRevD.105.043002)<br>
<a name="bnslib"></a>`[2]`: [BnsLib](https://github.com/MarlinSchaefer/BnsLib)
