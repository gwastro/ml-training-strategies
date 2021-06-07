#! /usr/bin/env python
from tensorflow import keras
from model import get_model
from generator import get_generator
import argparse
from callbacks import get_callbacks
from reg_loss import reg_loss
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--store-path', type=str,
                        help='The path under which all data created by this script is stored.')
    parser.add_argument('--data-path', type=str,
                        help='The path at which the training data is stored.')
    parser.add_argument('--epochs', type=int, default=200,
                        help="The number of epochs each network is trained for.")
    parser.add_argument('--run-key', type=str, default='loss',
                        help="A key known in callbacks.py to determine how the network is trained.")
    parser.add_argument('--network-author', type=str, default='gabbard',
                        help="Specify which network to use.")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="The batch size to use during training.")
    parser.add_argument('--noise-per-signal-train', type=int, default=1,
                        help="How many noise iterations to use for each signal during training.")
    parser.add_argument('--noise-per-signal-validation', type=int, default=1,
                        help="How many noise iterations to use for each signal during validation.")
    parser.add_argument('--sleep', type=int,
                        help="Sleep for this amount of time (in seconds) before starting. Helps with loading issues.")
    parser.add_argument('--frequency-training', action='store_true',
                        help="Train the network on frequency data instead of time series data.")
    
    args = parser.parse_args()
    
    if args.sleep is not None:
        time.sleep(args.sleep)
    
    if args.store_path is None:
        base_path = os.path.join(os.getcwd(), 'net_output', 'base_out')
    else:
        base_path = args.store_path
    
    if not os.path.isdir(base_path):
        os.makedirs(base_path, exist_ok=True)
    
    if args.data_path is None:
        data_dir = os.path.join(os.getcwd(), 'data')
    else:
        data_dir = args.data_path
    
    #Get generators
    train_generator = get_generator(data_dir, 'train',
                                    noise_per_signal=args.noise_per_signal_train,
                                    batch_size=args.batch_size,
                                    shuffle=True,
                                    freq_data=args.frequency_training)
    val_generator = get_generator(data_dir, 'val',
                                  noise_per_signal=args.noise_per_signal_validation,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  freq_data=args.frequency_training)
    sens_sig_generator = get_generator(data_dir, 'thr', n_noise=0,
                                       noise_per_signal=1,
                                       shuffle=True,
                                       freq_data=args.frequency_training)
    sens_noi_generator = get_generator(data_dir, 'thr', n_signals=0,
                                       shuffle=True,
                                       use_signal_files=False,
                                       freq_data=args.frequency_training)
    
    #Generate model
    model = get_model(num_detectors=1, author=args.network_author,
                      freq_data=args.frequency_training)
    opti = keras.optimizers.Adam(lr=1e-5, epsilon=1e-8)
    model.compile(optimizer=opti, loss=reg_loss, metrics=['acc'])
    
    #Train the network
    callbacks = get_callbacks(train_generator, val_generator,
                              sens_sig_generator, sens_noi_generator,
                              base_path=base_path, key=args.run_key)
    model.fit(train_generator, validation_data=val_generator,
              epochs=args.epochs, callbacks=callbacks)
    return

if __name__ == '__main__':
    main()
