import argparse
import h5py
import numpy as np
import logging
import os

import torch
nn = torch.nn

device = 'cpu'
dtype = torch.float32

def get_network(device=device, dtype=dtype, state_dict=None, replace_softmax=False):
	Network = nn.Sequential(
		nn.BatchNorm1d(1),	# 1x2048
		nn.Conv1d(1, 8, 64),	# 8x1985
		nn.ELU(),
		nn.Conv1d(8, 8, 32),	# 8x1954
		nn.MaxPool1d(4),	# 8x488
		nn.ELU(),
		nn.Conv1d(8, 16, 32),	# 16x457
		nn.ELU(),
		nn.Conv1d(16, 16, 16),	# 16x442
		nn.MaxPool1d(3),	# 16x147
		nn.ELU(),
		nn.Conv1d(16, 32, 16),	# 32x132
		nn.ELU(),
		nn.Conv1d(32, 32, 16),	# 32x117
		nn.MaxPool1d(2),	# 32x58
		nn.ELU(),
		nn.Flatten(),	#  1856
		nn.Linear(1856, 64),	# 64
		nn.Dropout(p=.5),
		nn.ELU(),
		nn.Linear(64, 64),	# 64
		nn.Dropout(p=.5),
		nn.ELU(),
		nn.Linear(64, 2),	# 2
		nn.Softmax(dim=1)
	)
	if not state_dict is None:
		Network.load_state_dict(state_dict)
	if replace_softmax:
		new_layer = torch.nn.Linear(2, 2, bias=False)
		new_layer._parameters['weight'] = torch.nn.Parameter(torch.Tensor([[1., -1.], [-1., 1.]]), requires_grad=False)
		new_layer.to(device=device)
		Network[-1] = new_layer
	Network.to(device=device, dtype=dtype)
	Network.eval()
	return Network


def main():
    #Process command-line arguments
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--input-dir', required=True, type=str, help="The path to a directory from which the files will be read.")
    parser.add_argument('--output-dir', required=True, type=str, help="The path to a directory in which the output files will be stored.")
    parser.add_argument('--verbose', action='store_true',
                        help="Print status updates while loading.")
    parser.add_argument('--debug', action='store_true',
                        help="Print debugging info")
    parser.add_argument('--model-file-path', type=str, default='', required=True, help="Path to file containing model state dictionary.")
    parser.add_argument('--batch-size', type=int, default=0, required=False, help="Batch size, 0 means entire data files. Default: 0")
    parser.add_argument('--device', type=str, default=device, required=False, help="Device to use for calculation, options are '%s' (default) or 'cuda', in case of more GPUs 'cuda:0', 'cuda:1', etc."%device)
    parser.add_argument('--remove-softmax', action='store_true', help="Replace the final softmax layer by a 'mutual subtraction layer' to remove the exponential saturation while preserving the order.")

    args = parser.parse_args()


    #Setup logger
    log_level = logging.INFO if args.verbose else logging.WARN
    if args.debug:
        log_level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s | %(asctime)s: %(message)s',
                        level=log_level, datefmt='%d-%m-%Y %H:%M:%S')

    #Initialize network
    logging.info(f'Initializing network with state dictionary from {args.model_file_path} on device {args.device}')
    if args.remove_softmax:
        logging.info('Removing the softmax layer')
    Network = get_network(device=args.device, dtype=dtype, state_dict=torch.load(args.model_file_path))
    
    logging.info(f'Network initialized, starting to load files')
    if args.batch_size == 0:
        logging.info('Each file will be processed as a single batch')
    else:
        logging.info(f'Files will be processed with batch size {args.batch_size}')

    #Check input and output directories
    if not os.path.isdir(args.input_dir):
        raise ValueError('Unknown input directory {}.'.format(args.input_dir))
    if not os.path.isdir(args.output_dir):
        raise ValueError('Missing output directory {}.'.format(args.output_dir))
    if args.input_dir == args.output_dir:
        raise ValueError('Cannot use the input directory as the output directory.')

    #Get filenames
    file_list = os.listdir(args.input_dir)
    logging.info('Found %i files in the input directory' % len(file_list))

    for fn in file_list:
        fin = os.path.join(args.input_dir, fn)
        logging.debug(f'Trying to load file from {fin}')
        #Load individual file
        with h5py.File(fin, 'r') as fp:
            in_data = fp['H1/data'][()]

        #Convert to PyTorch and into proper shape
        in_data = torch.from_numpy(in_data).to(device=args.device, dtype=dtype)
        assert len(in_data.shape)==2
        in_data = torch.unsqueeze(in_data, 1)
        logging.info(f'Successfully loaded data from {fn}')

        #Batch the data and run the model
        with torch.no_grad():
        	if args.batch_size==0:
        		out_data = Network(in_data).cpu().numpy()
        	else:
        		in_chunks = torch.split(in_data, args.batch_size, dim=0)
        		out_chunks = [Network(in_chunk).cpu().numpy() for in_chunk in in_chunks]
        		out_data = np.concatenate(out_chunks, axis=0)

        #Save output
        fout = os.path.join(args.output_dir, fn)
        with h5py.File(fout, 'w') as fp:
        	fp.create_dataset('data', data=out_data)

        logging.info(f'Successfully saved output to {fout}')

    # logging.info(f'Loaded {len(data)} samples')
    logging.info('Finished')
    return

if __name__ == "__main__":
    main()
