#!/usr/bin/env python
import argparse
import h5py
import numpy as np

def hdf_to_dict(fp):
    ret = {}
    for key in fp.keys():
        try:
            ret[key] = fp[key][()]
        except AttributeError:
            ret[key] = hdf_to_dict(fp[key])
    return ret

def dict_to_hdf(fp, dic):
    for key, val in dic.items():
        if isinstance(val, dict):
            gr = fp.create_group(key)
            dict_to_hdf(gr, val)
        else:
            fp.create_dataset(key, data=val)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--injection-file', required=True, type=str,
                        help="""The file where a parameter should be
                                stacked, i.e. where the parameter should
                                be turned into a cumsum. (Must be HDF5)""")
    parser.add_argument('--parameter', type=str, default='tc',
                        help="The parameter to which a cumsum should be applied.")
    parser.add_argument('--output-file', type=str, 
                        help="The path to the output file.")
    
    opts = parser.parse_args()
    
    with h5py.File(opts.injection_file, 'r') as f:
        params = hdf_to_dict(f)
        attrs = dict(f.attrs)
    
    if opts.parameter in params:
        params[opts.parameter] = np.cumsum(params[opts.parameter])
    
    if opts.output_file is None:
        opts.output_file = opts.injection_file
    
    with h5py.File(opts.output_file, 'w') as fp:
        dict_to_hdf(fp, params)
        for key, val in attrs.items():
            fp.attrs[key] = val

if __name__ == "__main__":
    main()
