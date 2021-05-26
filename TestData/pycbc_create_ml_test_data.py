#!/usr/bin/env python
#General imports
import argparse
import numpy as np
import h5py
import sys
import os
import logging
#PyCBC imports
import pycbc
import pycbc.workflow as wf
from pycbc.types import MultiDetOptionAction, MultiDetOptionAppendAction
from pycbc.workflow.jobsetup import PycbcCreateInjectionsExecutable
from ligo import segments
#BnsLib imports
#from BnsLib.types import str2bool

def main():
    #Setup basic parser
    parser = argparse.ArgumentParser()
    wf.add_workflow_command_line_group(parser)

    #Add arguments specific to this workflow
    parser.add_argument('--create-injections', action='store_true',
                        help="""Whether or not to create injections from
                                a config file.""")
    parser.add_argument('--cumsum-injection-times', action='store_true',
                        help="""Whether or not to apply a cumsum to the
                                injection times.""")
    parser.add_argument('--seed', type=int, default=0,
                        help="""The seed to use for data generation.
                                Default: 0""")
    parser.add_argument('--workers', type=int, default=1,
                        help="""How many nodes for slicing are created.
                                Default: 1""")
    parser.add_argument('--injection-config', type=str,
                        help="The config-file for pycbc_create_injections.")
    parser.add_argument('--min-separation', type=float, default=16.,
                        help="The minimum time between two injections. (in seconds)")
    
    #Main script
    opts = parser.parse_args()
    seed = opts.seed
    
    #Instantiate workflow
    if not hasattr(opts, 'workflow_name') or opts.workflow_name is None:
        wfn = "test-data"
    else:
        wfn = opts.workflow_name
    workflow = wf.Workflow(opts, wfn)
    ifos = workflow.ifos
    if not hasattr(opts, 'output_dir') or opts.output_dir is None:
        output_dir = wfn + '_output'
    else:
        output_dir = opts.output_dir
    wf.makedir(output_dir)
    config_dir = os.path.join(output_dir, 'config-files')
    wf.makedir(config_dir)
    
    gps_start_total = int(workflow.cp.get('workflow', 'start-time'))
    gps_end_total = int(workflow.cp.get('workflow', 'end-time'))
    duration_total = gps_end_total - gps_start_total
    
    #Take care of creating injections
    if opts.create_injections:
        injection_file_dir = os.path.join(output_dir, 'injection_files')
        wf.makedir(injection_file_dir)
        if opts.injection_config is None:
            raise ValueError('Must provide an injection configuration file when injections should be made.')
        inj_exe = wf.Executable(workflow.cp,
                                "inject",
                                ifos=ifos)
        node = inj_exe.create_node()
        ninj = int(duration_total / opts.min_separation)
        node.add_opt('--ninjections', ninj)
        node.add_opt('--seed', seed)
        node.add_opt('--config-files', opts.injection_config)
        seg = segments.segment(int(gps_start_total), int(gps_end_total))
        node.new_output_file_opt(seg, '.hdf', '--output-file')
        node.add_opt('--force')
        injection_file = node.output_files[0]
        
        seed += 1
        workflow += node
        
        #Fix the injection times
        if opts.cumsum_injection_times:
            fix_exe = wf.Executable(workflow.cp, "fix", ifos=ifos)
            node = fix_exe.create_node()
            node.add_input_opt('--injection-file', injection_file)
            node.new_output_file_opt(injection_file.segment, '.hdf',
                                     '--output-file')
            injection_file = node.output_files[0]
            workflow += node
    
    #Start slicing data
    slice_exe = wf.Executable(workflow.cp, "slice", ifos=ifos)
    
    duration_slice = int(np.ceil(duration_total / opts.workers))
    slice_size = float(workflow.cp.get('slice', 'slice-size'))
    step_size = float(workflow.cp.get('slice', 'step-size'))
    prev_end = gps_start_total + slice_size - step_size
    gps_end = gps_start_total + slice_size
    
    for worker in range(opts.workers):
        gps_start = int(gps_end - slice_size)
        gps_end = min(gps_end_total, gps_start_total + (worker + 1) * duration_slice)
        first_slice = prev_end - slice_size + step_size
        
        node = slice_exe.create_node()
        if opts.create_injections:
            node.add_input_opt('--injection-file', injection_file)
        node.add_opt('--gps-start-time', gps_start)
        node.add_opt('--gps-end-time', gps_end)
        node.add_opt('--first-slice-position', first_slice)
        
        prev_end = gps_end
        
        workflow += node
    
    workflow.save('test_data_generation.dax')

if __name__ == "__main__":
    main()
