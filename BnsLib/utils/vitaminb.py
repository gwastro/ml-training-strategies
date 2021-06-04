import h5py
#from vitamin_b.params_files import make_params_files
import warnings
from BnsLib.utils.config import config_to_dict, dict_to_string_dict
from BnsLib.utils.bounds import estimate_transformed_bounds
import configparser

massTrans = {'mass1': 'mass_1',
             'mass2': 'mass_2',
             'mass_1': 'mass_1',
             'mass_2': 'mass_2'}

pycbc_to_vitamin = {'mass1': 'mass_1',
                    'mass2': 'mass_2',
                    'distance': 'luminosity_distance',
                    'ra': 'ra',
                    'dec': 'dec',
                    'tc': 'geocent_time',
                    'inclination': 'theta_jn',
                    'coa_phase': 'phase',
                    'pol': 'psi',
                    'mass_1': 'mass_1',
                    'mass_2': 'mass_2'}

vitamin_to_pycbc = {pycbc_to_vitamin[key]: key for key in pycbc_to_vitamin.keys()}

def params_files_from_config(params_config_file, network_config_file,
                             translation=None):
    if translation is None:
        translation = {}
    params_out = {}
    bounds_out = {}
    fixed_vals_out = {}
    params_content = config_to_dict(params_config_file)
    network_content = config_to_dict(network_config_file)
    for key, val in network_content.items():
        params_out.update(val)
    for key, val in params_content['misc'].items():
        params_out[key] = val
    params_out['rand_pars'] = []
    params_out['inf_pars'] = []
    params_out['gauss_pars'] = []
    params_out['vonmise_pars'] = []
    params_out['sky_pars'] = []
    params_out['boost_pars'] = []
    params_out['no_dist_pars'] = []
    
    params_keys = list(params_content['variable'].keys())
    for key in params_keys:
        if key not in translation:
            translation[key] = key
        params_out['rand_pars'].append(translation[key])
        if params_content[key]['infer']:
            params_out['inf_pars'].append(translation[key])
        dist = params_content[key]['network_distribution']
        params_out[dist + '_pars'].append(translation[key])
        if params_content[key].get('boost', False):
            params_out['boost_pars'].append(translation[key])
        bounds_out[translation[key]+'_min'] = params_content[key]['min']
        bounds_out[translation[key]+'_max'] = params_content[key]['max']
        if 'fixed' in params_content[key]:
            fixed_vals_out[translation[key]] = params_content[key]['fixed']
        else:
            mean = (params_content[key]['max'] + params_content[key]['min'])
            mean /= 2
            fixed_vals_out[translation[key]] = mean
    
    fixed_keys = list(params_content['fixed'].keys())
    for key in fixed_keys:
        if key not in translation:
            translation[key] = key
        if key in params_content:
            if 'network_distribution' in params_content[key]:
                dist = params_content[key]['network_distribution']
                params_out[dist + '_pars'].append(translation[key])
            else:
                params_out['gauss_pars'].append(translation[key])
        fixed_vals_out[translation[key]] = params_content['fixed'][key]
        if key in params_content:
            if 'min' in params_content[key]:
                bounds_out[translation[key]+'_min'] = params_content[key]['min']
            else:
                if fixed_vals_out[translation[key]] is not None:
                    bounds_out[translation[key]+'_min'] = fixed_vals_out[translation[key]]
            if 'max' in params_content[key]:
                bounds_out[translation[key]+'_max'] = params_content[key]['max']
            else:
                if fixed_vals_out[translation[key]] is not None:
                    bounds_out[translation[key]+'_max'] = fixed_vals_out[translation[key]]
        else:
            if fixed_vals_out[translation[key]] is not None:
                bounds_out[translation[key]+'_min'] = fixed_vals_out[translation[key]]
                bounds_out[translation[key]+'_max'] = fixed_vals_out[translation[key]]
    
    if 'bounds' in params_content:
        bounds_keys = list(params_content['bounds'].keys())
        for key in bounds_keys:
            if key not in translation:
                translation[key] = key
            
            if key in params_content:
                if 'min' in params_content[key]:
                    bounds_out[translation[key]+'_min'] = params_content[key]['min']
                else:
                    raise RuntimeError
                if 'max' in params_content[key]:
                    bounds_out[translation[key]+'_max'] = params_content[key]['max']
                else:
                    raise RuntimeError
            else:
                raise RuntimeError
     
    return params_out, bounds_out, fixed_vals_out
        
        
    #from BnsLib.data.genenerate_train import WFParamGenerator
    ##-Save all parameters to a dict {name: bounds}
    ##-Go through the list of transformations and apply:
    ##    BnsLib.utils.bounds.estimate_transformed_bounds
    ## to them
    ##-Go through transformations and parameters and look which are
    ## actually output -> save into the params dictionary
    ##-Apply translation to params, bounds and fixed-params
    #params = make_params_files.get_params()
    #bounds = {}
    #fixed_vals = {}
    #gen = WFParamGenerator(config_file)
    #for dist in gen.params.pval.distributions:
        #for key, val in dist.items():
            #if 'mass' in key:
            #key = massTrans[key]
        #bounds[key + '_min'] = val.min
        #bounds[key + '_max'] = val.max
        #bounds['__definition__' + key] = f'{key} range'
        #fixed_vals['__definition__' + key] = f'{key} fixed value'
        #fixed_vals[key] = (val.max + val.min) / 2

def vitaminb_params_to_pycbc_params(in_config_file, out_config_file):
    config_dict = config_to_dict(in_config_file)
    
    config = configparser.ConfigParser()
    
    #Write static arguments
    static_args = config_dict["static_args"]
    static_args.update(config_dict["fixed"])
    static_args_set_dict = {}
    for key, val in static_args.items():
        if val is not None:
            static_args_set_dict[key] = val
    config["static_args"] = static_args_set_dict
    
    #Write variable arguments
    variable_args = config_dict['variable']
    config['variable_args'] = dict_to_string_dict(variable_args)
    
    for name in variable_args.keys():
        prior_dict = {}
        prior_dict['name'] = config_dict[name]['distribution']
        prior_dict['min-'+name] = config_dict[name]['min']
        prior_dict['max-'+name] = config_dict[name]['max']
        
        config['prior-'+name] = dict_to_string_dict(prior_dict)
    
    #Copy constraints
    for key in config_dict.keys():
        if key.startswith('constraint'):
            config[key] = dict_to_string_dict(config_dict[key])
    
    with open(out_config_file, 'w') as configfile:
        config.write(configfile)
