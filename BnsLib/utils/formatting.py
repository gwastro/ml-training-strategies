import numpy as np

def list_length(inp):
    """Returns the length of a list or 1, if the input is not a list.
    
    Arguments
    ---------
    inp : list or other
        The input.
    
    Returns
    -------
    int
        The length of the input, if the input is a list. Otherwise
        returns 1.
    
    Notes
    -----
    -A usecase for this function is to homologize function inputs. If
     the function is meant to operate on lists but can also accept a
     single instance, this function will give the length of the list the
     function needs to create. (Useful in combination with the function
     input_to_list)
    """
    if isinstance(inp, list):
        return len(inp)
    else:
        return 1

def input_to_list(inp, length=None):
    """Convert the input to a list of a given length.
    If the input is not a list, a list of the given length will be
    created. The contents of this list are all the same input value.
    
    Arguments
    ---------
    inp : list or other
        The input that should be turned into a list.
    length : {int or None, None}
        The length of the output list. If set to None this function will
        call list_length to determine the length of the list.
    
    Returns
    -------
    list
        Either returns the input, when the input is a list of matching
        length or a list of the wanted length filled with the input.
    """
    if length is None:
        length = list_length(inp)
    if isinstance(inp, list):
        if len(inp) != length:
            msg = f'Length of list {len(inp)} does not match the length'
            msg += f' requirement {length}.'
            raise ValueError(msg)
        else:
            return inp
    else:
        return [inp] * length

def field_array_to_dict(inp):
    """Convert a pycbc.io.record.FieldArray to a standard Python
    dictionary.
    
    Arguments
    ---------
    inp : pycbc.io.record.FieldArray or numpy.recarry
        The array to convert.
    
    Returns
    -------
    dict:
        A dict where each value is a list containing the values of the
        numpy array.
    """
    return {name: list(inp[name]) for name in inp.dtype.names}

def dict_to_field_array(inp):
    """Convert a Python dictionary to a numpy FieldArray.
    
    Arguments
    ---------
    inp : dict
        A dictionary with structure `<name>: <list of values>`. All
        lists must be of equal length.
    
    Returns
    -------
    numpy field array
    """
    assert isinstance(inp, dict)
    keys = list(inp.keys())
    assert all([len(inp[key]) == len(inp[keys[0]]) for key in keys])
    out = []
    for i in range(len(inp[keys[0]])):
        out.append(tuple([inp[key][i] for key in keys]))
    dtypes = [(key, np.dtype(type(out[0][i]))) for (i, key) in enumerate(keys)]
    return np.array(out, dtype=dtypes)

def split_str_by_vars(string):
    """Split a string into variables and strings, where each variable is
    is engulfed in curly brackets.
    
    Arguments
    ---------
    string : str
        The string to split into parts.
    
    Returns
    -------
    list of str
        A list of strings. Each string is either a variable or a
        substring.
    
    Example
    -------
    >>> split_str_by_vars('{hello} world')
        ['{hello}', ' world']
    
    Notes
    -----
    Escape the variable characters { and } by prepending them with \.
    """
    if len(string) == 0:
        return [string]
    if string[0] == '{':
        ret = []
    else:
        ret = ['']
    in_var = False
    esc = False
    for c in string:
        if c == '}' and not esc:
            if in_var:
                ret[-1] += c
                ret.append('')
                in_var = False
                if esc:
                    esc = False
            else:
                raise ValueError
        elif c == '{' and not esc:
            if in_var:
                raise ValueError
            else:
                ret.append('')
                in_var = True
                ret[-1] += c
                if esc:
                    esc = False
        elif c == '\\':
            esc = True
        else:
            ret[-1] += c
            if esc:
                esc = False
    else:
        if in_var:
            raise ValueError('Could not parse the string.')
    
    if ret[-1] == '':
        ret.pop(-1)
    return ret

def inverse_string_format(string, form, types=None,
                          dynamic_types=False):
    """Returns None if the string does not match the format. Otherwise
    returns a dictionary of the values.
    
    Arguments
    ---------
    string : str
        The string that should be checked against the format string.
    form : str
        The format string that should be matched with the input string.
    types : {dict or None, None}
        A dictionary containing the types of each variable in the format
        string. If a variable is not in this dictionary behavior is
        dictated by the argument dynamic_types. If set to None no
        variable types are assumed and the behavior is entirely dictated
        by the argument dynamic_types.
    dynamic_types : {bool, False}
        Try to interpret the types of variables which do not have a type
        stated explicitly. If set to True a the function
        BnsLib.utils.config.get_config_value will be used to infer the
        type of the variable. If set to False the return type will be
        string.
    
    Returns
    -------
    ret : dict
        A dictionary where the key is the variable name and the value is
        the value of said variable.
    
    Examples
    --------
    >>> inverse_string_format('hello_world', '{greet}_{target}')
        {'greet': 'hello', 'target': 'world'}
    >>>
    >>> inverse_string_format('file-0', 'file-{num}', types={'num': int})
        {'num': 0}
    >>>
    >>> inverse_string_format('fil-0', 'file-{num}', types={'num': int})
        None
    >>> inverse_string_format('file-0', 'file-{num}', dynamic_types=False)
        {'num': '0'}
    >>> inverse_string_format('file-0', 'file-{num}', dynamic_types=True)
        {'num': 0}
    """
    parts = split_str_by_vars(form)
    var_names = []
    str_parts = []
    for pt in parts:
        if pt.startswith('{') and pt.endswith('}'):
            var_names.append(pt[1:-1])
            str_parts.append(None)
        else:
            var_names.append(None)
            str_parts.append(pt)
    
    from BnsLib.utils.config import get_config_value
    if types is None:
        types = {}
    for var_name in var_names:
        if var_name not in types:
            if var_name is not None:
                if dynamic_types:
                    types[var_name] = get_config_value
                else:
                    types[var_name] = str
    
    if len(parts) == 1:
        if str_parts[0] is None == 1:
            return {var_names[0]: types[var_names[0]](string)}
        elif var_names[0] is None:
            if string == str_parts[0]:
                return {}
            else:
                return None
    
    values = []
    work_string = string
    part = 0
    if str_parts[0] is not None:
        if string.startswith(str_parts[0]):
            part = 1
            values.append(str_parts[0])
            work_string = work_string[len(str_parts[0]):]
        else:
            return None
    
    buff = ''
    work = ''
    i = 0
    sidx = 0
    while part + 1 < len(var_names):
        c = work_string[i]
        work += c
        if str_parts[part+1][sidx] == c:
            sidx += 1
        else:
            buff += work
            work = ''
            sidx = 0
        if len(work) == len(str_parts[part+1]):
            values.append(buff)
            values.append(work)
            buff = ''
            work = ''
            sidx = 0
            part += 2
        i += 1
        if i >= len(work_string):
            break;
    if part >= len(var_names) - 1:
        if var_names[-1] is not None:
            values.append(work_string[i:])
        elif i < len(work_string) -1:
            return None
            
    elif i >= len(work_string):
        return None
    
    if not len(values) == len(parts):
        raise RuntimeError
    
    ret = {}
    for i in range(len(parts)):
        if var_names[i] is not None:
            key = var_names[i]
            ret[key] = types[key](values[i])
    
    return ret
