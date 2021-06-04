import argparse
from ..utils.config import get_config_value
#########
#Actions#
#########

class TranslationAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        tmp = {}
        for pt in values:
            key, val = pt.split(':')
            tmp[key] = val
        setattr(namespace, self.dest, tmp)

class TypedDictAction(argparse.Action):
    type_dict = {'int': int,
                 'str': str,
                 'float': float,
                 'bool': bool}
    def __call__(self, parser, namespace, values, option_string=None):
        tmp = {}
        for pt in values:
            split = pt.split(':')
            if len(split) == 2:
                key, val = split
                tmp[key] = get_config_value(val)
            elif len(split) == 3:
                key, typ, val = split
                tmp[key] = self.type_dict[typ](val)
            else:
                msg = 'Argument must contain at least 2 and at most 3 '
                msg += 'part that are separated by `:`. Received {} '
                msg += 'arguments instead. ({})'
                msg = msg.format(len(split), split)
                raise RuntimeError(msg)
        setattr(namespace, self.dest, tmp)

#######
#Types#
#######

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
