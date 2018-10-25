import os
import json

#
# Global configuration
#
_this_dir = os.path.dirname(os.path.realpath(__file__))
_lambda_dir = os.path.abspath(_this_dir + '/../..')
_data_dir = _lambda_dir + '/data'

def lambda_dir():
    return lambda_dir

def data_dir():
    return _data_dir

def dirs(path):
    return sorted([x for x in os.listdir(path)
                   if os.path.isdir(path + '/' + x)])


def check_isnp(path, isnp):
    """Check if isnp is a valid snapshot index in a given path
    i.e., isnp directory exists in the path
    """
    
    ds = dirs(path)

    if isnp not in dirs(path):
        raise FileNotFoundError('isnp {} not found in {}'.format(isnp, path))


def check_sim(sim):
    """Check if sim is a valid simulation,
    i.e. directory `sim` exists in _data_dir
    """
    
    sims = sorted([x for x in os.listdir(_data_dir)
                   if os.path.isdir(_data_dir + '/' + x)])

    if sim not in sims:
        raise ValueError('sim = %s does not exist. Available sims %s' %
                         (sim, sims))

def isnp_str(isnp):
    """
    Convert to string if necessary;

    Args:
      isnp (str or int): snapshot number

    Returns
      isnp (str): snapshot number 000 - 010
    """

    if isinstance(isnp, str):
        return isnp
    elif isinstance(isnp, int):
        isnp = '%03d' % isnp
        return isnp
    else:
        raise TypeError('Unknown type for isnp: {}; must be int or str'.format(isnp))

def load_param(sim, isnp=None):
    """
    Return simulation info
    """

    #data_dir = lambdalib.util.data_dir()
    #data_dir = data_dir()
    check_sim(sim)

    with open('%s/%s/param.json' % (_data_dir, sim)) as f:
        d = json.load(f)

    if isnp is None:
        return d

    isnp = isnp_str(isnp)

    if not isnp in d['snapshot']:
        raise ValueError('isnp %s is not available in %s' % (isnp, sim))
    
    return d['snapshot'][isnp]

