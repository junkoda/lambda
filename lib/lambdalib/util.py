import os

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
