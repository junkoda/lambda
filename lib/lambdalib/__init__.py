#
# lambdalib package
#   The Libarary for the Lambda project
#
# Make this directory seen from Python
# e.g. export PYTHONPATH=/Users/junkoda/Research/lambda/lib


import os
import numbers
import numpy as np
import h5py


from lambdalib.lambda_fitting import fit_lambda
import lambdalib.power

#
# Global configuration
#
_this_dir = os.path.dirname(os.path.realpath(__file__))
_lambda_dir = os.path.abspath(_this_dir + '/../..')
_data_dir = _lambda_dir + '/data'
_sim_default = 'wizcola' # can be changed by select_sim()

#
# Internally used tools
#

# ...
def _check_sim(sim):
    sims = sorted([x for x in os.listdir(_data_dir)
                   if os.path.isdir(_data_dir + '/' + x)])

    if sim not in sims:
        raise ValueError('sim = %s does not exist. Available sims %s' %
                         (sim, sims))


#
# Functions (API)
#
def info():
    print('Project directory: ', _lambda_dir)
    print('Data directory: ', _data_dir)


def sims():
    """Returns a list of simulation data available, directories in the data directory"""
    
    return sorted([x for x in os.listdir(_data_dir)
                   if os.path.isdir(_data_dir + '/' + x)])

def select_sim(sim):
    """Set the default simulation data"""
    global _sim_default
    _sim_default = sim
    print('Set sim: ', _sim_default)


def load_lambda(sim, isnp):
    """
    Load one realisation of lambda data

    Args:
      isnp (str): snapshot index '000' to '010'
      sim (str): simulation name
    """

    _check_sim(sim)
    
    d ={}
    filename = '%s/%s/lambda/lambda_summary_%s.h5' % (_data_dir, sim, isnp)
    
    with h5py.File(filename, 'r') as f:
        d['PDD'] = f['PDD'][:]
        d['PDU'] = f['PDU'][:]
        d['PUU'] = f['PUU'][:]
        d['PDD0'] = f['PDD0'][:]
        d['dPDD'] = f['dPDD'][:]
        d['dPDU'] = f['dPDU'][:]
        d['dPUU'] = f['dPUU'][:]
        d['k'] = f['k'][:]
        d['mu'] = f['mu'][:]
        d['lambda'] = f['lambda'][:]
        d['redshift'] = f['parameters/z'][()]
        d['nrealisations'] = f['nrealisations']

    return d
        

def load_power_spectrum(kind, sim, isnp=None, *, nc=None):
    """
    Load real-space power spectrum

    Args:
      sim (str): wizcola, wizcola_particles, nbody
      kind (str): linear, matter, halo, DTFE
      isnp (str): snapshot index
    """

    _check_sim(sim)
    
    if kind == 'linear':
        return lambdalib.power.load_linear(sim, _data_dir)
    elif kind == 'halo':
        return lambdalib.power.load_halo_power(sim, isnp, _data_dir)
    elif kind == 'theta':
        return lambdalib.power.load_theta_power(sim, isnp, _data_dir, nc)
    else:
        raise ValueError('Unknown power spectrum name: %s' % kind)
