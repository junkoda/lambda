
# lambdalib package
#   The Libarary for the Lambda project
#
# Make this directory seen from Python
# e.g. export PYTHONPATH=/Users/junkoda/Research/lambda/lib


import os
import numbers
import numpy as np
import h5py
import json

from lambdalib.lambda_fitting import fit_lambda
import lambdalib.characteristic_function
import lambdalib.util
from lambdalib.util import load_param
from lambdalib.taruya import TaruyaModel
from lambdalib.power import load_linear_power, load_matter_power, load_halo_power, load_theta_power, load_bias, load_theta_power_bell_model, load_halofit_power
from lambdalib.dtfe import load_dtfe_A
from lambdalib.sigma import load_sigma_ab
from lambdalib.corr import load_corr_dduu

def _isnp_str(isnp):
    if isinstance(isnp, str):
        return isnp
    elif isinstance(isnp, int):
        isnp = '%03d' % isnp
    else:
        raise TypeError('Unknown type for isnp: {}; must be int or str'.format(isnp))


#
# Functions (API)
#
def info(sim=None, isnp=None):
    """
    Return simulation info
    """

    if sim is None:
        """Return a list of simulations"""

        data_dir = lambdalib.util.data_dir()
        
        print('Project directory: ', lambdalib.util.lambda_dir())
        print('Data directory: ', data_dir)
        
        return sorted([x for x in os.listdir(data_dir)
                   if os.path.isdir(data_dir + '/' + x)])

    lambdalib.util.check_sim(sim)

    data_dir = lambdalib.util.data_dir()

    with open('%s/%s/param.json' % (data_dir, sim)) as f:
        d = json.load(f)

    if isnp is None:
        return d

    isnp = _isnp_str(isnp)

    if not isnp in d['snapshot']:
        raise ValueError('isnp %s is not available in %s' % (isnp, sim))
    
    return d['snapshot'][isnp]


def load_lambda(sim, isnp):
    """
    Load one realisation of lambda data

    Args:
      isnp (str): snapshot index '000' to '010'
      sim (str): simulation name
    """

    lambdalib.util.check_sim(sim)
    data_dir = lambdalib.util.data_dir()
    isnp = _isnp_str(isnp)

    d = {}
    s = {}
    filename = '%s/%s/lambda/lambda_summary_%s.h5' % (data_dir, sim, isnp)
    
    with h5py.File(filename, 'r') as f:
        s['PDD'] = f['PDD'][:]
        s['PDU'] = f['PDU'][:]
        s['PUU'] = f['PUU'][:]
        s['PDD0'] = f['PDD0'][:]
        s['dPDD'] = f['dPDD'][:]
        s['dPDU'] = f['dPDU'][:]
        s['dPUU'] = f['dPUU'][:]
        d['k'] = f['k'][:]
        d['mu'] = f['mu'][:]
        d['lambda'] = f['lambda'][:]
        d['redshift'] = f['parameters/z'][()]
        d['nrealisations'] = f['nrealisations']

    d['summary'] = s

        
    return d
        

    
def load_characteristic_function(sim, isnp):
    """
    Load characteristic function data

    Args:
      sim (str): wizcola, wizcola_particles, nbody
      isnp (str): snapshot index
    """

    lambdalib.util.check_sim(sim)
    data_dir = lambdalib.util.data_dir()
    
    isnp = _isnp_str(isnp)

    path = '%s/%s/characteristic_function' % (data_dir, sim)

    return lambdalib.characteristic_function.load(path, isnp)


