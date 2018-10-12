"""
Characteristic function

phi(lambda) = < e^ilambda X >

data/<sim>/characteristic_function/
"""

import math
import numpy as np
import glob
from lambdalib.util import check_isnp

def _load_txt(filename):
    with open(filename, 'r') as f:
        v = f.readline().rstrip().split()

    assert(v[1] == 'sigma2_v')
    sigma2_v = float(v[2])

    a = np.loadtxt(filename)

    return a, sigma2_v
    
def load(path, isnp):
    """
    path (str): path to the characteristic_function data directory
    isnp (str): isnp index
    """

    check_isnp(path, isnp)

    filenames = glob.glob('%s/%s/char_*.txt' % (path, isnp))

    if not filenames:
        raise FileNotFoundError(
            'No characteristic function data found in: %s/%s/'
            % (path, isnp))

    n = len(filenames)
    
    sigma2_v = np.zeros(n)
    phi = None
    
    for i, filename in enumerate(filenames):
        a, s = _load_txt(filename)

        if phi is None:
            phi = np.empty((a.shape[0], a.shape[1], n))
            
        phi[:, :, i] = a
        sigma2_v[i] = s

    summary = {}
    summary['lambda'] = phi[:, 0, 0]
    summary['phi'] = np.mean(phi[:, 1, :], axis=1)
    summary['dphi'] = np.std(phi[:, 1, :], axis=1)/math.sqrt(n)
    summary['sigma2_v'] = np.mean(sigma2_v)

    d = {}
    d['summary'] = summary
    d['phi'] = phi

    return d

