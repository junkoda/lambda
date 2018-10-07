#
# Power spectra
#
import math
import numpy as np
import glob

def load_linear(sim, data_dir):
    """
    Returns:
      dictionary with
      'k': wavenumbeer [h/Mpc]
      'P': linear power spectrum [1/h Mpc]^3
    """
    
    filename = '%s/%s/linear_matterpower.dat' % (data_dir, sim)
    a = np.loadtxt(filename)

    d = {}
    d['a'] = a
    d['k'] = a[:, 0]
    d['P'] = a[:, 1]

    return d


def load_halo_power(sim, isnp, data_dir):
    """
    Returns:
      dictionary with
      'a[ik, iP, irealisation]' (array): array of all realisations 
      'k': wavenumber [h/Mpc]
      'nmodes': number of k modes in bins
      'P': mean real-space halo power spectrum
    """

    if isnp is None:
        raise ValueError('isnp is not specified')
    
    path = '%s/%s/halo_power/%s/halo_power_*.txt' % (data_dir, sim, isnp)
    filenames = glob.glob(path)

    if not filenames:
        raise FileNotFoundError('halo power spectrum not found: %s' %
                                path)

    filenames = sorted(filenames)

    n = len(filenames)
    P = None

    for i, filename in enumerate(filenames):
        a = np.loadtxt(filename)

        if P is None:
            P = np.zeros((a.shape[0], a.shape[1], n))

        P[:, :, i] = a

    d = {}
    d['a'] = P
    d['k'] = np.mean(P[:, 0, :], axis=1)
    d['nmodes'] = P[:, 1, 0]
    d['P'] = np.mean(P[:, 2, :], axis=1)
    d['dP'] = np.std(P[:, 2, :], axis=1)/math.sqrt(n)

    return d
    

def load_theta_power(sim, isnp, data_dir, nc=None):
    """
    Args:
      nc (int): resolution e.g. 324 for directory nc324
    """
    
    if isnp is None:
        raise ValueError('isnp is not specified')

    if nc is None:
        path = '%s/%s/theta_power/nc*/' % (data_dir, sim)
        dirs = glob.glob(path)
        
        if not dirs:
            raise FileNotFoundError('nc directories not found in %s' % path)
        nc_dir = sorted(dirs)[-1]
    else:
        nc_dir = '%s/%s/theta_power/nc%d' % (data_dir, sim, nc)
    
    path = '%s/%s/theta_power_*.txt' % (nc_dir, isnp)
    filenames = glob.glob(path)

    if not filenames:
        raise FileNotFoundError('Theta power spectra not found: %s' % path)

    filenames = sorted(filenames)

    n = len(filenames)
    P = None

    params = lambdalib.info(sim, isnp)
    aH = params['a']*params['H']

    for i, filename in enumerate(filenames):
        a = np.loadtxt(filename)
        
        if P is None:
            P = np.zeros((a.shape[0], a.shape[1], n))

        a[:, 3] /= aH
        a[:, 4] /= aH**2

        P[:, :, i] = a

    d = {}
    d['a'] = P
    d['k'] = P[:, 0, 0]
    d['nmodes'] = P[:, 1, 0]
    d['Pdd'] = np.mean(P[:, 2, :], axis=1)
    d['Pdt'] = np.mean(P[:, 3, :], axis=1)
    d['Ptt'] = np.mean(P[:, 4, :], axis=1)
    
    d['dPdd'] = np.std(P[:, 2, :], axis=1)/math.sqrt(n)
    d['dPdt'] = np.std(P[:, 3, :], axis=1)/math.sqrt(n)
    d['dPtt'] = np.std(P[:, 4, :], axis=1)/math.sqrt(n)

    return d
