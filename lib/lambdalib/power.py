#
# Power spectra
#
import math
import numpy as np
import glob
import lambdalib.util

def load_linear_power(sim):
    """
    Linearly extrapolated power spectrum at redshift=0
    Returns:
      dictionary with
      'k': wavenumbeer [h/Mpc]
      'P': linear power spectrum [1/h Mpc]^3
    """

    data_dir = lambdalib.util.data_dir()
    filename = '%s/%s/linear_matterpower.dat' % (data_dir, sim)
    a = np.loadtxt(filename)

    d = {}
    d['a'] = a
    d['k'] = a[:, 0]
    d['P'] = a[:, 1]

    return d


def _load_power(file_pattern, isnp):
    filenames = glob.glob(file_pattern)

    if not filenames:
        raise FileNotFoundError('halo power spectrum not found: %s' %
                                file_pattern)

    filenames = sorted(filenames)

    n = len(filenames)
    P = None

    for i, filename in enumerate(filenames):
        a = np.loadtxt(filename)

        if P is None:
            P = np.zeros((a.shape[0], a.shape[1], n))

        P[:, :, i] = a

    summary = {}
    summary['P'] = np.mean(P[:, 2, :], axis=1)
    summary['dP'] = np.std(P[:, 2, :], axis=1)/math.sqrt(n)
    
    d = {}
    d['k'] = P[:, 0, 0]
    d['nmodes'] = P[:, 1, 0]
    d['P'] = P[:, 2, :]
    d['summary'] = summary

    return d

    
def load_halo_power(sim, isnp):
    """
    Read real-space halo power spectrum

    Returns: d (dict)
      d['summary']['P'] (array): P[ik] mean power spectrum
      d['summary']['dP'] (array): dP[ik] standard error in the mean
      d['P'] (array): 'P[ik, irealisation]' array of all realisations 
      d['k'] (array): wavenumber [h/Mpc]
      d['nmodes']: number of k modes in bins
    """

    data_dir = lambdalib.util.data_dir()
    filename = '%s/%s/halo_power/%s/halo_power_*.txt' % (data_dir, sim, isnp)

    return _load_power(filename, isnp)

def load_matter_power(sim, isnp):
    """
    Read real-space matter power spectrum

    Returns: d (dict)
      d['summary']['P'] (array): P[ik] mean power spectrum
      d['summary']['dP'] (array): dP[ik] standard error in the mean
      d['P'] (array): 'P[ik, irealisation]' array of all realisations 
      d['k'] (array): wavenumber [h/Mpc]
      d['nmodes']: number of k modes in bins
    """

    data_dir = lambdalib.util.data_dir()
    filename = '%s/%s/matter_power/%s/matterpower_*.txt' % (data_dir, sim, isnp)

    return _load_power(filename, isnp)
    

def load_theta_power(sim, isnp):
    """
    Args:
      sim (str): simulation name
      isnp (str): snapshot index

    Returns: d (dict)
      d['k'] (array): wave number [h/Mpc]
      d['nmodes'] (array): number of modes in bins
      d['summary']['Pdd']: Pdd density-density power spectrum
      d['summary']['Pdt']: Pdt density-theta cross-power spectrum
      d['summary']['Pdd']: Ptt theta-theta power spectrum
      d['P']: P[ik, icol, irealisation]
              icol=0: Pdd
              icol=1: Pdt
              icol=2: Ptt
    """

    data_dir = lambdalib.util.data_dir()

    path = '%s/%s/theta_power' % (data_dir, sim)
    lambdalib.util.check_isnp(path, isnp)
    
    path = '%s/%s/theta_power/%s/theta_power_*.txt' % (data_dir, sim, isnp)
    filenames = glob.glob(path)

    if not filenames:
        raise FileNotFoundError('Theta power spectra not found: %s' % path)

    filenames = sorted(filenames)

    n = len(filenames)
    P = None

    params = lambdalib.info(sim, isnp)
    aHf = params['a']*params['H']*params['f']

    for i, filename in enumerate(filenames):
        a = np.loadtxt(filename)
        
        if P is None:
            P = np.zeros((a.shape[0], a.shape[1], n))

        a[:, 3] /= aH
        a[:, 4] /= aH**2

        P[:, :, i] = a

    summary = {}
    summary['Pdd'] = np.mean(P[:, 2, :], axis=1)
    summary['Pdt'] = np.mean(P[:, 3, :], axis=1)
    summary['Ptt'] = np.mean(P[:, 4, :], axis=1)
    
    summary['dPdd'] = np.std(P[:, 2, :], axis=1)/math.sqrt(n)
    summary['dPdt'] = np.std(P[:, 3, :], axis=1)/math.sqrt(n)
    summary['dPtt'] = np.std(P[:, 4, :], axis=1)/math.sqrt(n)
    
    d = {}
    d['P'] = P[:, 1:, :]
    d['k'] = P[:, 0, 0]
    d['nmodes'] = P[:, 1, 0]
    d['summary'] = summary

    return d


def load_bias(sim, isnp):
    halo = load_halo_power(sim, isnp)
    matter = load_matter_power(sim, isnp)

    b = np.sqrt(halo['P']/matter['P'])
    n= b.shape[1]

    summary = {}
    summary['b'] = np.mean(b, axis=1)
    summary['db'] = np.std(b, axis=1)/math.sqrt(n)

    d = {}
    d['k'] = halo['k']
    d['b'] = b
    d['summary'] = summary

    return d
