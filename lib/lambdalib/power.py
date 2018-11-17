#
# Power spectra
#
import math
import numpy as np
import glob
import lambdalib.util

def load_linear_power(sim, isnp=None):
    """
    Linearly extrapolated power spectrum at redshift of isnp
    Returns z=0 power spectrum if isnp = None

    Returns:
      dictionary with
      'k': wavenumbeer [h/Mpc]
      'P': linear power spectrum [1/h Mpc]^3
      'P_nowiggle': no wiggle power spectrum   
      'interp'['linear']: interpolation function P(k)
      'interp'['nowiggle']: interpolation function P_nowiggle(k)
    """

    data_dir = lambdalib.util.data_dir()
    filename = '%s/%s/linear_matterpower.dat' % (data_dir, sim)
    a = np.loadtxt(filename)

    d = {}
    d['a'] = a
    d['k'] = a[:, 0]
    d['P'] = a[:, 1]

    filename = '%s/%s/power_spectrum/nowiggle_matterpower.dat' % (data_dir, sim)
    nowiggle = np.loadtxt(filename)

    assert(np.all(nowiggle[:, 0] == a[:, 0]))
    d['P_nowiggle'] = nowiggle[:, 1]

    if isnp is not None:
        param = lambdalib.util.load_param(sim, isnp)
        d['P'] *= param['D']**2

    try:
        from scipy.interpolate import interp1d
        d['interp'] = {}
        d['interp']['linear'] = interp1d(d['k'], d['P'], kind='cubic')
        d['interp']['nowiggle'] = interp1d(nowiggle[:, 0], nowiggle[:, 1],
                                           kind='cubic')
    except ImportError:
        print('Warning: scipy.interpolate unabailable '
              'for linear power interpolation.')

    return d


def _load_power(file_pattern, isnp):
    """
    Load power spectrum ascii file of format
    k nmodes P0 [P2 P4]
    """
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

    d = {}
    summary = {}

    if P.shape[1] == 3: # monopole only
        summary['P'] = np.mean(P[:, 2, :], axis=1)
        summary['dP'] = np.std(P[:, 2, :], axis=1)/math.sqrt(n)
        d['P'] = P[:, 2, :]
    elif P.shape[1] == 5: # P0, P2, P4
        for l, i in [('0', 2), ('2', 3), ('4', 4)]:
            summary['P' + l] = np.mean(P[:, i, :], axis=1)
            summary['dP' + l] = np.std(P[:, i, :], axis=1)/math.sqrt(n)

        d['P0'] = P[:, 2, :]
        d['P2'] = P[:, 3, :]
        d['P4'] = P[:, 4, :]
    else:
        raise OSError('Unknown number of columns for power spectrum: %d'
                      % P.shape[1])

    summary['nmodes'] = np.sum(P[:, 1, :], axis=1)
    d['k'] = P[:, 0, 0]
    d['nmodes'] = P[:, 1, 0]
    d['summary'] = summary

    return d

    
def load_halo_power(sim, isnp, kind='real_space'):
    """
    Read real-space halo power spectrum

    Args:
      kind = 'real_space', 'zspace', or 'zspace_conventional_legendre'
      'zspace' is discrete Legendre Multipoles

    Returns: d (dict)
      d['summary']['P'] (array): P[ik] mean power spectrum
      d['summary']['dP'] (array): dP[ik] standard error in the mean
      d['summary']['nmodes'] (array): nmodes[ik]: number of independent
                                      modes; sum of all realisations
      d['P'] (array): 'P[ik, irealisation]' array of all realisations 
      d['k'] (array): wavenumber [h/Mpc]
      d['nmodes']: number of independent k modes in bins

      For zspace,
        P0, P2, P4, and, dP0, dP2, dP4 instead of P and dP
    """

    data_dir = lambdalib.util.data_dir()

    
    if kind == None:
        filename = '%s/%s/halo_power/%s/halo_power_*.txt' % (data_dir, sim, isnp)
    elif (kind == 'real_space' or kind == 'zspace'
          or kind == 'zspace_conventional_legendre'):
        filename = '%s/%s/halo_power/%s/%s/halo_power_*.txt' % (data_dir, sim, kind, isnp)
    else:
        raise ValueError('Unknown kind of halo power: %s' % kind)


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

    # Convert \nabla v to theta = \nabla v / (aHf)
    aHf = params['a']*params['H']*params['f']

    # Correct sqrt(a) factor in Gadget snapshot
    if sim == 'nbody':
        aHf /= math.sqrt(params['a'])

    for i, filename in enumerate(filenames):
        a = np.loadtxt(filename)
        
        if P is None:
            P = np.zeros((a.shape[0], a.shape[1], n))

        a[:, 3] /= aHf
        a[:, 4] /= aHf**2

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

def load_theta_power_bell_model(sim, isnp, *, kind='default',
                                Pdd=None, linear=None):
                                
    """
    Density- Velocity-divergence cross power using Bell et al. formula

    Args:
      sim (str): simulation name
      isnp (str): snapshot index
      Pdd (dict): [optional] Non-linear Pdd(k, z) dictionary with 'k' and 'P'
                  linear is used if not provided
      linear (dict): [optional] linear P(k, z) dictionary with 'k' and 'P'
                  linear is loaded if not provided
      kind (str): 'default'
                  'simple': simpler model of Ptt with exp only

    Returns:
      d['k']: wavenumber kdd if provided, linear['k'] otherwise
      d['Pdd']: Pdd if provided, linear['P'] otherwise
      d['Pdt']: P_delta_theta
      d['Ptt']: P_theta_theta

    Reference:
      Bell et al. https://arxiv.org/abs/1809.09338
    """

    lambdalib.util.check_sim(sim)

    if not (kind == 'default' or kind == 'simple'):
        raise ValueError('Unknown kind of Bell model.'
                         "Must be 'default' or 'simple': %s" % kind)
    
    param = lambdalib.util.load_param(sim)
    D = param['snapshot'][isnp]['D']
    sigma8 = D*param['sigma_8']

    a1 = -0.817 + 3.198*sigma8
    a2 = 0.877 - 4.191*sigma8
    a3 = -1.199 + 4.629*sigma8
    kd_inv = -0.111 + 3.811*sigma8**2
    b = 0.091 + 0.702*sigma8
    kt_inv = -0.048 + 1.917*sigma8**2

    if linear is None:
        linear = load_linear_power(sim, isnp)

    if not ('k' in Pdd and 'P' in Pdd):
        raise ValueError("dict linear must containt 'k' and 'P'")


    if Pdd is None:
        Pdd = linear['P']
        k = linear['k']
        P = linear['P']
    else:
        from scipy.interpolate import interp1d

        if not ('k' in Pdd and 'P' in Pdd):
            raise ValueError("dict Pdd must containt 'k' and 'P'")

        k = Pdd['k']
        Pdd = Pdd['P']
        P = interp1d(linear['k'], linear['P'], kind='cubic')(k)

    
    d = {}
    d['k'] = k
    d['Pdd'] = Pdd
    d['Pdt'] = np.sqrt(Pdd*P)*np.exp(-k*kd_inv - b*k**6)
    d['kind'] = kind
    
    if kind == 'simple':
        d['Ptt'] = P*np.exp(-k*kt_inv)
    else:
        d['Ptt'] = P*np.exp(-k*(a1 + a2*k + a3*k**2))

    return d

def load_halofit_power(sim, isnp):
    """
    Args:
      sim (str): simulation name
      isnp (str, int): snasphot index
    """
    lambdalib.util.check_sim(sim)

    isnp = lambdalib.util.isnp_str(isnp)

    data_dir = lambdalib.util.data_dir()
    
    filename = '%s/%s/power_spectrum/%s/halofit_matterpower.dat' % (data_dir, sim, isnp)
    a = np.loadtxt(filename)
    
    d = {}
    d['k'] = a[:, 0]
    d['P'] = a[:, 1]

    return d

def compute_sigma_v(sim, isnp):
    """
    compute linar sigma_v = \int P(k) dk/(6 pi^2)
    """
    lambdalib.util.check_sim(sim)

    linear = load_linear_power(sim, None)
    
    if isnp is None:
        fac = 1.0
    else:
        param = lambdalib.util.load_param(sim, isnp)
        fac = param['f']*param['D']

    k = linear['k']
    P = linear['P']
    dk = k[1:] - k[:-1]

    # Trapezoidal integral
    sigma2v = 0.5*np.sum((P[1:] + P[:-1])*(k[1:] - k[:-1]))/(6.0*math.pi**2)

    return fac*math.sqrt(sigma2v)

def load_halo_nbar(sim, isnp):
    """
    load number density of haloes
    """

    data_dir = lambdalib.util.data_dir()
    filename = '%s/%s/halo_power/%s/halo_power_*.txt' % (data_dir, sim, isnp)

    filenames = glob.glob(filename)

    if not filenames:
        raise FileNotFoundError('halo power spectrum not found: %s' %
                                file_pattern)

    filenames = sorted(filenames)

    n = len(filenames)
    nbar = np.empty(n)

    for i, filename in enumerate(filenames):
        for line in open(filename, 'r'):
            if line[0] != '#':
                raise OSError('nbar not found in %s' % filename)

            v = line.rstrip().split()
            if v[1] == 'nbar':
                nbar[i] = float(v[2])
                break

    summary = {}
    summary['nbar'] = np.mean(nbar)
    summary['dnbar'] = np.std(nbar)

    d = {}
    d['summary'] = summary
    d['nbar'] = nbar

    return d
        
