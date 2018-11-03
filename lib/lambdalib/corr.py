import numpy as np
import math
import glob
import lambdalib.util

def load_corr_dduu(sim, isnp):
    """
    Computes:
      xi_dd0: <d(x) d(y)> monopole
      xi_uu0: <u(x) u(y)> monopole
      xi_uu2: <u(x) u(y)> quadrupole
      xi_ww0: <[u(x) - u(y)]^2> monopole; w = u(x) - u(y)
      xi_ww2: <[u(x) - u(y)]^2> quadrupole
      xi_ddww0: <d(x) d(y) [u(x) - u(y)]^2> monopole
      xi_ddww2: <d(x) d(y) [u(x) - u(y)]^2> quadrupole
      xi_dduu0: <d(x) d(y) u(x) u(y)> monopole
      xi_dduu2: <d(x) d(y) u(x) u(y)> quadrupole
      xi_du1:    <d(x) u(y)> dipole
      xi_du3:    <d(x) u(y)> tripole

    Returns:
      d (dict):
        d['r'] (array): r [1/h Mpc] bin centre
        d['nrealisations'] (int): number of realisations
        d['xi_***'] (array): xi_***[ir, irealisation]
        d['summary']['xi_***'][ir]: mean of n realisations
        d['summary']['dxi_***'][ir]: standard error in mean
    """
    lambdalib.util.check_sim(sim)
    data_dir = lambdalib.util.data_dir()

    isnp=lambdalib.util.isnp_str(isnp)

    filenames = glob.glob('%s/%s/correlation_functions/dduu/%s/corr_dduu_*.txt' % (data_dir, sim, isnp))

    if not filenames:
        raise FileNotFoundError('No corr_dduu files in %s/%s/correlation_functions/dduu/%s/' % (data_dir, sim, isnp))

    xi = None
    n = len(filenames)
    for i, filename in enumerate(filenames):
        a = np.loadtxt(filename)

        if xi is None:
            xi = np.empty((a.shape[0], a.shape[1], n))

        xi[:, :, i] = a

    dat = [('dd', 1), ('ww0', 2), ('ww2', 3), ('uu0', 4), ('uu2', 5),
           ('ddww0', 7), ('ddww2', 8), ('dduu0', 9), ('dduu2', 10),
           ('du1', 11), ('du3', 12)]
    
    d = {}
    summary = {}
    
    d['r'] = xi[:, 0, 0]
    d['nrealisations'] = n
    d['_xi'] = xi

    for name, icol in dat:
        d['xi_' + name] = xi[:, icol, :]
        
        # summarise n realisations to mean and std
        summary['xi_' + name] = np.mean(xi[:, icol, :], axis=1)
        summary['dxi_' + name] = np.std(xi[:, icol, :], axis=1)/math.sqrt(n)

    d['summary'] = summary
    
    return d




    
