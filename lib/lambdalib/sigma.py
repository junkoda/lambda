import numpy as np
import lambdalib.util
import lambdalib
from scipy.special import legendre

def load_sigma_ab(sim, isnp):
    lambdalib.util.check_sim(sim)
    
    data_dir = lambdalib.util.data_dir()

    filename = '%s/%s/sigma_ab.txt' % (data_dir, sim)
    a = np.loadtxt(filename)

    param = lambdalib.info(sim, isnp)

    # rescale to redshift of isnp
    a[:, 1:] *= (param['f']*param['D'])**2

    # mu = kz/k
    nmu = 10
    mu = (np.arange(10) + 0.5)/nmu

    # Legendre polynomial
    P0 = np.ones_like(mu)
    P2 = legendre(2)(mu)
    P4 = legendre(4)(mu)
    P6 = legendre(6)(mu)

    sigma2_DD = np.outer(a[:, 2], P0) + np.outer(a[:, 3], P2)
    sigma2_DU = np.outer(a[:, 4], P0) + np.outer(a[:, 5], P2) + \
                np.outer(a[:, 6], P4)
    sigma2_UU = np.outer(a[:, 7], P0) + np.outer(a[:, 8], P2) + \
                np.outer(a[:, 9], P4) + np.outer(a[:, 10], P6)

    d = {}
    d['k'] = a[:, 0]
    d['mu'] = mu
    d['sigma2_v'] = a[:, 1]
    d['sigma2_DD'] = sigma2_DD
    d['sigma2_DU'] = sigma2_DU
    d['sigma2_UU'] = sigma2_UU

    d['sigma2_DD(0)'] = a[:, 2]
    d['sigma2_DD(2)'] = a[:, 3]
    d['sigma2_DU(0)'] = a[:, 4]
    d['sigma2_DU(2)'] = a[:, 5]
    d['sigma2_DU(4)'] = a[:, 6]
    d['sigma2_UU(0)'] = a[:, 7]
    d['sigma2_UU(2)'] = a[:, 8]
    d['sigma2_UU(4)'] = a[:, 9]
    d['sigma2_UU(6)'] = a[:, 10]    

    return d
