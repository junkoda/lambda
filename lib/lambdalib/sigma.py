import numpy as np
import lambdalib.util
import lambdalib
from scipy.special import legendre

def load_sigma_ab(sim, isnp=None):
    """
    Returns
      C-term contribution to sigma_DD and sigma_DU
      sigma_DD = C_DD/P
      sigma_DU = C_DU/(f mu^2 P)
      mu^2 sigma_UU = mu^2 (B_UU + C_UU)/(f^2 mu^4 P)

    Arg:
      sim: simulation name
      isnp: snapshot index; D=1 f=1 if isnp = None
    
    Returns:
      d (dict)
      d['k']: wavenumber
      d['P']: linear P(k)
      d['sigma_DD(0)'] sigma_DD monopole C_DD(0)/P
      d['sigma_DD(2)'] sigma_DD quadrupole
      d['sigma_DU(0)'] sigma_DU monopole
      d['sigma_DU(2)'] sigma_DU quadrupole
      d['sigma_DU(4)'] sigma_DU hexadecapole
      d['mu2_sigma_UU(0)'] mu^2 sigma_DU monopole
      d['mu2_sigma_UU(2)'] mu^2 sigma_DU quadrupole
      d['mu2_sigma_UU(4)'] mu^2 sigma_DU hexadecapole
    """
    lambdalib.util.check_sim(sim)
    
    data_dir = lambdalib.util.data_dir()

    filename = '%s/%s/sigma_ab.txt' % (data_dir, sim)
    a = np.loadtxt(filename)

    # rescale to redshift of isnp
    if isnp is not None:
        isnp = lambdalib.util.isnp_str(isnp)
        param = lambdalib.info(sim, isnp)
        a[:, 1] *= (param['D'])**2
        a[:, 2:] *= (param['f']*param['D'])**2

    # mu = kz/k
    #nmu = 10
    #mu = (np.arange(10) + 0.5)/nmu

    # Legendre polynomial
    #P0 = np.ones_like(mu)
    #P2 = legendre(2)(mu)
    #P4 = legendre(4)(mu)
    #P6 = legendre(6)(mu)

    #sigma2_DD = np.outer(a[:, 2], P0) + np.outer(a[:, 3], P2)
    #sigma2_DU = np.outer(a[:, 4], P0) + np.outer(a[:, 5], P2) + \
    #            np.outer(a[:, 6], P4)
    #sigma2_UU = np.outer(a[:, 7], P0) + np.outer(a[:, 8], P2) + \
    #            np.outer(a[:, 9], P4) + np.outer(a[:, 10], P6)

    d = {}
    d['k'] = a[:, 0]
    d['P'] = a[:, 1]
    d['sigma2_v'] = a[:, 2]
    d['sigma2_DD(0)'] = a[:, 3]
    d['sigma2_DD(2)'] = a[:, 4]
    d['sigma2_DU(0)'] = a[:, 5]
    d['sigma2_DU(2)'] = a[:, 6]
    d['sigma2_DU(4)'] = a[:, 7]
    d['mu2_sigma2_UU(0)'] = a[:, 8]
    d['mu2_sigma2_UU(2)'] = a[:, 9]
    d['mu2_sigma2_UU(4)'] = a[:, 10]

    #d['sigma2_DD'] = sigma2_DD
    #d['sigma2_DU'] = sigma2_DU
    #d['sigma2_UU'] = sigma2_UU
    #d['mu'] = mu


    return d
