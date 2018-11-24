import numpy as np
import lambdalib.util
import lambdalib
from scipy.special import legendre

def load_streaming_c(sim, isnp=None):
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
      d['sigma2_v']     sigma2_v = <u(x)^2>
      d['sigma2_DD(0)'] sigma_DD monopole C_DD(0)/P
      d['sigma2_DD(2)'] sigma_DD quadrupole
      d['sigma2_DU(0)'] sigma_DU monopole
      d['sigma2_DU(2)'] sigma_DU quadrupole
      d['sigma2_DU(4)'] sigma_DU hexadecapole
      d['mu2_sigma2_UU(0)'] mu^2 sigma_DU monopole
      d['mu2_sigma2_UU(2)'] mu^2 sigma_DU quadrupole
      d['mu2_sigma2_UU(4)'] mu^2 sigma_DU hexadecapole
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

def load_sigma_ab(sim, isnp):
    """
    Returns
      sigma_DD, sigma_DU, mu^2 sigma_UU
      Damping including B term is exp(-kz^2 sigma_ab)

    Arg:
      sim: simulation name
      isnp: snapshot index; D=1 f=1 if isnp = None
    
    Returns:
      d (dict)
      d['k'] (array): wavenumber k[ik]
      d['P']        : linear P[ik]
      d['sigma2_v']     sigma2_v = <u(x)^2>
      d['sigma2_DD(0)'] sigma_DD monopole
      d['sigma2_DD(2)'] sigma_DD quadrupole
      d['sigma2_DU(0)'] sigma_DU monopole
      d['sigma2_DU(2)'] sigma_DU quadrupole
      d['sigma2_DU(4)'] sigma_DU hexadecapole
      d['mu2_sigma2_UU(0)'] mu^2 sigma_DU monopole
      d['mu2_sigma2_UU(2)'] mu^2 sigma_DU quadrupole
      d['mu2_sigma2_UU(4)'] mu^2 sigma_DU hexadecapole

      d['sigma2_DD'] (func): sigma_DD(mu) function


      sigma2_ab = sigma2_v - b_ab - c_ab
    """
    lambdalib.util.check_sim(sim)
    
    data_dir = lambdalib.util.data_dir()

    # Load B-term
    taruya = lambdalib.load_taruya(sim, isnp)
    
    # Load C-term
    filename = '%s/%s/sigma_ab.txt' % (data_dir, sim)
    a = np.loadtxt(filename)

    # rescale to redshift of isnp
    if isnp is None:
        f = 1.0
    else:
        isnp = lambdalib.util.isnp_str(isnp)
        param = lambdalib.info(sim, isnp)
        f = param['f']
        a[:, 1] *= (param['D'])**2
        a[:, 2:] *= (f*param['D'])**2

    # make k same
    nk = min(a.shape[0], taruya['k'].shape[0])
    assert(np.all(np.abs(a[:nk, 0] - taruya['k'][:nk]) < 0.001))
    k = a[:nk, 0]
    P = a[:nk, 1]
    sigma2_v = a[:nk, 2]


    #
    # returning data
    #
    d = {}
    d['k'] = k
    d['P'] = a[:nk, 1]
    d['sigma2_v'] = a[:nk, 2]

    # compute bDD cDD
    b_dd0 = f**2*(taruya['B111'][:nk] + (1.0/3.0)*taruya['B211'][:nk])/(k**2*P)
    b_dd2 = f**2*(2.0/3.0)*taruya['B211'][:nk]/(k**2*P)
    c_dd0 = a[:nk, 3]
    c_dd2 = a[:nk, 4]

    d['bDD(0)'] = b_dd0
    d['bDD(2)'] = b_dd2
    d['cDD(0)'] = c_dd0
    d['cDD(2)'] = c_dd2

    #
    # compute -kz^2 sigma_ab = (B_ab + C_ab)/P_ab
    #
    # BDD = f^2 mu^2 (B111 + mu2 B211)
    # mu2 = 1/3 + 2/3 P2

    sigma_dd0 = sigma2_v - b_dd0 - c_dd0
    sigma_dd2 = -b_dd2 - c_dd2
    d['sigma2_DD(0)'] = sigma_dd0
    d['sigma2_DD(2)'] = sigma_dd2
    
    # Legendre polynomials
    P2 = legendre(2)
    #P4 = legendre(4)
    #P6 = legendre(6)




    
    #d['sigma2_DU(0)'] = a[:, 5]
    #d['sigma2_DU(2)'] = a[:, 6]
    #d['sigma2_DU(4)'] = a[:, 7]
    #d['mu2_sigma2_UU(0)'] = a[:, 8]
    #d['mu2_sigma2_UU(2)'] = a[:, 9]
    #d['mu2_sigma2_UU(4)'] = a[:, 10]

    # functions of mu

    d['sigma2_DD'] = lambda mu: sigma_dd0 + sigma_dd2*P2(mu)
    
    return d
