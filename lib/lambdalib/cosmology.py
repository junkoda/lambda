import math
import lambdalib.util

def a(sim, isnp):
    """
    Returns scale factor a

    Args:
      sim (str or None):  simulation name
      isnp (str or float): snapshot index or redshift
    """

    if sim is not None and isinstance(isnp, str):
        redshift = lambdalib.util.load_param(sim, isnp)['redshift']
        return 1.0/(1.0 + redshift)

    if sim is None:
        if not isinstance(isnp, float):
            raise TypeError('isnp must be redshift (float) when sim is None')

    redshift = isnp

    return 1.0/(1.0 + redshift)
    

def H(sim, redshift, *, omega_m=None):
    """
    Returns Hubble paramter H(redshift)

    Args:
      sim (str or None): simulation name
      redshift (str or float): isnp index or redshift
    
      omega_m (float): Omega matter at present when sim = None
    """

    if sim is None:
        if omega_m is None:
            raise ValueError('omega_m is necessary if sim is None')
        if not isinstance(redshift, float):
            raise ValueError('redshift must be float if sim is None')
    else:
        omega_m = lambdalib.load_param(sim)['omega_m']
    
        if isinstance(redshift, str):
            isnp = redshift
            param = lambdalib.load_param(sim, isnp)
            redshift = param['redshift']

    a = 1.0/(1.0 + redshift)
    omega_l = 1.0 - omega_m
    
    return 100.0*math.sqrt(omega_m*a**-3 + omega_l)


def f(sim, isnp):
    """
    Returns linear growth rate f

    Args:
      sim (str):  simulation name
      isnp (str): snapshot index
    """

    return lambdalib.load_param(sim, isnp)['f']

def D(sim, isnp):
    """
    Returns linear growth factor D

    Args:
      sim (str):  simulation name
      isnp (str): snapshot index
    """

    return lambdalib.load_param(sim, isnp)['D']
