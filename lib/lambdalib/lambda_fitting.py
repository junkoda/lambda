import numpy as np
from scipy.optimize import curve_fit
import sys

def fit_DD(d, ik, imu, f, fit=None, p0=None, ilambda_max=None):
    """
    Fit P_DD(k, mu, lambda) with an given function 
    f(lambda, y0, ...) = PDD_0 + f(lambda)

    Args:
      d  (dict):  lambda data returned by load_lambda()
      ik  (int):  k index
      imu (int): mu index
      f:   fitting function f(lambda, *params)
      fit: dictornary for results

    Returns:
      fit (dict)
        fit['lambda'] (array): lambda[ilambda]
        fit['PDD_params'] (array): best fitting *params
        fit['PDD'] (array): best fitting PDD[ilamba]
    """

    x = d['lambda'][:ilambda_max]
    y = d['summary']['PDD'][ik, imu, :ilambda_max]/d['summary']['PDD0'][ik, imu]

    def ff(x, *params):
        return PDD0*f(x, *params)

    # remove nans
    idx = np.isfinite(y)
    x = x[idx]
    y = y[idx]

    # fitting
    try:
        popt, pcov = curve_fit(f, x, y)
    except RuntimeError:
        return None

    if fit is None:
        fit = {}

    fit['PDD_amp'] = d['summary']['PDD0'][ik, imu]
    fit['PDD_params'] = popt
    fit['lambda'] = x
    fit['PDD'] = d['summary']['PDD0'][ik, imu]*f(x, *popt)

    return fit


def fit_DU(d, ik, imu, f, fit=None, p0=None, ilambda_max=None):
    """
    Fit P_DU(k, mu, lambda) with an given function 
    f(lambda, ...) = A*lambda*f(lambda, ...)

    Args:
      d    (dict):  lambda data returned by load_lambda()
      ik    (int): k index
      imu:  (int): mu index
      f:   (func): fitting function f(lambda, *params)
      fit  (dict): dictornary for results
    """

    def ff(x, A, *params):
        return A*x*f(x, *params)

    x = d['lambda'][:ilambda_max]
    y = d['summary']['PDU'][ik, imu, :ilambda_max]

    # remove nans
    idx = np.isfinite(y)
    x = x[idx]
    y = y[idx]

    # initial guess
    if p0 is None:
        p0 = [0,]*(f.__code__.co_argcount)
    else:
        p0 = [0,] + p0

    p0[0] = y[10]/x[10]

    # fitting
    try:
        popt, pcov = curve_fit(ff, x, y, p0=p0)
    except RuntimeError:
        sys.stderr.write('Warning: unable to fit DU with %s; ik=%d imu=%d\n' %
                         (f.__name__, ik, imu))
        return None

    if fit is None:
        fit = {}

    fit['PDU_amp'] = popt[0]
    fit['PDU_params'] = popt[1:]
    fit['lambda'] = x
    fit['PDU'] = ff(x, *popt)

    return fit


def fit_UU(d, ik, imu, f, fit=None, p0=None, ilambda_max=None):
    """
    Fit P_UU(k, mu, lambda) with an given function 
    f(lambda, ...) = A*lambda**2*f(lambda, ...)

    Args:
      d   (dict): lambda data returned by load_lambda()
      ik   (int): k index
      imu  (int): mu index
      f   (func): fitting function f(lambda, *params)
      fit (dict): dictionary for the result
    """

    def ff(x, A, *params):
        return A*x**2*f(x, *params)

    x = d['lambda'][:ilambda_max]
    y = d['summary']['PUU'][ik, imu, :ilambda_max]

    # remove nans
    idx = np.isfinite(y)
    x = x[idx]
    y = y[idx]

    # initial guess
    if p0 is None:
        p0 = [0,]*(f.__code__.co_argcount)
    else:
        p0 = [0.0,] + p0

    p0[0] = y[10]/x[10]**2
    assert(len(p0) == f.__code__.co_argcount)

    # fitting
    try:
        popt, pcov = curve_fit(ff, x, y, p0=p0)
    except RuntimeError:
        sys.stderr.write('Warning: unable to fit UU with %s; ik=%d imu=%d\n' %
                         (f.__name__, ik, imu))
        return None
    

    if fit is None:
        fit = {}

    fit['PUU_amp'] = popt[0]
    fit['PUU_params'] = popt[1:]

    fit['lambda'] = x
    fit['PUU'] = ff(x, *popt)

    return fit

def _nans(shape):
    a = np.empty(shape)
    a[:] = np.nan
    return a

def fit_lambda(d, ik, imu, f, *,
               kind=('DD', 'DU', 'UU'),
               p0=None, ilambda_max=None):
    """
    Fit lambda plot with a fitting function f for a pair of k, mu

    P_DD(k, mu, lambda) = P_DD(k, mu, lambda=0)*f(lambda)
    P_DU(k, mu, lambda) = P_DU_amp*lambda*f(lambda)
    P_UU(k, mu, lambda) = P_UU_amp*lamba**2*f(lambda)

    Args:
      data (dict): lambda data loaded by load_lambda
      ik   (array-like): index of k
      imu  (array-like): index of mu
      f    (func): fitting function f(lambda, fitting parameters ...)
      kind (list): fitting P_**, subset of ('DD', 'DU', 'UU')
      p0   (list): initial parameter guess
      
    ik, imu can be:
      integer, 1D array, or 2D array.

    Result:
      fit (dict)
        fit['PDD'] (np.array): fitted P_DD
        fit['PDU'] (np.array): fitted P_DU
        fit['PUU'] (np.array): fitted P_DU
       
        fit['PDD_params']: best fitting parameters in f
        fit['PDU_params']: best fitting parameters in f
        fit['PUU_params']: best fitting parameters in f

        fit['PDU_amp']:    amplitude A in PDU = A*lambda*f(lambda)
        fit['PUU_amp']:    amplitude A in PDU = A*lambda**2*f(lambda)

      None if fitting failed
    """

    # single pair of (ik, imu)
    if isinstance(ik, int) and isinstance(imu, int):
        fit = {}

        if np.isnan(d['summary']['PDD'][ik, imu, 0]):
            return None

        if 'DD' in kind:
            fit_DD(d, ik, imu, f, fit, p0=p0, ilambda_max=ilambda_max)

        if 'DU' in kind:
            fit_DU(d, ik, imu, f, fit, p0=p0, ilambda_max=ilambda_max)

        if 'UU' in kind:
            fit_UU(d, ik, imu, f, fit, p0=p0, ilambda_max=ilambda_max)
        
        return fit

    # Convert ik, imu to np.array if they are array-like
    if type(ik) != np.ndarray:
        ik = np.array(ik, ndmin=1)
        
    if len(ik.shape) == 1:
        if type(imu) != np.ndarray:
            imu = np.array(imu, ndmin=1)
            
        if len(imu.shape) != 1:
            raise TypeError('If ik is an 1D array, '
                            'imu must also be an 1D array: '
                            'imu.shape {}'.format(imu.shape))

        nk = len(ik)
        nmu = len(imu)

        # Convert ik and imu to 2D arrays by repeating same row/column
        ik = ik.reshape((nk, 1)).repeat(nmu, axis=1)
        imu = imu.reshape((1, nmu)).repeat(nk, axis=0)

    # 2D arrays of ik imu
    if ik.shape != imu.shape:
        raise TypeError('2D arrays ik imu must have the same shape: '
                        '{} != {}'.format(ik.shape, imu.shape))

    nk = ik.shape[0]
    nmu = ik.shape[1]
    nparam = f.__code__.co_argcount
    # number of free paramters for f + linear RSD amplitude

    nlambda = len(d['lambda'][:ilambda_max])

    # Arrays for fitting results

    if 'DD' in kind:
        PDD_params = _nans((nk, nmu, nparam))
        PDD = _nans((nk, nmu, nlambda))

    if 'DU' in kind:
        PDU_params = _nans((nk, nmu, nparam))
        PDU = _nans((nk, nmu, nlambda))

    if 'UU' in kind:
        PUU_params = _nans((nk, nmu, nparam))
        PUU = _nans((nk, nmu, nlambda))

    for i in range(nk):
        for j in range(nmu):
            ik_ij = ik[i, j]
            imu_ij = imu[i, j]

            if 'DD' in kind:
                fit = fit_DD(d, ik_ij, imu_ij, f, p0=p0,
                             ilambda_max=ilambda_max)
                if fit:
                    PDD_params[i, j, 0]  = fit['PDD_amp']
                    PDD_params[i, j, 1:] = fit['PDD_params']
                    PDD[i, j, :] = fit['PDD']

            if 'DU' in kind:
                fit = fit_DU(d, ik_ij, imu_ij, f, p0=p0,
                             ilambda_max=ilambda_max)
                if fit:
                    PDU_params[i, j, 0]  = fit['PDU_amp']
                    PDU_params[i, j, 1:] = fit['PDU_params']
                    PDU[i, j, :] = fit['PDU']

            if 'UU' in kind:
                fit = fit_UU(d, ik_ij, imu_ij, f, p0=p0,
                             ilambda_max=ilambda_max)
                if fit:
                    PUU_params[i, j, 0]  = fit['PUU_amp']
                    PUU_params[i, j, 1:] = fit['PUU_params']
                    PUU[i, j, :] = fit['PUU']

    fit = {}
    fit['ik'] = ik
    fit['imu'] = imu
    fit['lambda'] = d['lambda'][:ilambda_max]

    if 'DD' in kind:
        fit['PDD'] = PDD
        fit['PDD_params'] = PDD_params
    if 'DU' in kind:
        fit['PDU'] = PDU
        fit['PDU_params'] = PDU_params
    if 'UU' in kind:
        fit['PUU'] = PUU
        fit['PUU_params'] = PUU_params

    return fit
