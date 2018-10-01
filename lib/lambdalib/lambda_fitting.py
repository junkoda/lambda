import numpy as np
from scipy.optimize import curve_fit

def fit_DD(d, ik, imu, f, fit=None):
    """
    Fit P_DD(k, mu, lambda) with an given function 
    f(lambda, y0, ...) = PDD_0 + f(lambda)

    Args:
      d:   lambda data returned by load_lambda()
      ik:  k index
      imu: mu index
      f:   fitting function f(lambda, *params)
      fit: dictornary of fitting results
    """

    x = d['lambda']
    y = d['PDD'][ik, imu, :]/d['PDD0'][ik, imu]

    def ff(x, *params):
        return PDD0*f(x, *params)

    # remove nans
    idx = np.isfinite(y)
    x = x[idx]
    y = y[idx]

    # fitting
    popt, pcov = curve_fit(f, x, y)

    if fit is None:
        fit = {}

    fit['PDD_params'] = popt
    fit['PDD'] = d['PDD0'][ik, imu]*f(x, *popt)

    return fit


def fit_DU(d, ik, imu, f, fit=None):
    """
    Fit P_DU(k, mu, lambda) with an given function 
    f(lambda, ...) = A*lambda*f(lambda, ...)

    Args:
      d:   lambda data returned by load_lambda()
      ik:  k index
      imu: mu index
      f:   fitting function f(lambda, *params)
      fit: dictornary of fitting results
    """

    def ff(x, A, *params):
        return A*x*f(x, *params)

    x = d['lambda']
    y = d['PDU'][ik, imu, :]

    # remove nans
    idx = np.isfinite(y)
    x = x[idx]
    y = y[idx]

    # initial guess
    p0 = [0,]*(f.__code__.co_argcount)
    p0[0] = y[10]/x[10]

    # fitting
    popt, pcov = curve_fit(ff, x, y, p0=p0)

    if fit is None:
        fit = {}

    fit['PDU_amp'] = popt[0]
    fit['PDU_params'] = popt[1:]
    fit['PDU'] = ff(x, *popt)

    return fit


def fit_UU(d, ik, imu, f, fit=None):
    """
    Fit P_UU(k, mu, lambda) with an given function 
    f(lambda, ...) = A*lambda**2*f(lambda, ...)

    Args:
      d:   lambda data returned by load_lambda()
      ik:  k index
      imu: mu index
      f:   fitting function f(lambda, *params)
    """

    def ff(x, A, *params):
        return A*x**2*f(x, *params)

    x = d['lambda']
    y = d['PUU'][ik, imu, :]

    # remove nans
    idx = np.isfinite(y)
    x = x[idx]
    y = y[idx]

    # initial guess
    p0 = [0,]*(f.__code__.co_argcount)
    p0[0] = y[10]/x[10]**2

    # fitting
    popt, pcov = curve_fit(ff, x, y, p0=p0)

    if fit is None:
        fit = {}

    fit['PUU_amp'] = popt[0]
    fit['PUU_params'] = popt[1:]

    fit['PUU'] = ff(x, *popt)

    return fit

def fit_lambda(d, ik, imu, f, *, kind=('DD', 'DU', 'UU')):
    """
    Fit lambda plot with a fitting function f:

    P_DD(k, mu, lambda) = P_DD(k, mu, lambda=0)*f(lambda)
    P_DU(k, mu, lambda) = P_DU_amp*lambda*f(lambda)
    P_UU(k, mu, lambda) = P_UU_amp*lamba**2*f(lambda)

    Args:
      data (dict): lambda data loaded by load_lambda
      ik (int):    index of k
      imu (int):   index of mu
      f (func):    fitting function f(lambda, fitting parameters ...)
      kind (list): fitting P_**, subset of ('DD', 'DU', 'UU')

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
    """

    fit = {}

    if 'DD' in kind:
        fit_DD(d, ik, imu, f, fit)

    if 'DU' in kind:
        fit_DU(d, ik, imu, f, fit)

    if 'UU' in kind:
        fit_UU(d, ik, imu, f, fit)        

    return fit
