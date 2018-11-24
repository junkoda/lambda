import math
import numpy as np
import lambdalib
import h5py
import glob

def load_dtfe_A(sim, isnp):
    """
    Load Taruya A term measured by DTFE

    Args:
      sim (str):  simulation name 'wizcola-particles'
      isnp (str): snapshot index '010', '008', '006', '002'

    Returns:
      d (dict)
      d['k']: k[ik, imu] mean wavenumber in 2D bin
      d['mu']: mu[ik, imu] mean mu = kz/k i 2D bin
      d['summary'] (dict): mean and standard error of ADD, ADU, AUU
      d['ADD']: ADD[ik, imu, irealisation]
      d['ADU']: ADU[ik, imu, irealisation]
      d['AUU']: AUU[ik, imu, irealisation]
    """
    params = lambdalib.info(sim, isnp)

    # Convert velocity vz to RSD displacemennt u = vz/aH
    aH = params['a']*params['H']

    if sim == 'nbody':
        aH /= math.sqrt(params['a']) # sqrt(a) in Gadget snasphot
    


    data_dir = lambdalib.util.data_dir()

    filenames = glob.glob('%s/%s/dtfe_A/%s/taruya_bispectrum_*.h5' %
                          (data_dir, sim, isnp))

    if not filenames:
        raise FileNotFoundError('File not found in %s/%s/%s' %
                                (data_dir, sim, isnp))

    n = len(filenames)
    ADD = None
    ADU = None
    AUU = None
    
    for i, filename in enumerate(filenames):
        with h5py.File(filename, 'r') as f:
            shape = f['Add'].shape
            
            if ADD is None:
                k = f['k'][:]
                mu = f['mu'][:]
                ADD = np.empty((shape[0], shape[1], n))
                ADU = np.empty_like(ADD)
                AUU = np.empty_like(ADD)

            ADD[:, :, i] = f['Add'][:]/(aH)
            ADU[:, :, i] = f['Adu'][:]/aH**2
            AUU[:, :, i] = f['Auu'][:]/(aH**3)

    if sim == 'nbody':
        ADD *= 2.0
        AUU *= 2.0
            
    summary = {}
    summary['ADD'] = np.mean(ADD, axis=2)
    summary['ADU'] = np.mean(ADU, axis=2)
    summary['AUU'] = np.mean(AUU, axis=2)

    summary['dADD'] = np.std(ADD, axis=2)/np.sqrt(n)
    summary['dADU'] = np.std(ADU, axis=2)/np.sqrt(n)
    summary['dAUU'] = np.std(AUU, axis=2)/np.sqrt(n)

    d = {}
    d['summary'] = summary
    d['k'] = k
    d['mu'] = mu
    d['ADD'] = ADD
    d['ADU'] = ADU
    d['AUU'] = AUU

    return d

def load_dtfe_higher_order_2d(sim, isnp):
    """
    Load 2D 4th-order moments.

    Fourier transform of:
      du2d: <delta(x) u2(x) delta(y)>
      dudu: <delta(x) u(x) delta(y) u(y)>
    """
    params = lambdalib.info(sim, isnp)

    # Convert velocity vz to RSD displacemennt u = vz/aH
    aH = params['a']*params['H']

    if sim == 'nbody':
        aH /= math.sqrt(params['a']) # sqrt(a) in Gadget snasphot

    data_dir = lambdalib.util.data_dir()

    filenames = glob.glob('%s/%s/higher_order_dtfe/%s/higher_order_dtfe_*.h5' %
                          (data_dir, sim, isnp))

    if not filenames:
        raise FileNotFoundError('h5 files not found in %s/%s/%s' %
                                (data_dir, sim, isnp))

    n = len(filenames)

    P = None
    du2d = None
    dudu = None
    
    for i, filename in enumerate(filenames):
        with h5py.File(filename, 'r') as f:
            shape = f['dd'].shape
            
            if P is None:
                k = f['k'][:]
                mu = f['mu'][:]
                P = np.empty((shape[0], shape[1], n))
                du2d = np.empty_like(P)
                dudu = np.empty_like(P)

            P[:, :, i]   = f['dd'][:]
            du2d[:, :, i] = f['du2d'][:]/aH**2
            dudu[:, :, i] = f['dudu'][:]/aH**2

    summary = {}
    summary['P'] = np.mean(P, axis=2)
    summary['Pdu2d'] = np.mean(du2d, axis=2)
    summary['Pdudu'] = np.mean(dudu, axis=2)

    summary['dP'] = np.std(P, axis=2)/np.sqrt(n)
    summary['dPdu2d'] = np.std(du2d, axis=2)/np.sqrt(n)
    summary['dPdudud'] = np.std(dudu, axis=2)/np.sqrt(n)

    d = {}
    d['summary'] = summary
    d['k'] = k
    d['mu'] = mu
    d['P'] = P
    d['Pdu2d'] = du2d
    d['Pdudu'] = dudu

    return d


def load_dtfe_higher_order_multipoles(sim, isnp):
    """
    Load 4th-order moment multipoles.

    Fourier transform of:
      Pdu2d: <delta(x) u2(x) delta(y)>
      Pdudu: <delta(x) u(x) delta(y) u(y)>

    Returns: d (dict)
      d['k']:                 : mean k[ik] in bin
      d['nmodes']             : nmodes[ik] number of independent modes in bin

      d['summary']['Pdd']     : Pdd[ik]   delta delta power
      d['summary']['Pdu2d(0)']: Pdu2d[ik] monopole, mean of realisations
      d['summary']['Pdu2d(2)']: Pdu2d[ik] quadrupole
      d['summary']['Pdudu(0)']: Pdudu[ik] monopole
      d['summary']['Pdudu(2)']: Pdudu[ik] quadrupole
      d['summary']['dPdd']: standard error in the mean Pdd
      same for dPdu2d(0), dPdu2d(2), ...

      d['Pdd']: Pdd[ik, irealisation] for each realisation
      same for Pdu2d(0), Pdu2d(2), ...
    """
    params = lambdalib.info(sim, isnp)

    # Convert velocity vz to RSD displacemennt u = vz/aH
    aH = params['a']*params['H']

    if sim == 'nbody':
        aH /= math.sqrt(params['a']) # sqrt(a) in Gadget snasphot

    data_dir = lambdalib.util.data_dir()

    filenames = glob.glob('%s/%s/higher_order_dtfe/%s/higher_order_dtfe_*.txt' %
                          (data_dir, sim, isnp))

    if not filenames:
        raise FileNotFoundError('txt files not found in %s/%s/%s' %
                                (data_dir, sim, isnp))

    n = len(filenames)

    P = None
    
    for i, filename in enumerate(filenames):
        a = np.loadtxt(filename)
            
        if P is None:
            P = np.empty((a.shape[0], a.shape[1], n))

        P[:, :, i]   = a

    d = {}
    summary = {}

    for name, icol in [('dd', 2), ('du2d(0)', 3), ('du2d(2)', 4),
                       ('dudu(0)', 5), ('dudu(2)', 6)]:
        d['P' + name] = P[:, icol, :]
        summary['P' + name] = np.mean(P[:, icol], axis=1)
        summary['dP' + name] = np.std(P[:, icol], axis=1)

    d['summary'] = summary
    d['k'] = P[:, 0, 0]
    d['nmodes'] = P[:, 1, 0]

    return d



