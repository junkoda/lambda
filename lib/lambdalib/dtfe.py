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
