#
# TNS model
#

import numpy as np
import json
from numbers import Number
import lambdalib.util


class TaruyaModel:
    def __init__(self, sim, *, dk=None):
        lambdalib.util.check_sim(sim)
        data_dir = lambdalib.util.data_dir()

        # Load Taruya AB data
        if dk is None:
            filename = '%s/%s/taruyaAB.txt' % (data_dir, sim)
        else:
            filename = '%s/%s/taruyaAB_%s.txt' % (data_dir, sim, str(dk))

        # Load param
        with open('%s/%s/param.json' % (data_dir, sim)) as f:
            params = json.load(f)

        self.param = params['snapshot']

        a = np.loadtxt(filename)
        
        self.k =   a[:, 0]
        self.A11 = a[:, 1]
        self.A12 = a[:, 2]
        self.A22 = a[:, 3]
        self.A23 = a[:, 4]
        self.A33 = a[:, 5]
        self.B111 = a[:, 6]
        self.B211 = a[:, 7]
        self.B112 = a[:, 8]  # B1_12 + B1_21
        self.B212 = a[:, 9]  # B2_12 + B2_21
        self.B312 = a[:, 10] # B3_12 + B3_21
        self.B122 = a[:, 11]
        self.B222 = a[:, 12]
        self.B322 = a[:, 13]
        self.B422 = a[:, 14]

        self.a = a

    def ADD(self, isnp, mu):
        assert(np.all(0.0 <= mu) and np.all(mu <= 1.0))
        fac = self.param[isnp]['f']*self.param[isnp]['D']**4

        if isinstance(mu, Number):
            return fac*mu**2*self.A11

        return fac*np.outer(self.A11, mu**2)

    def ADU(self, isnp, mu):
        assert(np.all(0.0 <= mu) and np.all(mu <= 1.0))
        fac = 0.5*self.param[isnp]['f']**2*self.param[isnp]['D']**4

        if isinstance(mu, Number):
            return fac*mu**2*(self.A12 + mu**2*self.A22)

        return fac*(np.outer(self.A12, mu**2) + np.outer(self.A22, mu**4))

    def AUU(self, isnp, mu):
        assert(np.all(0.0 <= mu) and np.all(mu <= 1.0))
        fac = self.param[isnp]['f']**3*self.param[isnp]['D']**4

        if isinstance(mu, Number):
            return fac*mu**4*(self.A23 + mu**2*self.A33)
        return fac*(np.outer(self.A23, mu**4) + np.outer(self.A33, mu**6))

    def BDD(self, isnp, mu):
        assert(np.all(0.0 <= mu) and np.all(mu <= 1.0))
        fac = self.param[isnp]['f']**2*self.param[isnp]['D']**4

        if isinstance(mu, Number):
            return fac*mu**2*(self.B111 + mu**2*self.B211)
        return fac*(np.outer(self.B111, mu**2) + np.outer(self.B211, mu**4))
        

    def BDU(self, isnp, mu):
        assert(np.all(0.0 <= mu) and np.all(mu <= 1.0))
        # (-1)^{a + b} sign
        fac = -0.5*self.param[isnp]['f']**3*self.param[isnp]['D']**4

        if isinstance(mu, Number):
            return fac*(mu**2*self.B112 + mu**4*self.B212 + mu**6*self.B312)

        return fac*(np.outer(self.B112, mu**2) + np.outer(self.B212, mu**4)
                    + np.outer(self.B312, mu**6))
        
    def BUU(self, isnp, mu):
        assert(np.all(0.0 <= mu) and np.all(mu <= 1.0))
        fac = self.param[isnp]['f']**4*self.param[isnp]['D']**4

        if isinstance(mu, Number):
            return fac*(mu**2*self.B122 + mu**4*self.B222 +
                        mu**6*self.B322 + mu**8*self.B422)

        return fac*(np.outer(self.B122, mu**2) + np.outer(self.B222, mu**4) +
                    np.outer(self.B322, mu**6) + np.outer(self.B422, mu**8))


def load_taruya(sim, isnp, *, dk=None):
    """
    Load precomputed Taruya AB terms

    Args:
      sim (str): simulation name
      isnp (str): simulation index

    Returns:
      d (dict):
      d['k'] (array): k[ik] wavenumber [h/Mpc]
      d['A11'] - d['B422'] (array): A11[ik] Taruya AB terms
        A11, A12, A22, A22, A33
        B111, B211, B112 + B121, B212 + B221, B312 + B321, B112, B222, B422

      d['ADD'] - d['BUU'] (function): ADD(mu) returns array ADD(mu)[ik]
        ADD, ADU, AUU, BDD, BDU, BUU
    
    Note:
      The order of Bnab is different from original Taruya's Fortran code output
      Also, our Bnab does not contain sign (-1)^{a + b}
    """
    lambdalib.util.check_sim(sim)
    data_dir = lambdalib.util.data_dir()

    # Load Taruya AB data
    if dk is None:
        filename = '%s/%s/taruyaAB.txt' % (data_dir, sim)
    else:
        filename = '%s/%s/taruyaAB_%s.txt' % (data_dir, sim, str(dk))

    a = np.loadtxt(filename)

    d = {}
    
    isnp = lambdalib.util.isnp_str(isnp)
    param = lambdalib.util.load_param(sim, isnp)
    f = param['f']
    D = param['D']
    a[:, 1:] *= param['D']**4

    #
    # Data
    #
    d['AB']   = a
    d['k']    = a[:, 0]
    d['A11']  = a[:, 1]
    d['A12']  = a[:, 2]
    d['A22']  = a[:, 3]
    d['A23']  = a[:, 4]
    d['A33']  = a[:, 5]
    d['B111'] = a[:, 6]
    d['B211'] = a[:, 7]
    d['B112'] = a[:, 8]  # B1_12 + B1_21
    d['B212'] = a[:, 9]  # B2_12 + B2_21
    d['B312'] = a[:, 10] # B3_12 + B3_21
    d['B122'] = a[:, 11]
    d['B222'] = a[:, 12]
    d['B322'] = a[:, 13]
    d['B422'] = a[:, 14]

    #
    # Functions
    #
    d['ADD'] = lambda mu: f*mu**2*d['A11']
    d['ADU'] = lambda mu: 0.5*f**2*mu**2*(d['A12'] + mu**2*d['A22'])
    d['AUU'] = lambda mu: f**3*mu**4*(d['A23'] + mu**2*d['A33'])
    d['BDD'] = lambda mu: f**2*mu**2*(d['B111'] + mu**2*d['B211'])
    d['BDU'] = lambda mu: -0.5*f**3*(mu**2*d['B112'] + mu**4*d['B212']
                                     + mu**6*d['B312'])
    d['BUU'] = lambda mu: f**4*(mu**2*d['B122'] + mu**4*d['B222'] +
                                mu**6*d['B322'] + mu**8*d['B422'])

    return d
