#
# TNS model
#

import numpy as np
import json

import lambdalib.util


class TaruyaModel:
    def __init__(self, sim):
        lambdalib.util.check_sim(sim)
        data_dir = lambdalib.util.data_dir()

        # Load Taruya AB data
        filename = '%s/%s/taruyaAB.txt' % (data_dir, sim)

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
        assert(0.0 <= mu <= 1.0)
        fac = self.param[isnp]['f']*self.param[isnp]['D']**4
        return fac*mu**2*self.A11

    def ADU(self, isnp, mu):
        assert(0.0 <= mu <= 1.0)
        fac = 0.5*self.param[isnp]['f']**2*self.param[isnp]['D']**4
        return fac*mu**2*(self.A12 + mu**2*self.A22)

    def AUU(self, isnp, mu):
        assert(0.0 <= mu <= 1.0)
        fac = self.param[isnp]['f']**3*self.param[isnp]['D']**4
        return fac*mu**4*(self.A23 + mu**2*self.A33)

    def BDD(self, isnp, mu):
        assert(0.0 <= mu <= 1.0)
        fac = self.param[isnp]['f']**2*self.param[isnp]['D']**4
        return fac*mu**2*(self.B111 + mu**2*self.B211)

    def BDU(self, isnp, mu):
        assert(0.0 <= mu <= 1.0)
        fac = -0.5*self.param[isnp]['f']**3*self.param[isnp]['D']**4
        return fac*(mu**2*self.B112 + mu**4*self.B212 + mu**6*self.B312)
        
    def BUU(self, isnp, mu):
        assert(0.0 <= mu <= 1.0)
        fac = self.param[isnp]['f']**4*self.param[isnp]['D']**4
        return fac*(mu**2*self.B122 + mu**4*self.B222 +
                    mu**6*self.B322 + mu**8*self.B422)


def load_taruya(sim, isnp):
    """
    Args:
      sim (str): simulation name
      isnp (str): simulation index

    Returns:
      d (dict):
      d['k'] (array): k[ik] wavenumber [h/Mpc]
      d['A11'] - d['B422'] (array): A11[ik] Taruya AB terms
        A11, A12, A22, A22, A33
        B111, B211, B112, B212, B312, B112, B222, B422
        B112 is for B112 + B121. Same for B212 and B312

      d['ADD'] - d['BUU'] (function): ADD(mu) returns array ADD(mu)[ik]
        ADD, ADU, AUU, BDD, BDU, BUU
    """
    lambdalib.util.check_sim(sim)
    data_dir = lambdalib.util.data_dir()

    # Load Taruya AB data
    filename = '%s/%s/taruyaAB.txt' % (data_dir, sim)
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
