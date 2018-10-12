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
        fac = self.param[isnp]['f']*self.param[isnp]['D']**4
        return fac*mu**2*self.A11

    def ADU(self, isnp, mu):
        fac = 0.5*self.param[isnp]['f']**2*self.param[isnp]['D']**4
        return fac*mu**2*(self.A12 + mu**2*self.A22)

    def AUU(self, isnp, mu):
        fac = self.param[isnp]['f']**3*self.param[isnp]['D']**4
        return fac*mu**4*(self.A23 + mu**2*self.A33)

    def BDD(self, isnp, mu):
        fac = self.param[isnp]['f']**2*self.param[isnp]['D']**4
        return fac*mu**2*(self.B111 + mu**2*self.B211)

    def BDU(self, isnp, mu):
        fac = -0.5*self.param[isnp]['f']**3*self.param[isnp]['D']**4
        return fac*(mu**2*self.B112 + mu**4*self.B212 + mu**6*self.B312)
        
    def BUU(self, isnp, mu):
        fac = self.param[isnp]['f']**4*self.param[isnp]['D']**4
        return fac*(mu**2*self.B122 + mu**4*self.B222 +
                    mu**6*self.B322 + mu**8*self.B422)

