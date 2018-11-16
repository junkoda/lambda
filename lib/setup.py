#
# $ make
#

from distutils.core import setup, Extension
import os

setup(name='lambdalib',
      version='0.0.1',
      author='Jun Koda',
      py_modules=['lambdalib.power',
                  'lambdalib.characteristic_function',
                  'lambdalib.sigma',
                  'lambdalib.corr', 'lambdalib.taruya.py',
                  'lambdalib.dtfe', 'lambdalib.util',
                  'lambdalib.lambda_fitting',
      ],
      packages=['lambdalib'],
)
