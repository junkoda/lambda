#
# $ make
#

from distutils.core import setup, Extension
import os

this_dir = os.path.dirname(os.path.realpath(__file__))
lambda_dir = os.path.abspath(this_dir + '/..')

with open('lambdalib/lambdalib_dir.py', 'w') as f:
    f.write("_lambda_dir='%s'\n" % lambda_dir)


setup(name='lambdalib',
      version='0.0.1',
      author='Jun Koda',
      py_modules=['lambdalib.power',
                  'lambdalib.characteristic_function',
                  'lambdalib.sigma',
                  'lambdalib.corr', 'lambdalib.taruya',
                  'lambdalib.dtfe', 'lambdalib.util',
                  'lambdalib.lambda_fitting',
      ],
      packages=['lambdalib'],
)
