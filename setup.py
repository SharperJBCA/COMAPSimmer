from numpy.distutils.core import setup
from numpy.distutils.core import Extension
from numpy import get_include
from Cython.Build import cythonize
import os

try:
    slalib_path = os.environ['SLALIB_LIBS']
except KeyError:
    slalib_path = '/star/lib' # default path of Manchester machines
    print('Warning: No SLALIB_LIBS environment variable set, assuming: {}'.format(slalib_path))

pysla = Extension(name = 'COMAPSimmer.Tools.pysla', 
                  sources = ['COMAPSimmer/Tools/pysla.f90'],
                  libraries=['sla'],
                  library_dirs =['{}'.format(slalib_path)],
                  f2py_options = [],
                  extra_f90_compile_args=['-L{}'.format(slalib_path),'-lsla'])
quickbin = Extension(name='COMAPSimmer.Tools.quickbin',
                     include_dirs=[get_include()],
                     sources=['COMAPSimmer/Tools/quickbin.pyx'])
truepos = Extension(name='COMAPSimmer.Instrument.TruePosition',
                     include_dirs=[get_include()],
                     sources=['COMAPSimmer/Instrument/TruePosition.pyx'])
                     
extensions = [pysla, quickbin, truepos]
setup(name='COMAPSimmer', ext_modules=cythonize(extensions))

# config = {'name':'COMAPSimmer',
#           'version':'0.1dev',
#           'ext_modules':[cythonize("quickbin.pyx"),cythonize("TruePosition.pyx")]}

# setup(ext_modules=com
