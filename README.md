# COMAPSimmer

### Installing

'''
python setup.py build_ext --inplace
'''
You must set a SLALIB_LIBS env variable to point to your local installation of the SLALIB libraries.

### Running

'''
python createLevel1.py
'''

Modify Parameters.py to change inputs.

Run createLevel1.py above COMAPSimmer directory (that is within the repo). It will create two directories:
* data - contains the mock level 1 TOD files
* COCubes - contains the mean Tco files as npz's, and the CO cubes as hdf5 files.


