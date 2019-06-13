import numpy as np
# Parameters read into createLevel1.py 

# Data directories
COCubeDir    = 'COCubes'
COCubePrefix = 'Li2015'
DataDir      = 'data'
DataPrefix   = 'fulltest'

# Reload previous run
selectCOCube = 1 # if None, generate new cube, else 0...N

# Instrument setup
freqMin   = 26 # GHz
freqMax   = 34 # GHz
nchannels = 16 # total number of channels (over all sidebands)
nsbs      = 4 #  divide number of channels over this many sidebands
nspix     = 512
cdelt     = 1./60. # degrees, pixel size
ctype     = ['RA---TAN','DEC--TAN'] # Projection type
Tsys      = 0/np.sqrt(360.) # K, if None then no noise, 360 assuming we observing 360 days
sampleRate= 50

# Parameters for the scan strategy
field = 'CO4'
totalTime = 1.          # Days (integer numbers only)
obsType = 'ces'         # ces, circular, lissajous
rv = 0.47               # degrees
rate = 100.             # percentage of max
offsetPct = 30.         # percentage of radius (random offset)
lowerRadiusPct = 100.   # percent of avg. radius (lower bound random radii)
upperRadiusPct = 100.   # percent of avg. radius (upper bound random radii)
fieldsize = np.sqrt(3./np.pi) 
