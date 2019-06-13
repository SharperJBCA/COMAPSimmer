import numpy as np
import pickle
import numpy as np
from matplotlib import pyplot
import h5py
import h5py_cache as h5py
import glob

import Parameters

from COMAPSimmer.Tools import utils
from COMAPSimmer.Tools import  quickbin
from COMAPSimmer.Tools import  Mapping
from COMAPSimmer.Tools import Coordinates

from COMAPSimmer.Instrument import FocalPlane
from COMAPSimmer.Instrument import Telescope

from COMAPSimmer.Sky import Li2015GaussCO

fields = {'CO1':[(14. + 30./60.+ 0)*15.          , (20. + 45./60.)         ], 
          'CO2':[(1. + 41/60. + 44.4/60.**2)*15. , 0.                      ],
          'CO3':[(12. + 36./60. + 55./60.**2)*15., 62. + 14./60.+15./60.**2],
          'CO4':[(14. + 17./60.)*15.             , 52. + 30./60.           ]}

def createDataFile():
    # load in the level 1 structure...
    with open('COMAPSimmer/AncilData/level1-struct.p','rb') as fp:
        data = pickle.load(fp)
        fp.close()
    output = h5py.File('{}/{}_{:04d}.hd5'.format(Parameters.DataDir, Parameters.DataPrefix, i),
                       chunk_cache_mem_size=1024**3)

    for k, v in data['attrs'].items():
        if not k in output:
            grp = output.create_group(k)
        else:
            grp = output[k]
            
        for k2, v2 in v['attrs'].items():
            grp.attrs[k2] = ''
            
    output['comap'].attrs['level'] = 'simulation_v1'
    
    def testval(vi):
        if vi > 20:
            return None
        else:
            return vi

    for k, v in data['datasets'].items():
        if not k in output:
            dtype = v[-1]
            shape = tuple([testval(vi) for vi in v[0:len(v)-1]])
            if 'tod' in k:
                output.create_dataset(k, v[0:len(v)-1], maxshape=shape,
                                      dtype=dtype)
            else:
                output.create_dataset(k, v[0:len(v)-1], maxshape=shape, dtype=dtype)

    return output

def addNoise(shape):
    dnu = (Parameters.freqMax - Parameters.freqMin)/Parameters.nchannels * 1e9
    rms = Parameters.Tsys/np.sqrt(dnu/Parameters.sampleRate)

    return np.random.normal(size=shape, scale=rms)

import time
if __name__ == "__main__": 

    # create directories
    utils.createDirectories(Parameters.COCubeDir, Parameters.DataDir)

    # Create sky model
    t0 = time.time()
    cubedata, wcs = Li2015GaussCO.selectCOCube(fields[Parameters.field][0],
                                               fields[Parameters.field][1],
                                               Parameters)
    
    t1 = time.time()
    # Create telescope tracks
    az, el, mjd, ra, dec, Nobs = Telescope.getCoordinates(fields[Parameters.field][0],
                                                          fields[Parameters.field][1],
                                                          Parameters.totalTime, 
                                                          Parameters.obsType, 
                                                          Parameters.rv, 
                                                          Parameters.rate, 
                                                          Parameters.offsetPct, 
                                                          Parameters.upperRadiusPct, 
                                                          Parameters.lowerRadiusPct, 
                                                          Parameters.fieldsize,
                                                          Parameters.sampleRate)

    t2 = time.time()
    # save h5py file...
    for i in range(Nobs):
        t3 = time.time()
        print('Pct Complete {:.1f}'.format(i/float(Nobs)*100.))
        output = createDataFile()
    
        # flip sidebands 0 and 2
        frequencies = np.reshape(cubedata['frequencies'][:],
                                 (Parameters.nsbs, 
                                  cubedata['frequencies'].shape[0]//Parameters.nsbs))
        for j in [0,2]:
            frequencies[j,:] = np.flip(frequencies[j,:])
        
        # Data structures to write
        pvals = {'/spectrometer/pixel_pointing/pixel_el' : el[:,i,:],
                 '/spectrometer/pixel_pointing/pixel_az' : az[:,i,:],
                 '/spectrometer/pixel_pointing/pixel_ra' : ra[:,i,:],
                 '/spectrometer/pixel_pointing/pixel_dec': dec[:,i,:],
                 '/spectrometer/MJD': mjd[i,:],
                 '/spectrometer/frequency': frequencies,
                 '/spectrometer/feeds':np.arange(az.shape[0]),
                 '/spectrometer/features':np.zeros(az.shape[-1]),
                 '/spectrometer/bands':np.array(['A:LSB', 'A:USB', 'B:LSB', 'B:USB'])}

        # Set data structures
        for k, v in pvals.items():
            output[k].resize(v.shape)
            dtype = output[k].dtype
            output[k][...] = v.astype(dtype)

        # TOD is too big to generate at once, so this is a special case...
        ypixels, xpixels = wcs.wcs_world2pix(ra[:,i,:],dec[:,i,:],0)
        pflat  = (xpixels.astype(int) + Parameters.nspix*ypixels.astype(int)).astype(int)

        toddata = output['spectrometer/tod']
        toddata.resize((ra.shape[0],
                        frequencies.shape[0],
                        frequencies.shape[1],
                        ra.shape[-1]))

        dtype   = toddata.dtype
        tod = np.zeros((toddata.shape[1],toddata.shape[2], toddata.shape[-1]))
        cube = cubedata['cube'][...]
        
        cube = np.reshape(cube, (cube.shape[0], Parameters.nsbs, Parameters.nchannels//Parameters.nsbs))
        cube[:,0,:] = np.flip(cube[:,0,:],axis=1)
        cube[:,2,:] = np.flip(cube[:,2,:],axis=1)
        #select = np.reshape(np.arange(Parameters.nchannels),(Parameters.nsbs,
        #                                                     Parameters.nchannels//Parameters.nsbs)).astype(int)
        #select[0,:] = select[0,::-1]
        #select[2,:] = select[2,::-1]
        t4 = time.time()
        #import time
        for j in range(toddata.shape[0]):
            t5 = time.time()
            #t0 = time.time()
            quickbin.cube2TOD(pflat[j,:], 
                              Parameters.nsbs,
                              Parameters.nchannels//Parameters.nsbs,
                              cube,
                              tod)
            #tod = ((cube[pflat[j,:],:]).T[select,:])
            t6 = time.time()
            tod += addNoise(tod.shape)
            t7 = time.time()
            #print(tod.shape, toddata.shape)
            #toddata[j:j+1,:,:,:] = tod#.astype(dtype)
            toddata.write_direct(tod,np.s_[0:tod.shape[0],0:tod.shape[1],0:tod.shape[2]],
                                 np.s_[j:j+1,0:tod.shape[0],0:tod.shape[1],0:tod.shape[2]])
            print(t1-t0, t2-t1, t4-t3,t6-t5,t7-t6, time.time()-t7)
            #print('loop {} took {} seconds'.format(j, time.time()-t0))
        # for j in range(Parameters.nchannels):
        #     c = Tools.nspace(j, list(frequencies.shape))
        #     #tod = cubedata['cube'][:,j]
        #     tod = cube[pflat,j]
        #     tod += addNoise(tod.shape)
            
        #     if c[0] == 0 | c[0] == 2:
        #        toddata[:,c[0],toddata.shape[2]-c[1]-1,:] = tod.astype(dtype)
        #     else:
        #        toddata[:,c[0],c[1],:] = tod.astype(dtype)
        # output[:,0,:,:] = output
        output.close()
