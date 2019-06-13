import numpy as np
from matplotlib import pyplot
from scipy.interpolate import RectBivariateSpline
from scipy.integrate import simps
import hmf
from astropy import wcs as wcsModule
from scipy.interpolate import InterpolatedUnivariateSpline
from pathlib import Path
from scipy.ndimage import gaussian_filter

import h5py
import glob
import copy

import scipy.fftpack as sfft
import pyfftw
import multiprocessing
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()
sfft = pyfftw.interfaces.scipy_fftpack
pyfftw.interfaces.cache.enable()

#Constants
pc = 3.086e16
H0 = 100.*1e3 * 1e-6  / pc  #1/s
c  = 299792458.
k = 1.38e-23
v0 = 115e9
L0 = 3.828e26

def selectCOCube(ra0, dec0, Parameters):
    """
    Return HDF5 object containing CO Cube.
    """
    filelist = glob.glob('{}/{}_N{:04d}_*'.format(Parameters.COCubeDir,
                                                  Parameters.COCubePrefix,
                                                  Parameters.nchannels))
    N = len(filelist)
    if isinstance(Parameters.selectCOCube, type(None)):
        Parameters.selectCOCube = N

    if '{}/{}_N{:04d}_{:04d}.hd5'.format(Parameters.COCubeDir,
                                         Parameters.COCubePrefix,
                                         Parameters.nchannels,
                                         Parameters.selectCOCube) in filelist:
        cubedata = h5py.File('{}/{}_N{:04d}_{:04d}.hd5'.format(Parameters.COCubeDir,
                                                               Parameters.COCubePrefix,
                                                               Parameters.nchannels,
                                                               Parameters.selectCOCube),'r')
        wcs = wcsModule.WCS(naxis=2)
        wcs.wcs.crval = cubedata['cube'].attrs['crval']
        wcs.wcs.cdelt = cubedata['cube'].attrs['cdelt']
        wcs.wcs.crpix = cubedata['cube'].attrs['crpix']
        wcs.wcs.ctype = [cubedata['cube'].attrs['ctype1'],cubedata['cube'].attrs['ctype2'] ]
    else:
        # First generate a CO cube
        coCube = Li2015GaussCO(freqmin=Parameters.freqMin,
                               freqmax=Parameters.freqMax,
                               nchannels=Parameters.nchannels,
                               nspix=Parameters.nspix, 
                               x0=ra0,
                               y0=dec0,
                               save=True,
                               CODataDir=Parameters.COCubeDir)
        cube = coCube()
        d = h5py.File('{}/{}_N{:04d}_{:04d}.hd5'.format(Parameters.COCubeDir,
                                                        Parameters.COCubePrefix,
                                                        Parameters.nchannels, N))
        d.create_dataset('cube',data=np.reshape(cube, (cube.shape[0]*cube.shape[1], cube.shape[2])))
        d['frequencies'] = coCube.freqs
        d['cube'].attrs['nxpix'] = Parameters.nspix
        d['cube'].attrs['nypix'] = Parameters.nspix
        d['cube'].attrs['cdelt'] = coCube.wcs.wcs.cdelt
        d['cube'].attrs['ctype1'] = coCube.wcs.wcs.ctype[0]
        d['cube'].attrs['ctype2'] = coCube.wcs.wcs.ctype[1]
        d['cube'].attrs['crval'] = coCube.wcs.wcs.crval
        d['cube'].attrs['crpix'] = coCube.wcs.wcs.crpix
        d.close()
        wcs = copy.deepcopy(coCube.wcs)
        del cube
        del coCube
        cubedata = h5py.File('{}/{}_N{:04d}_{:04d}.hd5'.format(Parameters.COCubeDir,
                                                               Parameters.COCubePrefix,
                                                               Parameters.nchannels, N),'r')
    return cubedata, wcs

class Li2015GaussCO:
    """
    Generate Gaussian CO cube realisations using the Li 2015 CO model.
    """

    def __init__(self, freqmin = 26, freqmax= 34, nchannels=64, Mmin = 9, Mmax=14, save=False,
                 x0 = 0, y0 = 0, dtheta = 1./60., nspix = 256, ctype = ['RA---TAN', 'DEC--TAN'],
                 CODataDir='.', SFRdata='COMAPSimmer/AncilData/sfr_release.dat'):
        self.SFRdata = SFRdata

        # Instrument
        self.freqDelta = (freqmax - freqmin)/nchannels * 1e9 # Hz
        self.freqMin   = freqmin * 1e9 # Hz
        self.freqMax   = freqmax * 1e9 # Hz
        self.nchannels = int(nchannels)
        self.freqs     = (np.arange(nchannels) + 0.5)*self.freqDelta + self.freqMin

        # Setup image
        self.crval = [x0, y0]
        self.cdelt = [dtheta, dtheta]
        self.npixs = [int(nspix), int(nspix)]
        self.crpix = [nspix/2, nspix/2]
        self.ctype = ctype
        self.setWCS(self.crval, self.cdelt, self.crpix, self.ctype)

        # Cosmology
        self.mycosmo   = hmf.cosmo.Cosmology()
        self.Pk   = hmf.transfer.Transfer()

        self.Mmin = Mmin
        self.Mmax = Mmax
        self.mf   = hmf.hmf.MassFunction(Mmin=self.Mmin,Mmax=self.Mmax)
        self.z    = v0/self.freqs - 1
        self.rc   = self.mycosmo.cosmo.comoving_distance(self.z).value *1e6 *pc
        dk   = self.Pk.k[1:] - self.Pk.k[:-1]
        Pmk  = self.Pk.delta_k/self.Pk.k**3 * 2 * np.pi
        Pmk  = (Pmk[1:] + Pmk[:-1])/2.
        self.inputPk = self.Pk.delta_k/self.Pk.k**3 * 2* np.pi/np.sum(Pmk*dk)
        self.pmdl = InterpolatedUnivariateSpline(self.Pk.k,self.inputPk)


        # pyFFTw
        self.cube = None

        # Initalisation
        if save:
            self.filename = '{}/MeanTco_freq{:.1f}-{:.1f}_N{:d}_M{:.0f}-{:.0f}.npz'.format(CODataDir,
                                                                                           freqmin,
                                                                                           freqmax,
                                                                                           int(nchannels),
                                                                                           Mmin, Mmax)
        else:
            self.filename = ''
        my_file = Path(self.filename)

        # If a large job was already done, just reload that model.
        if save:
            if my_file.is_file():
                self.Tco = np.load(self.filename)['Tco']
            else:
                self.Tco = self.Halo2Tco()
                np.savez(self.filename, Tco=self.Tco)
        else:
            self.Tco = self.Halo2Tco()

    def __call__(self):
        """
        Each call to the class should generate a new cube.
        """
        # Setup power spectrum step sizes
        dt = self.cdelt[0]*np.pi/180. * np.mean(self.rc) / (1e6 *pc) # assume a mean comove dist, reasonable approximation?
        dr = np.abs(np.mean(self.rc[1:] - self.rc[:-1]))  / (1e6 *pc)
        self.step  = [dt,dt,dr]
        self.shape = [self.npixs[1],self.npixs[0],self.nchannels]
        self.cube = self.GenCube(self.step, self.shape, self.pmdl) # returns a normalised cube of haloes

        # Fudge factor just to scale correctly for COMAP meeting...
        fudge = 1e5
        self.cube *= self.Tco * fudge
        return self.cube

    def sphAvgPwr(self, nbins=30):
        """
        Perform 3D FFT to generate a gaussian realisation of the CO cube
        """
        kEdges = np.linspace(np.min(self.k),np.max(self.k),nbins+1)
        kMids  = (kEdges[1:] + kEdges[:-1])/2.

        self.cube -= np.median(self.cube)
        Pcube = np.abs(sfft.fftn(self.cube))**2 #/ self.cube.size**2
        Pk = np.histogram(self.k.flatten(), kEdges, weights=Pcube.flatten())[0]/np.histogram(self.k.flatten(), kEdges)[0]

        return kMids, Pk


    def setWCS(self, crval, cdelt, crpix, ctype=['RA---TAN', 'DEC--TAN']):
        """
        Setup WCS
        """
        self.wcs = wcsModule.WCS(naxis=2)
        self.wcs.wcs.crval = crval
        self.wcs.wcs.cdelt = cdelt
        self.wcs.wcs.crpix = crpix
        self.wcs.wcs.ctype = ctype

    def GenCube(self, step, shape, pmodel, random=np.random.normal):
        """
        Perform 3D FFT to generate a gaussian realisation of the CO cube
        """
        x = sfft.fftfreq(shape[0], d=step[0])
        y = sfft.fftfreq(shape[1], d=step[1])
        z = sfft.fftfreq(shape[2], d=step[2])
        i = np.array(np.meshgrid(x,y,z, indexing='ij'))

        self.k = np.array([i[j,...]**2 for j in range(i.shape[0]) ])
        self.k = np.sqrt(np.sum(self.k, axis=0))

        i = np.array(np.meshgrid(x,y, indexing='ij'))
        self.kp = np.array([i[j,...]**2 for j in range(i.shape[0]) ])
        self.kp = np.sqrt(np.sum(self.kp, axis=0))

        Atemp = 180**2/np.pi
        beam = lambda sigma,k: np.exp(-0.5*k*Atemp*(k*Atemp+1)*sigma**2) # Check this, close enough for now

        vals = pyfftw.empty_aligned(self.k.shape, dtype='float64')
        img  = pyfftw.empty_aligned(self.k.shape, dtype='float64')
        img[...]  = np.sqrt(pmodel(self.k)/vals.size**2) * beam(5./60./2.355*np.pi/180.,self.kp)[:,:,np.newaxis] #/self.k.shape[2])
        vals[...] = random(size=self.k.shape)
        vals -= np.median(vals)
        #vals /= np.std(vals)
        phases = sfft.fftn(vals)
        if isinstance(self.cube, type(None)):
            self.cube = pyfftw.empty_aligned(self.k.shape, dtype='float64')
        self.cube[...] =  np.real(sfft.ifftn(np.sqrt(img)*phases))
        #cube = sfft.fftconvolve(img, phases)
        return self.cube


    def Halo2SFR(self, z, mass):
        """
        Interpolate SFR from Behroozi, Wechsler, & Conroy 2013a (http://arxiv.org/abs/1207.6105) and 2013b ((http://arxiv.org/abs/1209.3013)
        """
        zp1, logm, logsfr, _  = np.loadtxt(self.SFRdata).T

        logm   = np.unique(logm)
        zp1    = np.unique(np.log10(zp1))
        logsfr = np.reshape(logsfr, (logm.size, zp1.size))

        sfr_interp = RectBivariateSpline(logm, zp1,10**logsfr, kx=1,ky=1)

        return sfr_interp.ev(mass, np.log10(z+1))

    def SFR2IR(self, SFR):
        """
        Start formation rate to IR from Li et al...
        """
        return SFR * 1e10

    def IR2Lco(self, IR, alpha=1.37, beta=-1.74):
        """
        Lco in units of K km/s /pc^2
        """

        A = (np.log10(IR) - beta)/alpha
        return 10**A

    def Halo2Tco(self):
        """
        Derive mean co brightness temperature with redshift
        """
        # Here derive relation between mass and Lco for each redshift bin
        self.Tco  = np.zeros(self.z.size)

        norm = simps(self.mf.dndm, self.mf.m)
        Jy2K = c**2 / (2. * k * self.freqs**2)
        for i in range(self.z.size):
            SFR = self.Halo2SFR(self.z[i], np.log10(self.mf.m))
            IR  = self.SFR2IR(SFR)
            _Lco = self.IR2Lco(IR)

            self.mf.update(z=self.z[i])
            Lco = simps(_Lco**1*self.mf.dndm, self.mf.m)/norm*4.9e-5

            self.Tco[i] = Jy2K[i]* Lco*L0/(4*np.pi*(1+self.z[i])**2*self.rc[i]**2)
        return self.Tco


if __name__ == "__main__":
    import time

    t0 = time.time()
    test = Li2015GaussCO(nchannels=32,nspix=512, save=True)
    t1 = time.time()
    cube = test()
    t2 = time.time()
    k, Pk = test.sphAvgPwr(nbins=256)
    t3 = time.time()
    print(t1 - t0, t2-t1, t3-t2, np.sum(cube**2))
    pyplot.subplot(121, projection=test.wcs)
    pyplot.imshow(gaussian_filter(cube[:,:,0],3)*1e6, origin='lower')
    pyplot.grid()
    pyplot.colorbar(label=r'$\mu$K')
    pyplot.subplot(122)
    print(test.cube.size,)
    pyplot.plot(k, Pk * k**3 *1e12,color='k',zorder=1,linestyle='--')
    pyplot.yscale('log')
    pyplot.xscale('log')
    ylim = pyplot.ylim()
    pyplot.plot(test.Pk.k,test.inputPk*test.Pk.k**3 * 1e12* np.mean(test.Tco*1e5)**2, zorder=0) #* np.mean(test.Tco)**2
    pyplot.xlim(np.min(k),np.max(k))
    pyplot.ylim(ylim)
    pyplot.grid()
    pyplot.xlabel(r'$k$')
    pyplot.ylabel(r'$P_k$ $k^3$')
    pyplot.tight_layout()
    pyplot.show()
