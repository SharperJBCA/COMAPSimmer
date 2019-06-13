from astropy import wcs
import numpy as np

def makemap(d,x,y,w, ra0=0,dec0=0, cd=1./60., nxpix=600, nypix=400):

    xy = np.zeros((x.size,2))
    xy[:,0] = x.flatten()
    xy[:,1] = y.flatten()

    nxpix = int(w.wcs.crpix[1]*2)
    nypix = int(w.wcs.crpix[0]*2)
    cd    = w.wcs.cdelt[0]

    pixels = w.wcs_world2pix(xy,0)
    ygrid, xgrid = np.meshgrid(np.arange(nypix),np.arange(nxpix))

    pixCens = w.wcs_pix2world(np.array([xgrid.flatten(), ygrid.flatten()]).T,0)
    pixCens[:,0] += 1./2.*cd
    pixCens[:,1] += 1./2.*cd
    pflat = (pixels[:,1].astype(int) + (nypix)*pixels[:,0].astype(int)).astype(int)


    pEdges = np.arange(nxpix*nypix+1)
    m = np.histogram(pflat,pEdges, weights=d)[0]
    h = np.histogram(pflat,pEdges)[0]
    m = m/h
    return np.reshape(m,(nypix, nxpix)),np.reshape(h,(nypix,nxpix))
