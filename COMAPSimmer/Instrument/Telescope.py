import numpy as np

from COMAPSimmer.Instrument import ObservingModes
from COMAPSimmer.Instrument import SourcePosition
from COMAPSimmer.Instrument import FocalPlane

from COMAPSimmer.Tools import Coordinates

siderealDay = (23.*3600. + 56.*60. + 4.0905) # seconds

def getCoordinates(ra0, dec0, totalTime, obsType, rv, rate, offsetPct, 
                   upperRadiusPct, lowerRadiusPct, fieldsize,sampleRate=50):
    """
    ra0       - Right ascension of field center
    dec0      - Declination of field center
    totalTime - Total observing time in days
    obsType   - Observing mode (ces, circular, lissajous)
    rv        - Scan radius (az and el)
    rate      - Pct of maximum
    offsetPct - Pct of radius
    upperRadiusPct - Pct of radius
    lowerRadiusPct - Pct of radius
    fieldsize - Target field width
    sampleRate- Sample rate in Hz
    """

    stepSize = (1./sampleRate)/86400.
    
    siderealRate = 360./siderealDay
    obsLen = fieldsize/siderealRate/np.cos(dec0*np.pi/180.)/60 # minutes
    dT = ((obsLen*60.)//sampleRate) * sampleRate # 20 minutes, seconds



    sp = SourcePosition.AcquireSource(ra0 - obsLen*15./60./2., dec0, minEl=30, maxEl=90, minLHA=5)
    sp.sourceHorizonRing()
    sp.getSourceCenters(58400, 58400+totalTime, dT/3600./24., rv, rv)

    if 'lissajous' in obsType.lower():
        liss1 = ObservingModes.Lissajous(rv,
                                         rv,
                                         rate,
                                         rate*3./4.,
                                         0.,
                                         180.,
                                         offsetPct,
                                         offsetPct,
                                         lowerRadiusPct,
                                         upperRadiusPct)
    elif 'circular' in obsType.lower():
        liss1 = ObservingModes.Lissajous(rv,
                                         rv,
                                         rate,
                                         rate,
                                         0.,
                                         180.,
                                         offsetPct,
                                         offsetPct,
                                         lowerRadiusPct,
                                         upperRadiusPct)
    elif 'ces' in obsType.lower():
        liss1 = ObservingModes.Lissajous(rv,
                                         rv,
                                         rate,
                                         0.,
                                         0.,
                                         180.,
                                         offsetPct,
                                         offsetPct,
                                         lowerRadiusPct,
                                         upperRadiusPct)
    else:
        raise 'No allowed obsType chosen'



    obsSize = int(dT*sampleRate)
    az  = np.zeros(obsSize * sp.totalNobs)
    el  = np.zeros(obsSize * sp.totalNobs)
    utc = np.zeros(obsSize * sp.totalNobs)
    

    for i in range(sp.totalNobs):
        utc[i*obsSize:(i+1)*obsSize] = np.arange(obsSize)*stepSize + sp.obsStartUTC[i]
        az[i*obsSize:(i+1)*obsSize], el[i*obsSize:(i+1)*obsSize] = liss1(sp.obsAz[i],
                                                                         sp.obsEl[i],
                                                                         utc[i*obsSize:(i+1)*obsSize],
                                                                         1./float(sampleRate))

    nHorns = 19
    focalPlane = FocalPlane.FocalPlane()


    azAll = np.zeros((nHorns, az.size))
    elAll = azAll*0.
    raAll = azAll*0.
    decAll= azAll*0.
    offsets = np.arange(az.size)//(2*sampleRate) # offset every 2 seconds
    maxOffset = np.max(offsets)
    offsetsAll = azAll*0.
    for i in range(nHorns):
        elAll[i,:] = el+focalPlane.offsets[i][1] - focalPlane.eloff
        azAll[i,:] = az+focalPlane.offsets[i][0]/np.cos(elAll[i,:]*np.pi/180.) - focalPlane.azoff
        raAll[i,:], decAll[i,:] = Coordinates.h2e(azAll[i,:],elAll[i,:], utc, sp.lon, sp.lat)

        offsetsAll[i,:] = offsets + maxOffset*i


    return azAll.reshape((nHorns, sp.totalNobs, obsSize)), elAll.reshape((nHorns, sp.totalNobs, obsSize)), utc.reshape((sp.totalNobs, obsSize)), raAll.reshape((nHorns, sp.totalNobs, obsSize)), decAll.reshape((nHorns, sp.totalNobs, obsSize)), sp.totalNobs
