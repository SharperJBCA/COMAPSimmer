import numpy as np
import healpy as hp
from scipy.optimize import leastsq
from matplotlib import pyplot
import time

from COMAPSimmer.Instrument import TruePosition

def accelModel(v0, accel, Ta):
    return v0*Ta + 0.5 * accel * Ta**2

def driftModel(v0, Td):
    return v0*Td

def model(P, dTe, v0, accel):
    Ta = P[0]
    Td = dTe - Ta
    v1 = v0 + accel*Ta
    s = accelModel(v0, accel, Ta) + driftModel(v1, Td)
    return s 

def error(P, s, dTe, v0, accel, maxSpeed):
    Ta = P[0]

    if (np.abs(v0 + Ta*accel) > maxSpeed) | (Ta < dTe):
        return 1e24
    else:
        return s - model(P, dTe, v0, accel)


# consts
deg2rad = np.pi/180.
rad2deg = 180./np.pi
masPerRad = 3.6e6 / deg2rad

class ObservingMode:

    def __init__(self):
        pass

class Lissajous(ObservingMode):
    
    def __init__(self,
                 azRadiusInDeg, # Radius in az
                 elRadiusInDeg, # Radius in el
                 dazdtPct, # rate * radius (max 1.0 degree /minute)
                 deldtPct, # rate * radius (max 0.5 degree/minute)
                 azPhaseInDeg, # lissajous phase angle
                 phaseInDeg, # starting phase
                 azDistributionWidth, # az offset width as pct of az radius (~20%)
                 elDistributionWidth, # el offset width as pct of el radius 
                 minPctRadiusDistribution,
                 maxPctRadiusDistribution,
                 MAXRATE_=30., # force both drives to not exceed 30 arcmin/minute
                 MAXRATE_CES_=60., # if just az drives use 60 arcmin/minute
                 keepRadiusRatio=True):
        
        
        if elRadiusInDeg != 0:
            self.MAXRATEinRadPerSec_ = MAXRATE_ * deg2rad / 60.
        else:
            self.MAXRATEinRadPerSec_ = MAXRATE_CES_ * deg2rad / 60.

        self.azInitialRadiusInRad_ = azRadiusInDeg * deg2rad
        self.elInitialRadiusInRad_ = elRadiusInDeg * deg2rad
        # why the 0.01? Since it is a pct input
        self.azRateInRadPerSec_ = dazdtPct * 0.01 * self.MAXRATEinRadPerSec_
        self.elRateInRadPerSec_ = deldtPct * 0.01 * self.MAXRATEinRadPerSec_

        #print(self.azRateInRadPerSec_/deg2rad,self.elRateInRadPerSec_/deg2rad)

        self.azRateAdjInRadPerSec_ = self.azRateInRadPerSec_*1. # Catch changes in el
        self.elRateAdjInRadPerSec_ = self.elRateInRadPerSec_*1.
        self.azPhaseInRad_ = azPhaseInDeg * deg2rad
        self.phaseInRad_ = phaseInDeg * deg2rad

        self.azDistributionWidth_ = azDistributionWidth * 0.01 # multiply by 0.01 to convert
        self.elDistributionWidth_ = elDistributionWidth * 0.01

        # Radius randomisation widths:
        self.minRadiusDistribution = minPctRadiusDistribution * 0.01
        self.maxRadiusDistribution = maxPctRadiusDistribution * 0.01
        self.keepRadiusRatio = keepRadiusRatio

        # These are the relative scan centers
        self.azScanCenter_ = 0.
        self.elScanCenter_ = 0.
    

        self.accel = 0.5

    def __call__(self, az0, el0, utc, dTs):
        #t0 = time.time()

        self.rot = hp.rotator.Rotator(rot=[az0,el0], deg=True, inv=True)
        self.time = (utc-np.min(utc))*86400. # seconds
        
        self.updateScanRadius()
        self.updateScanCenter()
        self.adjustAzElRate(el0*deg2rad) # Perform AFTER updateRadius/Center

        az, el = self.computeOnSkyScanPosition()


        az = np.mod(az*rad2deg,360)
        az[az > 180] -= 360.
        el = el*rad2deg

        az = az/np.cos(el0*np.pi/180.)

        #pyplot.plot(az)
        #pyplot.plot(np.gradient(az)/dTs)

        #print(self.azRadiusInRad_ *self.azRateAdjInRadPerSec_*180./np.pi)

        #_az,_vaz = TruePosition.TruePosition(az,
        #                                     self.azRadiusInRad_ *self.azRateAdjInRadPerSec_*180./np.pi, 
        #                                     self.accel, 
        #                                     0.5, dTs)
        #az = np.array(_az)
        #vaz = np.array(_vaz)
        #pyplot.plot(az)
        #pyplot.plot(vaz)
        #pyplot.show()

        #print(self.azScanCenter_, self.elScanCenter_)
        if (self.elRateAdjInRadPerSec_ >0) & (self.elRadiusInRad_ > 0):
            # _el,_vel = TruePosition.TruePosition(el,
            #                                         self.elRadiusInRad_ *self.elRateAdjInRadPerSec_*180./np.pi, 
            #                                         self.accel,
            #                                         0.5, dTs)
            # el = np.array(_el)
            # vel = np.array(_vel)
            pass
        else:
            return np.mod(az+az0,360), el+el0
        return np.mod(az+az0,360), el+el0

        

    def computeOnSkyScanPosition(self):
        """
        Compute scan az/el positions
        """
        azOffsets = self.azScanCenter_ + self.azRadiusInRad_ * np.sin(np.mod(self.time * self.azRateAdjInRadPerSec_ + self.phaseInRad_ + self.azPhaseInRad_,2*np.pi))
        elOffsets = self.elScanCenter_ + self.elRadiusInRad_ * np.sin(np.mod(self.time * self.elRateAdjInRadPerSec_ + self.phaseInRad_ + np.pi/2.,2*np.pi))

        return azOffsets, elOffsets


    def updateScanCenter(self):
        """
        Uniform (to match std::cmath::rand()) randomise the scan center
        """

        # Make this a fraction of the updated scan radius
        self.azScanCenter_ =  self.azDistributionWidth_ * self.azRadiusInRad_ * np.random.uniform(low=-0.5,high=0.5)
        if self.elRateAdjInRadPerSec_ > 0:
            self.elScanCenter_ =  self.elDistributionWidth_ * self.elRadiusInRad_ * np.random.uniform(low=-0.5,high=0.5)
        else:
            self.elScanCenter_ =  self.elDistributionWidth_ * self.azRadiusInRad_ * np.random.uniform(low=-0.5,high=0.5)


    def updateScanRadius(self):
        """
        Uniform (to match std::cmath::rand()) randomise the scan center
        """

        # Make this a fraction of the updated scan radius
        offset = np.random.uniform(low =self.minRadiusDistribution,
                                   high=self.maxRadiusDistribution)
       # print(self.minRadiusDistribution, self.maxRadiusDistribution)
        self.azRadiusInRad_ =  self.azInitialRadiusInRad_ * offset

        if not self.keepRadiusRatio:
            offset = np.random.uniform(low =self.minRadiusDistribution,
                                       high=self.maxRadiusDistribution)

        if self.elRateAdjInRadPerSec_ > 0:
            self.elRadiusInRad_ =  self.elInitialRadiusInRad_ * offset
        else:
            self.elRadiusInRad_ = 0

    def adjustAzElRate(self, elInRad):
        """
        Ensure that scan rate is reasonable given mean elevation
        """

        if elInRad != np.pi/2.:
            telAzRadius = self.azRadiusInRad_ / np.cos(elInRad) # So account for the cos(el) for the drive azimuth not on sky azimuth
        else:
            telAzRadius = np.pi
        

        if telAzRadius != 0:
            self.azRateAdjInRadPerSec_ = np.min([self.MAXRATEinRadPerSec_,
                                                 self.azRateInRadPerSec_]) / telAzRadius

        if self.elRadiusInRad_ != 0:
            self.elRateAdjInRadPerSec_ = np.min([self.MAXRATEinRadPerSec_,
                                                 self.elRateInRadPerSec_]) / self.elRadiusInRad_

        if (self.elRateInRadPerSec_ != 0) and (self.elRadiusInRad_ !=0) and ( self.azRateInRadPerSec_ != 0) and (self.azRadiusInRad_ !=0):
            rateRatio = self.azRateInRadPerSec_ / self.elRateInRadPerSec_

            if rateRatio != 0:
                minRate = np.min([self.azRateAdjInRadPerSec_, self.elRateAdjInRadPerSec_])
                if rateRatio > 1:
                    self.azRateAdjInRadPerSec_ = minRate
                    self.elRateAdjInRadPerSec_ = minRate/rateRatio
                else:
                    self.elRateAdjInRadPerSec_ = minRate
                    self.azRateAdjInRadPerSec_ = minRate*rateRatio

        
