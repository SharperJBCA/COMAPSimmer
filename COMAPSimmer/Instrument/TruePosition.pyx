import numpy as np
cimport numpy as np
from libc.math cimport sqrt, pow, abs, sin, cos, tan

from scipy.optimize import leastsq, minimize_scalar
from cpython cimport array
import array

def accelModel(v0, accel, Ta):
    return v0*Ta + 0.5*accel*Ta**2
def driftModel(v0, Td):
    return v0*Td

def model(Ta, dTe, v0, accel):
    Td = dTe - Ta
    v1 = v0 + accel*Ta
    s = accelModel(v0,accel,Ta) + driftModel(v1, Td)
    return s

def scal_err(Ta, s, dTe, v0, accel, maxSpeed):
    #if abs(v0 + accel*Ta) > abs(maxSpeed):
    #    return 1e32
    #else:
        
    return np.sum((s-model(Ta,dTe, v0, accel))**2)


def error(P, s, dTe, v0, accel, maxSpeed):
    Ta = P[0]
    if (np.abs(v0+ Ta*accel) > maxSpeed) | (Ta > dTe):
        return 1e24
    else:
        return s-model(Ta,dTe, v0, accel)


def error2Params(P, s, dTe, v0, maxAccel, maxSpeed):
    Ta, accel = P[0], P[1]
    if (np.abs(v0+ Ta*maxAccel) > maxSpeed) | (Ta > dTe) | (accel > maxAccel):
        return 1e24
    else:
        return s-model(Ta,dTe, v0, accel)



def createProfile(double[:] profile,
                  double x1, double x0, double dTe, double dTp,
                  double maxAccel, double maxSpeed, double v0):

    cdef int i
    cdef double dx = (x1-x0)
    cdef double xdir = dx/abs(dx)
    #cdef double v0 = 0#profile[profile.size-2]
    cdef double v1 = 0
    cdef double adir = xdir
    cdef double ratio
    #P0 = [0.]
    #P1, s = leastsq(error, P0, args=(dx, dTe, v0, minAccel*xdir, maxSpeed))

    # Check if we need to accelerate or decelerate
    if v0 != 0:
        ratio = dx/v0
    else:
        ratio = 0
    if (abs(v0*dTe)-abs(dx) > 0) & (ratio > 0):
        # we must decelerate
        adir *= -1


    res= minimize_scalar(scal_err, args=(dx, dTe, v0, maxAccel*adir, maxSpeed*xdir),
                         method='bounded', bounds=(0, dTe))

    cdef double Ta = res.x
    
    if (abs(v0 + maxAccel*xdir*Ta) > maxSpeed) & (xdir/adir > 0):
        #print('Will Exceed max')
        Ta = abs(maxSpeed*xdir - v0)/maxAccel

    cdef int Np = profile.size
    for i in range(Np):
        if i*dTp < Ta:
            profile[i] = v0 + float(i)*dTp*maxAccel*adir
        else:
            profile[i] = v0 + Ta*maxAccel*adir
            v1 = v0 + Ta*maxAccel*adir

    return Ta, xdir, v1


def intProfile(double[:] profile, int pos, int PPS, double dTp):
    
    cdef int i
    cdef double dist = 0
    for i in range(pos*PPS, (pos+1)*PPS):
        dist += profile[i] * dTp


    return dist

def TruePosition(double[:] coord, double maxSpeed, double maxAccel, double dTe, double dTs):
    cdef int k = 0
    cdef int i

    cdef int Ns = coord.size
    cdef int Np = 1000

    # dTe = 500 ms
    #cdef double dTs = 0.02 # 20 ms
    cdef int EncPerSample = int(dTe/dTs)

    cdef double Ta, _
    cdef double dTp = dTe/float(Np)
    cdef int PPS = 500 # P samples per encoder sample
 
    cdef array.array double_array_template = array.array('d',[])
    profile = array.clone(double_array_template, Np, zero=True)

    cout = array.clone(double_array_template, Ns, zero=True)
    cout[0] = coord[0]
    vout = array.clone(double_array_template, Ns, zero=True)

    cdef double x0, x1, xs0

    x0 = coord[0]
    x1 = coord[EncPerSample]
    xs0 = coord[0]
    cdef double xdir = (x1 - x0)/abs(x1-x0)
    cdef double v1 = maxSpeed*xdir

    _,_,v1 = createProfile(profile, x1, x0, dTe, dTp, maxAccel, maxSpeed, v1)
    vout[0] =  maxSpeed*xdir

    cdef int count =1
    k=0
    for i in range(1,Ns):

        if i >= Ns-EncPerSample:
            cout[i] = cout[i-1]
            continue
        if k >= EncPerSample:
            k = 0
            x0 = cout[i-1]
            x1 = coord[i+EncPerSample]
            Ta,_,v1 = createProfile(profile, x1, x0, dTe, dTp, maxAccel, maxSpeed,v1)
            ###print(x0, x1, Ta, profile[0], v1)
            #print('SUM',np.sum(profile))
        cout[i] = cout[i-1] + intProfile(profile, k, int(dTs/dTp), dTp)
        vout[i] = profile[k]
        k+=1

    return cout, vout
