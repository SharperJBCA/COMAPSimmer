import numpy as np
import os

def createDirectories(COCubeDir, DataDir):

    if not os.path.exists(COCubeDir):
        os.makedirs(COCubeDir)

    if not os.path.exists(DataDir):
        os.makedirs(DataDir)



def nspace(position, dims):
    ndims = len(dims)
    _dims = [int(1)] + dims
    coords = [np.mod(int(position//np.product(_dims[:i+1])), dims[i]) for i in range(ndims)]
    return coords
