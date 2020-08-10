import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import random

# functions
def im2dec(arr):
    """takes in an n-by-8 array of 16-bit compressed representation of 2ds pixels and returns an n-by-128 array 
    of the expanded visual field
    """
    def dec2binarr(arr):
        return np.array(list(''.join([bin(int(i))[2:].zfill(16) for i in arr]))).astype(int)
    
    assert arr.shape[1] == 8
    return np.array([dec2binarr(i) for i in arr])