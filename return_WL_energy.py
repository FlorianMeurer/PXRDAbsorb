import numpy as np
import scipy as sp
from scipy import constants as conts

def ret_energy(wavelength):
    return (conts.h * conts.c) / (wavelength*1.60218e-19) * 1E10

def ret_wl(energy):
    return (conts.h * conts.c) / (energy*1.60218e-19) * 1E10