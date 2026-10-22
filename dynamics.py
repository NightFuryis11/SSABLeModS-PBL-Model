import numpy as np
from pint import Quantity
import constants
from typing import Union
from pint.errors import UnitStrippedWarning
import warnings
from scipy import interpolate
import sys
from datetime import datetime
import os
from netCDF4 import Dataset as NCDF
import numba as nb
import math
import matplotlib.pyplot as plt
from utility_equations import _ensure_arrays

warnings.filterwarnings('ignore', category=UnitStrippedWarning)

def Coriolis_u(lat : float, v_ag : Union[int, float, np.ndarray]) -> np.ndarray:
    v_ag = _ensure_arrays(v_ag)
    
    f = 2 * constants.omega * np.sin(np.deg2rad(lat))
    return f * v_ag

def Coriolis_v(lat : float, u_ag : Union[int, float, np.ndarray]) -> np.ndarray:
    u_ag = _ensure_arrays(u_ag)
    
    f = 2 * constants.omega * np.sin(np.deg2rad(lat))
    return -f * u_ag

if __name__ == "__main__":

    pass