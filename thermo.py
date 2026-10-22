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

def poissons_T_to_theta(T : Union[int, float, np.ndarray], p : Union[int, float, np.ndarray], r_v : Union[int, float, np.ndarray]) -> np.ndarray:
    '''
    Parameters
    ----------------------
    T  -  int, float, np.ndarray, or pint.Quantity
        The temperature of the parcel to find the potential temperature of.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    p  -  int, float, np.ndarray, or pint.Quantity
        The pressure of the parcel to find the potential temperature of.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    r_v  -  int, float, np.ndarray, or pint.Quantity
        The mixing ratio of the parcel to find the potential temperature of.
        If provided as a pint.Quantity, may be in any units of mass ratio. If provided as an int or float, expects units of kilograms per kilogram.
    
    Returns
    ----------------------
    theta  -  pint.Quantity
        The potential temperature of the given parcel, in Kelvin.
    '''

    T = _ensure_arrays(T)
    p = _ensure_arrays(p)
    r_v = _ensure_arrays(r_v)
    
    return T * (constants.p_0/p)**((constants.R_d/constants.c_p)*(1 - 0.28 * r_v))

def poissons_theta_to_T(theta : Union[int, float, np.ndarray], p : Union[int, float, np.ndarray], r_v : Union[int, float, np.ndarray]) -> np.ndarray:
    '''
    Parameters
    ----------------------
    theta  -  int, float, np.ndarray, or pint.Quantity
        The potential temperature of the parcel to find the temperature of.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    p  -  int, float, np.ndarray, or pint.Quantity
        The pressure of the parcel to find the temperature of.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    r_v  -  int, float, np.ndarray, or pint.Quantity
        The mixing ratio of the parcel to find the temperature of.
        If provided as a pint.Quantity, may be in any units of mass ratio. If provided as an int or float, expects units of kilograms per kilogram.
    
    Returns
    ----------------------
    T  -  pint.Quantity
        The temperature of the given parcel at reference pressure p, in Kelvin.
    '''

    theta = _ensure_arrays(theta)
    p = _ensure_arrays(p)
    r_v = _ensure_arrays(r_v)
    
    return theta / (constants.p_0/p)**((constants.R_d/constants.c_p)*(1 - 0.28 * r_v))

def calc_e_s(T : Union[int, float, np.ndarray]) -> np.ndarray:
    '''
    Parameters
    ----------------------
    T  -  int, float, np.ndarray, or pint.Quantity
        The temperature at which to find the saturation vapor pressure.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    Returns
    ----------------------
    e_s  -  pint.Quantity
        The saturation vapor pressure for the given temperature, in Pascals.
    '''

    T = _ensure_arrays(T)
    
    term_1 = constants.L_tp/constants.R_v
    term_2 = (1/constants.T_tp) - (1/T)
    term_3 = ((constants.c_pl - constants.c_pv)/constants.R_v)
    term_4 = constants.T_tp/T
    term_5 = np.log(T/constants.T_tp)
    return constants.e_s_tp * np.e**((term_1 * term_2) + (term_3 * (1 - term_4 - term_5)))


def r_v_from_e(e : Union[int, float, np.ndarray], p : Union[int, float, np.ndarray]) -> np.ndarray:
    e = _ensure_arrays(e)
    p = _ensure_arrays(p)

    r_v = ((constants.R_d/constants.R_v)*e)/(p - e)
    return r_v

def e_from_r_v(r_v : Union[int, float, np.ndarray], p : Union[int, float, np.ndarray]) -> np.ndarray:
    r_v = _ensure_arrays(r_v)
    p = _ensure_arrays(p)

    e = ((r_v * p)/((constants.R_d/constants.R_v) + r_v))
    return e

def r_v_from_T_d(T_d : Union[int, float, np.ndarray], p : Union[int, float, np.ndarray]) -> np.ndarray:
    T_d = _ensure_arrays(T_d)
    p = _ensure_arrays(p)
    
    e = calc_e_s(T_d)
    r_v = r_v_from_e(e, p)
    return r_v

def r_v_from_RH(T : Union[int, float, np.ndarray], RH : Union[int, float, np.ndarray]) -> np.ndarray:
    T = _ensure_arrays(T)
    RH = _ensure_arrays(RH)
    
    e_s = calc_e_s(T)
    r_v = RH * r_v_from_e(e_s)
    return r_v

def e_from_q_v(q_v : Union[int, float, np.ndarray], p : Union[int, float, np.ndarray]) -> np.ndarray:
    q_v = _ensure_arrays(q_v)
    p = _ensure_arrays(p)

    e = (q_v * p)/((constants.R_d/constants.R_v) + q_v * (1 - (constants.R_d/constants.R_v)))
    return e

def e_from_rho_v(rho_v : Union[int, float, np.ndarray], T : Union[int, float, np.ndarray]) -> np.ndarray:
    rho_v = _ensure_arrays(rho_v)
    T = _ensure_arrays(T)

    e = rho_v * constants.R_v * T
    return e