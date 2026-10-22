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

def _ensure_arrays(input_quantity : Union[int, float, list, np.ndarray], units : Union[None, tuple[str]] = None) -> Union[np.ndarray, Quantity]:
    if isinstance(input_quantity, int) or isinstance(input_quantity, float) or isinstance(input_quantity, list) or isinstance(input_quantity, np.ndarray):
        input_quantity = np.array(input_quantity, ndmin = 1)
        if units is not None:
            input_quantity = Quantity(input_quantity, units[0]).to_base_units().to(units[1]).m
    return input_quantity

def get_config(filepath : str = f"{os.path.dirname(os.path.realpath(__file__))}/pbl.config") -> Union[dict, None]:
    if os.path.exists(filepath):
        cfg_dict = {}
        with open(filepath, "r") as F:
            for line in F.readlines():
                if line =="\n":
                    pass
                elif line.startswith("#") or line.startswith("\\"):
                    pass
                else:
                    splitline = line.split("=")
                    val, unit, datatype = [x.strip() for x in splitline[1].split(",")]
                    if datatype == "int":
                        cfg_dict[f"{splitline[0].strip()}"] = np.int64(val)
                    elif datatype == "float":
                        cfg_dict[f"{splitline[0].strip()}"] = np.float64(val)
                    elif datatype == "bool":
                        cfg_dict[f"{splitline[0].strip()}"] = bool(int(val))
                    
                    if not unit == "":
                        if unit == "degree":
                            cfg_dict[f"{splitline[0].strip()}"] = Quantity(cfg_dict[f"{splitline[0].strip()}"], unit).m
                        else:
                            cfg_dict[f"{splitline[0].strip()}"] = Quantity(cfg_dict[f"{splitline[0].strip()}"], unit).to_base_units().m
        return cfg_dict
    else:
        return None

def hypsometric_P_to_Z(T_bar : Union[int, float, np.ndarray], p_1 : Union[int, float, np.ndarray], p_2 : Union[int, float, np.ndarray], r_v_bar : Union[int, float, np.ndarray], z_1 : Union[None, int, float, np.ndarray] = None) -> np.ndarray:
    '''
    Parameters
    ----------------------
    T_bar  -  int, float, np.ndarray, or pint.Quantity
        The layer average temperature of the layer.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    p_1  -  int, float, np.ndarray, or pint.Quantity
        The pressure of the initial pressure level.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    p_2  -  int, float, np.ndarray, or pint.Quantity
        The pressure of the final pressure level.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    r_v_bar  -  int, float, np.ndarray, or pint.Quantity
        The layer average mixing ratio of the layer.
        If provided as a pint.Quantity, may be in any units of mass ratio. If provided as an int or float, expects units of kilograms per kilogram or dimensionless.
    
    z_1  -  None, int, float, np.ndarray, or pint.Quantity (Optional, defaults to None)
        The height at p_1.
        If provided as a pint.Quantity, may be in any units of length. If provided as an int or float, expects units of meters.

    Returns
    ----------------------
    z_2 OR dz  -  pint.Quantity
        If z_1 was specified, returns the height z_2, in meters.     
        If z_1 was not specified, returns the height difference dz between the two pressure levels, in meters.
    '''

    T_bar = _ensure_arrays(T_bar)
    p_1 = _ensure_arrays(p_1)
    p_2 = _ensure_arrays(p_2)
    r_v_bar = _ensure_arrays(r_v_bar)
    
    if z_1 is not None:
        z_1 = _ensure_arrays(z_1)
    
        return z_1 + ((constants.R_d * (T_bar * (1 + (0.61 * r_v_bar)))/constants.g_0) * np.log(p_1/p_2)) # returns z_2
    
    else:
        return (constants.R_d * (T_bar * (1 + (0.61 * r_v_bar)))/constants.g_0) * np.log(p_1/p_2) # returns dz
    
def hypsometric_Z_to_P(T_bar : Union[int, float, np.ndarray], z_1 : Union[int, float, np.ndarray], z_2 : Union[int, float, np.ndarray], r_v_bar : Union[int, float, np.ndarray], p_1 : Union[None, int, float, np.ndarray] = None, downward : Union[bool, np.ndarray] = False, profile : bool = True) -> np.ndarray:
    '''
    Parameters
    ----------------------
    T_bar  -  int, float, np.ndarray, or pint.Quantity
        The layer average temperature of the layer.
        If provided as a pint.Quantity, may be in any units of temperature. If provided as an int or float, expects units of Kelvin.
    
    z_1  -  int, float, np.ndarray, or pint.Quantity
        The height of the initial height level.
        If provided as a pint.Quantity, may be in any units of length. If provided as an int or float, expects units of meters.
    
    z_2  -  int, float, np.ndarray, or pint.Quantity
        The height of the final height level.
        If provided as a pint.Quantity, may be in any units of length. If provided as an int or float, expects units of meters.
    
    r_v_bar  -  int, float, np.ndarray, or pint.Quantity
        The layer average mixing ratio of the layer.
        If provided as a pint.Quantity, may be in any units of mass ratio. If provided as an int or float, expects units of kilograms per kilogram or dimensionless.
    
    p_1  -  None, int, float, np.ndarray, or pint.Quantity (Optional, defaults to None)
        The pressure at z_1.
        If provided as a pint.Quantity, may be in any units of pressure. If provided as an int or float, expects units of Pascals.
    
    downward - bool or np.ndarray (Optional, defaults to False)
        Whether the given p_1 is the higher (z) pressure or the lower (z) pressure.
        Must be the same length as z_1.
    
    profile - bool (Optional, defaults to True)
        Whether the given data represents a single column.
        If true, will perform the calculation recursively on the previous p_2 (only uses p_1[0]).
        If false, considers all calculations separately, requiring that p_1 be the same size as z_1.

    Returns
    ----------------------
    p_2 OR p_1/p_2  -  pint.Quantity
        If p_1 was specified, returns the pressure p_2, in pascals.     
        If p_1 was not specified, returns the ratio of the initial to final pressure levels, as a dimensionless number.
    '''
    
    T_bar = _ensure_arrays(T_bar)
    z_1 = _ensure_arrays(z_1)
    z_2 = _ensure_arrays(z_2)
    r_v_bar = _ensure_arrays(r_v_bar)
    
    if p_1 is not None:
        p_1 = _ensure_arrays(p_1)

        p_2 = np.array([])
        #print(z_1.m.size, downward)
        if (z_1.size == 1) and (type(downward) == bool):
            if downward:
                p_2 = np.append(p_2, (p_1*(np.e**(((z_2 - z_1) * constants.g_0)/(constants.R_d * (T_bar * (1 + (0.61 * r_v_bar)))))))) # returns a p_2 that is below the original p_1
            else:
                p_2 = np.append(p_2, (p_1/(np.e**(((z_2 - z_1) * constants.g_0)/(constants.R_d * (T_bar * (1 + (0.61 * r_v_bar)))))))) # returns p_2
        else:
            for i in range(len(z_1)):
                if profile:
                    if p_1.size != 1:
                        p_1 = p_1[0]
                    if i == 0:
                        if downward[i]:
                            p_2 = np.append(p_2, (p_1*(np.e**(((z_2[i] - z_1[i]) * constants.g_0)/(constants.R_d * (T_bar[i] * (1 + (0.61 * r_v_bar[i])))))))) # returns a p_2 that is below the original p_1
                        else:
                            p_2 = np.append(p_2, (p_1/(np.e**(((z_2[i] - z_1[i]) * constants.g_0)/(constants.R_d * (T_bar[i] * (1 + (0.61 * r_v_bar[i])))))))) # returns p_2
                    else:
                        if downward[i]:
                            p_2 = np.append(p_2, (p_2[i-1]*(np.e**(((z_2[i] - z_1[i]) * constants.g_0)/(constants.R_d * (T_bar[i] * (1 + (0.61 * r_v_bar[i])))))))) # returns a p_2 that is below the original p_1
                        else:
                            p_2 = np.append(p_2, (p_2[i-1]/(np.e**(((z_2[i] - z_1[i]) * constants.g_0)/(constants.R_d * (T_bar[i] * (1 + (0.61 * r_v_bar[i])))))))) # returns p_2

        return p_2
    else:
        return 1/(np.e**(((z_2 - z_1) * constants.g_0)/(constants.R_d * (T_bar * (1 + (0.61 * r_v_bar)))))) # returns p_2/p_1, i.e. the ratio of the pressures at z_1 to z_2





if __name__ == "__main__":
    A =  _ensure_arrays([55], "degC", "degF")
    print(A, type(A), A.size)