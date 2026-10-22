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

def Monin_Obukhov_L(theta_bar : Union[int, float, np.ndarray], u_star : Union[int, float, np.ndarray], theta_star : Union[int, float, np.ndarray]) -> np.ndarray:
    theta_bar = _ensure_arrays(theta_bar)
    u_star = _ensure_arrays(u_star)
    theta_star = _ensure_arrays(theta_star)

    if theta_star != 0:
        L = (theta_bar * u_star**2)/(constants.von_Karman * constants.g_0 * theta_star)
    else:
        L = _ensure_arrays(np.inf)

    return L

def stars_alt(theta_z : Union[int, float, np.ndarray], theta_g : Union[int, float, np.ndarray], r_v_z : Union[int, float, np.ndarray], r_v_g : Union[int, float, np.ndarray], z_s : Union[int, float, np.ndarray], z_0 : Union[int, float, np.ndarray], u_bar : Union[int, float, np.ndarray], v_bar : Union[int, float, np.ndarray]) -> np.ndarray:
    theta_z = _ensure_arrays(theta_z)
    theta_g = _ensure_arrays(theta_g)
    r_v_z = _ensure_arrays(r_v_z)
    r_v_g = _ensure_arrays(r_v_g)
    z_s = _ensure_arrays(z_s)
    z_0 = _ensure_arrays(z_0)
    u_bar = _ensure_arrays(u_bar)
    v_bar = _ensure_arrays(v_bar)

    drag = constants.von_Karman**2 / (np.log(z_s/z_0))**2
    drag_ratio = 0.74
    V = np.array(np.sqrt(u_bar**2 + v_bar**2))

    theta_bar = (theta_g + theta_z)/2
    dthetatz = (theta_z - theta_g)/(z_s)
    dudz = (u_bar)/(z_s)
    dvdz = (v_bar)/(z_s)

    R_i = bulk_richardson_layer(theta_bar, dthetatz, dudz, dvdz)
    const_b = 9.4
    const_b_prime = 4.7
    const_c_star_momentum = 7.4
    const_c_star_heat = 5.3

    if R_i < 0:
        F_m = 1 - ((const_b * R_i)/(1 + const_c_star_momentum * drag * const_b * np.sqrt(z_s/z_0) * np.sqrt(np.abs(R_i))))
        F_h = 1 - ((const_b * R_i)/(1 + const_c_star_heat * drag * const_b * np.sqrt(z_s/z_0) * np.sqrt(np.abs(R_i))))
    else:
        F_m = 1/((1 + const_b_prime * R_i)**2)
        F_h = 1/((1 + const_b_prime * R_i)**2)
    
    
    u_star_squared = drag * V**2 * F_m
    u_star = np.sqrt(u_star_squared) * np.sign(F_m)
    if u_star == 0:
        theta_star = 0
        r_v_star = 0
    else:
        theta_star = ((drag/drag_ratio) * V * (theta_z - theta_g) * F_h)/u_star
        r_v_star = ((drag/drag_ratio) * V * (r_v_z - r_v_g) * F_h)/u_star
    theta_star = theta_star

    return u_star, theta_star, r_v_star

def fluxes_surface_layer_alt(u_star : Union[int, float, np.ndarray], theta_star : Union[int, float, np.ndarray], r_v_star : Union[int, float, np.ndarray], u_z_s : Union[int, float, np.ndarray], v_z_s : Union[int, float, np.ndarray]) -> tuple[np.ndarray]:
    u_star = _ensure_arrays(u_star)
    theta_star = _ensure_arrays(theta_star)
    r_v_star = _ensure_arrays(r_v_star)
    u_z_s = _ensure_arrays(u_z_s)
    v_z_s = _ensure_arrays(v_z_s)

    mu = np.arctan2(v_z_s, u_z_s)

    uw = -u_star**2 * np.cos(mu)
    vw = -u_star**2 * np.sin(mu)
    thetaw = -u_star * theta_star
    r_vw = -u_star * r_v_star

    return uw, vw, thetaw, r_vw

def bulk_richardson_layer(theta_bar : Union[int, float, np.ndarray], dthetadz : Union[int, float, np.ndarray], dudz : Union[int, float, np.ndarray], dvdz : Union[int, float, np.ndarray]) -> np.ndarray:
    theta_bar = _ensure_arrays(theta_bar)
    dthetadz = _ensure_arrays(dthetadz)
    dudz = _ensure_arrays(dudz)
    dvdz = _ensure_arrays(dvdz)

    R_i = ((constants.g_0 * dthetadz))/(theta_bar * (dudz**2 + dvdz**2 + 1e-6))

    return R_i

def boundary_layer_height_tendency(u_star : Union[int, float, np.ndarray], theta_star : Union[int, float, np.ndarray], theta_field : Union[int, float, np.ndarray], z_field : Union[int, float, np.ndarray], theta_g : Union[int, float, np.ndarray], z_s : Union[int, float, np.ndarray], z_i : Union[int, float, np.ndarray], z_0 : Union[int, float, np.ndarray], lat : Union[int, float, np.ndarray]) -> np.ndarray:
    u_star = _ensure_arrays(u_star)
    theta_star = _ensure_arrays(theta_star)
    theta_field = _ensure_arrays(theta_field)
    theta_g = _ensure_arrays(theta_g)
    z_s = _ensure_arrays(z_s)
    z_i = _ensure_arrays(z_i)
    z_0 = _ensure_arrays(z_0)

    theta_z_s = interpolate.interp1d(z_field, theta_field)(z_s)
    theta_above_z_i = interpolate.interp1d(z_field, theta_field)(z_i + 25)
    theta_below_z_i = interpolate.interp1d(z_field, theta_field)(z_i - 25)
    dtheta_idz = (theta_above_z_i - theta_below_z_i)/50

    f = 2 * constants.omega * np.sin(np.deg2rad(lat))

    w_star = wstar(theta_z_s, u_star, theta_star, z_s)
    dz_idt = (1.8 * (w_star**3 + 1.1 * u_star**3 - 3.3 * u_star**2 * f * z_i))/(constants.g_0 * (z_i**2/theta_z_s) * dtheta_idz + 9 * w_star**2 + 7.2 * u_star**2)

    return dz_idt

def wstar(theta_z : Union[int, float, np.ndarray], u_star : Union[int, float, np.ndarray], theta_star : Union[int, float, np.ndarray], z : Union[int, float, np.ndarray]) -> np.ndarray:
    theta_z = _ensure_arrays(theta_z)
    u_star = _ensure_arrays(u_star)
    theta_star = _ensure_arrays(theta_star)
    z = _ensure_arrays(z)
    
    if theta_star <= 0:
        w_star = (-(constants.g_0/theta_z) * u_star * theta_star * z)**(1/3)
    else:
        w_star = 0
    
    return w_star

def eddy_momentum_diffusivity(theta_field : Union[int, float, np.ndarray], theta_g : Union[int, float, np.ndarray], z_field : Union[int, float, np.ndarray], z_s : Union[int, float, np.ndarray], z_i : Union[int, float, np.ndarray], u_bar_field : Union[int, float, np.ndarray], v_bar_field : Union[int, float, np.ndarray], z_0 : Union[int, float, np.ndarray], L : Union[int, float, np.ndarray], u_star : Union[int, float, np.ndarray], height_index : int) -> np.ndarray:
    theta_field = _ensure_arrays(theta_field)
    theta_g = _ensure_arrays(theta_g)
    L = _ensure_arrays(L)
    u_star = _ensure_arrays(u_star)
    z_field = _ensure_arrays(z_field)
    z_s = _ensure_arrays(z_s)
    z_i = _ensure_arrays(z_i)
    z_0 = _ensure_arrays(z_0)
    u_bar_field = _ensure_arrays(u_bar_field)
    v_bar_field = _ensure_arrays(v_bar_field)

    dtheta = (theta_field[1:][height_index[1:]] - theta_field[:-1][height_index[1:]])
    theta_bar = (theta_field[1:][height_index[1:]] + theta_field[:-1][height_index[1:]])/2
    du = (u_bar_field[1:][height_index[1:]] - u_bar_field[:-1][height_index[1:]])
    u_bar = (u_bar_field[1:][height_index[1:]] + u_bar_field[:-1][height_index[1:]])/2
    dv = (v_bar_field[1:][height_index[1:]] - v_bar_field[:-1][height_index[1:]])
    v_bar = (v_bar_field[1:][height_index[1:]] + v_bar_field[:-1][height_index[1:]])/2
    dz = (z_field[1:][height_index[1:]] - z_field[:-1][height_index[1:]])
    z_bar = (z_field[1:][height_index[1:]] + z_field[:-1][height_index[1:]]) / 2
    dthetadz = dtheta/dz
    dudz = du/dz
    dvdz = dv/dz
    
    unstable = ((z_bar/L) <= 0) & ~(L == np.inf)

    K_m = np.zeros(dthetadz.size)
    if L < 0:
        dKmdz = (-15 * constants.von_Karman * u_star * z_s)/(4 * L * (1 - 15 * ((z_s)/L)))
        K_Ms = ((constants.von_Karman * u_star * z_s)/((1 - 15 * (z_s/L))**(-1/4)))
        K_m[unstable] = 0.001 + ((z_i - z_bar[unstable])**2/(z_i - z_s)**2) * (K_Ms - 0.001 + (z_bar[unstable] - z_s) * (dKmdz + 2 * ((K_Ms - 0.001)/(z_i - z_s))))
        #print(L, (z_i - z_bar[unstable])**2/(z_i - z_s)**2, dKmdz)
    else:
        K_m[unstable] = 0

    R_ic = 0.115 * (dz[~unstable])**0.175
    R_i = bulk_richardson_layer(theta_bar[~unstable], dthetadz[~unstable], dudz[~unstable], dvdz[~unstable])
    l = np.zeros(len(R_i))

    high = z_bar[~unstable] > 200
    low_Ri = R_i <= R_ic
    l[(low_Ri) & (~high)] = constants.von_Karman * z_bar[~unstable][(low_Ri) & (~high)]
    l[(low_Ri) & (high)] = np.ones(len(l[(low_Ri) & (high)])) * 70
    K_m[~unstable][low_Ri] = 1.1 * ((R_ic[low_Ri] - R_i[low_Ri])/(R_ic[low_Ri])) * l[low_Ri]**2 * np.sqrt(dudz[~unstable][low_Ri]**2 + dvdz[~unstable][low_Ri]**2)
    K_m[~unstable][~low_Ri] = np.zeros(len(K_m[~unstable][~low_Ri]))

    return K_m

def eddy_heat_diffusivity(theta_field : Union[int, float, np.ndarray], theta_g : Union[int, float, np.ndarray], z_field : Union[int, float, np.ndarray], z_s : Union[int, float, np.ndarray], z_i : Union[int, float, np.ndarray], u_bar_field : Union[int, float, np.ndarray], v_bar_field : Union[int, float, np.ndarray], z_0 : Union[int, float, np.ndarray], L : Union[int, float, np.ndarray], u_star : Union[int, float, np.ndarray], height_index : int) -> np.ndarray:
    theta_field = _ensure_arrays(theta_field)
    theta_g = _ensure_arrays(theta_g)
    L = _ensure_arrays(L)
    u_star = _ensure_arrays(u_star)
    z_field = _ensure_arrays(z_field)
    z_s = _ensure_arrays(z_s)
    z_i = _ensure_arrays(z_i)
    z_0 = _ensure_arrays(z_0)
    u_bar_field = _ensure_arrays(u_bar_field)
    v_bar_field = _ensure_arrays(v_bar_field)

    dtheta = (theta_field[1:][height_index[1:]] - theta_field[:-1][height_index[1:]])
    theta_bar = (theta_field[1:][height_index[1:]] + theta_field[:-1][height_index[1:]])/2
    du = (u_bar_field[1:][height_index[1:]] - u_bar_field[:-1][height_index[1:]])
    u_bar = (u_bar_field[1:][height_index[1:]] + u_bar_field[:-1][height_index[1:]])/2
    dv = (v_bar_field[1:][height_index[1:]] - v_bar_field[:-1][height_index[1:]])
    v_bar = (v_bar_field[1:][height_index[1:]] + v_bar_field[:-1][height_index[1:]])/2
    dz = (z_field[1:][height_index[1:]] - z_field[:-1][height_index[1:]])
    z_bar = (z_field[1:][height_index[1:]] + z_field[:-1][height_index[1:]]) / 2
    dthetadz = dtheta/dz
    dudz = du/dz
    dvdz = dv/dz

    unstable = ((z_bar/L) <= 0) & ~(L == np.inf)

    K_h = np.zeros(dthetadz.size)
    if L < 0:
        dKhdz = (-9 * constants.von_Karman * u_star * z_s)/(1.48 * L * (1 - 9 * ((z_s)/L)))
        K_Hs = ((constants.von_Karman * u_star * z_s)/((1 - 9 * (z_s/L))**(-1/2)))
        K_h[unstable] = 0.001 + ((z_i - z_bar[unstable])**2/(z_i - z_s)**2) * (K_Hs - 0.001 + (z_bar[unstable] - z_s) * (dKhdz + 2 * ((K_Hs - 0.001)/(z_i - z_s))))
    else:
        K_h[unstable] = 0

    R_ic = 0.115 * (dz[~unstable])**0.175
    R_i = bulk_richardson_layer(theta_bar[~unstable], dthetadz[~unstable], dudz[~unstable], dvdz[~unstable])
    l = np.zeros(len(R_i))

    high = z_bar[~unstable] > 200
    low_Ri = R_i <= R_ic
    l[(low_Ri) & (~high)] = constants.von_Karman * z_bar[~unstable][(low_Ri) & (~high)]
    l[(low_Ri) & (high)] = np.ones(len(l[(low_Ri) & (high)])) * 70
    K_h[~unstable][low_Ri] = 1.1 * ((R_ic[low_Ri] - R_i[low_Ri])/(R_ic[low_Ri])) * l[low_Ri]**2 * np.sqrt(dudz[~unstable][low_Ri]**2 + dvdz[~unstable][low_Ri]**2)
    K_h[~unstable][~low_Ri] = np.zeros(len(K_h[~unstable][~low_Ri]))

    return K_h

def sensible_heat(p_1, T_1, u_star, theta_star) -> np.ndarray:
    p_1 = _ensure_arrays(p_1)
    T_1 = _ensure_arrays(T_1)
    theta_star = _ensure_arrays(theta_star)
    u_star = _ensure_arrays(u_star)

    H = - (p_1 / (constants.R_d * T_1)) * constants.c_p * u_star * theta_star
    return H