import hapi
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset as NCDF
from pint import Quantity
import sys
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

def create_radiation_database(download_new_data : bool = False) -> None:
    hapi.db_begin("local_radiation_data")

    if download_new_data:
        hapi.fetch("CO2", 2, 1, 1, 100000)
        hapi.fetch("N2", 22, 1, 1, 100000)
        hapi.fetch("H2O", 1, 1, 1, 100000)
        hapi.fetch("O2", 7, 1, 1, 100000)
        hapi.fetch("O3", 3, 1, 1, 100000)

    nu = np.arange(1, 100000.0001, 0.01)

    _, abscoef_CO2 = hapi.absorptionCoefficient_Lorentz(SourceTables = ["CO2"], OmegaGrid = nu, HITRAN_units = False)
    _, abscoef_N2 = hapi.absorptionCoefficient_Lorentz(SourceTables = ["N2"], OmegaGrid = nu, HITRAN_units = False)
    _, abscoef_H2O = hapi.absorptionCoefficient_Lorentz(SourceTables = ["H2O"], OmegaGrid = nu, HITRAN_units = False)
    _, abscoef_O2 = hapi.absorptionCoefficient_Lorentz(SourceTables = ["O2"], OmegaGrid = nu, HITRAN_units = False)
    _, abscoef_O3 = hapi.absorptionCoefficient_Lorentz(SourceTables = ["O3"], OmegaGrid = nu, HITRAN_units = False)

    #_, absorbtion_CO2 = hapi.absorptionSpectrum(nu, abscoef_CO2, Environment = {"l":5000})
    #_, absorbtion_N2 = hapi.absorptionSpectrum(nu, abscoef_N2, Environment = {"l":5000})
    #_, absorbtion_H2O = hapi.absorptionSpectrum(nu, abscoef_H2O, Environment = {"l":5000})
    #_, absorbtion_O2 = hapi.absorptionSpectrum(nu, abscoef_O2, Environment = {"l":5000})
    #_, absorbtion_O2 = hapi.absorptionSpectrum(nu, abscoef_O2, Environment = {"l":5000})

    window_size = 1000
    weights = np.ones(window_size)/window_size

    wavelength = 1/nu

    DS = NCDF("SABRES-PBL-Model/radiation-data/radiation.nc", "w", format='NETCDF4')
    DS.title = "SSABLeModS Radiation Database"
    DS.description = "The absorbtivity spectra for the gasses used within the SSABLeModS PBL Model, retrieved via the HITRAN API (HAPI) version 1."
    DS.version = "1.0.0"

    nu_dim = DS.createDimension("wavenum", nu.size)


    nu_var = DS.createVariable("wavenumber", np.float64, ("wavenum"))
    nu_var[:] = Quantity(nu[:], "cm^(-1)").to("m^(-1)").m
    nu_var.units = "m^(-1)"

    abscoef_CO2_var = DS.createVariable("CO2_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_CO2_var[:] = abscoef_CO2[:]
    abscoef_CO2_var.units = "dimensionless"

    abscoef_N2_var = DS.createVariable("N2_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_N2_var[:] = abscoef_N2[:]
    abscoef_N2_var.units = "dimensionless"

    abscoef_H2O_var = DS.createVariable("H2O_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_H2O_var[:] = abscoef_H2O[:]
    abscoef_H2O_var.units = "dimensionless"

    abscoef_O2_var = DS.createVariable("O2_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_O2_var[:] = abscoef_O2[:]
    abscoef_O2_var.units = "dimensionless"

    abscoef_O3_var = DS.createVariable("O3_absorbtion_coefficient", np.float64, ("wavenum"))
    abscoef_O3_var[:] = abscoef_O3[:]
    abscoef_O3_var.units = "dimensionless"

    abscoef_CO2_smoo_var = DS.createVariable("CO2_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_CO2_smoo_var[:] = np.convolve(abscoef_CO2, weights, mode = "same")[:]
    abscoef_CO2_smoo_var.units = "dimensionless"

    abscoef_N2_smoo_var = DS.createVariable("N2_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_N2_smoo_var[:] = np.convolve(abscoef_N2, weights, mode = "same")[:]
    abscoef_N2_smoo_var.units = "dimensionless"

    abscoef_H2O_smoo_var = DS.createVariable("H2O_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_H2O_smoo_var[:] = np.convolve(abscoef_H2O, weights, mode = "same")[:]
    abscoef_H2O_smoo_var.units = "dimensionless"

    abscoef_O2_smoo_var = DS.createVariable("O2_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_O2_smoo_var[:] = np.convolve(abscoef_O2, weights, mode = "same")[:]
    abscoef_O2_smoo_var.units = "dimensionless"

    abscoef_O3_smoo_var = DS.createVariable("O3_absorbtion_coefficient_smoothed", np.float64, ("wavenum"))
    abscoef_O3_smoo_var[:] = np.convolve(abscoef_O3, weights, mode = "same")[:]
    abscoef_O3_smoo_var.units = "dimensionless"

    DS.close()

def ground_incoming_shortwave(ToA_incoming_exitance : np.ndarray, current_model_time : datetime, lat : float, shortwave_reduction : float) -> np.ndarray:
    local_hour = current_model_time.hour + current_model_time.minute / 60 + current_model_time.second / 3600
    solar_declination = -23.44 * np.cos((np.deg2rad(360)/365) * (int(current_model_time.strftime("%j")) + local_hour/24) + 10) # Very simple approximation
    solar_hour_angle = (local_hour - 12) * (360/24)
    solar_zenith_angle = np.arccos(np.sin(np.deg2rad(lat)) * np.sin(np.deg2rad(solar_declination)) + np.cos(np.deg2rad(lat)) * np.cos(np.deg2rad(solar_declination)) * np.cos(np.deg2rad(solar_hour_angle)))
    if -90 <= np.rad2deg(solar_zenith_angle) <= 90:
        incoming_shortwave = ToA_incoming_exitance * (0.6 * np.cos(solar_zenith_angle) + 0.2 * np.cos(solar_zenith_angle)**2)  * (1 - shortwave_reduction)
    else:
        incoming_shortwave = np.zeros_like(ToA_incoming_exitance)
    return incoming_shortwave

def get_absorbtion_coefficient(min_wavenumber : Union[int, float, np.ndarray], max_wavenumber : Union[int, float, np.ndarray], gasses = list[str]) -> list[np.ndarray]:
    ret = []
    dr = os.path.dirname(os.path.realpath(__file__))
    Data = NCDF(f"{dr}/radiation-data/radiation.nc", "r", format='NETCDF4')
    wavenumber = Data["wavenumber"][:]
    selection = (wavenumber >= min_wavenumber) & (wavenumber <= max_wavenumber)

    for gas in gasses:
        ret.append(Data[f"{gas.upper()}_absorbtion_coefficient"][selection])

    return ret, wavenumber[selection]

def weighted_absorbtion_coefficients(e : Union[int, float, np.ndarray], p : Union[int, float, np.ndarray], absorbtion_coefficient_CO2 : np.ndarray, absorbtion_coefficient_H2O : np.ndarray, absorbtion_coefficient_O2 : np.ndarray, absorbtion_coefficient_N2 : np.ndarray) -> np.ndarray:
    e = _ensure_arrays(e)
    p = _ensure_arrays(p)

    p_2d = np.tile(p, (absorbtion_coefficient_N2.size, 1))
    #p_2d = (p_2d[:,1:] + p_2d[:,:-1]) / 2
    e_2d = np.tile(e, (absorbtion_coefficient_N2.size, 1))
    #e_2d = (e_2d[:,1:] + e_2d[:,:-1]) / 2

    absorbtion_coefficient_N2_2d = np.tile(absorbtion_coefficient_N2, (e.size, 1)).transpose()
    absorbtion_coefficient_O2_2d = np.tile(absorbtion_coefficient_O2, (e.size, 1)).transpose()
    absorbtion_coefficient_CO2_2d = np.tile(absorbtion_coefficient_CO2, (e.size, 1)).transpose()
    absorbtion_coefficient_H2O_2d = np.tile(absorbtion_coefficient_H2O, (e.size, 1)).transpose()

    tau = (((p_2d - e_2d)/p_2d) * 0.000426 * absorbtion_coefficient_CO2_2d) + (((p_2d - e_2d)/p_2d) * 0.78084 * absorbtion_coefficient_N2_2d) + (((p_2d - e_2d)/p_2d) * 0.20948 * absorbtion_coefficient_O2_2d) + ((e_2d/p_2d) * absorbtion_coefficient_H2O_2d)

    return tau

def ozone_absorbtion_coefficients(ozone_layer_average_ozone : float, ozone_layer_average_water_vapor : float, absorbtion_coefficient_CO2 : np.ndarray, absorbtion_coefficient_H2O : np.ndarray, absorbtion_coefficient_O2 : np.ndarray, absorbtion_coefficient_N2 : np.ndarray, absorbtion_coefficient_O3 : np.ndarray) -> np.ndarray:

    tau = 0.000426 * absorbtion_coefficient_CO2 + 0.78084 * absorbtion_coefficient_N2 + 0.20948 * absorbtion_coefficient_O2 + ozone_layer_average_water_vapor * absorbtion_coefficient_H2O + ozone_layer_average_ozone * absorbtion_coefficient_O3

    return tau

def point_to_point_exitance(T : Union[int, float, np.ndarray], layer_depth : Union[int, float, np.ndarray], ground_emissivity : float, absorbtion_coefficients : Union[int, float, np.ndarray], wavenumber : Union[int, float, np.ndarray], M_in : np.ndarray, ozone_absorbtion_coeff : np.ndarray, ozone_layer_temperature : float, ozone_layer_depth : float) -> np.ndarray:
    T = _ensure_arrays(T)
    layer_depth = _ensure_arrays(layer_depth)
    absorbtion_coefficients = _ensure_arrays(absorbtion_coefficients)

    M = spectral_exitance(T, wavenumber)

    #M_path_upwards = np.zeros((M.shape[0], M.shape[1]-1))
    #M_path_downwards = np.zeros((M.shape[0], M.shape[1]-1))
    M_net = np.zeros_like(M)
    transmittance = np.exp(-absorbtion_coefficients*(layer_depth[np.newaxis,:]))
    ozone_transmittance = np.exp(-ozone_absorbtion_coeff*ozone_layer_depth)
    ozone_emission = spectral_exitance(np.array(ozone_layer_temperature, ndmin = 1), wavenumber)
    absobrtivity = 1 - transmittance

    up = M[:,0] * ground_emissivity
    M_net[:,0] = -M[:,0] * ground_emissivity
    for i in range(1, layer_depth.size - 1):
        transmitted = up * transmittance[:,i]
        produced = M[:,i] * absobrtivity[:,i]
        absorbed = up * absobrtivity[:,i]
        up = transmitted + produced
        M_net[:,i] = absorbed - produced
    


    down = M_in[:,0] * ozone_transmittance + ozone_emission[:,0] * (1 - ozone_transmittance)
    M_net[:,-1] = down * absobrtivity[:,-1]
    down = down * transmittance[:,-1] + M[:,-1] * absobrtivity[:,-1]
    for i in range(layer_depth.size-1, 0, -1):
        transmitted = down * transmittance[:,i]
        produced = M[:,i] * absobrtivity[:,i]
        absorbed = down * absobrtivity[:,i]
        down = transmitted + produced
        M_net[:,i] = M_net[:,i] + (absorbed - produced)

    absorbed = down * ground_emissivity
    M_net[:,0] = M_net[:,0] + (absorbed)

    return M_net

def weighted_absorbtivity_spectra(e : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity], emissivity_CO2 : np.ndarray, emissivity_H2O : np.ndarray, emissivity_O2 : np.ndarray, emissivity_N2 : np.ndarray) -> Quantity:
    e = _ensure_arrays(e, "Pa")
    p = _ensure_arrays(p, "Pa")

    p_2d = np.tile(p, (emissivity_N2.size, 1))
    e_2d = np.tile(e, (emissivity_N2.size, 1))
    emissivity_N2_2d = np.tile(emissivity_N2, (e.size, 1)).transpose()
    emissivity_O2_2d = np.tile(emissivity_O2, (e.size, 1)).transpose()
    emissivity_CO2_2d = np.tile(emissivity_CO2, (e.size, 1)).transpose()
    emissivity_H2O_2d = np.tile(emissivity_H2O, (e.size, 1)).transpose()

    absorbtivity_spectra = 1 - np.e**(-(-((((p_2d - e_2d)/p_2d) * 0.000426)[:,:] * np.log(1-emissivity_CO2_2d[:,:]+0.000000001))-((((p_2d - e_2d)/p_2d) * 0.78084)[:,:] * np.log(1-emissivity_N2_2d[:,:]+0.000000001))-((((p_2d - e_2d)/p_2d) * 0.20948)[:,:] * np.log(1-emissivity_O2_2d[:,:]+0.000000001))-((e_2d/p_2d)[:,:] * np.log(1-emissivity_H2O_2d[:,:]+0.000000001))))

    return absorbtivity_spectra

@nb.njit(cache=True,fastmath=True,parallel=True)
def planck(T : np.ndarray, wavenumber : np.ndarray, const_1 : float, const_2 : float) -> np.ndarray:
    num_layers = T.size
    num_wn = wavenumber.size
    B = np.empty((num_wn, num_layers))

    for i in nb.prange(num_wn):
        wn = wavenumber[i]
        wn_cubed = wn*wn*wn
        for j in range(num_layers):
            exponent = (const_1 * wn)/(T[j])
            B[i,j] = (const_2 * wn_cubed) / (np.exp(exponent) - 1)
    
    return B

def spectral_exitance(T : Union[int, float, np.ndarray], wavenumber : Union[int, float, np.ndarray]) -> np.ndarray:
    #T = _ensure_unit_aware_arrays(T, "K")
    #wavenumber = _ensure_unit_aware_arrays(wavenumber, "m^(-1)")

    B = planck(T, wavenumber, ((constants.Planck * constants.c)/constants.Boltzmann), (2 * constants.Planck * constants.c**2))
    M = B * np.pi

    return M

def filtered_exitance(M : Union[int, float, np.ndarray, Quantity], absorbtivity_spectra : np.ndarray) -> Quantity:
    M = _ensure_arrays(M, "W m^(-1)")

    absorbtivity_spectra2d = np.tile(absorbtivity_spectra, (M.shape[1], 1)).transpose()

    M_prime = M * absorbtivity_spectra2d

    return M_prime

def irradiance(M_prime : Union[int, float, np.ndarray]) -> np.ndarray:
    M_prime = _ensure_arrays(M_prime)

    I = np.sum(M_prime, axis = 0) * Quantity(10, "cm^(-1)").to("m^(-1)").m

    return I

def net_radiant_longwave_power(M_prime : Union[int, float, np.ndarray, Quantity], T_g : Union[int, float, np.ndarray, Quantity], ground_emissivity : float, filters : list[np.ndarray], wavenumber : Union[int, float, np.ndarray, Quantity], air_emissivity : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    M_prime = _ensure_arrays(M_prime, "W m^(-1)")
    T_g = _ensure_arrays(T_g, "K")
    wavenumber = _ensure_arrays(wavenumber, "m^(-1)")

    lower_filter_inner, upper_filter_inner, upper_absorbtive_filter, lower_absorbtive_filter = filters

    weighted_absorbtivity = np.expand_dims(weighted_absorbtivity, axis = 0)
    weighted_absorbtivity = np.tile(weighted_absorbtivity, (lower_filter_inner.shape[0], 1, 1))

    weighted_absorbtivity = np.expand_dims(weighted_absorbtivity, axis = 3)
    weighted_absorbtivity = np.tile(weighted_absorbtivity, (1, 1, 1, ()-1))

    weighted_absorbtivity_product_upper = np.where(upper_filter_inner[:,:,:,-1], np.nanprod(1 - np.where(upper_absorbtive_filter[:,:,:,:], weighted_absorbtivity, np.nan), axis = 3), np.nan)
    weighted_absorbtivity_product_lower = np.where(lower_filter_inner[:,:,:,-1], np.nanprod(1 - np.where(lower_absorbtive_filter[:,:,:,:], weighted_absorbtivity, np.nan), axis = 3), np.nan)

    M_prime = np.expand_dims(M_prime, axis = 0)
    M_prime = np.tile(M_prime, (lower_filter_inner.shape[0]-1, 1, 1))
    M_prime_upper = np.where(upper_filter_inner[1:,:,:,-1], M_prime, np.nan)
    M_prime_lower = np.where(lower_filter_inner[1:,:,:,-1], M_prime, np.nan)

    M_g = spectral_exitance(T_g, wavenumber)
    M_g = np.expand_dims(M_g, axis = 1)
    M_g = np.tile(M_g, (1, lower_filter_inner.shape[2]))
    M_g_prime = ground_emissivity * M_g

    M_prime_prime = np.nansum(weighted_absorbtivity_product_upper[1:,:,:] * M_prime_upper, axis = 0) + np.nansum(weighted_absorbtivity_product_lower[1:,:,:] * M_prime_lower, axis = 0)
    M_g_prime_prime = np.nansum(weighted_absorbtivity_product_upper[:weighted_absorbtivity_product_upper.shape[2],:,0].transpose() * M_g_prime, axis = 0)


    I_prime_air = np.sum(weighted_absorbtivity[0,:,:,-1] * M_prime_prime, axis = 0) * Quantity(1, "cm^(-1)").to_base_units()

    I_prime_g = np.sum(weighted_absorbtivity[0,:,:,-1] * M_g_prime_prime, axis = 0) * Quantity(1, "cm^(-1)").to_base_units()

    P = np.zeros(weighted_absorbtivity.shape[2])

    P[0] = (- irradiance(M_g_prime[:,0]) + ground_emissivity * I_prime_air[1:])# * Quantity(1, "m^2")
    P[1:] = (- 2 * irradiance(M_prime[1:]) + I_prime_air[1:] + I_prime_g[1:])# * Quantity(1, "m^2")

    del weighted_absorbtivity_product_lower
    del weighted_absorbtivity_product_upper
    del M_prime_lower
    del M_prime_upper
    del M_prime_prime
    del M_g_prime
    del M_g_prime_prime

    return P

def net_irradience_simplified(T : Union[int, float, np.ndarray, Quantity], e : Union[int, float, np.ndarray, Quantity], p : Union[int, float, np.ndarray, Quantity], ground_emissivity : float, epsilon_a : np.ndarray, wavenumber : Union[int, float, np.ndarray, Quantity]) -> Quantity:
    T = _ensure_arrays(T, "K")
    e = _ensure_arrays(e, "Pa")
    p = _ensure_arrays(p, "Pa")
    wavenumber = _ensure_arrays(wavenumber, "m^(-1)")


    M = spectral_exitance(T, wavenumber)
    #print(M.shape)
    #print(epsilon_a.shape)
    M_prime = np.zeros(M.shape)
    M_prime[:,0] = ground_emissivity * M[:,0].m
    M_prime[:,1:] = epsilon_a[:,1:] * M[:,1:].m
    M_prime = Quantity(M_prime, M.units)

    I = np.zeros(M.shape[1])
    I[0] = - irradiance(M_prime[:,0]).m + ground_emissivity * irradiance(M_prime[:,1]).m
    I[1] = - 2 * irradiance(M_prime[:,1]).m + irradiance(epsilon_a[:,1] * M_prime[:,0].m).m + irradiance(M_prime[:,2].m).m
    I[2:-1] = - 2 * irradiance(M_prime[:,2:-1]).m + irradiance(np.roll(M_prime[:,2:-1].m, (0, 1)) + np.roll(M_prime[:,2:-1].m, (0, -1))).m
    I[-1] = - irradiance(M_prime[:,-1]).m + irradiance(M_prime[:,-2]).m

    I = Quantity(I, "W m^(-2)")

    return I