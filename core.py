import numpy as np
from netCDF4 import Dataset as NCDF
from netCDF4 import date2num, num2date
import os
from typing import Union
from datetime import datetime, timedelta
from pytz import timezone
import grid
import ingest
import dynamics
import constants
import radiation
import thermo
import turbulence
from pint import Quantity
import matplotlib.pyplot as plt
from timezonefinder import TimezoneFinder
import sys
import pandas as pd
from time import sleep
import shutil
from scipy import interpolate
import figures
from utility_equations import _ensure_arrays, get_config

np.set_printoptions(linewidth=350)

def progress(total_time : float, timestep : float, current_step : int) -> None:
    sys.stdout.write("\r")

    num_steps = np.ceil(total_time / timestep)
    fraction = current_step / num_steps


    sys.stdout.write("[%-40s] %s" % ('='*int(40*fraction), f"{current_step:>4}/{int(num_steps):<4}"))

def run_model(config : dict, prepared_filename : str, **kwargs) -> str:
    if "progress_tracker" in kwargs.keys():
        progress_tracker = kwargs['progress_tracker']
    else:
        progress_tracker = True
    
    if "fudge_time" in kwargs.keys():
        fudge_time = timedelta(seconds = kwargs["fudge_time"].to("s").m)
    else:
        fudge_time = timedelta(seconds = 0)

    init_time = datetime.now(timezone("UTC"))
    if progress_tracker:
        print("Initializing model, this may take a moment...", end = "\r")

    output_filename = prepared_filename.replace("INPUT", "OUTPUT")
    if os.path.isfile(output_filename):
        output_filename = output_filename.split(".")[0] + "__" + init_time.strftime("%H-%H-%S") + ".nc"
    shutil.copy(prepared_filename, output_filename)

    dynamics_schema = config["dynamics_schema"]

    moisture_processes_enabled = config["moisture_processes_enabled"]

    turbulence_schema = config["turbulence_schema"]
    radiation_schema = config["radiation_schema"]
    boundary_layer_height_schema = config["boundary_layer_height_schema"]

    soil_emissivity = config["soil_emissivity"]
    soil_density = config["soil_density"]
    soil_heat_capacity = config["soil_heat_capacity"]
    soil_depth = config["soil_depth"]
    
    ozone_layer_depth = config["ozone_layer_depth"]
    ozone_layer_average_ozone = config["ozone_layer_average_ozone"]
    ozone_layer_average_water_vapor = config["ozone_layer_average_water_vapor"]
    ozone_layer_temp = config["ozone_layer_temp"]

    nu_min = config["wavenumber_minimum"]
    nu_max = config["wavenumber_maximum"]
    nu = np.arange(nu_min, nu_max, 1000)

    Dataset = NCDF(output_filename, "r+", format = "NETCDF4")

    data_tz = Dataset["timezone"][0]
    current_time_index = np.count_nonzero(~np.isnan(Dataset["potential_temperature"][:,1])) - 1
    current_model_state_time_utc = num2date(Dataset["time"][0], Dataset["time"].units, only_use_cftime_datetimes=False).replace(tzinfo = timezone("UTC")) + fudge_time
    current_model_state_time_local = current_model_state_time_utc.astimezone(timezone(data_tz))
    timestep = np.float64(config["time_step"])
    total_time = np.float64(config["total_time"])
    time_units = Dataset["time"].units

    max_timesteps, ts_remainder = divmod(total_time, timestep)
    if ts_remainder > 0:
        max_timesteps += 1
    max_timesteps = int(max_timesteps + 1)

    z = Dataset["height"][:] #
    dz = Dataset["height_change"][:] #
    depth = Dataset["layer_depth"][:] #

    p = np.full((max_timesteps, Dataset["pressure"].shape[1]), np.nan, dtype = np.float64)
    p[0,:] = Dataset["pressure"][0,:]
    dpdt = np.full((max_timesteps, Dataset["dpdt"].shape[1]), np.nan, dtype = np.float64)
    dpdt[0,:] = Dataset["dpdt"][0,:]
    T = np.full((max_timesteps, Dataset["temperature"].shape[1]), np.nan, dtype = np.float64)
    T[0,:] = Dataset["temperature"][0,:]
    rho = np.full((max_timesteps, Dataset["density"].shape[1]), np.nan, dtype = np.float64)
    rho[0,:] = Dataset["density"][0,:]
    theta = np.full((max_timesteps, Dataset["potential_temperature"].shape[1]), np.nan, dtype = np.float64)
    theta[0,:] = Dataset["potential_temperature"][0,:]
    r_v = np.full((max_timesteps, Dataset["mixing_ratio"].shape[1]), np.nan, dtype = np.float64)
    r_v[0,:] = Dataset["mixing_ratio"][0,:]
    e = np.full((max_timesteps, Dataset["vapor_pressure"].shape[1]), np.nan, dtype = np.float64)
    e[0,:] = Dataset["vapor_pressure"][0,:]
    dr_vdt = np.full((max_timesteps, Dataset["dr_vdt"].shape[1]), np.nan, dtype = np.float64)
    dr_vdt[0,:] = Dataset["dr_vdt"][0,:]
    u_ag = np.full((max_timesteps, Dataset["u-component_of_ageostrophic_wind"].shape[1]), np.nan, dtype = np.float64)
    u_ag[0,:] = Dataset["u-component_of_ageostrophic_wind"][0,:]
    v_ag = np.full((max_timesteps, Dataset["v-component_of_ageostrophic_wind"].shape[1]), np.nan, dtype = np.float64)
    v_ag[0,:] = Dataset["v-component_of_ageostrophic_wind"][0,:]
    u_g = np.full((max_timesteps, Dataset["u-component_of_geostrophic_wind"].shape[1]), np.nan, dtype = np.float64)
    u_g[0,:] = Dataset["u-component_of_geostrophic_wind"][0,:]
    v_g = np.full((max_timesteps, Dataset["v-component_of_geostrophic_wind"].shape[1]), np.nan, dtype = np.float64)
    v_g[0,:] = Dataset["v-component_of_geostrophic_wind"][0,:]
    u_bar = np.full((max_timesteps, Dataset["u-component_of_mean_wind"].shape[1]), np.nan, dtype = np.float64)
    u_bar[0,:] = Dataset["u-component_of_mean_wind"][0,:]
    v_bar = np.full((max_timesteps, Dataset["v-component_of_mean_wind"].shape[1]), np.nan, dtype = np.float64)
    v_bar[0,:] = Dataset["v-component_of_mean_wind"][0,:]
    duagdt = np.full((max_timesteps, Dataset["duagdt"].shape[1]), np.nan, dtype = np.float64)
    duagdt[0,:] = Dataset["duagdt"][0,:]
    dvagdt = np.full((max_timesteps, Dataset["dvagdt"].shape[1]), np.nan, dtype = np.float64)
    dvagdt[0,:] = Dataset["dvagdt"][0,:]
    dugdt = np.full((max_timesteps, Dataset["dugdt"].shape[1]), np.nan, dtype = np.float64)
    dugdt[0,:] = Dataset["dugdt"][0,:]
    dvgdt = np.full((max_timesteps, Dataset["dvgdt"].shape[1]), np.nan, dtype = np.float64)
    dvgdt[0,:] = Dataset["dvgdt"][0,:]
    dudt = np.full((max_timesteps, Dataset["dudt"].shape[1]), np.nan, dtype = np.float64)
    dudt[0,:] = Dataset["dudt"][0,:]
    dvdt = np.full((max_timesteps, Dataset["dvdt"].shape[1]), np.nan, dtype = np.float64)
    dvdt[0,:] = Dataset["dvdt"][0,:]
    dthetadt = np.full((max_timesteps, Dataset["dthetadt"].shape[1]), np.nan, dtype = np.float64)
    dthetadt[0,:] = Dataset["dthetadt"][0,:]
    dTdt = np.full((max_timesteps, Dataset["dTdt"].shape[1]), np.nan, dtype = np.float64)
    dTdt[0,:] = Dataset["dTdt"][0,:]
    z_s = np.full((max_timesteps), np.nan, dtype = np.float64)
    z_s[0] = Dataset["surface_layer_height"][0]
    z_i = np.full((max_timesteps), np.nan, dtype = np.float64)
    z_i[0] = Dataset["pbl_top_height"][0]
    dz_idt = np.full((max_timesteps), np.nan, dtype = np.float64)
    dz_idt[0] = Dataset["dz_idt"][0]
    uw = np.full((max_timesteps, Dataset["u-momentum_flux"].shape[1]), np.nan, dtype = np.float64)
    uw[0,:] = Dataset["u-momentum_flux"][0,:]
    vw = np.full((max_timesteps, Dataset["v-momentum_flux"].shape[1]), np.nan, dtype = np.float64)
    vw[0,:] = Dataset["v-momentum_flux"][0,:]
    thetaw = np.full((max_timesteps, Dataset["potential_temperature_flux"].shape[1]), np.nan, dtype = np.float64)
    thetaw[0,:] = Dataset["potential_temperature_flux"][0,:]
    r_vw = np.full((max_timesteps, Dataset["moisture_flux"].shape[1]), np.nan, dtype = np.float64)
    r_vw[0,:] = Dataset["moisture_flux"][0,:]
    K_m = np.full((max_timesteps, Dataset["eddy_momentum_diffusivity"].shape[1]), np.nan, dtype = np.float64)
    K_m[0,:] = Dataset["eddy_momentum_diffusivity"][0,:]
    K_h = np.full((max_timesteps, Dataset["eddy_heat_diffusivity"].shape[1]), np.nan, dtype = np.float64)
    K_h[0,:] = Dataset["eddy_heat_diffusivity"][0,:]
    puwpz = np.full((max_timesteps, Dataset["differential_u-momentum_flux"].shape[1]), np.nan, dtype = np.float64)
    puwpz[0,:] = Dataset["differential_u-momentum_flux"][0,:]
    pvwpz = np.full((max_timesteps, Dataset["differential_v-momentum_flux"].shape[1]), np.nan, dtype = np.float64)
    pvwpz[0,:] = Dataset["differential_v-momentum_flux"][0,:]
    pthetawpz = np.full((max_timesteps, Dataset["differential_potential_temperature_flux"].shape[1]), np.nan, dtype = np.float64)
    pthetawpz[0,:] = Dataset["differential_potential_temperature_flux"][0,:]
    pr_vwpz = np.full((max_timesteps, Dataset["differential_moisture_flux"].shape[1]), np.nan, dtype = np.float64)
    pr_vwpz[0,:] = Dataset["differential_moisture_flux"][0,:]
    time = np.full((max_timesteps), np.nan, dtype = np.float64)
    time[0] = Dataset["time"][0]
    model_time = np.full((max_timesteps), np.nan, dtype = np.float64)
    model_time[0] = Dataset["model_time"][0]
    shortwave_reduction = np.full((max_timesteps), np.nan, dtype = np.float64)
    shortwave_reduction[0] = Dataset["shortwave_reduction"][0]
    SW_in = np.full((max_timesteps), np.nan, dtype = np.float64)
    SW_in[0] = Dataset["incoming_solar_radiation"][0]
    I = np.full((max_timesteps, Dataset["net_irradiance"].shape[1]), np.nan, dtype = np.float64)
    I[0,:] = Dataset["net_irradiance"][0,:]
    u_star = np.full((max_timesteps), np.nan, dtype = np.float64)
    u_star[0] = Dataset["u_star"][0]
    theta_star = np.full((max_timesteps), np.nan, dtype = np.float64)
    theta_star[0] = Dataset["theta_star"][0]
    r_v_star = np.full((max_timesteps), np.nan, dtype = np.float64)
    r_v_star[0] = Dataset["r_v_star"][0]
    L = np.full((max_timesteps), np.nan, dtype = np.float64)
    L[0] = Dataset["monin_obukhov_length_scale"][0]

    solar_exitance = radiation.spectral_exitance(np.array([5773], ndmin=0), nu[:])

    if "lat" in Dataset.variables.keys():
        lat = Dataset["latitude"][0]
    else:
        lat = config["default_lat"]
    
    if "lon" in Dataset.variables.keys():
        lon = Dataset["longitude"][0]
    else:
        lon = config["default_lon"]

    elapsed_time = 0
    
    z_0 = config["roughness_length"]

    if radiation_schema:
        air_absorbtion_coefficients, _ = radiation.get_absorbtion_coefficient(nu_min, nu_max, ["CO2", "H2O", "O2", "N2", "O3"])
        air_absorbtion_coefficients = [x[::1000] for x in air_absorbtion_coefficients]

    init_duration = (datetime.now(timezone('UTC'))-init_time)
    if progress_tracker:
        print(f"Model successfully initialized after {init_duration.seconds:4.1f} seconds")
        progress(total_time, timestep, current_time_index)
    runtime = datetime.now(timezone("UTC"))

    Dataset.close()

    # Primary Model Loop
    for current_time_index in range(0, max_timesteps - 2):
        # Current timestep details
        elapsed_time += timestep
        current_model_state_time_utc += timedelta(seconds = timestep)
        current_model_state_time_local = current_model_state_time_utc.astimezone(timezone(data_tz))
        current_time_index += 1
        model_time[current_time_index] = model_time[current_time_index-1] + timestep
        time[current_time_index] = date2num(current_model_state_time_utc, time_units)

        # Model layer-based filters
        model_bottom = (z == z[0])
        surface_layer = (z > z[0]) & (z <= z_s[current_time_index-1])
        above_ground = surface_layer & (z > z[1])
        boundary_layer = (z > z_s[current_time_index-1]) & (z <= z_i[current_time_index-1]) & (z < z[-1])
        free_atmosphere = (z > z_i[current_time_index-1]) & (z < z[-1])
        model_top = (z == z[-1])

        # Initializing time-rate-of-change variables for the current time
        dpdt[current_time_index,:] = 0 #"Pa s^(-1)"
        duagdt[current_time_index,:] = 0 #"m s^(-2)"
        dvagdt[current_time_index,:] = 0 #"m s^(-2)"
        dugdt[current_time_index,:] = 0 #"m s^(-2)"
        dvgdt[current_time_index,:] = 0 #"m s^(-2)"
        dudt[current_time_index,:] = 0 #"m s^(-2)"
        dvdt[current_time_index,:] = 0 #"m s^(-2)"
        dthetadt[current_time_index,:] = 0 #"K s^(-1)"
        dTdt[current_time_index,:] = 0 #"K s^(-1)"
        dr_vdt[current_time_index,:] = 0 #"kg kg^(-1) s^(-1)"

        if dynamics_schema != 0:
            if dynamics_schema == 1:
                # Rates of change
                # Pressure
                #dpdt[current_time_index,model_bottom] += Quantity(0, "Pa s^(-1)") # Pressure change isn't actually supported here and now; this is an example.

                # Winds
                duagdt[current_time_index,model_bottom] = duagdt[current_time_index,model_bottom] - (u_ag[current_time_index-1,model_bottom]/timestep) # Momentum AT the ground should be zero, so simply get rid of it. dynamics.Coriolis_u(lat, v_bar[current_time_index-1,j], v_g[current_time_index-1,j]) - Quantity((puwpz[current_time_index,j]), puwpz.units) #
                dvagdt[current_time_index,model_bottom] = dvagdt[current_time_index,model_bottom] - (v_ag[current_time_index-1,model_bottom]/timestep) # dynamics.Coriolis_v(lat, u_bar[current_time_index-1,j], u_g[current_time_index-1,j]) - Quantity((pvwpz[current_time_index,j]), pvwpz.units) #
                duagdt[current_time_index,surface_layer] = duagdt[current_time_index,surface_layer] + dynamics.Coriolis_u(lat, v_ag[current_time_index-1,surface_layer])
                dvagdt[current_time_index,surface_layer] = dvagdt[current_time_index,surface_layer] + dynamics.Coriolis_v(lat, u_ag[current_time_index-1,surface_layer])
                duagdt[current_time_index,boundary_layer] = duagdt[current_time_index,boundary_layer] + dynamics.Coriolis_u(lat, v_ag[current_time_index-1,boundary_layer])
                dvagdt[current_time_index,boundary_layer] = dvagdt[current_time_index,boundary_layer] + dynamics.Coriolis_v(lat, u_ag[current_time_index-1,boundary_layer])
                duagdt[current_time_index,free_atmosphere] = duagdt[current_time_index,free_atmosphere] + dynamics.Coriolis_u(lat, v_ag[current_time_index-1,free_atmosphere])
                dvagdt[current_time_index,free_atmosphere] = dvagdt[current_time_index,free_atmosphere] + dynamics.Coriolis_v(lat, u_ag[current_time_index-1,free_atmosphere])
                duagdt[current_time_index,model_top] = duagdt[current_time_index,model_top] + dynamics.Coriolis_u(lat, v_ag[current_time_index-1,model_top])
                dvagdt[current_time_index,model_top] = dvagdt[current_time_index,model_top] + dynamics.Coriolis_v(lat, u_ag[current_time_index-1,model_top])
        
        if (turbulence_schema != 0) or (radiation_schema != 0):
            theta_z_s = np.interp(z_s[current_time_index-1], z[:], theta[current_time_index-1,:])
            u_z_s = np.interp(z_s[current_time_index-1], z[:], u_bar[current_time_index-1,:])
            v_z_s = np.interp(z_s[current_time_index-1], z[:], v_bar[current_time_index-1,:])
            r_v_s = np.interp(z_s[current_time_index-1], z[:], r_v[current_time_index-1,:])
            u_star[current_time_index], theta_star[current_time_index], r_v_star[current_time_index] = turbulence.stars_alt(theta_z_s, theta[current_time_index-1,0], r_v_s, r_v[current_time_index-1,0], z_s[current_time_index-1], z_0, u_z_s, v_z_s)


        if turbulence_schema != 0:
            # Calculate surface layer-based stars
            #theta_z_s = interpolate.interp1d(z[:], theta[current_time_index-1,:])(z_s[current_time_index-1])
            #u_z_s = interpolate.interp1d(z[:], u_bar[current_time_index-1,:])(z_s[current_time_index-1])
            #v_z_s = interpolate.interp1d(z[:], v_bar[current_time_index-1,:])(z_s[current_time_index-1])
            #r_v_s = interpolate.interp1d(z[:], r_v[current_time_index-1,:])(z_s[current_time_index-1])
            L[current_time_index] = turbulence.Monin_Obukhov_L((theta_z_s + theta[current_time_index-1,0])/2, u_star[current_time_index], theta_star[current_time_index])

            if turbulence_schema == 1:
                # Calculate turbulent fluxes for the current time step
                # Within the surface layer
                uw[current_time_index,surface_layer[1:]], vw[current_time_index,surface_layer[1:]], thetaw[current_time_index,surface_layer[1:]], r_vw[current_time_index,surface_layer[1:]] = turbulence.fluxes_surface_layer_alt(u_star[current_time_index], theta_star[current_time_index], r_v_star[current_time_index], u_z_s, v_z_s)
                
                # Within the boundary layer
                K_m[current_time_index,boundary_layer[1:]] = turbulence.eddy_momentum_diffusivity(theta[current_time_index-1,:], theta[current_time_index-1,0], z[:], z_s[current_time_index-1], z_i[current_time_index-1], u_bar[current_time_index-1,:], v_bar[current_time_index-1,:], z_0, L[current_time_index], u_star[current_time_index], boundary_layer)
                K_h[current_time_index,boundary_layer[1:]] = turbulence.eddy_heat_diffusivity(theta[current_time_index-1,:], theta[current_time_index-1,0], z[:], z_s[current_time_index-1], z_i[current_time_index-1], u_bar[current_time_index-1,:], v_bar[current_time_index-1,:], z_0, L[current_time_index], u_star[current_time_index], boundary_layer)
                uw[current_time_index,boundary_layer[1:]] = - K_m[current_time_index,boundary_layer[1:]] * (u_bar[current_time_index-1,1:][boundary_layer[1:]] - u_bar[current_time_index-1,:-1][boundary_layer[1:]])/(z[1:][boundary_layer[1:]] - z[:-1][boundary_layer[1:]])
                vw[current_time_index,boundary_layer[1:]] = - K_m[current_time_index,boundary_layer[1:]] * (v_bar[current_time_index-1,1:][boundary_layer[1:]] - v_bar[current_time_index-1,:-1][boundary_layer[1:]])/(z[1:][boundary_layer[1:]] - z[:-1][boundary_layer[1:]])
                thetaw[current_time_index,boundary_layer[1:]] = - K_h[current_time_index,boundary_layer[1:]] * (theta[current_time_index-1,1:][boundary_layer[1:]] - theta[current_time_index-1,:-1][boundary_layer[1:]])/(z[1:][boundary_layer[1:]] - z[:-1][boundary_layer[1:]])
                r_vw[current_time_index,boundary_layer[1:]] = - K_h[current_time_index,boundary_layer[1:]] * (r_v[current_time_index-1,1:][boundary_layer[1:]] - r_v[current_time_index-1,:-1][boundary_layer[1:]])/(z[1:][boundary_layer[1:]] - z[:-1][boundary_layer[1:]])
                
                # Within the free atmosphere
                uw[current_time_index,free_atmosphere[1:]] = 0
                vw[current_time_index,free_atmosphere[1:]] = 0
                thetaw[current_time_index,free_atmosphere[1:]] = 0
                r_vw[current_time_index,free_atmosphere[1:]] = 0

                # At the top of the model
                uw[current_time_index,model_top[1:]] = 0
                vw[current_time_index,model_top[1:]] = 0
                thetaw[current_time_index,model_top[1:]] = 0
                r_vw[current_time_index,model_top[1:]] = 0

                # Calculate differential fluxes
                # At the ground
                puwpz[current_time_index,model_bottom] = 0 / dz[model_bottom[:-1]]
                pvwpz[current_time_index,model_bottom] = 0 / dz[model_bottom[:-1]]
                pthetawpz[current_time_index,model_bottom] = 0 / dz[model_bottom[:-1]]
                pr_vwpz[current_time_index,model_bottom] = 0 / dz[model_bottom[:-1]]
                
                # At the model top
                puwpz[current_time_index,model_top] = - uw[current_time_index,model_top[1:]] / (dz[model_top[1:]]/2)
                pvwpz[current_time_index,model_top] = - vw[current_time_index,model_top[1:]] / (dz[model_top[1:]]/2)
                pthetawpz[current_time_index,model_top] = - thetaw[current_time_index,model_top[1:]] / (dz[model_top[1:]]/2)
                pr_vwpz[current_time_index,model_top] = - r_vw[current_time_index,model_top[1:]] / (dz[model_top[1:]]/2)
                
                # Everywhere else
                puwpz[current_time_index,1:-1] = (uw[current_time_index,1:] - uw[current_time_index,:-1]) / dz[1:]
                pvwpz[current_time_index,1:-1] = (vw[current_time_index,1:] - vw[current_time_index,:-1]) / dz[1:]
                pthetawpz[current_time_index,1:-1] = (thetaw[current_time_index,1:] - thetaw[current_time_index,:-1]) / dz[1:]
                pr_vwpz[current_time_index,1:-1] = (r_vw[current_time_index,1:] - r_vw[current_time_index,:-1]) / dz[1:]

                # Rates of change
                # Winds
                duagdt[current_time_index,surface_layer] = duagdt[current_time_index,surface_layer] - puwpz[current_time_index,surface_layer]
                dvagdt[current_time_index,surface_layer] = dvagdt[current_time_index,surface_layer] - pvwpz[current_time_index,surface_layer]
                duagdt[current_time_index,boundary_layer] = duagdt[current_time_index,boundary_layer] - puwpz[current_time_index,boundary_layer]
                dvagdt[current_time_index,boundary_layer] = dvagdt[current_time_index,boundary_layer] - pvwpz[current_time_index,boundary_layer]
                duagdt[current_time_index,free_atmosphere] = duagdt[current_time_index,free_atmosphere] - puwpz[current_time_index,free_atmosphere]
                dvagdt[current_time_index,free_atmosphere] = dvagdt[current_time_index,free_atmosphere] - pvwpz[current_time_index,free_atmosphere]
                duagdt[current_time_index,model_top] = duagdt[current_time_index,model_top] - puwpz[current_time_index,model_top]
                dvagdt[current_time_index,model_top] = dvagdt[current_time_index,model_top] - pvwpz[current_time_index,model_top]

                # Moisture
                if moisture_processes_enabled:
                    dr_vdt[current_time_index,model_bottom] = dr_vdt[current_time_index,model_bottom] - pr_vwpz[current_time_index,model_bottom]
                    dr_vdt[current_time_index,surface_layer] = dr_vdt[current_time_index,surface_layer] - pr_vwpz[current_time_index,surface_layer]
                    dr_vdt[current_time_index,boundary_layer] = dr_vdt[current_time_index,boundary_layer] - pr_vwpz[current_time_index,boundary_layer]
                    dr_vdt[current_time_index,free_atmosphere] = dr_vdt[current_time_index,free_atmosphere] - pr_vwpz[current_time_index,free_atmosphere]
                    dr_vdt[current_time_index,model_top] = dr_vdt[current_time_index,model_top] - pr_vwpz[current_time_index,model_top]

                # Temperature
                dthetadt[current_time_index,surface_layer] = dthetadt[current_time_index,surface_layer] - pthetawpz[current_time_index,surface_layer]
                dthetadt[current_time_index,boundary_layer] = dthetadt[current_time_index,boundary_layer] - pthetawpz[current_time_index,boundary_layer]
                dthetadt[current_time_index,free_atmosphere] = dthetadt[current_time_index,free_atmosphere] - pthetawpz[current_time_index,free_atmosphere]
                dthetadt[current_time_index,model_top] = dthetadt[current_time_index,model_top] - pthetawpz[current_time_index,model_top]


        if radiation_schema!= 0:
            if radiation_schema == 1:
                # Fetch absorbtion coefficients for each atmospheric layer from the included database
                weighted_absorbtion_coefficients = radiation.weighted_absorbtion_coefficients(e[current_time_index-1,:], p[current_time_index-1,:], air_absorbtion_coefficients[0], air_absorbtion_coefficients[1], air_absorbtion_coefficients[2], air_absorbtion_coefficients[3])
                ozone_absorbtion_coeff = radiation.ozone_absorbtion_coefficients(ozone_layer_average_ozone, ozone_layer_average_water_vapor, air_absorbtion_coefficients[0], air_absorbtion_coefficients[1], air_absorbtion_coefficients[2], air_absorbtion_coefficients[3], air_absorbtion_coefficients[4])

                # Radiative transfer
                M_in = radiation.ground_incoming_shortwave(((constants.r_sun**2 * solar_exitance) / constants.D_sun**2), current_model_state_time_local, lat, shortwave_reduction[current_time_index-1])
                SW_in[current_time_index] = radiation.irradiance(M_in)
                I[current_time_index,:] = radiation.irradiance(radiation.point_to_point_exitance(T[current_time_index-1,:], depth[:], soil_emissivity, weighted_absorbtion_coefficients[:,:], nu[:], M_in, ozone_absorbtion_coeff, ozone_layer_temp, ozone_layer_depth))

                # Rates of change
                # Temperature
                dTdt[current_time_index,model_bottom] = dTdt[current_time_index,model_bottom] + (1/(soil_density * soil_heat_capacity * soil_depth)) * (- turbulence.sensible_heat(p[current_time_index-1,1], T[current_time_index-1,1], u_star[current_time_index], theta_star[current_time_index])[0] + I[current_time_index,model_bottom])
                dTdt[current_time_index,1] = dTdt[current_time_index,1] + (1/(rho[current_time_index-1,1] * constants.c_p * depth[1])) * (turbulence.sensible_heat(p[current_time_index-1,1], T[current_time_index-1,1], u_star[current_time_index], theta_star[current_time_index])[0] + I[current_time_index,1])
                dTdt[current_time_index,above_ground] = dTdt[current_time_index,above_ground] + ((1/(rho[current_time_index-1,above_ground] * constants.c_p * depth[above_ground])) * I[current_time_index,above_ground])
                dTdt[current_time_index,boundary_layer] = dTdt[current_time_index,boundary_layer] + ((1/(rho[current_time_index-1,boundary_layer] * constants.c_p * depth[boundary_layer])) * I[current_time_index,boundary_layer])
                dTdt[current_time_index,free_atmosphere] = dTdt[current_time_index,free_atmosphere] + ((1/(rho[current_time_index-1,free_atmosphere] * constants.c_p * depth[free_atmosphere])) * I[current_time_index,free_atmosphere])
                dTdt[current_time_index,model_top] = dTdt[current_time_index,model_top] + ((1/(rho[current_time_index-1,model_top] * constants.c_p * depth[model_top])) * I[current_time_index,model_top])

        # Sum the contributions to the mean wind
        dudt[current_time_index,:] = duagdt[current_time_index,:] + dugdt[current_time_index,:]
        dvdt[current_time_index,:] = dvagdt[current_time_index,:] + dvgdt[current_time_index,:]

        # Calculate the new primary model values
        # Pressure
        p[current_time_index,:] = p[current_time_index-1,:] + timestep * dpdt[current_time_index,:]

        # Winds
        u_ag[current_time_index,:] = u_ag[current_time_index-1,:] + timestep * duagdt[current_time_index,:]
        v_ag[current_time_index,:] = v_ag[current_time_index-1,:] + timestep * dvagdt[current_time_index,:]
        u_g[current_time_index,:] = u_g[current_time_index-1,:] + timestep * dugdt[current_time_index,:]
        v_g[current_time_index,:] = v_g[current_time_index-1,:] + timestep * dvgdt[current_time_index,:]
        u_bar[current_time_index,:] = u_bar[current_time_index-1,:] + timestep * dudt[current_time_index,:]
        v_bar[current_time_index,:] = v_bar[current_time_index-1,:] + timestep * dvdt[current_time_index,:]

        # Moisture
        r_v[current_time_index,:] = r_v[current_time_index-1,:] + timestep * dr_vdt[current_time_index,:]
        e[current_time_index,:] = thermo.e_from_r_v(r_v[current_time_index,:], p[current_time_index,:])

        # Temperature
        theta[current_time_index,:] = theta[current_time_index-1,:] + timestep * dthetadt[current_time_index,:] + thermo.poissons_T_to_theta(timestep * dTdt[current_time_index,:], p[current_time_index-1,:], r_v[current_time_index-1,:])
        T[current_time_index,:] = thermo.poissons_theta_to_T(theta[current_time_index,:], p[current_time_index,:], r_v[current_time_index,:])

        # Density
        rho[current_time_index,:] = p[current_time_index,:]/(constants.R_d * T[current_time_index,:] * (1 + 0.608 * r_v[current_time_index,:]))

        # Cloud Fraction?
        shortwave_reduction[current_time_index] = shortwave_reduction[current_time_index-1]

        # Handle PBL (and surface layer, if enabled) depth changes
        if boundary_layer_height_schema != 0:
            if boundary_layer_height_schema in [1, 2]:
                # Calculate new PBL depth
                dz_idt[current_time_index] = turbulence.boundary_layer_height_tendency(u_star[current_time_index], theta_star[current_time_index], theta[current_time_index-1,:], z[:], theta[current_time_index-1,0], z_s[current_time_index-1], z_i[current_time_index-1], z_0, lat)
                z_i[current_time_index] = z_i[current_time_index-1] + dz_idt[current_time_index] * timestep
                
                # Clip within the model
                if boundary_layer_height_schema != 2:
                    if z_i[current_time_index] <= z_s[current_time_index]:
                        z_i[current_time_index] = z_i[current_time_index] + 1
                
                if z_i[current_time_index] > (z[-1] - 25):
                    z_i[current_time_index] = (z[-1] - 25)
                elif z_i[current_time_index] < 25:
                    z_i[current_time_index] = 25
            
                if boundary_layer_height_schema == 2:
                    # Calculate new surface layer depth
                    z_s[current_time_index] = z_i[current_time_index] / 10

                    # Clip within the model
                    if z_s[current_time_index] > (z[-1] - 25):
                        z_s[current_time_index] = (z[-1] - 25)
                    elif z_s[current_time_index] < 25:
                        z_s[current_time_index] = 25
                else:
                    z_s[current_time_index] = z_s[0]
        else:
            z_i[current_time_index] = z_i[current_time_index-1]


        if progress_tracker:
            progress(total_time, timestep, current_time_index)
    
    Dataset = NCDF(output_filename, "r+", format = "NETCDF4")
    Dataset["pressure"][:,:] = p[:,:]
    Dataset["dpdt"][:,:] = dpdt[:,:]
    Dataset["temperature"][:,:] = T[:,:]
    Dataset["density"][:,:] = rho[:,:]
    Dataset["potential_temperature"][:,:] = theta[:,:]
    Dataset["mixing_ratio"][:,:] = r_v[:,:]
    Dataset["vapor_pressure"][:,:] = e[:,:]
    Dataset["dr_vdt"][:,:] = dr_vdt[:,:]
    Dataset["u-component_of_ageostrophic_wind"][:,:] = u_ag[:,:]
    Dataset["v-component_of_ageostrophic_wind"][:,:] = v_ag[:,:]
    Dataset["u-component_of_geostrophic_wind"][:,:] = u_g[:,:]
    Dataset["v-component_of_geostrophic_wind"][:,:] = v_g[:,:]
    Dataset["u-component_of_mean_wind"][:,:] = u_bar[:,:]
    Dataset["v-component_of_mean_wind"][:,:] = v_bar[:,:]
    Dataset["duagdt"][:,:] = duagdt[:,:]
    Dataset["dvagdt"][:,:] = dvagdt[:,:]
    Dataset["dugdt"][:,:] = dugdt[:,:]
    Dataset["dvgdt"][:,:] = dvgdt[:,:]
    Dataset["dudt"][:,:] = dudt[:,:]
    Dataset["dvdt"][:,:] = dvdt[:,:]
    Dataset["dthetadt"][:,:] = dthetadt[:,:]
    Dataset["dTdt"][:,:] = dTdt[:,:]
    Dataset["surface_layer_height"][:] = z_s[:]
    Dataset["pbl_top_height"][:] = z_i[:]
    Dataset["dz_idt"][:] = dz_idt[:]
    Dataset["u-momentum_flux"][:,:] = uw[:,:]
    Dataset["v-momentum_flux"][:,:] = vw[:,:]
    Dataset["potential_temperature_flux"][:,:] = thetaw[:,:]
    Dataset["moisture_flux"][:,:] = r_vw[:,:]
    Dataset["eddy_momentum_diffusivity"][:,:] = K_m[:,:]
    Dataset["eddy_heat_diffusivity"][:,:] = K_h[:,:]
    Dataset["differential_u-momentum_flux"][:,:] = puwpz[:,:]
    Dataset["differential_v-momentum_flux"][:,:] = pvwpz[:,:]
    Dataset["differential_potential_temperature_flux"][:,:] = pthetawpz[:,:]
    Dataset["differential_moisture_flux"][:,:] = pr_vwpz[:,:]
    Dataset["time"][:] = time[:]
    Dataset["model_time"][:] = model_time[:]
    Dataset["shortwave_reduction"][:] = shortwave_reduction[:]
    Dataset["incoming_solar_radiation"][:] = SW_in[:]
    Dataset["net_irradiance"][:,:] = I[:,:]
    Dataset["u_star"][:] = u_star[:]
    Dataset["theta_star"][:] = theta_star[:]
    Dataset["r_v_star"][:] = r_v_star[:]
    Dataset["monin_obukhov_length_scale"][:] = L[:]
    Dataset.close()
    
    duration = (datetime.now(timezone("UTC")) - runtime).total_seconds()
    print(f"\n   Model run completed in {int(duration // 3600):02d}:{int((duration % 3600) // 60):02d}:{int(duration % 60):02d}.")
    print(f"   Model efficiency: 1:{(total_time/duration):03.1f}")
    return output_filename


if __name__ == "__main__":
    config = get_config()

    if not os.path.exists("./radiation-data/radiation.nc"):
        os.mkdir("./radiation-data")
        new_data = False
        if not os.path.exists("./local_radiation_data"):
            new_data = True
        radiation.create_radiation_database(new_data)

    #dummy_datetime = datetime(2025, 10, 16, 0).replace(tzinfo = timezone("UTC"))
    #dummy_inputs = {"z" : [0, 25, 50, 75], "u_g" : 5, "v_g" : 5, "T" : [280, 282, 284, 286], "p" : [1000, 995, 990, 985], "u_bar" : [0, 2, 4, 6], "v_bar" : [0, 2, 4, 6]}
    #filename = ingest.ingest_1d(config, dummy_datetime, dummy_inputs)

    #input_filename = ingest.ingest_file(config, "https://www.nsstc.uah.edu/~nair/files/ats681/pbl1D_init.csv", input_data_time = datetime(2025, 7, 16, 1, 45).replace(tzinfo = timezone("UTC")), column_names = ['z', 'p', 't'])
    
    #input_filename = ingest.ingest_previous_run("PBL_MODEL_OUTPUT_2025-10-23_08-17-32.nc", "continue", time = Quantity(3.5, "hr"))

    sounding_datetime = datetime(2025, 10, 25, 12)#.replace(tzinfo = timezone("UTC"))
    sounding_station = "72230"
    input_filename = ingest.ingest_sounding(config, sounding_datetime, sounding_station)

    #input_filename = "SABRES-PBL-Model\PBL_MODEL_INPUT_2025-12-23_16-03-18.nc"

    output_filename = run_model(config, input_filename)#, progress_tracker = False)

    #figures.plot_timeseries(output_filename, config)
    figures.plot_radiation(output_filename)
    figures.make_gif(output_filename)

    #figures.plot_timeseries("SABRES-PBL-Model\PBL_MODEL_OUTPUT_2025-11-19_13-39-17.nc")
    #figures.plot_radiation("SABRES-PBL-Model\PBL_MODEL_OUTPUT_2025-11-19_13-39-17.nc")

    #figures.make_gif("D:\CodeProjects\PBLModeling\PBL_MODEL_OUTPUT_2026-02-03_11-45-49.nc")