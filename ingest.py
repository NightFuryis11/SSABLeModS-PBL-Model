from netCDF4 import Dataset as NCDF
from pint import Quantity
import dynamics
import constants
from typing import Union
import numpy as np
from scipy import interpolate
from datetime import datetime
import grid
from pytz import timezone
import shutil
import os
import pandas as pd
from siphon.simplewebservice.igra2 import IGRAUpperAir
import sys
from timezonefinder import TimezoneFinder
from utility_equations import _ensure_arrays, hypsometric_Z_to_P
import thermo

def pressure_reference_levels(ref_z : Union[int, float, np.ndarray], z : Union[int, float, np.ndarray], p : Union[int, float, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    ref_z = _ensure_arrays(ref_z)
    z = _ensure_arrays(z)
    p = _ensure_arrays(p)

    ref_p = []
    is_below = []

    for i in range(len(ref_z)):
        level = np.where(z >= ref_z[i])[0][0]
        ref_p.append(p[level])
        if (z[level] > ref_z[i]) and not (level == 0):
            level -= 1
            is_below.append(False)
        elif (z[level] == ref_z[i]):
            is_below.append(False)
        else:
            is_below.append(True)
        
    return ref_p, np.array(is_below)

def ingest_file(config : dict, input_filename : str, input_data_time : datetime = datetime.now(tz = timezone("UTC")), column_names : list = None) -> str:
    input_vals = {}

    if input_filename.endswith(".csv"):
        data = pd.read_csv(input_filename, header = None)
        if column_names is not None:
            data.columns = column_names
    
        for col_name in data.columns:
            if col_name.lower().strip() in ["z", "height", "altitude"]:
                if (data[col_name].values[-1] < 100) and (np.count_nonzero(data[col_name].values[data[col_name].values < 1]) >= 2):
                    input_vals["z"] = Quantity(data[col_name].values, "km")
                else:
                    input_vals["z"] = Quantity(data[col_name].values, "m")
            elif col_name.lower().strip() in ["p", "pressure", "press"]:
                if (data[col_name].values[0] > 1100):
                    input_vals["p"] = Quantity(data[col_name].values, "Pa")
                else:
                    input_vals["p"] = Quantity(data[col_name].values, "hPa")
            elif col_name.lower().strip() in ["temp", "t", "tmpc", "tmpk", "temperature"]:
                if (data[col_name].values[-1] > 100):
                    input_vals["T"] = Quantity(data[col_name].values, "K")
                elif (data[col_name].values[-1] < -100):
                    input_vals["T"] = Quantity(data[col_name].values, "degF")
                else:
                    input_vals["T"] = Quantity(data[col_name].values, "degC")
            elif col_name.lower().strip() in ["dwpt", "dewpoint", "dewpoint_temperature", "td", "t_d"]:
                if (data[col_name].values[-1] > 100):
                    input_vals["T_d"] = Quantity(data[col_name].values, "K")
                elif (data[col_name].values[-1] < -100):
                    input_vals["T_d"] = Quantity(data[col_name].values, "degF")
                else:
                    input_vals["T_d"] = Quantity(data[col_name].values, "degC")
            elif col_name.lower().strip() in ["lat", "latitude"]:
                input_vals["lat"] = data[col_name].values[0]
            elif col_name.lower().strip() in ["lon", "longitude"]:
                input_vals["lon"] = data[col_name].values[0]
        
        if ("z" not in input_vals.keys()) or ("p" not in input_vals.keys()) or ("T" not in input_vals.keys()):
            sys.exit("<ingest> Missing a critical value from file ingest!")
    
    output_filename = ingest_1d(config, input_data_time, input_vals)
    return output_filename

def ingest_sounding(config : dict, input_data_time : datetime, station : str) -> str:
    input_vals = {}

    stations = pd.read_fwf("https://www.ncei.noaa.gov/pub/data/igra/igra2-station-list.txt", header = None, names = ["ID", "lat", "lon", "elevation", "loc", "start_year", "end_year", "num"], widths = [11, 9, 10, 7, 34, 5, 5, 7])
    station_data = stations.loc[stations["ID"].str.contains(station, na=False, case=True)]
    station_row  = station_data.index[0]
    station_id = station_data.loc[station_row, "ID"]
    data, header = IGRAUpperAir.request_data(input_data_time, station_id)
    input_vals["z"] = data["height"].values
    input_vals["p"] = Quantity(data["pressure"].values, data.units["pressure"]).to("Pa").to_base_units().m
    input_vals["T"] = data["temperature"].values + 273.15
    input_vals["T_d"] = data["dewpoint"].values + 273.15

    input_vals["lat"] = header["latitude"].values[0]
    input_vals["lon"] = header["longitude"].values[0]

    mask = ~np.isnan(input_vals["T"])

    input_vals["z"] = input_vals["z"][mask]
    input_vals["p"] = input_vals["p"][mask]
    input_vals["T"] = input_vals["T"][mask]
    input_vals["T_d"] = input_vals["T_d"][mask]

    reorder = np.argsort(input_vals["z"])

    input_vals["z"] = input_vals["z"][reorder]
    input_vals["p"] = input_vals["p"][reorder]
    input_vals["T"] = input_vals["T"][reorder]
    input_vals["T_d"] = input_vals["T_d"][reorder]
    
    output_filename = ingest_1d(config, input_data_time, input_vals)
    return output_filename

def ingest_1d(config : dict, input_data_time : datetime, input_vals : dict) -> str:
    in_z = _ensure_arrays(input_vals["z"])
    in_T = _ensure_arrays(input_vals["T"])
    in_p = _ensure_arrays(input_vals["p"])

    if config["constant_geostrophic_flow"]:
        in_u_g = _ensure_arrays(config["geostrophic_u"])
        in_v_g = _ensure_arrays(config["geostrophic_v"])
    else:
        if "u_g" in input_vals.keys():
            in_u_g = _ensure_arrays(input_vals["u_g"])
        else:
            in_u_g = 0
        if "v_g" in input_vals.keys():
            in_v_g = _ensure_arrays(input_vals["v_g"])
        else:
            in_v_g = 0

    if "lat" in input_vals.keys():
        in_lat = input_vals["lat"]
    else:
        in_lat = config["default_lat"]

    if "lon" in input_vals.keys():
        in_lon = input_vals["lon"]
    else:
        in_lon = config["default_lon"]

    if "u_bar" in input_vals.keys():
        in_u_bar = _ensure_arrays(input_vals["u_bar"])
    else:
        in_u_bar = np.array([0])
    
    if "v_bar" in input_vals.keys():
        in_v_bar = _ensure_arrays(input_vals["v_bar"])
    else:
        in_v_bar = np.array([0])

    if "r_v" in input_vals.keys():
        in_r_v = _ensure_arrays(input_vals["r_v"])
    elif "e" in input_vals.keys():
        in_e = _ensure_arrays(input_vals["e"])
        
        in_r_v = thermo.r_v_from_e(in_e, in_p)
    elif "RH" in input_vals.keys():
        in_RH = _ensure_arrays(input_vals["RH"])
        
        in_r_v = thermo.r_v_from_RH(in_T, in_RH)
    elif "T_d" in input_vals.keys():
        in_T_d = _ensure_arrays(input_vals["T_d"])
        
        in_r_v = thermo.r_v_from_T_d(in_T_d, in_p)
    elif "q_v" in input_vals.keys():
        in_q_v = _ensure_arrays(input_vals["q_v"])
        
        in_r_v = thermo.r_v_from_e(thermo.e_from_q_v(in_q_v, in_p), in_p)
    elif "rho_v" in input_vals.keys():
        in_rho_v = _ensure_arrays(input_vals["rho_v"])
        
        in_r_v = thermo.r_v_from_e(thermo.e_from_rho_v(in_rho_v, in_T), in_p)
    else:
        print("No moisture variable found.")
        in_r_v = Quantity(np.full(in_z.shape, 0), "kg kg^(-1)")

    in_z_s = config["starting_surface_layer_depth"]
    in_z_i = config["starting_boundary_layer_depth"]

    intended_dz = np.minimum(np.full(config["num_heights"]-1, config["min_height_change"]) * config["height_change_exponential_base"]**np.arange(config["num_heights"]-1), config["max_height_change"])
    intended_z = np.append(np.array([0]), intended_dz.cumsum())

    if np.min(in_z):
        in_p_0m = hypsometric_Z_to_P(in_T[0], 0, in_z[0], in_r_v[0], in_p[0], True)
        in_z = np.insert(in_z, 0, 0)
        in_T = np.insert(in_T, 0, in_T[0])
        in_r_v = np.insert(in_r_v, 0, in_r_v[0])
        in_p = np.insert(in_p, 0, in_p_0m)

    intended_z = intended_z[intended_z <= np.max(in_z)]

    # Initialize the grid
    filename = grid.create_grid(datetime.now(timezone("UTC")), input_data_time, len(intended_z), config["min_height_change"], config["max_height_change"], config["height_change_exponential_base"])
    DS = NCDF(filename, "r", format = "NETCDF4")

    # Interpolate values to the desired heights
    interp_z = DS["height"][:].data

    interp_T = interpolate.interp1d(in_z, in_T)(interp_z)
    interp_r_v = interpolate.interp1d(in_z, in_r_v)(interp_z)
    if in_u_g.size == 1:
        interp_u_g = np.full(interp_z.shape, in_u_g)
    else:
        interp_u_g = interpolate.interp1d(in_z, in_u_g)(interp_z)
    if in_v_g.size == 1:
        interp_v_g = np.full(interp_z.shape, in_v_g)
    else:
        interp_v_g = interpolate.interp1d(in_z, in_v_g)(interp_z)
    if in_u_bar.size == 1:
        interp_u_bar = np.full(interp_z.shape, in_u_bar)
    else:
        interp_u_bar = interpolate.interp1d(in_z, in_u_bar)(interp_z)
    if in_v_bar.size == 1:
        interp_v_bar = np.full(interp_z.shape, in_v_bar)
    else:
        interp_v_bar = interpolate.interp1d(in_z, in_v_bar)(interp_z)

    ref_p, is_downward = pressure_reference_levels(interp_z, in_z, in_p)

    interp_p = hypsometric_Z_to_P((interp_T[:-1] + interp_T[1:])/2, interp_z[:-1], interp_z[1:], (interp_r_v[:-1] + interp_r_v[1:])/2, ref_p[0], is_downward)
    interp_p = np.insert(interp_p, 0, ref_p[0])

    interp_theta = thermo.poissons_T_to_theta(interp_T, interp_p, interp_r_v)

    DS.close()
    # Add new variables here for the wind, temp, pressure, etc.
    DS = NCDF(filename, "a", format = "NETCDF4")

    tz_var = DS.createVariable("timezone", str)
    DS["timezone"][0] = TimezoneFinder().timezone_at(lng = in_lon, lat = in_lat)

    lat_var = DS.createVariable("latitude", np.float64)
    lat_var.units = "degrees north"
    DS["latitude"][:] = in_lat

    lon_var = DS.createVariable("longitude", np.float64)
    lon_var.units = "degrees east"
    DS["longitude"][:] = in_lon

    z_s_var = DS.createVariable("surface_layer_height", np.float64, ("time"))
    z_s_var.units = "m"
    DS["surface_layer_height"][0] = in_z_s

    z_i_var = DS.createVariable("pbl_top_height", np.float64, ("time"))
    z_i_var.units = "m"
    DS["pbl_top_height"][0] = in_z_i

    u_star_var = DS.createVariable("u_star", np.float64, ("time"))
    u_star_var.units = "m s^(-1)"
    DS["u_star"][0] = 0

    theta_star_var = DS.createVariable("theta_star", np.float64, ("time"))
    theta_star_var.units = "K"
    DS["theta_star"][0] = 0

    r_v_star_var = DS.createVariable("r_v_star", np.float64, ("time"))
    r_v_star_var.units = "kg kg^(-1)"
    DS["r_v_star"][0] = 0

    sw_in_var = DS.createVariable("incoming_solar_radiation", np.float64, ("time"))
    sw_in_var.units = "W m^(-2)"
    DS["incoming_solar_radiation"][0] = 0

    irr_var = DS.createVariable("net_irradiance", np.float64, ("time", "z"))
    irr_var.units = "W m^(-2)"
    DS["net_irradiance"][0,:] = np.zeros(interp_z.size)[:]

    #M_var = DS.createVariable("transmitting_exitance", np.float64, ("time", "dz"))
    #M_var.units = "W m^(-1)"
    #DS["transmitting_exitance"][0,:] = np.zeros(interp_z.m.size-1)[:]

    L_var = DS.createVariable("monin_obukhov_length_scale", np.float64, ("time"))
    L_var.units = "m"
    DS["monin_obukhov_length_scale"][0] = 0

    p_var = DS.createVariable("pressure", np.float64, ("time", "z"))
    p_var.units = "Pa"
    DS["pressure"][0,:] = interp_p[:]
    
    T_var = DS.createVariable("temperature", np.float64, ("time", "z"))
    T_var.units = "K"
    DS["temperature"][0,:] = interp_T[:]

    r_v_var = DS.createVariable("mixing_ratio", np.float64, ("time", "z"))
    r_v_var.units = "kg kg^(-1)"
    DS["mixing_ratio"][0,:] = interp_r_v[:]

    rho_var = DS.createVariable("density", np.float64, ("time", "z"))
    rho_var.units = "kg m^(-3)"
    DS["density"][0,:] = interp_p[:] / (constants.R_d * interp_T[:] * (1 + 0.608 * interp_r_v[:]))

    r_vs_var = DS.createVariable("saturation_mixing_ratio", np.float64, ("time", "z"))
    r_vs_var.units = "kg kg^(-1)"
    DS["saturation_mixing_ratio"][0,:] = thermo.r_v_from_e(thermo.calc_e_s(interp_T[:]), interp_p[:])

    theta_var = DS.createVariable("potential_temperature", np.float64, ("time", "z"))
    theta_var.units = "K"
    DS["potential_temperature"][0,:] = interp_theta[:]

    sw_red_var = DS.createVariable("shortwave_reduction", np.float64, ("time"))
    DS["shortwave_reduction"][0] = 0

    dpdt_var = DS.createVariable("dpdt", np.float64, ("time", "z"))
    dpdt_var.units = "Pa s^(-1)"
    DS["dpdt"][0,:] = np.zeros(interp_z.size)[:]

    e_var = DS.createVariable("vapor_pressure", np.float64, ("time", "z"))
    e_var.units = "Pa"
    DS["vapor_pressure"][0,:] = thermo.e_from_r_v(interp_r_v[:], interp_p[:])

    u_g_var = DS.createVariable("u-component_of_geostrophic_wind", np.float64, ("time", "z"))
    u_g_var.units = "m s^(-1)"
    DS["u-component_of_geostrophic_wind"][0,:] = interp_u_g[:]

    v_g_var = DS.createVariable("v-component_of_geostrophic_wind", np.float64, ("time", "z"))
    v_g_var.units = "m s^(-1)"
    DS["v-component_of_geostrophic_wind"][0,:] = interp_v_g[:]

    u_bar_var = DS.createVariable("u-component_of_mean_wind", np.float64, ("time", "z"))
    u_bar_var.units = "m s^(-1)"
    DS["u-component_of_mean_wind"][0,:] = interp_u_bar[:]

    v_bar_var = DS.createVariable("v-component_of_mean_wind", np.float64, ("time", "z"))
    v_bar_var.units = "m s^(-1)"
    DS["v-component_of_mean_wind"][0,:] = interp_v_bar[:]

    u_ag_var = DS.createVariable("u-component_of_ageostrophic_wind", np.float64, ("time", "z"))
    u_ag_var.units = "m s^(-1)"
    DS["u-component_of_ageostrophic_wind"][0,:] = interp_u_bar[:] - interp_u_g[:]

    v_ag_var = DS.createVariable("v-component_of_ageostrophic_wind", np.float64, ("time", "z"))
    v_ag_var.units = "m s^(-1)"
    DS["v-component_of_ageostrophic_wind"][0,:] = interp_v_bar[:] - interp_v_g[:]

    duagdt_var = DS.createVariable("duagdt", np.float64, ("time", "z"))
    duagdt_var.units = "m s^(-2)"
    DS["duagdt"][0,:] = np.zeros(interp_z.size)[:]

    dvagdt_var = DS.createVariable("dvagdt", np.float64, ("time", "z"))
    dvagdt_var.units = "m s^(-2)"
    DS["dvagdt"][0,:] = np.zeros(interp_z.size)[:]

    dugdt_var = DS.createVariable("dugdt", np.float64, ("time", "z"))
    dugdt_var.units = "m s^(-2)"
    DS["dugdt"][0,:] = np.zeros(interp_z.size)[:]

    dvgdt_var = DS.createVariable("dvgdt", np.float64, ("time", "z"))
    dvgdt_var.units = "m s^(-2)"
    DS["dvgdt"][0,:] = np.zeros(interp_z.size)[:]

    dudt_var = DS.createVariable("dudt", np.float64, ("time", "z"))
    dudt_var.units = "m s^(-2)"
    DS["dudt"][0,:] = np.zeros(interp_z.size)[:]

    dvdt_var = DS.createVariable("dvdt", np.float64, ("time", "z"))
    dvdt_var.units = "m s^(-2)"
    DS["dvdt"][0,:] = np.zeros(interp_z.size)[:]

    dthetadt_var = DS.createVariable("dthetadt", np.float64, ("time", "z"))
    dthetadt_var.units = "K s^(-1)"
    DS["dthetadt"][0,:] = np.zeros(interp_z.size)[:]

    dTdt_var = DS.createVariable("dTdt", np.float64, ("time", "z"))
    dTdt_var.units = "K s^(-1)"
    DS["dTdt"][0,:] = np.zeros(interp_z.size)[:]

    dr_vdt_var = DS.createVariable("dr_vdt", np.float64, ("time", "z"))
    dr_vdt_var.units = "kg kg^(-1) s^(-1)"
    DS["dr_vdt"][0,:] = np.zeros(interp_z.size)[:]

    uw_var = DS.createVariable("u-momentum_flux", np.float64, ("time", "dz"))
    uw_var.units = "m^(2) s^(-2)"
    DS["u-momentum_flux"][0,:] = np.zeros(interp_z.size-1)[:]

    vw_var = DS.createVariable("v-momentum_flux", np.float64, ("time", "dz"))
    vw_var.units = "m^(2) s^(-2)"
    DS["v-momentum_flux"][0,:] = np.zeros(interp_z.size-1)[:]

    thetaw_var = DS.createVariable("potential_temperature_flux", np.float64, ("time", "dz"))
    thetaw_var.units = "K m s^(-1)"
    DS["potential_temperature_flux"][0,:] = np.zeros(interp_z.size-1)[:]

    r_vw_var = DS.createVariable("moisture_flux", np.float64, ("time", "dz"))
    r_vw_var.units = "kg m kg^(-1) s^(-1)"
    DS["moisture_flux"][0,:] = np.zeros(interp_z.size-1)[:]

    Tw_var = DS.createVariable("temperature_flux", np.float64, ("time", "dz"))
    Tw_var.units = "K m s^(-1)"
    DS["temperature_flux"][0,:] = np.zeros(interp_z.size-1)[:]

    km_var = DS.createVariable("eddy_momentum_diffusivity", np.float64, ("time", "dz"))
    km_var.units = "m^(2) s^(-1)"
    DS["eddy_momentum_diffusivity"][0,:] = np.zeros(interp_z.size-1)[:]

    kh_var = DS.createVariable("eddy_heat_diffusivity", np.float64, ("time", "dz"))
    kh_var.units = "m^(2) s^(-1)"
    DS["eddy_heat_diffusivity"][0,:] = np.zeros(interp_z.size-1)[:]

    puwpz_var = DS.createVariable("differential_u-momentum_flux", np.float64, ("time", "z"))
    puwpz_var.units = "m s^(-2)"
    DS["differential_u-momentum_flux"][0,:] = np.zeros(interp_z.size)[:]

    pvwpz_var = DS.createVariable("differential_v-momentum_flux", np.float64, ("time", "z"))
    pvwpz_var.units = "m s^(-2)"
    DS["differential_v-momentum_flux"][0,:] = np.zeros(interp_z.size)[:]

    pthetawpz_var = DS.createVariable("differential_potential_temperature_flux", np.float64, ("time", "z"))
    pthetawpz_var.units = "K s^(-1)"
    DS["differential_potential_temperature_flux"][0,:] = np.zeros(interp_z.size)[:]

    pr_vwpt_var = DS.createVariable("differential_moisture_flux", np.float64, ("time", "z"))
    pr_vwpt_var.units = "kg kg^(-1) s^(-1)"
    DS["differential_moisture_flux"][0,:] = np.zeros(interp_z.size)[:]

    DS.close()
    print(f"Successfully ingested data to {filename}")
    return filename


def ingest_previous_run(filename : str, mode : str, **kwargs) -> str:
    new_filename = filename.split(".")[0] + "_REINPUT.nc" 
    shutil.copy(filename, new_filename)

    Dataset = NCDF(new_filename, "r+", format = "NETCDF4")

    if "step" in kwargs.keys():
        sl = kwargs["step"]
    if "time" in kwargs.keys():
        timestep = Quantity(Dataset["time"][1] - Dataset["time"][0], "hr")
        sl = int(np.floor(kwargs["time"].to("s").m/timestep.to("s").m))

    
    if mode == "continue":
        Dataset["pressure"][sl:,:] = np.nan
        Dataset["dpdt"][sl:,:] = np.nan
        Dataset["temperature"][sl:,:] = np.nan
        Dataset["potential_temperature"][sl:,:] = np.nan
        Dataset["surface_temperature"][sl:] = np.nan
        Dataset["dTdt"][sl:] = np.nan
        Dataset["mixing_ratio"][sl:,:] = np.nan
        Dataset["dr_vdt"][sl:,:] = np.nan
        Dataset["u-component_of_geostrophic_wind"][sl:] = np.nan
        Dataset["v-component_of_geostrophic_wind"][sl:] = np.nan
        Dataset["u-component_of_mean_wind"][sl:,:] = np.nan
        Dataset["v-component_of_mean_wind"][sl:,:] = np.nan
        Dataset["dudt"][sl:,:] = np.nan
        Dataset["dvdt"][sl:,:] = np.nan
        Dataset["dthetadt"][sl:,:] = np.nan
        Dataset["surface_layer_height"][sl:] = np.nan
        Dataset["pbl_top_height"][sl:] = np.nan
        Dataset["dz_idt"][sl:] = np.nan
        Dataset["u-momentum_flux"][sl:,:] = np.nan
        Dataset["v-momentum_flux"][sl:,:] = np.nan
        Dataset["potential_temperature_flux"][sl:,:] = np.nan
        Dataset["eddy_momentum_diffusivity"][sl:,:] = np.nan
        Dataset["eddy_heat_diffusivity"][sl:,:] = np.nan
        Dataset["differential_u-momentum_flux"][sl:,:] = np.nan
        Dataset["differential_v-momentum_flux"][sl:,:] = np.nan
        Dataset["differential_potential_temperature_flux"][sl:,:] = np.nan
        Dataset["time"][sl:] = np.nan
        Dataset["local_time"][sl:] = np.nan
        Dataset["model_time"][sl:] = np.nan
        Dataset["shortwave_reduction"][sl:] = np.nan
        Dataset["incoming_solar_radiation"][sl:] = np.nan
    
    if mode == "init":
        p = Dataset["pressure"][sl,:]
        dpdt = Dataset["dpdt"][sl,:]
        T = Dataset["temperature"][sl,:]
        theta = Dataset["potential_temperature"][sl,:]
        T_g = Dataset["surface_temperature"][sl]
        dTdt = Dataset["dTdt"][sl]
        r_v = Dataset["mixing_ratio"][sl,:]
        dr_vdt = Dataset["dr_vdt"][sl,:]
        u_g = Dataset["u-component_of_geostrophic_wind"][sl]
        v_g = Dataset["v-component_of_geostrophic_wind"][sl]
        u_bar = Dataset["u-component_of_mean_wind"][sl,:]
        v_bar = Dataset["v-component_of_mean_wind"][sl,:]
        dudt = Dataset["dudt"][sl,:]
        dvdt = Dataset["dvdt"][sl,:]
        dthetadt = Dataset["dthetadt"][sl,:]
        z_s = Dataset["surface_layer_height"][sl]
        z_i = Dataset["pbl_top_height"][sl]
        dz_idt = Dataset["dz_idt"][sl]
        uw = Dataset["u-momentum_flux"][sl,:]
        vw = Dataset["v-momentum_flux"][sl,:]
        thetaw = Dataset["potential_temperature_flux"][sl,:]
        K_m = Dataset["eddy_momentum_diffusivity"][sl,:]
        K_h = Dataset["eddy_heat_diffusivity"][sl,:]
        puwpz = Dataset["differential_u-momentum_flux"][sl,:]
        pvwpz = Dataset["differential_v-momentum_flux"][sl,:]
        pthetawpz = Dataset["differential_potential_temperature_flux"][sl,:]
        t = Dataset["time"][sl]
        t_l = Dataset["local_time"][sl]
        t_m = Dataset["model_time"][sl]
        alpha = Dataset["shortwave_reduction"][sl]
        SW_in = Dataset["incoming_solar_radiation"][sl]

        Dataset["pressure"][:,:] = np.nan
        Dataset["dpdt"][:,:] = np.nan
        Dataset["temperature"][:,:] = np.nan
        Dataset["potential_temperature"][:,:] = np.nan
        Dataset["surface_temperature"][:] = np.nan
        Dataset["dTdt"][:] = np.nan
        Dataset["mixing_ratio"][:,:] = np.nan
        Dataset["dr_vdt"][:,:] = np.nan
        Dataset["u-component_of_geostrophic_wind"][:] = np.nan
        Dataset["v-component_of_geostrophic_wind"][:] = np.nan
        Dataset["u-component_of_mean_wind"][:,:] = np.nan
        Dataset["v-component_of_mean_wind"][:,:] = np.nan
        Dataset["dudt"][:,:] = np.nan
        Dataset["dvdt"][:,:] = np.nan
        Dataset["dthetadt"][:,:] = np.nan
        Dataset["surface_layer_height"][:] = np.nan
        Dataset["pbl_top_height"][:] = np.nan
        Dataset["dz_idt"][:] = np.nan
        Dataset["u-momentum_flux"][:,:] = np.nan
        Dataset["v-momentum_flux"][:,:] = np.nan
        Dataset["potential_temperature_flux"][:,:] = np.nan
        Dataset["eddy_momentum_diffusivity"][:,:] = np.nan
        Dataset["eddy_heat_diffusivity"][:,:] = np.nan
        Dataset["differential_u-momentum_flux"][:,:] = np.nan
        Dataset["differential_v-momentum_flux"][:,:] = np.nan
        Dataset["differential_potential_temperature_flux"][:,:] = np.nan
        Dataset["time"][:] = np.nan
        Dataset["local_time"][:] = np.nan
        Dataset["model_time"][:] = np.nan
        Dataset["shortwave_reduction"][:] = np.nan
        Dataset["incoming_solar_radiation"][:] = np.nan
    
        Dataset["pressure"][0,:] = p
        Dataset["dpdt"][0,:] = dpdt
        Dataset["temperature"][0,:] = T
        Dataset["potential_temperature"][0,:] = theta
        Dataset["surface_temperature"][0] = T_g
        Dataset["dTdt"][0] = dTdt
        Dataset["mixing_ratio"][0,:] = r_v
        Dataset["dr_vdt"][0,:] = dr_vdt
        Dataset["u-component_of_geostrophic_wind"][0] = u_g
        Dataset["v-component_of_geostrophic_wind"][0] = v_g
        Dataset["u-component_of_mean_wind"][0,:] = u_bar
        Dataset["v-component_of_mean_wind"][0,:] = v_bar
        Dataset["dudt"][0,:] = dudt
        Dataset["dvdt"][0,:] = dvdt
        Dataset["dthetadt"][0,:] = dthetadt
        Dataset["surface_layer_height"][0] = z_s
        Dataset["pbl_top_height"][0] = z_i
        Dataset["dz_idt"][0] = dz_idt
        Dataset["u-momentum_flux"][0,:] = uw
        Dataset["v-momentum_flux"][0,:] = vw
        Dataset["potential_temperature_flux"][0,:] = thetaw
        Dataset["eddy_momentum_diffusivity"][0,:] = K_m
        Dataset["eddy_heat_diffusivity"][0,:] = K_h
        Dataset["differential_u-momentum_flux"][0,:] = puwpz
        Dataset["differential_v-momentum_flux"][0,:] = pvwpz
        Dataset["differential_potential_temperature_flux"][0,:] = pthetawpz
        Dataset["time"][0] = t
        Dataset["local_time"][0] = t_l
        Dataset["model_time"][0] = t_m
        Dataset["shortwave_reduction"][0] = alpha
        Dataset["incoming_solar_radiation"][0] = SW_in
    
    Dataset.close()
    return new_filename