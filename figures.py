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
from siphon.simplewebservice.wyoming import WyomingUpperAir
import sys
from timezonefinder import TimezoneFinder
from netCDF4 import date2num, num2date
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import imageio.v2 as imageio
import subprocess
from PIL import Image, ImageOps
from utility_equations import _ensure_arrays



def make_gif(filename : str):
    Dataset = NCDF(filename, "r", format = "NETCDF4")

    T = Dataset["temperature"]
    theta = Dataset["potential_temperature"]
    dTdt = Dataset["dTdt"]
    dthetadt = Dataset["dthetadt"]
    z_i = Dataset["pbl_top_height"]
    z_s = Dataset["surface_layer_height"]
    u_star = Dataset["u_star"]
    theta_star = Dataset["theta_star"]
    z_i = Dataset["pbl_top_height"]
    #LW = Dataset["net_longwave_radiation"]

    z = Dataset["height"]
    model_time = Dataset["model_time"]
    time = Dataset["time"]
    dr = os.path.dirname(os.path.realpath(__file__))
    if os.path.exists(f"{dr}/gif-frames/"):
        shutil.rmtree(f"{dr}/gif-frames/")
    
    os.mkdir(f"{dr}/gif-frames/")

    for i in [x.astype(np.int32) for x in np.floor(np.linspace(0, len(model_time), 50, endpoint = False))]:
        fig = plt.figure(1, figsize = (12, 18), dpi = 150)
        fig.subplots_adjust(hspace = 0.4)
        ax1 = fig.add_subplot(521)
        ax2 = fig.add_subplot(522)
        ax4 = fig.add_subplot(523)
        ax5 = fig.add_subplot(524)
        ax6 = fig.add_subplot(5, 2, (5, 6))
        ax7 = fig.add_subplot(5, 2, (7, 8))
        ax8 = fig.add_subplot(5, 2, (9, 10))

        for ax in [ax4, ax5]:
            ax.plot(np.zeros(z[:].size), z[:], color = "grey", linestyle = "dashed")

        ax1.plot(T[i,:], z[:], color = "orange", label = r"$T$")
        ax1.set_xlim(np.nanmin(T[:,:]), np.nanmax(T[:,:]))
        ax1.set_title("Temperature")
        ax1.set_xlabel(r"$T$ (K)")
        ax1.set_ylabel(r"$z$ (m)")
        ax2.plot(theta[i,:], z[:], color = "red", label = r"$\theta$")
        ax2.set_xlabel(r"$\theta$ (K)")
        ax2.set_xlim(np.nanmin(theta[:,:]), np.nanmax(theta[:,:]))
        ax2.set_title("Potential Temperature")
        #ax3.plot(LW[i,:], z[:], color = "salmon", label = r"$F^{net}_{LW}$")
        #ax3.set_xlabel(r"$F^{net}_{LW}$ $\left(\dfrac{W}{m^2}\right)$")
        #ax3.set_xlim(np.nanmin(LW[:,:]), np.nanmax(LW[:,:]))
        #ax3.set_title("Net Longwave Radiation")
        ax4.plot(dTdt[i,:], z[:], color = "green", label = r"$\dfrac{\partial T}{\partial t}$")
        ax4.set_xlim(np.nanmin(dTdt[:,:]), np.nanmax(dTdt[:,:]))
        ax4.set_title("Temperature Tendency")
        ax4.set_xlabel(r"$\dfrac{\partial T}{\partial t}$ $\left(\dfrac{K}{s}\right)$")
        ax4.set_ylabel(r"$z$ (m)")
        ax5.plot(dthetadt[i,:], z[:], color = "blue", label = r"$\dfrac{\partial \theta}{\partial t}$")
        ax5.set_xlim(np.nanmin(dthetadt[:,:]), np.nanmax(dthetadt[:,:]))
        ax5.set_title("Potential Temperature Tendency")
        ax5.set_xlabel(r"$\dfrac{\partial \theta}{\partial t}$ $\left(\dfrac{K}{s}\right)$")

        for ax in [ax1, ax2, ax4, ax5]:
            x_min, x_max = ax.get_xlim()
            ax.hlines([z_s[i], z_i[i]], xmin = x_min, xmax = x_max, colors = ["thistle", "powderblue"], linestyles = ["dotted", "dashed"])

        ax6.plot(model_time[:i], u_star[:i], color = "blue", label = r"$u_\star$")
        ax6.set_title(r"$u_\star$")
        ax6.yaxis.set_major_formatter(FormatStrFormatter('%9.6f'))
        ax7.plot(model_time[:i], theta_star[:i], color = "red", label = r"$\theta_\star$")
        ax7.set_title(r"$\theta_\star$")
        ax7.yaxis.set_major_formatter(FormatStrFormatter('%9.6f'))
        ax8.plot(model_time[:i], z_i[:i], color = "powderblue", label = r"z_i")
        ax8.set_title(r"z_i")

        ax6.vlines([model_time[0]], ymin = np.min(u_star), ymax = np.max(u_star), alpha = [0.0])
        ax7.vlines([model_time[0]], ymin = np.min(theta_star), ymax = np.max(theta_star), alpha = [0.0])
        ax8.vlines([model_time[0]], ymin = z[0], ymax = z[-1], alpha = [0.0])

        for ax in [ax6, ax7, ax8]:
            ax.plot(model_time[:], np.zeros(model_time[:].size), color = "white", alpha = 0.0)
            #ax.vlines([model_time[0]], ymin = )

        #fig.legend()
        fig.suptitle(f"Frame {i}", y = 0.9)
        fig.savefig(f"{dr}/gif-frames/frame{i:06d}.png", dpi = 150, bbox_inches = "tight")
        plt.close()

        with Image.open(f"{dr}/gif-frames/frame{i:06d}.png") as img:
            print(i, img.size)
            padded_img = ImageOps.pad(img, size = (1550,2200), method = Image.Resampling.LANCZOS, color = "white")
        
        padded_img.save(f"{dr}/gif-frames/frame{i:06d}.png")


    img_dir = f"{dr}/gif-frames/"
    images = []
    for file_name in sorted(os.listdir(img_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(img_dir, file_name)
            images.append(imageio.imread(file_path))

    # Make it pause at the end so that the viewers can ponder
    for _ in range(10):
        images.append(imageio.imread(file_path))

    imageio.mimsave(f"{dr}/evolution.gif", images, duration = 5, loop = 65535)

def plot_timeseries(filename : str, config : dict, title : str = "Latest_Output", vertical_line : list = [-1]) -> None:
    Dataset = NCDF(filename, "r", format = "NETCDF4")

    z = Dataset["height"]
    dz = Dataset["height_change"]
    p = Dataset["pressure"]
    dpdt = Dataset["dpdt"]
    T = Dataset["temperature"]
    theta = Dataset["potential_temperature"]
    r_v = Dataset["mixing_ratio"]
    dr_vdt = Dataset["dr_vdt"]
    u_g = Dataset["u-component_of_geostrophic_wind"]
    v_g = Dataset["v-component_of_geostrophic_wind"]
    u_bar = Dataset["u-component_of_mean_wind"]
    v_bar = Dataset["v-component_of_mean_wind"]
    dudt = Dataset["dudt"]
    dvdt = Dataset["dvdt"]
    dthetadt = Dataset["dthetadt"]
    z_s = Dataset["surface_layer_height"]
    z_i = Dataset["pbl_top_height"]
    dz_idt = Dataset["dz_idt"]
    uw = Dataset["u-momentum_flux"]
    vw = Dataset["v-momentum_flux"]
    thetaw = Dataset["potential_temperature_flux"]
    K_m = Dataset["eddy_momentum_diffusivity"]
    K_h = Dataset["eddy_heat_diffusivity"]
    puwpz = Dataset["differential_u-momentum_flux"]
    pvwpz = Dataset["differential_v-momentum_flux"]
    pthetawpz = Dataset["differential_potential_temperature_flux"]
    pr_vwpz = Dataset["differential_moisture_flux"]
    r_vw = Dataset["moisture_flux"]
    data_time = Dataset["time"]
    model_time = Dataset["model_time"]
    shortwave_reduction = Dataset["shortwave_reduction"]
    SW_in = Dataset["incoming_solar_radiation"]
    data_tz = Dataset["timezone"][0]

    if "lat" in Dataset.variables.keys():
        lat = Dataset["lat"]
    else:
        lat = config["default_lat"]
    
    if "lon" in Dataset.variables.keys():
        lon = Dataset["lon"]
    else:
        lon = config["default_lon"]

    local_time = np.array([num2date(x, data_time.units, only_use_cftime_datetimes=False).replace(tzinfo = timezone("UTC")).astimezone(timezone(data_tz)).strftime("%Y-%m-%d %H:%M:%S") for x in data_time[:].data])

    #fig3 = plt.figure(3)
    #ax100 = fig3.add_subplot(111)
    #moisture_sum = np.sum(r_v[:,:], axis = 1)
    #ax100.plot(local_time[100::], moisture_sum[100::], color = "green")
    #fig3.savefig("SABRES-PBL-Model/Latest_Output_Moisture_Sum.png", dpi = 150, bbox_inches = "tight")


    fig = plt.figure(1, figsize = (20, 10))
    #ax1 = fig.add_subplot(4, 3, 1)
    #ax2 = fig.add_subplot(4, 3, 2)
    #ax3 = fig.add_subplot(4, 3, 3)
    #ax4 = fig.add_subplot(4, 3, (4, 6))
    #ax5 = fig.add_subplot(4, 3, (7, 9))
    #ax6 = fig.add_subplot(4, 3, (10, 12))

    ax1 = fig.add_subplot(4, 3, 3)
    ax2 = fig.add_subplot(4, 3, 6)
    ax3 = fig.add_subplot(4, 3, 12)
    ax10 = fig.add_subplot(4, 3, 9)

    ax4 = fig.add_subplot(4, 3, 1)
    ax5 = fig.add_subplot(4, 3, 4)
    ax6 = fig.add_subplot(4, 3, 10)
    ax11 = fig.add_subplot(4, 3, 7)

    ax7 = fig.add_subplot(4, 3, 2)
    ax8 = fig.add_subplot(4, 3, 5)
    ax9 = fig.add_subplot(4, 3, 11)
    ax12 = fig.add_subplot(4, 3, 8)

    #print(Dataset["differential_v-momentum_flux"][:,:])
    #print(Dataset["differential_potential_temperature_flux"][:,:])

    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11, ax12]:
        ax.plot(local_time, z_s[:].data, color = "grey", linestyle = "dotted", linewidth = 0.5, label = "Surface Layer Height", zorder = 3)
        ax.plot(local_time, z_i[:].data, color = "black", linestyle = "dashed", linewidth = 0.5, label = "Boundary Layer Height", zorder = 4)
        for val in vertical_line:
            if val != -1:
                ax.plot(np.ones(z[:].data.shape) * data_time[val].data, z[:].data, color = "cyan", linewidth = 0.5, zorder = 5)

    #ax3.plot(Dataset["time"][:].data, surface_potential_temperature, color = "lime")
    #ax3.plot(Dataset["time"][:].data, Dataset["potential_temperature"][:,1].data, color = "orange")

    u_bar_plot = ax1.pcolormesh(local_time, z[:].data, np.swapaxes(u_bar[:,:], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(u_bar[:,:])), np.abs(np.nanmax(u_bar[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(u_bar[:,:])), np.abs(np.nanmax(u_bar[:,:]))]), cmap = "PuOr", zorder = 1)
    v_bar_plot = ax2.pcolormesh(local_time, z[:].data, np.swapaxes(v_bar[:,:], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(v_bar[:,:])), np.abs(np.nanmax(v_bar[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(v_bar[:,:])), np.abs(np.nanmax(v_bar[:,:]))]), cmap = "PuOr", zorder = 1)
    theta_plot = ax3.pcolormesh(local_time, z[:].data, np.swapaxes(theta[:,:], 0, 1), shading='nearest', vmin = np.nanmin(theta[:,:]), vmax = np.nanmax(theta[:,:]), cmap = "Reds", zorder = 1)
    r_v_plot = ax10.pcolormesh(local_time, z[:].data, np.swapaxes(r_v[:,:], 0, 1), shading='nearest', vmin = 0, vmax = np.nanmax([np.abs(np.nanmin(r_v[:,:])), np.abs(np.nanmax(r_v[:,:]))]), cmap = "YlGn", zorder = 1)

    uw_plot = ax4.pcolormesh(local_time, z[:].data, np.swapaxes(uw[1:,:], 0, 1), shading="flat", vmin = -np.nanmax([np.abs(np.nanmin(uw[1:,:])), np.abs(np.nanmax(uw[1:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(uw[1:,:])), np.abs(np.nanmax(uw[1:,:]))]), cmap = "PuOr", zorder = 1)
    vw_plot = ax5.pcolormesh(local_time, z[:].data, np.swapaxes(vw[1::,:], 0, 1), shading="flat", vmin = -np.nanmax([np.abs(np.nanmin(vw[1:,:])), np.abs(np.nanmax(vw[1:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(vw[1:,:])), np.abs(np.nanmax(vw[1:,:]))]), cmap = "PuOr", zorder = 1)
    thetaw_plot = ax6.pcolormesh(local_time, z[:].data, np.swapaxes(thetaw[1:,:], 0, 1), shading="flat", vmin = -np.nanmax([np.abs(np.nanmin(thetaw[1:,:])), np.abs(np.nanmax(thetaw[1:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(thetaw[1:,:])), np.abs(np.nanmax(thetaw[1:,:]))]), cmap = "bwr", zorder = 1)
    r_vw_plot = ax11.pcolormesh(local_time, z[:].data, np.swapaxes(r_vw[1:,:], 0, 1), shading="flat", vmin = -np.nanmax([np.abs(np.nanmin(r_vw[1:,:])), np.abs(np.nanmax(r_vw[1:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(r_vw[1:,:])), np.abs(np.nanmax(r_vw[1:,:]))]), cmap = "BrBG", zorder = 1)

    puwpz_plot = ax7.pcolormesh(local_time, z[:].data, np.swapaxes(puwpz[:,::], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(puwpz[:,:])), np.abs(np.nanmax(puwpz[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(puwpz[:,:])), np.abs(np.nanmax(puwpz[:,:]))]), cmap = "PuOr", zorder = 1)
    pvwpz_plot = ax8.pcolormesh(local_time, z[:].data, np.swapaxes(pvwpz[:,:], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(pvwpz[:,:])), np.abs(np.nanmax(pvwpz[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(pvwpz[:,:])), np.abs(np.nanmax(pvwpz[:,:]))]), cmap = "PuOr", zorder = 1)
    pthetawpz_plot = ax9.pcolormesh(local_time, z[:].data, np.swapaxes(pthetawpz[:,:], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(pthetawpz[:,:])), np.abs(np.nanmax(pthetawpz[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(pthetawpz[:,:])), np.abs(np.nanmax(pthetawpz[:,:]))]), cmap = "bwr", zorder = 1)
    pr_vwpz_plot = ax12.pcolormesh(local_time, z[:].data, np.swapaxes(pr_vwpz[:,:], 0, 1), shading='nearest', vmin = -np.nanmax([np.abs(np.nanmin(pr_vwpz[:,:])), np.abs(np.nanmax(pr_vwpz[:,:]))]), vmax = np.nanmax([np.abs(np.nanmin(pr_vwpz[:,:])), np.abs(np.nanmax(pr_vwpz[:,:]))]), cmap = "BrBG", zorder = 1)

    tick_ids = np.round(np.linspace(0, len(local_time) - 1, 8)).astype(int)
    
    ax1.set_xticks(local_time[tick_ids])
    ax1.set_xticklabels([])
    ax1.set_title("Output")
    ax2.set_xticks(local_time[tick_ids])
    ax2.set_xticklabels([])
    ax3.set_xticks(local_time[tick_ids])
    ax3.set_xticklabels(local_time[tick_ids], rotation = 80)
    ax10.set_xticks(local_time[tick_ids])
    ax10.set_xticklabels([])

    ax4.set_xticks(local_time[tick_ids])
    ax4.set_xticklabels([])
    ax4.set_title("Fluxes")
    ax4.set_ylabel("z")
    ax5.set_xticks(local_time[tick_ids])
    ax5.set_xticklabels([])
    ax5.set_ylabel("z")
    ax6.set_xticks(local_time[tick_ids])
    ax6.set_xticklabels(local_time[tick_ids], rotation = 80)
    ax6.set_ylabel("z")
    ax11.set_xticks(local_time[tick_ids])
    ax11.set_xticklabels([])
    ax11.set_ylabel("z")

    ax7.set_xticks(local_time[tick_ids])
    ax7.set_xticklabels([])
    ax7.set_title("Differential Fluxes")
    ax8.set_xticks(local_time[tick_ids])
    ax8.set_xticklabels([])
    ax9.set_xticks(local_time[tick_ids])
    ax9.set_xticklabels(local_time[tick_ids], rotation = 80)
    ax12.set_xticks(local_time[tick_ids])
    ax12.set_xticklabels([])

    cbar1 = fig.colorbar(u_bar_plot, pad = 0.01)
    cbar1.set_label(r"$\overline{u}$")
    cbar2 = fig.colorbar(v_bar_plot, pad = 0.01)
    cbar2.set_label(r"$\overline{v}$")
    cbar3 = fig.colorbar(theta_plot, pad = 0.01)
    cbar3.set_label(r"$\overline{\theta}$")
    cbar10 = fig.colorbar(r_v_plot, pad = 0.01)
    cbar10.set_label(r"$\overline{r_v}$")

    cbar4 = fig.colorbar(uw_plot, pad = 0.01)
    cbar4.set_label(r"$u'w'$")
    cbar5 = fig.colorbar(vw_plot, pad = 0.01)
    cbar5.set_label(r"$v'w'$")
    cbar6 = fig.colorbar(thetaw_plot, pad = 0.01)
    cbar6.set_label(r"$\theta' w'$")
    cbar11 = fig.colorbar(r_vw_plot, pad = 0.01)
    cbar11.set_label(r"$r_v' w'$")

    cbar7 = fig.colorbar(puwpz_plot, pad = 0.01)
    cbar7.set_label(r"$\frac{\partial \overline{u'w'}}{\partial z}$")
    cbar8 = fig.colorbar(pvwpz_plot, pad = 0.01)
    cbar8.set_label(r"$\frac{\partial \overline{v'w'}}{\partial z}$")
    cbar9 = fig.colorbar(pthetawpz_plot, pad = 0.01)
    cbar9.set_label(r"$\frac{\partial \overline{\theta' w'}}{\partial z}$")
    cbar12 = fig.colorbar(pr_vwpz_plot, pad = 0.01)
    cbar12.set_label(r"$\frac{\partial \overline{r_v' w'}}{\partial z}$")

    fig.suptitle(f'PBL Model Output from {local_time[0]} to {local_time[-1]}')
    fig.savefig(f"{os.path.dirname(os.path.realpath(__file__))}/{title}.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    Dataset.close()

def plot_radiation(filename : str, title : str = "Latest_Output_Radiation"):
    Dataset = NCDF(filename, "r", format = "NETCDF4")

    z = _ensure_arrays(Dataset["height"][:])
    theta = _ensure_arrays(Dataset["potential_temperature"][:,:])
    u_bar = _ensure_arrays(Dataset["u-component_of_mean_wind"][:,:])
    v_bar = _ensure_arrays(Dataset["v-component_of_mean_wind"][:,:])
    T = _ensure_arrays(Dataset["temperature"][:])
    z_s = _ensure_arrays(Dataset["surface_layer_height"][:])
    z_i = _ensure_arrays(Dataset["pbl_top_height"][:])
    p = _ensure_arrays(Dataset["pressure"][:,:])
    r_v = _ensure_arrays(Dataset["mixing_ratio"][:,:])
    SW = _ensure_arrays(Dataset["incoming_solar_radiation"][:])
    I = _ensure_arrays(Dataset["net_irradiance"][:,:])
    u_star = _ensure_arrays(Dataset["u_star"][:])
    theta_star = _ensure_arrays(Dataset["theta_star"][:])
    data_time = Dataset["time"]
    data_tz = Dataset["timezone"][:]
    z_0 = Quantity(0.1, "m")
    lat = Quantity(33.16, "degree")
    model_time = Dataset["model_time"]
    rho = Quantity(1.2, "kg m^(3)")
    # p[this_ind,1].to_base_units() / (constants.R_d * T_1)
    # rho

    local_time = np.array([num2date(float(x), data_time.units, only_use_cftime_datetimes=False).replace(tzinfo = timezone("UTC")).astimezone(timezone(data_tz)).strftime("%Y-%m-%d %H:%M:%S") if np.isfinite(x) else "" for x in data_time[:].data])

    T_1 = T[:,1]
    H = - (Quantity(p[:,1], "Pa").to_base_units() / (constants.R_d * T_1)) * constants.c_p * u_star[:] * theta_star[:]
    #print((p[:,1].to_base_units() / (constants.R_d * T_1[:])), u_star[:], theta_star[:])
    SW_in = SW[:]

    fig = plt.figure(2, figsize = (12, 8))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()
    ax.plot(model_time[1:]/3600, -H[1:].m, color = "green", label = r"$H$")
    ax.plot(model_time[1:]/3600, SW_in[1:], color = "yellow", label = r"$F^\downarrow_{SW}$")
    ax.plot(model_time[1:]/3600, I[1:,0], color = "orange", label = r"$I^{net}_{LW}$")
    ax.plot(model_time[1:]/3600, -H[1:].m + I[1:,0] + SW_in[1:], color = "black", label = "Sum")
    ax.set_ylabel(r"Surface Energy Contribution ($\frac{W}{m^{2}}$)")
    ax.set_xlabel("Hours Since Model Initialization")

    ax2.plot(model_time[1:]/3600, T[1:,1], color = "cyan", label = r"$T_1$", linestyle = "dashed")
    ax2.plot(model_time[1:]/3600, T[1:,0], color = "blue", label = r"$T_g$", linestyle = "dashed")
    ax2.set_ylabel(r"Temperature ($K$)")

    fig.legend()
    fig.suptitle(f'PBL Model SFC Energy Budget from {local_time[0]} to {local_time[-1]}')
    fig.savefig(f"{os.path.dirname(os.path.realpath(__file__))}/{title}.png", dpi = 150, bbox_inches = "tight")
    plt.close()
    Dataset.close()

if __name__ == "__main__":
    #make_gif("SABRES-PBL-Model\PBL_MODEL_OUTPUT_2025-12-27_23-28-57.nc")

    Dataset = NCDF("SABRES-PBL-Model\PBL_MODEL_INPUT_2026-02-10_12-13-54.nc", "r", format = "NETCDF4")

    for var_name, var in Dataset.variables.items():
        print(f"{var_name} - {var.shape}")