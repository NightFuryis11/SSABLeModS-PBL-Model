from netCDF4 import Dataset as NCDF
from netCDF4 import date2num
from datetime import datetime
from pytz import timezone
from timezonefinder import TimezoneFinder
import numpy as np
import os
from typing import Union
from pint import Quantity

def create_grid(runtime : datetime, input_data_time : datetime, num_heights : int = 160, min_height_change : Union[int, float] = 50, max_height_change : Union[int, float] = 50, height_change_exponential_base : float = 1.0, ) -> str:
    timestamp = runtime.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{os.path.dirname(os.path.realpath(__file__))}/PBL_MODEL_INPUT_{timestamp}.nc"
    DS = NCDF(filename, "w", format='NETCDF4')

    time_dim = DS.createDimension("time", None) # Forward-in-time model progression dimension
    z_dim = DS.createDimension("z", num_heights) # Vertical spatial dimension that model parameters are calculated on
    dz_dim = DS.createDimension("dz", num_heights-1) # Vertical layers that fluxes are calculated over

    DS.title = "1-D PBL Model Output"
    DS.description = "1-dimensional single-column-model output from Sam Bailey's modified planetary boundary layer model."
    DS.version = "1.0.0"

    ### Calculate grid base parameters ###

    dz = np.minimum(np.full(num_heights-1, min_height_change) * height_change_exponential_base**np.arange(num_heights-1), max_height_change)
    z = np.append(np.array([0]), dz.cumsum())

    ### Construct the variables which handle the dimensions' base values ###

    time_var = DS.createVariable("time", np.float64, ("time"))
    time_var.units = "hours since 1970-01-01T00:00:00"
    DS["time"][0] = date2num(input_data_time, DS["time"].units)

    model_time_var = DS.createVariable("model_time", np.float64, ("time"))
    model_time_var.units = "seconds since model initialization"
    DS["model_time"][0] = 0

    z_var = DS.createVariable("height", np.float32, ("z"))
    z_var.units = "m"
    DS["height"][:] = z[:]

    dzi_var = DS.createVariable("dz_idt", np.float32, ("time"))
    dzi_var.units = "m s^(-1)"

    z_layer_var = DS.createVariable("height_layer", np.float32, ("z"))
    DS["height_layer"][:] = np.arange(len(z[:]))

    z_depth_var = DS.createVariable("layer_depth", np.float32, ("z"))
    z_depth_var.units = "m"
    DS["layer_depth"][1:-1] = (z[2:] - z[:-2]) / 2
    DS["layer_depth"][0] = (z[1] - z[0]) / 2
    DS["layer_depth"][-1] = (z[-1] - z[-2])

    dz_var = DS.createVariable("height_change", np.float32, ("dz"))
    dz_var.units = "m"
    DS["height_change"][:] = dz[:]

    mdz_var = DS.createVariable("midpoint_height_change", np.float32, ("dz"))
    mdz_var.units = "m"
    DS["midpoint_height_change"][:] = z[1:] - z[:-1]

    dz_layer_var = DS.createVariable("height_change_layer", np.float32, ("dz"))
    DS["height_change_layer"][:] = np.arange(len(dz[:]))

    DS.close()

    return filename



if __name__ == "__main__":
    filename = create_grid(datetime.now(timezone("UTC")))
    DS = NCDF(filename, "r", format='NETCDF4')
    print(DS["height"])
    print(DS["height_change"])
    DS.close()
    os.remove(filename)